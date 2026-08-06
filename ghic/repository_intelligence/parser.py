"""Repository walking, metadata detection, and structural chunking.

Chunking is the part that decides whether retrieval works at all. Fixed-size
windows are cheap and wrong: they cut mid-function, so a retrieved chunk
often contains the tail of one function and the head of another, which
reads as noise to both the ranker and the LLM. Everything here splits on
the language's own declaration boundaries instead:

  - Python   -> the `ast` module (stdlib, exact)
  - Markdown -> heading hierarchy
  - C-family (JS/TS/Go/Rust/Java/C#) -> a declaration-regex + brace-matching
    scanner (see `_chunk_brace_language`)

The C-family scanner is a heuristic, not a parser, and is documented as
such: it finds declaration lines and then tracks brace depth while skipping
string and comment content. It is deliberately not tree-sitter -- that's a
compiled dependency per grammar, and this project's deployment target
(Vercel, ~295MB of a 500MB bundle already used) can't absorb it. When the
scanner finds no declarations it degrades to whole-file chunks rather than
producing garbage, so a language it reads badly is *less useful*, never
*wrong*.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

from .. import utils
from .config import (
    DEPENDENCY_MANIFESTS,
    ENTRY_POINT_CANDIDATES,
    EXCLUDED_DIRECTORIES,
    EXCLUDED_FILE_SUFFIXES,
    EXCLUDED_FILENAMES,
    FRAMEWORK_MARKERS,
    LANGUAGE_BY_EXTENSION,
    RepositoryIntelligenceConfig,
    is_sensitive_path,
    redact_secrets,
)
from .models import CodeChunk, RepositoryMetadata

logger = utils.get_logger(__name__)

_MARKDOWN_LANGUAGES = {"Markdown", "reStructuredText"}
_PYTHON = "Python"

# A named declaration only has to be non-empty to be worth indexing; see
# _is_substantive().
_MIN_NAMED_CHUNK_CHARS = 12

# Declaration patterns per C-family language. Each must capture the symbol
# name in group "name". Anchored at line start (allowing indentation) so a
# call like `foo(function(){})` mid-expression isn't mistaken for one.
_DECLARATION_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "JavaScript": (
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>\w+)"),
        re.compile(r"^\s*(?:export\s+)?class\s+(?P<name>\w+)"),
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*=\s*"
            r"(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|\w+\s*=>)"
        ),
        re.compile(r"^\s{2,}(?:async\s+)?(?P<name>\w+)\s*\([^;]*\)\s*\{"),   # class method
    ),
    "Go": (
        re.compile(r"^func\s+(?:\([^)]*\)\s*)?(?P<name>\w+)"),
        re.compile(r"^type\s+(?P<name>\w+)\s+(?:struct|interface)"),
    ),
    "Rust": (
        re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(?P<name>\w+)"),
        re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+(?P<name>\w+)"),
        re.compile(r"^\s*impl(?:<[^>]*>)?\s+(?P<name>[\w:<>]+)"),
    ),
    "Java": (
        re.compile(r"^\s*(?:public|private|protected|abstract|final|static|\s)*"
                   r"(?:class|interface|enum|record)\s+(?P<name>\w+)"),
        re.compile(r"^\s+(?:public|private|protected|static|final|synchronized|abstract|\s)+"
                   r"[\w<>\[\],.?]+\s+(?P<name>\w+)\s*\([^;]*\)\s*(?:throws [\w,\s.]+)?\{"),
    ),
    "C#": (
        re.compile(r"^\s*(?:public|private|protected|internal|abstract|sealed|static|partial|\s)*"
                   r"(?:class|interface|struct|record|enum)\s+(?P<name>\w+)"),
        re.compile(r"^\s+(?:public|private|protected|internal|static|virtual|override|async|\s)+"
                   r"[\w<>\[\],.?]+\s+(?P<name>\w+)\s*\([^;]*\)\s*\{"),
    ),
}
_DECLARATION_PATTERNS["TypeScript"] = _DECLARATION_PATTERNS["JavaScript"] + (
    re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?(?:interface|type|enum)\s+(?P<name>\w+)"),
)


# ---------------------------------------------------------------------------
# Walking
# ---------------------------------------------------------------------------
def is_indexable(path: Path, root: Path, cfg: RepositoryIntelligenceConfig) -> bool:
    """Cheap, side-effect-free filter used by both the walker and its tests."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if any(part in EXCLUDED_DIRECTORIES for part in relative.parts[:-1]):
        return False
    if path.name in EXCLUDED_FILENAMES:
        return False
    # Checked before the extension allowlist, not after: a credential file
    # must be excluded because of what it is, not because its extension
    # happens to be absent from the allowlist today.
    if is_sensitive_path(relative.as_posix()):
        return False
    name_lower = path.name.lower()
    if any(name_lower.endswith(suffix.lower()) for suffix in EXCLUDED_FILE_SUFFIXES):
        return False
    if path.suffix.lower() not in LANGUAGE_BY_EXTENSION:
        return False
    try:
        if path.stat().st_size > cfg.max_file_bytes:
            return False
    except OSError:
        return False
    return True


def _git_tracked_files(root: Path) -> list[Path] | None:
    """Tracked files via `git ls-files`, or None if this isn't a git repo.

    Strongly preferred over walking the filesystem: it respects .gitignore
    for free, which is the difference between indexing a repository and
    indexing whatever happens to be sitting in someone's working tree
    (build output, a nested unrelated project, a local scratch directory).
    Found the hard way -- the first smoke test of this engine indexed a
    gitignored sibling project and returned its files as "evidence".
    """
    if not (root / ".git").exists():
        return None
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--exclude-standard"],
            cwd=str(root), capture_output=True, text=True, timeout=60, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    return [root / name for name in completed.stdout.split("\0") if name]


def _priority(path: Path, root: Path) -> tuple[int, int, str]:
    """Sort key deciding what survives the `max_files` cap.

    A flat alphabetical truncation is systematically wrong: on a repository
    larger than the cap it keeps whatever sorts early and silently drops
    everything else, so a repo with an `_archive/` directory indexes the
    archive and not the source. Ranking instead means the cap degrades by
    *relevance* -- real source first, then docs, then tests and examples,
    shallower paths before deeper ones.
    """
    relative = path.relative_to(root)
    parts = [p.lower() for p in relative.parts]
    suffix = path.suffix.lower()
    language = LANGUAGE_BY_EXTENSION.get(suffix, "")

    is_test = any(
        p.startswith("test") or p.endswith(("_test", "-test", ".test", ".spec"))
        or p in {"tests", "spec", "specs", "examples", "example", "samples", "benchmarks"}
        for p in parts
    )
    is_archive = any(p in {"_archive", "archive", "deprecated", "legacy", "attic"} for p in parts)

    if is_archive:
        tier = 4
    elif is_test:
        tier = 3
    elif language in _MARKDOWN_LANGUAGES:
        # A root README is top-tier context; a deep docs page is not.
        tier = 0 if len(parts) == 1 else 2
    else:
        tier = 1
    return (tier, len(parts), relative.as_posix())


def walk_repository(root: Path, cfg: RepositoryIntelligenceConfig) -> list[Path]:
    """Indexable files under `root`, deterministically ordered.

    Prefers git's own file list (see `_git_tracked_files`) and falls back to
    a filesystem walk for a plain directory. When the result exceeds
    `max_files` it is ranked by `_priority` before truncation, then re-sorted
    by path so the surviving set is stable and diffable regardless of which
    files were cut.
    """
    candidates = _git_tracked_files(root)
    if candidates is None:
        candidates = list(root.rglob("*"))

    found = [p for p in candidates if p.is_file() and is_indexable(p, root, cfg)]

    if len(found) > cfg.max_files:
        logger.warning(
            "%s has %d indexable files, above max_files=%d; keeping the "
            "highest-priority %d (source before docs, tests and archives last)",
            root, len(found), cfg.max_files, cfg.max_files,
        )
        found.sort(key=lambda p: _priority(p, root))
        found = found[:cfg.max_files]

    return sorted(found)


def read_text(path: Path) -> str:
    """UTF-8 with replacement -- a repository is untrusted input and a stray
    latin-1 byte in one file must not abort an entire indexing run."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.debug("unreadable file %s: %s", path, e)
        return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_file(
    repo: str, relative_path: str, text: str, cfg: RepositoryIntelligenceConfig
) -> list[CodeChunk]:
    """Structural chunks for one file, dispatched by language."""
    language = LANGUAGE_BY_EXTENSION.get(Path(relative_path).suffix.lower(), "")
    if not text.strip() or not language:
        return []

    # Redact before chunking, so a secret is never embedded, never stored
    # in the vector database, and never quotable as evidence -- there is no
    # later stage where it could be stripped, because by then it exists in
    # an index and possibly in an embedding provider's logs.
    text = redact_secrets(text)

    if language == _PYTHON:
        chunks = _chunk_python(repo, relative_path, text, language, cfg)
    elif language in _MARKDOWN_LANGUAGES:
        chunks = _chunk_markdown(repo, relative_path, text, language, cfg)
    elif language in _DECLARATION_PATTERNS:
        chunks = _chunk_brace_language(repo, relative_path, text, language, cfg)
    else:
        chunks = []

    if not chunks:
        chunks = _whole_file_chunks(repo, relative_path, text, language, cfg)
    return [c for c in chunks if _is_substantive(c, cfg)]


def _is_substantive(chunk: CodeChunk, cfg: RepositoryIntelligenceConfig) -> bool:
    """Filter trivia without discarding short *named* declarations.

    `min_chunk_chars` exists to drop noise: a stray brace, a one-line
    module fragment, a blank section. Applying it uniformly also dropped
    things that are short but highly retrievable -- a three-line
    `decode_utf8()` method, an `## Installation` section with one sentence
    -- because a chunk's value comes from its *name* as much as its body,
    and the name is what an issue reporter tends to type. So a chunk with a
    symbol only has to clear a much lower bar.
    """
    body = chunk.text.strip()
    if chunk.symbol:
        return len(body) >= _MIN_NAMED_CHUNK_CHARS
    return len(body) >= cfg.min_chunk_chars


def _lines_slice(lines: list[str], start: int, end: int) -> str:
    """1-indexed inclusive line span -> text."""
    return "\n".join(lines[start - 1:end])


def _split_oversized(
    repo: str, path: str, language: str, text: str, start_line: int,
    kind: str, symbol: str, parent_symbol: str, cfg: RepositoryIntelligenceConfig,
) -> list[CodeChunk]:
    """A single declaration bigger than max_chunk_chars gets split on line
    boundaries. Rare (a 6000-char function is already a code smell), but a
    god-function must not silently blow the embedding provider's token
    limit."""
    lines = text.split("\n")
    if len(text) <= cfg.max_chunk_chars:
        return [CodeChunk(
            repo=repo, path=path, language=language, text=text,
            start_line=start_line, end_line=start_line + len(lines) - 1,
            kind=kind, symbol=symbol, parent_symbol=parent_symbol,
        )]

    out: list[CodeChunk] = []
    buffer: list[str] = []
    buffer_start = start_line
    size = 0
    for offset, line in enumerate(lines):
        if buffer and size + len(line) + 1 > cfg.max_chunk_chars:
            out.append(CodeChunk(
                repo=repo, path=path, language=language, text="\n".join(buffer),
                start_line=buffer_start, end_line=start_line + offset - 1,
                kind=kind, symbol=symbol, parent_symbol=parent_symbol,
            ))
            buffer, size, buffer_start = [], 0, start_line + offset
        buffer.append(line)
        size += len(line) + 1
    if buffer:
        out.append(CodeChunk(
            repo=repo, path=path, language=language, text="\n".join(buffer),
            start_line=buffer_start, end_line=start_line + len(lines) - 1,
            kind=kind, symbol=symbol, parent_symbol=parent_symbol,
        ))
    return out


def _whole_file_chunks(
    repo: str, path: str, text: str, language: str, cfg: RepositoryIntelligenceConfig
) -> list[CodeChunk]:
    return _split_oversized(repo, path, language, text.strip("\n"), 1, "module", "", "", cfg)


def _chunk_python(
    repo: str, path: str, text: str, language: str, cfg: RepositoryIntelligenceConfig
) -> list[CodeChunk]:
    """AST-based: exact function/method/class boundaries, no heuristics.

    Module-level code outside any declaration (imports, constants, a
    `if __name__` block) becomes its own chunk -- that's where a
    configuration constant an issue is really about tends to live.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        logger.debug("python parse failed for %s (%s); falling back to whole-file", path, e)
        return []

    lines = text.split("\n")
    chunks: list[CodeChunk] = []
    covered: set[int] = set()

    def emit(node: ast.AST, kind: str, symbol: str, parent: str) -> None:
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", 0) or start
        # Pull the decorators in with the declaration they decorate: a
        # @router.post("/webhook") line is often the most retrievable text
        # in the whole function.
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start = min(start, min(d.lineno for d in decorators))
        if not start or end < start:
            return
        covered.update(range(start, end + 1))
        chunks.extend(_split_oversized(
            repo, path, language, _lines_slice(lines, start, end), start,
            kind, symbol, parent, cfg,
        ))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            emit(node, "function", node.name, "")
        elif isinstance(node, ast.ClassDef):
            methods = [
                child for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                # The class header (docstring, class-level attributes) is
                # its own chunk; each method is separate. A 40-method class
                # embedded as one blob retrieves for everything and
                # discriminates nothing.
                header_end = min(m.lineno for m in methods) - 1
                for dec in (getattr(methods[0], "decorator_list", []) or []):
                    header_end = min(header_end, dec.lineno - 1)
                if header_end >= node.lineno:
                    covered.update(range(node.lineno, header_end + 1))
                    chunks.extend(_split_oversized(
                        repo, path, language, _lines_slice(lines, node.lineno, header_end),
                        node.lineno, "class", node.name, "", cfg,
                    ))
                for method in methods:
                    emit(method, "method", method.name, node.name)
            else:
                emit(node, "class", node.name, "")

    remainder = sorted(set(range(1, len(lines) + 1)) - covered)
    for start, end in _contiguous_spans(remainder):
        body = _lines_slice(lines, start, end)
        if body.strip():
            chunks.extend(_split_oversized(
                repo, path, language, body, start, "module", "", "", cfg
            ))
    return sorted(chunks, key=lambda c: c.start_line)


def _contiguous_spans(numbers: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for n in numbers:
        if spans and n == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], n)
        else:
            spans.append((n, n))
    return spans


def _chunk_markdown(
    repo: str, path: str, text: str, language: str, cfg: RepositoryIntelligenceConfig
) -> list[CodeChunk]:
    """Split on ATX headings; each section keeps its own heading as the symbol.

    Fenced code blocks are tracked so a `# comment` inside a ```bash fence
    isn't mistaken for a heading -- a mistake that shreds exactly the
    install/usage sections most worth retrieving.
    """
    lines = text.split("\n")
    chunks: list[CodeChunk] = []
    section_start = 1
    heading = ""
    in_fence = False
    fence_marker = ""

    def flush(end_line: int) -> None:
        if end_line < section_start:
            return
        body = _lines_slice(lines, section_start, end_line)
        if body.strip():
            chunks.extend(_split_oversized(
                repo, path, language, body, section_start, "section", heading, "", cfg
            ))

    for index, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            continue
        if in_fence:
            continue
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            flush(index - 1)
            section_start = index
            heading = match.group(2).strip()
    flush(len(lines))
    return chunks


def _strip_noncode(line: str, in_block_comment: bool) -> tuple[str, bool]:
    """Blank out string literals and comments so brace counting is reliable.

    Not a lexer: it handles the cases that actually break brace matching in
    real source (a `{` inside a string or a comment) and accepts that an
    exotic case (a regex literal containing an unbalanced brace) may still
    fool it. When it does, the affected declaration's end line is wrong and
    that chunk is oversized or short -- degraded retrieval, never a crash.
    """
    out: list[str] = []
    i = 0
    quote = ""
    while i < len(line):
        two = line[i:i + 2]
        if in_block_comment:
            if two == "*/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if quote:
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == quote:
                quote = ""
            i += 1
            continue
        if two == "/*":
            in_block_comment = True
            i += 2
            continue
        if two == "//" or line[i] == "#":
            break
        if line[i] in "\"'`":
            quote = line[i]
            i += 1
            continue
        out.append(line[i])
        i += 1
    return "".join(out), in_block_comment


def _chunk_brace_language(
    repo: str, path: str, text: str, language: str, cfg: RepositoryIntelligenceConfig
) -> list[CodeChunk]:
    """Declaration-regex + brace-depth scanner for C-family languages."""
    patterns = _DECLARATION_PATTERNS[language]
    lines = text.split("\n")

    # Pre-compute comment/string-stripped lines once: the scanner reads each
    # line twice (declaration match, brace depth) and block-comment state is
    # inherently sequential.
    stripped: list[str] = []
    in_block = False
    for line in lines:
        clean, in_block = _strip_noncode(line, in_block)
        stripped.append(clean)

    declarations: list[tuple[int, str]] = []
    for index, line in enumerate(lines, start=1):
        if not stripped[index - 1].strip():
            continue
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                declarations.append((index, match.group("name")))
                break

    if not declarations:
        return []

    chunks: list[CodeChunk] = []
    covered: set[int] = set()
    for start, name in declarations:
        if start in covered:
            continue  # a method already swallowed by its enclosing class chunk
        end = _find_block_end(stripped, start)
        covered.update(range(start, end + 1))
        body = _lines_slice(lines, start, end)
        kind = "class" if re.search(r"\b(class|struct|interface|enum|trait|impl)\b",
                                    stripped[start - 1]) else "function"
        chunks.extend(_split_oversized(
            repo, path, language, body, start, kind, name, "", cfg
        ))

    remainder = sorted(set(range(1, len(lines) + 1)) - covered)
    for start, end in _contiguous_spans(remainder):
        body = _lines_slice(lines, start, end)
        if body.strip():
            chunks.extend(_split_oversized(
                repo, path, language, body, start, "module", "", "", cfg
            ))
    return sorted(chunks, key=lambda c: c.start_line)


def _find_block_end(stripped: list[str], start_line: int) -> int:
    """Last line of the brace-delimited block opening at/after `start_line`.

    Falls back to the declaration line itself for brace-less declarations
    (a Go `type X int`, a Rust `pub fn f();`) and gives up at end-of-file
    for an unbalanced block rather than looping.
    """
    depth = 0
    seen_open = False
    for index in range(start_line - 1, len(stripped)):
        for char in stripped[index]:
            if char == "{":
                depth += 1
                seen_open = True
            elif char == "}":
                depth -= 1
                if seen_open and depth <= 0:
                    return index + 1
        if not seen_open and index > start_line - 1 and stripped[index].strip().endswith(";"):
            return index + 1
    return start_line if not seen_open else len(stripped)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def detect_metadata(
    root: Path, repo: str, files: list[Path], cfg: RepositoryIntelligenceConfig,
    *, default_branch: str = "", commit_sha: str = "",
) -> RepositoryMetadata:
    """Infer language/framework/toolchain from files actually on disk.

    Everything returned is evidence-backed: languages are counted from the
    indexed file list, frameworks come from dependency names read out of a
    manifest, entry points are files confirmed to exist. Nothing is inferred
    from the repository's name or description.
    """
    languages = Counter(
        LANGUAGE_BY_EXTENSION[p.suffix.lower()]
        for p in files if p.suffix.lower() in LANGUAGE_BY_EXTENSION
    )
    code_languages = Counter({
        lang: n for lang, n in languages.items() if lang not in _MARKDOWN_LANGUAGES
    })
    primary = (code_languages or languages).most_common(1)
    primary_language = primary[0][0] if primary else ""

    managers: list[str] = []
    frameworks: set[str] = set()
    for manifest, manager in DEPENDENCY_MANIFESTS.items():
        manifest_path = root / manifest
        if not manifest_path.is_file():
            continue
        managers.append(manager)
        frameworks.update(_frameworks_in(manifest_path))

    entry_points = [name for name in ENTRY_POINT_CANDIDATES if (root / name).is_file()]
    if not entry_points:
        for candidate in ("src", "app", "cmd"):
            sub = root / candidate
            if sub.is_dir():
                entry_points.extend(
                    f"{candidate}/{name}" for name in ENTRY_POINT_CANDIDATES
                    if (sub / name).is_file()
                )

    top_level = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name not in EXCLUDED_DIRECTORIES and not p.name.startswith(".")
    )

    readme_summary, readme_path = _readme_summary(root, cfg)

    return RepositoryMetadata(
        repo=repo,
        default_branch=default_branch,
        commit_sha=commit_sha,
        primary_language=primary_language,
        languages=dict(languages.most_common()),
        frameworks=sorted(frameworks),
        dependency_managers=sorted(set(managers)),
        project_type=_project_type(root, entry_points, readme_path),
        entry_points=entry_points[:5],
        top_level_dirs=top_level[:15],
        readme_summary=readme_summary,
        file_count=len(files),
    )


def _frameworks_in(manifest_path: Path) -> set[str]:
    """Framework labels for dependency names present in one manifest.

    package.json is parsed as JSON so a marker only matches a real
    dependency key -- substring-matching the raw text would label any repo
    whose README example mentions "react". Other manifest formats (TOML,
    go.mod, XML) are substring-matched against the dependency section, which
    is imprecise enough that only high-signal markers live in
    FRAMEWORK_MARKERS.
    """
    text = read_text(manifest_path)
    if not text:
        return set()

    if manifest_path.name == "package.json":
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return set()
        names: set[str] = set()
        for section in ("dependencies", "devDependencies", "peerDependencies"):
            block = data.get(section)
            if isinstance(block, dict):
                names.update(str(k).lower() for k in block)
        return {label for marker, label in FRAMEWORK_MARKERS.items() if marker in names}

    lowered = text.lower()
    return {
        label for marker, label in FRAMEWORK_MARKERS.items()
        if re.search(rf"(?<![\w-]){re.escape(marker)}(?![\w-])", lowered)
    }


def _readme_summary(root: Path, cfg: RepositoryIntelligenceConfig) -> tuple[str, Path | None]:
    """First substantive prose from the README, truncated.

    Skips the title, badge lines, and blockquotes to land on the sentence
    that actually says what the project does.
    """
    for name in ("README.md", "README.rst", "README.txt", "README", "readme.md"):
        path = root / name
        if not path.is_file():
            continue
        text = read_text(path)
        collected: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", ">", "!", "|", "---", "===", "<")):
                continue
            if re.fullmatch(r"[\[\]!()\w\s./:-]*\)\s*", stripped) and "](" in stripped:
                continue  # a bare badge/link line
            collected.append(stripped)
            if sum(len(c) for c in collected) >= cfg.readme_summary_chars:
                break
        summary = " ".join(collected)[:cfg.readme_summary_chars].strip()
        return summary, path
    return "", None


def _project_type(root: Path, entry_points: list[str], readme_path: Path | None) -> str:
    """Coarse and deliberately conservative: only claims what a marker file
    proves, and returns "" rather than guessing."""
    if (root / "Dockerfile").is_file() or (root / "docker-compose.yml").is_file():
        return "service"
    if (root / "setup.py").is_file() or (root / "pyproject.toml").is_file():
        if entry_points:
            return "application"
        return "library"
    if (root / "package.json").is_file():
        return "application" if entry_points else "library"
    if entry_points:
        return "application"
    if readme_path is not None:
        return "project"
    return ""
