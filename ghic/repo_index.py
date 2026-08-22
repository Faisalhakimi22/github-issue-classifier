"""CLI for the Repository Intelligence Engine.

    python -m ghic.repo_index --repo owner/name              # clone + index
    python -m ghic.repo_index --repo owner/name --path .     # index a local tree
    python -m ghic.repo_index --repo owner/name --query "csv import crashes"
    python -m ghic.repo_index --list
    python -m ghic.repo_index --repo owner/name --purge

Exists because auto-indexing needs the QStash queue (see app.py's
`_schedule_indexing`): a deploy without one -- Docker with a volume, a
local dev box, a CI warm-up step -- indexes here instead. Also the fastest
way to see what retrieval actually returns for a given issue before
turning the feature on for real, which is worth doing: retrieval quality
varies a lot by repository.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import utils
from .repository_intelligence import (
    RepositoryIntelligenceConfig,
    RepositoryIntelligenceService,
    RepositoryMetadata,
    build_service,
)

logger = utils.get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index and query repositories for GHIC.")
    parser.add_argument("--repo", help="owner/name")
    parser.add_argument("--path", help="index this local directory instead of cloning")
    parser.add_argument("--token", default="", help="GitHub token for a private clone")
    parser.add_argument("--query", help="run a retrieval query against the index")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="re-index even if current")
    parser.add_argument("--purge", action="store_true", help="drop this repo's indexes")
    parser.add_argument("--list", action="store_true", help="list indexed repositories")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    cfg = RepositoryIntelligenceConfig.from_env()
    service = build_service(cfg)
    if service is None:
        print("repository intelligence is not available with the current configuration",
              file=sys.stderr)
        return 1

    if args.list:
        try:
            return _list_indexes(service, cfg, as_json=args.json)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 1

    if not args.repo:
        parser.error("--repo is required unless --list is given")

    try:
        if args.purge:
            service.forget(args.repo)
            print(f"removed the index and clone for {args.repo}")
            return 0

        if args.query:
            return _query(service, args.repo, args.query, args.top_k, as_json=args.json)

        if args.path:
            from pathlib import Path

            metadata = service.index_local_path(args.repo, Path(args.path).resolve())
            ok = metadata is not None
        else:
            ok = service.index_repository(args.repo, token=args.token, force=args.force)
            metadata = None
            if ok:
                metadata = _metadata_from_state(service, args.repo)
                if metadata is None:
                    loaded = service.index_cache.get_latest(args.repo)
                    if loaded:
                        pair = service.index_cache.load(args.repo, *loaded)
                        metadata = pair[1] if pair else None
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    if not ok:
        print(f"indexing failed for {args.repo} (see logs)", file=sys.stderr)
        return 1
    if metadata is not None:
        if args.json:
            print(json.dumps(metadata.as_dict(), indent=2))
        else:
            print(f"indexed {args.repo}: {metadata.file_count} files, "
                  f"{metadata.chunk_count} chunks")
            print(f"  language: {metadata.primary_language or '(unknown)'}")
            if metadata.frameworks:
                print(f"  frameworks: {', '.join(metadata.frameworks)}")
            if metadata.project_type:
                print(f"  project type: {metadata.project_type}")
    return 0


def _query(
    service: RepositoryIntelligenceService, repo: str, query: str, top_k: int, *, as_json: bool
) -> int:
    context = service.get_context(repo, query, "")
    if as_json:
        print(json.dumps(context.as_dict(), indent=2))
        return 0
    if not context.indexed:
        print(f"{repo} is not indexed. Run: python -m ghic.repo_index --repo {repo}")
        return 1
    if context.is_empty:
        print(context.note)
        return 0
    for retrieved in context.chunks[:top_k]:
        chunk = retrieved.chunk
        symbol = f"  {chunk.kind} {chunk.qualified_symbol}" if chunk.qualified_symbol else ""
        print(f"{retrieved.score:.3f}  {chunk.path}:{chunk.start_line}-{chunk.end_line}{symbol}")
    return 0


def _metadata_from_state(
    service: RepositoryIntelligenceService, repo: str
) -> RepositoryMetadata | None:
    record = service.status(repo)
    if record is None or not record.metadata_json:
        return None
    try:
        return RepositoryMetadata.from_dict(record.metadata_json)
    except (KeyError, TypeError, ValueError):
        return None


def _list_indexes(
    service: RepositoryIntelligenceService,
    cfg: RepositoryIntelligenceConfig,
    *,
    as_json: bool,
) -> int:
    records = service.list_repositories()
    if records:
        payload = [record.as_dict() for record in records]
        if as_json:
            print(json.dumps(payload, indent=2))
        else:
            for record in records:
                print(
                    f"{record.repo}  {record.indexed_commit_sha[:8]}  "
                    f"{record.embedding_signature or '(unknown embedding)'}  "
                    f"{record.state.value}"
                )
        return 0

    pointer_dir = cfg.index_dir / "_latest"
    entries: list[dict[str, object]] = []
    if pointer_dir.is_dir():
        for path in sorted(pointer_dir.glob("*.json")):
            try:
                entries.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    if as_json:
        print(json.dumps(entries, indent=2))
        return 0
    if not entries:
        print("no repositories indexed")
        return 0
    for entry in entries:
        print(f"{entry.get('repo')}  {str(entry.get('commit_sha'))[:8]}  {entry.get('embedder')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
