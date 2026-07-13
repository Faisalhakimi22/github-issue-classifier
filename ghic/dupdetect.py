"""Duplicate detection: flag newly opened issues that likely duplicate a
prior issue, and surface the candidates.

Design decisions (stated, per the build discipline):

- **Exact search, no vector database.** The corpus is ~6k issues; a
  normalized matrix product computes every cosine similarity in
  milliseconds. FAISS/pgvector would add a dependency to solve a problem
  this corpus does not have. The interface (`DupIndex.query`) is the
  seam where an ANN index would slot in if the corpus grows 100x.
- **Two representations, same evaluation.** A TF-IDF baseline and (when
  the optional sentence-transformers extra is installed) MiniLM sentence
  embeddings. The evaluation reports both; if the baseline is
  competitive, that is the finding, not an embarrassment.
- **Causal evaluation.** An issue may only match issues created strictly
  before it, in the same repo — mirroring what the webhook could actually
  have known. Vocabulary/embedding choices are fit on the training window.
- **Ground truth is rule-derived and lopsided.** Duplicate labels exist
  almost only in microsoft/vscode (369/372); react and tensorflow close
  duplicates without labeling them. The metric is therefore validated on
  vscode and reported as unvalidated elsewhere. The task evaluated is
  "will this issue be closed as a duplicate?" scored by max similarity to
  any prior issue — pairwise duplicate targets are not recorded in the
  dataset, so retrieval hit-rate cannot be honestly measured. Documented,
  not smoothed over.

CLI:
  python -m ghic.dupdetect --evaluate      # both representations, honest table
  python -m ghic.dupdetect --build-index   # models/dup_index.joblib for serving
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import evaluate, train, utils

logger = utils.get_logger(__name__)

INDEX_PATH = utils.PROJECT_ROOT / "models" / "dup_index.joblib"
DEFAULT_SUGGEST_THRESHOLD = 0.55   # min cosine sim before we surface a candidate
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MAX_TEXT_CHARS = 4000             # embeddings truncate anyway; keep TF-IDF comparable


def _blob(title: Any, body: Any) -> str:
    t = str(title) if title is not None and title == title else ""
    b = str(body) if body is not None and body == body else ""
    return (t + "\n\n" + b)[:_MAX_TEXT_CHARS]


def is_duplicate_labeled(labels_json: Any) -> bool:
    """Ground truth: the issue carried a duplicate label at close."""
    try:
        labels = json.loads(labels_json) if isinstance(labels_json, str) else (labels_json or [])
    except json.JSONDecodeError:
        return False
    return any("duplicate" in (label or "").strip().lower() for label in labels)


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------
def _tfidf_vectors(train_texts: list[str], all_texts: list[str]) -> np.ndarray | Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), min_df=2,
        sublinear_tf=True, strip_accents="unicode", stop_words="english",
    )
    vec.fit(train_texts)
    return normalize(vec.transform(all_texts)), vec


def _embedding_vectors(all_texts: list[str]) -> np.ndarray | None:
    """MiniLM sentence embeddings, L2-normalized. None if the extra isn't installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — skipping the embedding "
            "representation. pip install -e '.[embeddings]' to enable it."
        )
        return None
    model = SentenceTransformer(EMBEDDING_MODEL)
    emb = model.encode(all_texts, batch_size=64, show_progress_bar=False,
                       normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


# ---------------------------------------------------------------------------
# Serving index
# ---------------------------------------------------------------------------
@dataclass
class DupIndex:
    """Exact-cosine index over prior issues. `vectors` are L2-normalized."""
    vectorizer: Any                 # fitted TfidfVectorizer (serving uses TF-IDF)
    vectors: Any                    # sparse matrix, rows align with meta
    meta: list[dict[str, Any]]      # repo, number, title, created_at per row

    def query(self, repo: str, title: str, body: str, k: int = 3,
              min_sim: float = DEFAULT_SUGGEST_THRESHOLD) -> list[dict[str, Any]]:
        from sklearn.preprocessing import normalize

        rows = [i for i, m in enumerate(self.meta) if m["repo"] == repo]
        if not rows:
            return []
        q = normalize(self.vectorizer.transform([_blob(title, body)]))
        sims = np.asarray((self.vectors[rows] @ q.T).todense()).ravel()
        order = np.argsort(sims)[::-1][:k]
        out = []
        for j in order:
            if sims[j] < min_sim:
                break
            m = self.meta[rows[j]]
            out.append({"number": m["number"], "title": m["title"],
                        "similarity": round(float(sims[j]), 4)})
        return out


def build_index(labeled_path: Path | None = None, out_path: Path = INDEX_PATH) -> DupIndex:
    import joblib

    df = train.load_labeled(labeled_path or (utils.DATA_PROCESSED / "labeled.csv"))
    texts = [_blob(t, b) for t, b in zip(df["title"], df["body"])]
    vectors, vec = _tfidf_vectors(texts, texts)
    meta = [
        {"repo": r, "number": int(n), "title": str(t)[:120], "created_at": c}
        for r, n, t, c in zip(df["repo_name"], df["number"], df["title"], df["created_at"])
    ]
    index = DupIndex(vectorizer=vec, vectors=vectors, meta=meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, out_path)
    logger.info("wrote %s (%d issues indexed)", out_path, len(meta))
    return index


def load_index(path: Path = INDEX_PATH) -> DupIndex | None:
    import joblib

    if not path.exists():
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Evaluation: does max-similarity-to-prior predict duplicate closure?
# ---------------------------------------------------------------------------
def _max_prior_similarity(vectors: Any, df: pd.DataFrame, test_positions: np.ndarray) -> np.ndarray:
    """For each test row: max cosine sim to any SAME-REPO issue created earlier."""
    dense = not hasattr(vectors, "todense")
    repos = df["repo_name"].to_numpy()
    created = df["created_at"].to_numpy()
    scores = np.zeros(len(test_positions))
    for out_i, pos in enumerate(test_positions):
        mask = (repos == repos[pos]) & (created < created[pos])
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        if dense:
            sims = vectors[idx] @ vectors[pos]
        else:
            sims = np.asarray((vectors[idx] @ vectors[pos].T).todense()).ravel()
        scores[out_i] = float(np.max(sims))
    return scores


def evaluate_detector(labeled_path: Path | None = None) -> dict[str, Any]:
    df = train.load_labeled(labeled_path or (utils.DATA_PROCESSED / "labeled.csv"))
    df = df.reset_index(drop=True)
    df["is_dup"] = df["labels_at_close"].map(is_duplicate_labeled).astype(int)
    train_df, test_df = train.chronological_split(df)
    train_pos = train_df.index.to_numpy()
    test_pos = test_df.index.to_numpy()
    texts = [_blob(t, b) for t, b in zip(df["title"], df["body"])]
    train_texts = [texts[i] for i in train_pos]

    representations: dict[str, Any] = {}
    tfidf_vectors, _ = _tfidf_vectors(train_texts, texts)
    representations["tfidf_cosine"] = tfidf_vectors
    emb = _embedding_vectors(texts)
    if emb is not None:
        representations[f"minilm ({EMBEDDING_MODEL.split('/')[-1]})"] = emb

    y = test_df["is_dup"].to_numpy()
    results: dict[str, Any] = {
        "task": "predict duplicate-labeled closure from max similarity to prior same-repo issues",
        "test_n": int(len(test_pos)),
        "test_positives": int(y.sum()),
        "representations": {},
    }
    for name, vectors in representations.items():
        logger.info("scoring representation: %s", name)
        scores = _max_prior_similarity(vectors, df, test_pos)
        rep: dict[str, Any] = {"overall": _auc_block(y, scores)}
        for repo in sorted(test_df["repo_name"].unique()):
            mask = (test_df["repo_name"] == repo).to_numpy()
            rep[repo] = _auc_block(y[mask], scores[mask])
        results["representations"][name] = rep
    return results


def _auc_block(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    # Cosine of near-identical texts can exceed 1.0 by float error, which the
    # probabilistic metric validation rejects.
    scores = np.clip(scores, 0.0, 1.0)
    m = evaluate.compute_metrics(y, scores, threshold=0.5)
    return {"n": int(len(y)), "positives": int(y.sum()),
            "roc_auc": None if m.roc_auc != m.roc_auc else round(m.roc_auc, 4),
            "pr_auc": None if m.pr_auc != m.pr_auc else round(m.pr_auc, 4)}


def format_report(results: dict[str, Any]) -> str:
    lines = ["", "=" * 72, "  DUPLICATE DETECTION — honest evaluation", "=" * 72,
             f"\nTask: {results['task']}",
             f"Test slice: {results['test_n']} issues, {results['test_positives']} duplicate-labeled",
             "(duplicate labels exist almost only in microsoft/vscode — other repos",
             " have no positives and report n/a; this capability is vscode-validated)", ""]
    header = f"{'representation':38s} {'subset':24s} {'n':>5} {'pos':>4} {'roc':>7} {'prauc':>7}"
    lines += [header, "-" * len(header)]
    for name, rep in results["representations"].items():
        for subset, block in rep.items():
            roc = f"{block['roc_auc']:.3f}" if block["roc_auc"] is not None else "  n/a"
            pr = f"{block['pr_auc']:.3f}" if block["pr_auc"] is not None else "  n/a"
            lines.append(f"{name:38s} {subset:24s} {block['n']:5d} {block['positives']:4d} {roc:>7} {pr:>7}")
    return "\n".join(lines)


def write_card(results: dict[str, Any]) -> Path:
    lines = [
        "# Model card — duplicate detector", "",
        "**Method:** max cosine similarity of a new issue against all prior",
        "same-repo issues (exact search — at ~6k issues a normalized matrix",
        "product is milliseconds; an ANN index is unjustified complexity).", "",
        f"**Task:** {results['task']}.", "",
        "## Ground truth caveats",
        "- Duplicate labels exist almost exclusively in microsoft/vscode",
        "  (369/372 in the corpus). The evaluation is valid there and",
        "  unvalidated elsewhere.",
        "- The dataset records *that* an issue was closed as duplicate, not",
        "  *which* issue it duplicated — so retrieval hit-rate against true",
        "  targets cannot be honestly measured, only duplicate-closure",
        "  prediction. Candidate surfacing is assistive, for a maintainer to",
        "  confirm.", "",
        "## Results (chronological test slice)", "",
        "| representation | subset | n | positives | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for name, rep in results["representations"].items():
        for subset, block in rep.items():
            roc = f"{block['roc_auc']:.3f}" if block["roc_auc"] is not None else "n/a"
            pr = f"{block['pr_auc']:.3f}" if block["pr_auc"] is not None else "n/a"
            lines.append(f"| `{name}` | {subset} | {block['n']} | {block['positives']} | {roc} | {pr} |")
    lines += [
        "",
        "## Interpretation (2026-07-13 run)",
        "Both representations are near chance at predicting duplicate-labeled",
        "closure (ROC ≈ 0.52–0.56; PR-AUC ≈ 0.06 vs a 7.3% base rate), and",
        "MiniLM embeddings do **not** beat the TF-IDF baseline. The likely",
        "mechanism: at vscode's volume nearly every new issue has *some*",
        "highly similar prior issue, so max similarity doesn't separate",
        "duplicate closures — being closed as duplicate is triager behavior,",
        "not a text property. Product consequences, applied:",
        "- similarity candidates stay **assistive context** (\"possibly",
        "  related\", verify yourself) — that claim is true by construction;",
        "- **no** automatic \"likely duplicate\" flag or label ships off this",
        "  score, and none should until pairwise duplicate-target ground",
        "  truth (issue-to-issue timeline references) is collected — that is",
        "  the roadmap item this evaluation motivates.",
        "",
        "_Auto-generated by `python -m ghic.dupdetect --evaluate`._",
    ]
    path = utils.PROJECT_ROOT / "models" / "DUPLICATE_CARD.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Duplicate detection: evaluate or build the serving index.")
    parser.add_argument("--evaluate", action="store_true", help="evaluate both representations honestly")
    parser.add_argument("--build-index", action="store_true", help="build models/dup_index.joblib for serving")
    args = parser.parse_args(argv)

    if args.build_index:
        build_index()
    if args.evaluate:
        results = evaluate_detector()
        utils.write_json(evaluate.REPORTS_DIR / "dupdetect.json", results)
        write_card(results)
        print(format_report(results))
    if not (args.evaluate or args.build_index):
        parser.print_help()
    return 0


if __name__ == "__main__":
    # Re-enter through the package path so DupIndex pickles as
    # ghic.dupdetect.DupIndex, not __main__.DupIndex (which would make the
    # saved index unloadable from the service process).
    from ghic.dupdetect import main as _main

    sys.exit(_main())
