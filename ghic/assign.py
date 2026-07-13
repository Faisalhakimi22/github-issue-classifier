"""Maintainer assignment recommendation — data collection and honest evaluation.

The plan of record (BUILD_PLAN_2 Phase 16): start with the honest baseline,
not a model. Surface the maintainers who were assigned to / closed the most
similar prior issues (reusing the duplicate-detection similarity machinery),
and validate that against the naive "most-active maintainer" baseline. Only
if similarity meaningfully beats the naive baseline does a learned ranker
earn its complexity; if the baseline wins, the baseline is the shipped
feature.

Data: the original collection has no assignee information, so this module
adds a supplemental GraphQL pass (`--collect`) fetching, per issue:
assignees, conversation participants, and the closing actor. Batched 50
issues per request with the same caching/rate-limit discipline as
`collect.py`; output is `data/processed/assignments.json`.

Honesty notes, carried into the card:
- Assignee/participant/closer state is fetched *now* (close-time state, like
  labels_at_close). Using prior issues' assignees as a candidate pool is
  slightly acausal for issues assigned late; the dataset does not record
  assignment timestamps.
- The spec's "most-active-committer" baseline is approximated by
  most-active-assignee: commit history is not collected, and prior
  assignment volume is the closer analog of triage workload.
- Recommendations are assistive only. A wrong automatic assignment affects
  a real person's workload; nothing here ever calls the assignment API.

CLI:
  python -m ghic.assign --collect     # supplemental GraphQL pass (needs GH_TOKEN)
  python -m ghic.assign --evaluate    # top-k hit rates vs baseline + card
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import evaluate, train, utils
from .collect import _graphql, _gate_rate_limit
from .config import Config, get_config
from .dupdetect import _blob, _tfidf_vectors

logger = utils.get_logger(__name__)

ASSIGNMENTS_PATH = utils.DATA_PROCESSED / "assignments.json"
CARD_PATH = utils.PROJECT_ROOT / "models" / "ASSIGNMENT_CARD.md"
REPORT_PATH = evaluate.REPORTS_DIR / "assign.json"
BATCH_SIZE = 50
N_SIMILAR = 20          # prior issues consulted by the similarity recommender
K_VALUES = (1, 3, 5)

_ISSUE_FIELDS = """
      number
      assignees(first: 10) { nodes { login } }
      participants(first: 20) { nodes { login } }
      timelineItems(first: 10, itemTypes: [CLOSED_EVENT]) {
        nodes { ... on ClosedEvent { actor { login } } }
      }
""".rstrip()


# ---------------------------------------------------------------------------
# Supplemental collection
# ---------------------------------------------------------------------------
def _build_batch_query(numbers: list[int]) -> str:
    aliases = "\n".join(
        f"    i{n}: issue(number: {n}) {{{_ISSUE_FIELDS}\n    }}" for n in numbers
    )
    return (
        "query Assignments($owner: String!, $name: String!) {\n"
        "  rateLimit { remaining resetAt cost }\n"
        "  repository(owner: $owner, name: $name) {\n"
        f"{aliases}\n"
        "  }\n"
        "}"
    )


def _extract(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if not node:
        return None
    closer = None
    for ev in ((node.get("timelineItems") or {}).get("nodes") or []):
        actor = (ev or {}).get("actor") or {}
        if actor.get("login"):
            closer = actor["login"]
    return {
        "assignees": [
            a["login"] for a in ((node.get("assignees") or {}).get("nodes") or [])
            if a and a.get("login")
        ],
        "participants": [
            p["login"] for p in ((node.get("participants") or {}).get("nodes") or [])
            if p and p.get("login")
        ],
        "closer": closer,
    }


def collect_assignments(
    cfg: Config | None = None, combined_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch assignees/participants/closer for every collected issue.

    Batched per repo, cached per batch (same idempotent-resume behavior as
    collect.py: re-runs skip already-cached batches).
    """
    cfg = cfg or get_config()
    combined_path = combined_path or (utils.DATA_PROCESSED / "combined.csv")
    df = pd.read_csv(combined_path, usecols=["repo_name", "number"])

    out: dict[str, dict[str, Any]] = {}
    started = time.time()
    for repo, grp in df.groupby("repo_name", sort=True):
        owner, name = str(repo).split("/", 1)
        namespace = f"assignments/{owner}__{name}"
        numbers = sorted(int(n) for n in grp["number"].unique())
        repo_out: dict[str, Any] = {}
        for start in range(0, len(numbers), BATCH_SIZE):
            batch = numbers[start:start + BATCH_SIZE]
            query = _build_batch_query(batch)
            variables = {"owner": owner, "name": name}
            key = utils.cache_key(query, variables)
            payload = utils.cache_get(namespace, key)
            if payload is None:
                logger.info("fetch %s assignments %d-%d of %d",
                            repo, start + 1, start + len(batch), len(numbers))
                payload = _graphql(cfg, query, variables)
                utils.cache_put(namespace, key, payload)
                _gate_rate_limit(payload["data"]["rateLimit"], cfg.collection.rate_limit_floor)
            repository = payload["data"].get("repository") or {}
            for n in batch:
                rec = _extract(repository.get(f"i{n}"))
                if rec is not None:
                    repo_out[str(n)] = rec
        out[str(repo)] = repo_out
        logger.info("%s: %d issues, %d with >=1 assignee", repo, len(repo_out),
                    sum(1 for r in repo_out.values() if r["assignees"]))
    utils.write_json(ASSIGNMENTS_PATH, out)
    logger.info("wrote %s (%.0fs)", ASSIGNMENTS_PATH, time.time() - started)
    return out


def load_assignments(path: Path = ASSIGNMENTS_PATH) -> dict[str, dict[str, Any]] | None:
    return utils.read_json(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Evaluation: similarity recommender vs most-active baseline
# ---------------------------------------------------------------------------
def _is_human(login: str, bot_logins: set[str]) -> bool:
    return bool(login) and login not in bot_logins and not login.endswith("[bot]")


def evaluate_recommender(
    combined_path: Path | None = None,
    cfg: Config | None = None,
    assignments: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_config(require_token=False)
    assignments = assignments if assignments is not None else load_assignments()
    if assignments is None:
        raise FileNotFoundError(
            f"{ASSIGNMENTS_PATH} not found — run `python -m ghic.assign --collect` first."
        )
    bot_logins = set(cfg.labeling.bot_logins)

    df = pd.read_csv(combined_path or (utils.DATA_PROCESSED / "combined.csv"))
    df = df.reset_index(drop=True)
    train_df, test_df = train.chronological_split(df)
    texts = [_blob(t, b) for t, b in zip(df["title"], df["body"])]
    train_texts = [texts[i] for i in train_df.index.to_numpy()]
    vectors, _ = _tfidf_vectors(train_texts, texts)

    repos = df["repo_name"].to_numpy()
    created = df["created_at"].to_numpy()
    numbers = df["number"].to_numpy()

    def maintainers_of(repo: str, number: int) -> list[str]:
        rec = assignments.get(repo, {}).get(str(int(number)))
        if not rec:
            return []
        pool = list(rec["assignees"]) + ([rec["closer"]] if rec["closer"] else [])
        return [m for m in pool if _is_human(m, bot_logins)]

    per_repo: dict[str, dict[str, Any]] = {}
    for repo in sorted(test_df["repo_name"].unique()):
        hits_sim = {k: 0 for k in K_VALUES}
        hits_freq = {k: 0 for k in K_VALUES}
        n_eval = 0

        # Prior-assignment counts, updated as we walk the test slice in time
        # order so the frequency baseline is causal too.
        prior_counts: Counter = Counter()
        for i in train_df.index[train_df["repo_name"] == repo]:
            for m in maintainers_of(repo, numbers[i]):
                prior_counts[m] += 1

        repo_test = test_df[test_df["repo_name"] == repo].sort_values("created_at")
        for pos in repo_test.index.to_numpy():
            truth = {
                m for m in (assignments.get(repo, {})
                            .get(str(int(numbers[pos])), {})
                            .get("assignees") or [])
                if _is_human(m, bot_logins)
            }
            if truth:
                n_eval += 1
                # similarity recommender
                mask = (repos == repo) & (created < created[pos])
                idx = np.flatnonzero(mask)
                scores: Counter = Counter()
                if len(idx):
                    sims = np.asarray((vectors[idx] @ vectors[pos].T).todense()).ravel()
                    top = idx[np.argsort(sims)[::-1][:N_SIMILAR]]
                    simmap = dict(zip(idx, sims))
                    for j in top:
                        for m in maintainers_of(repo, numbers[j]):
                            scores[m] += float(simmap[j])
                ranked_sim = [m for m, _ in scores.most_common(max(K_VALUES))]
                ranked_freq = [m for m, _ in prior_counts.most_common(max(K_VALUES))]
                for k in K_VALUES:
                    hits_sim[k] += bool(truth & set(ranked_sim[:k]))
                    hits_freq[k] += bool(truth & set(ranked_freq[:k]))
            # issues become history for later test issues either way
            for m in maintainers_of(repo, numbers[pos]):
                prior_counts[m] += 1

        per_repo[repo] = {
            "n_test_with_assignee": n_eval,
            "distinct_prior_assignees": len(prior_counts),
            "similarity_hit_at_k": {str(k): round(hits_sim[k] / n_eval, 4) if n_eval else None
                                    for k in K_VALUES},
            "most_active_hit_at_k": {str(k): round(hits_freq[k] / n_eval, 4) if n_eval else None
                                     for k in K_VALUES},
        }

    total = sum(r["n_test_with_assignee"] for r in per_repo.values())
    overall = {
        "n_test_with_assignee": total,
        "similarity_hit_at_k": {},
        "most_active_hit_at_k": {},
    }
    for k in K_VALUES:
        for field in ("similarity_hit_at_k", "most_active_hit_at_k"):
            weighted = sum(
                (r[field][str(k)] or 0) * r["n_test_with_assignee"]
                for r in per_repo.values()
            )
            overall[field][str(k)] = round(weighted / total, 4) if total else None

    return {
        "task": ("recommend maintainers for a new issue; ground truth = the "
                 "issue's eventual human assignee(s)"),
        "recommenders": {
            "similarity": f"assignees+closers of the {N_SIMILAR} most similar prior "
                          "same-repo issues (TF-IDF cosine), similarity-weighted",
            "most_active": "maintainers ranked by prior assignment count (causal)",
        },
        "overall": overall,
        "per_repo": per_repo,
    }


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------
def write_card(results: dict[str, Any]) -> Path:
    def row(label: str, block: dict[str, Any]) -> str:
        sim = block["similarity_hit_at_k"]
        freq = block["most_active_hit_at_k"]

        def f(v: Any) -> str:
            return f"{v:.3f}" if v is not None else "n/a"

        return (f"| {label} | {block['n_test_with_assignee']} "
                f"| {f(sim['1'])} | {f(sim['3'])} | {f(sim['5'])} "
                f"| {f(freq['1'])} | {f(freq['3'])} | {f(freq['5'])} |")

    o = results["overall"]
    sim3, freq3 = o["similarity_hit_at_k"]["3"], o["most_active_hit_at_k"]["3"]
    lines = [
        "# Card — maintainer assignment recommender",
        "",
        f"**Task:** {results['task']}.",
        "",
        f"- similarity recommender: {results['recommenders']['similarity']}",
        f"- naive baseline: {results['recommenders']['most_active']}",
        "",
        "## Ground truth caveats",
        "- Assignee/closer state is fetched at collection time (close-time",
        "  state); assignment timestamps are not recorded, so prior issues'",
        "  assignees are slightly acausal for late-assigned issues.",
        "- \"Most-active-committer\" from the spec is approximated by",
        "  most-active-assignee — commit history is not collected.",
        "- Only issues with at least one human assignee are scored.",
        "",
        "## Results (chronological test slice, hit@k)",
        "",
        "| subset | n | sim@1 | sim@3 | sim@5 | active@1 | active@3 | active@5 |",
        "|---|---|---|---|---|---|---|---|",
        row("overall", o),
    ]
    for repo, block in results["per_repo"].items():
        lines.append(row(repo, block))
    lines += [
        "",
        "## Decision",
        _decision_text(sim3, freq3),
        "",
        "Recommendations are **assistive only** — surfaced in the webhook",
        "response for a maintainer to consider. The service never calls the",
        "assignment API; a wrong automatic assignment costs a real person's",
        "time, and this evaluation does not justify that risk.",
        "",
        "_Auto-generated by `python -m ghic.assign --evaluate`._",
    ]
    CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    CARD_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", CARD_PATH)
    return CARD_PATH


def _decision_text(sim3: float | None, freq3: float | None) -> str:
    if sim3 is None or freq3 is None:
        return "Insufficient assignee ground truth to decide — nothing ships."
    if sim3 >= freq3 + 0.05:
        return (f"The similarity recommender wins (hit@3 {sim3:.3f} vs {freq3:.3f}) "
                "and is the shipped mechanism. A learned ranker is not built: "
                "it would need to beat this number first, on this protocol.")
    if freq3 >= sim3 + 0.05:
        return (f"The naive most-active baseline wins (hit@3 {freq3:.3f} vs "
                f"{sim3:.3f}); per the plan, **the baseline is the shipped "
                "feature** — a learned ranker that can't beat prior-frequency "
                "isn't earning its complexity.")
    return (f"Similarity and most-active are within noise of each other "
            f"(hit@3 {sim3:.3f} vs {freq3:.3f}). The similarity recommender "
            "ships because its suggestions carry per-issue evidence (the "
            "similar issues), which the frequency list cannot provide.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Maintainer assignment: supplemental collection + evaluation."
    )
    parser.add_argument("--collect", action="store_true",
                        help="fetch assignees/participants/closer (needs GH_TOKEN)")
    parser.add_argument("--evaluate", action="store_true",
                        help="hit@k vs naive baseline; writes card + report")
    args = parser.parse_args(argv)

    if args.collect:
        collect_assignments()
    if args.evaluate:
        results = evaluate_recommender()
        utils.write_json(REPORT_PATH, results)
        write_card(results)
        print(json.dumps(results["overall"], indent=2))
    if not (args.collect or args.evaluate):
        parser.print_help()
    return 0


if __name__ == "__main__":
    from ghic.assign import main as _main  # package path for any pickled artifacts

    sys.exit(_main())
