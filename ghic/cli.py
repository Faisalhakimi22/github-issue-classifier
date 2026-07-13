"""`ghic` — the unified command-line interface.

Thin wrappers only: every subcommand delegates to a module that already
exists and is already tested; no logic lives here beyond argument routing.

  ghic train [...]        -> ghic.train      (add --champion for the full protocol)
  ghic predict [...]      -> score one issue with the local model, print JSON
  ghic explain [...]      -> predict + top feature contributions
  ghic serve              -> ghic.service.app (uvicorn)
  ghic benchmark [...]    -> ghic.backtest    (held-out replay through the webhook)
  ghic dashboard          -> open the running service's /dashboard
  ghic collect / label    -> data pipeline stages
"""
from __future__ import annotations

import argparse
import json
import sys


def _predict_once(args: argparse.Namespace, explain: bool) -> int:
    from datetime import datetime, timezone

    from .service.inference import IssuePredictor
    from .service.settings import default_model_path

    predictor = IssuePredictor(
        args.model or default_model_path(), threshold=args.threshold
    )
    pred = predictor.predict(
        repo_full_name=args.repo,
        issue_number=0,
        title=args.title,
        body=args.body,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        explain=explain,
    )
    out = pred.as_dict()
    if not explain:
        out.pop("top_features", None)
        out.pop("signed_contributions", None)
    print(json.dumps(out, indent=2))
    return 0


def dashboard_url() -> str:
    import os

    host = os.environ.get("GHIC_HOST", "127.0.0.1")
    port = os.environ.get("PORT", os.environ.get("GHIC_PORT", "8000"))
    return f"http://{host}:{port}/dashboard"


def _passthrough(command: str, rest: list[str]) -> int:
    """Dispatch to the target module's own CLI, forwarding all arguments.

    Routed before argparse ever sees `rest` — argparse.REMAINDER stopped
    capturing leading options in recent Pythons, and these commands own
    their argument parsing anyway.
    """
    if command == "train":
        from .train import main as target
    elif command == "collect":
        from .collect import main as target
    elif command == "label":
        from .label import main as target
    elif command == "benchmark":
        from .backtest import main as target
    elif command == "serve":
        from .service.app import main as serve_main

        sys.argv = ["ghic-serve", *rest]
        return serve_main()
    else:  # pragma: no cover — guarded by the caller
        raise ValueError(command)
    return target(rest)


_PASSTHROUGH_HELP = {
    "train": "train models (--champion for the rigorous protocol)",
    "collect": "collect issues via the GitHub GraphQL API",
    "label": "apply the deterministic labeling rules",
    "benchmark": "replay the held-out set through the real webhook (ghic.backtest)",
    "serve": "run the webhook service",
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in _PASSTHROUGH_HELP:
        return _passthrough(argv[0], argv[1:])

    parser = argparse.ArgumentParser(
        prog="ghic", description="GitHub issue triage: pipeline, models, service."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in _PASSTHROUGH_HELP.items():
        sub.add_parser(name, help=help_text + " (all further args forwarded)")

    for name, help_text in [
        ("predict", "score one issue with the local model"),
        ("explain", "score one issue and show top feature contributions"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--title", required=True)
        p.add_argument("--body", default="")
        p.add_argument("--repo", default="cli/adhoc")
        p.add_argument("--model", default=None)
        p.add_argument("--threshold", type=float, default=0.5)

    sub.add_parser("dashboard", help="open the running service's dashboard")

    args = parser.parse_args(argv)
    if args.command in ("predict", "explain"):
        return _predict_once(args, explain=args.command == "explain")
    if args.command == "dashboard":
        import webbrowser

        url = dashboard_url()
        print(url)
        webbrowser.open(url)
        return 0
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
