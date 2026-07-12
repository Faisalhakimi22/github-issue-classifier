# Support

- **Bugs / feature requests:** open an issue on this repository (yes, the bot
  will score it).
- **Deployment help:** see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) first —
  it covers GitHub App registration, staged rollout, threshold calibration,
  and operations.
- **Model behavior questions:** `models/MODEL_CARD.md` documents the selection
  protocol, metrics, and known limitations. For a specific prediction, the
  bot's comment lists the top contributing features.

When reporting a service problem, include the `/healthz` output and the
relevant log lines (one line per scored issue). Never include your webhook
secret or private key.
