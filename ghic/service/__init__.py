"""Production webhook service: score newly opened GitHub issues with the
trained classifier and (optionally) comment on / label them.

Modules:
  settings   — 12-factor env-var configuration (GHIC_* variables)
  github_app — GitHub App auth (JWT -> installation token) + REST helpers
  inference  — model loading and single-issue prediction with explanations
  app        — FastAPI application: /webhook, /healthz, /api/predict
"""
