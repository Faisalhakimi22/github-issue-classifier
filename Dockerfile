# GitHub Issue Triage Bot — webhook service image.
#
# Build (a trained model must exist in models/ first — run `python -m ghic.train`):
#   docker build -t ghic .
# Run:
#   docker run -p 8000:8000 \
#     -e GHIC_WEBHOOK_SECRET=... -e GHIC_APP_ID=... -e GHIC_PRIVATE_KEY="$(cat key.pem)" \
#     -e GHIC_DRY_RUN=false -e GHIC_POST_COMMENT=true \
#     ghic

FROM python:3.12-slim AS runtime

# Never run a network-facing service as root.
RUN useradd --create-home --uid 1000 ghic
WORKDIR /app

COPY pyproject.toml README.md ./
COPY ghic/ ghic/
RUN pip install --no-cache-dir ".[service]"

# ML/labeling config + trained model artifacts (champion preferred at runtime).
COPY config.yaml ./
COPY models/*.joblib models/

# Writable dir for the online-evaluation ledger (mount a volume to persist
# live precision/recall across container replacements).
RUN mkdir -p /app/data && chown ghic:ghic /app/data
VOLUME /app/data

ENV GHIC_HOST=0.0.0.0 \
    PORT=8000

USER ghic
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"

CMD ["python", "-m", "ghic.service.app"]
