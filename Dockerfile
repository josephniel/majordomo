# OPTIONAL app image (one persona = one container, same image, different
# env). The documented flow runs the bot on the HOST — Claude Code
# subscription auth needs ~/.claude, and run_code needs the host's Docker —
# so this image suits API-key-only deployments. All connectors are
# in-process Python; no Node required.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the rest of the app. In dev, docker-compose mounts these as volumes
# so edits don't require a rebuild.
COPY . .

# Default command — overridden per-service in docker-compose.yml.
CMD ["python", "-m", "runtime", "--persona", "personal_assistant"]
