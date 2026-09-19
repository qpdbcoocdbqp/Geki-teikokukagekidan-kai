# syntax=docker/dockerfile:1.7

ARG UV_VERSION=0.12.1
ARG PROJECT_DIR=jev-ultrafast
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.12-slim-bookworm

ARG PROJECT_DIR=jev-ultrafast

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0 \
    PATH="/app/.venv/bin:${PATH}" \
    TYPESAFE_DEMO_HOST=0.0.0.0 \
    TYPESAFE_DEMO_PORT=8766 \
    BU_CDP_URL=http://127.0.0.1:9222

# Google Chrome's Linux package is currently published for amd64.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://dl.google.com/linux/linux_signing_key.pub \
        | gpg --dearmor -o /etc/apt/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/google-chrome.gpg] https://dl.google.com/linux/chrome/deb/ stable main" \
        > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

RUN useradd --create-home --uid 10001 app \
    && install -d -o app -g app /app

WORKDIR /app

COPY --chown=app:app ${PROJECT_DIR}/pyproject.toml ${PROJECT_DIR}/uv.lock ${PROJECT_DIR}/README.md ./
USER app
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=10001,gid=10001 \
    uv sync --locked --no-install-project

COPY --chown=app:app ${PROJECT_DIR}/ ./
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=10001,gid=10001 \
    chmod +x /app/docker-entrypoint.sh \
    && uv sync --locked

EXPOSE 8766

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uv", "run", "--no-sync", "jev"]
