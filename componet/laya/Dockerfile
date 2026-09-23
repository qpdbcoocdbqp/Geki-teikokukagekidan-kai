ARG UV_VERSION=0.12.1
ARG PROJECT_DIR=laya
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.12-slim-bookworm

COPY --from=uv /uv /uvx /bin/
ARG PROJECT_DIR=laya

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY --chown=user ${PROJECT_DIR}/requirements.txt /home/user/app/requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

COPY --chown=user ${PROJECT_DIR}/server /home/user/app/server

USER user

ENV PORT=8000 \
    HF_HOME=/home/user/.cache/huggingface

EXPOSE 8000

CMD ["python", "-m", "server.main"]
