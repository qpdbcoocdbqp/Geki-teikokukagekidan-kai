ARG UV_VERSION=0.12.1
ARG PROJECT_DIR=rlcd
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.12-slim-bookworm

COPY --from=uv /uv /uvx /bin/

ARG PROJECT_DIR=rlcd

# Create standard non-root user with UID 1000 for Hugging Face Spaces
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    BACKEND=torch \
    MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct

# Install Python requirements
COPY --chown=user ${PROJECT_DIR}/requirements.txt /home/user/app/requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

# Copy source tree
COPY --chown=user ${PROJECT_DIR}/ /home/user/app

# Run the application as the non-root user
USER user

EXPOSE 7860

CMD ["python3", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
