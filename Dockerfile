# syntax=docker/dockerfile:1

ARG PYTHON_LATEST_VERSION=3.13
ARG PYTHON_VERSION=${PYTHON_LATEST_VERSION}.5

# ----------------------------------------------
# Stage 1: Builder
# ----------------------------------------------
FROM ghcr.io/astral-sh/uv:python${PYTHON_LATEST_VERSION}-bookworm-slim AS builder
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-default-groups --no-editable

COPY pyproject.toml uv.lock README.md /app/
COPY src/ /app/src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable

# ----------------------------------------------
# Stage 2: Runtime
# ----------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
WORKDIR /app

# For real time logs
ENV PYTHONUNBUFFERED=1

# Define directory environment variables
ENV DATASETS_DIR=/app/data/datasets \
    MODELS_DIR=/app/data/models \
    DB_DIR=/app/db \
    LOG_DIR=/app/log \
    HF_HOME=/app/data/hf_home \
    CACHE_DIR=/app/data/cache

# Install dependencies, create user, and prepare directories in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates git-core && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    groupadd --system appuser && useradd --system --gid appuser --no-create-home --shell /usr/sbin/nologin appuser && \
    install -d -o appuser -g appuser -m 755 ${MODELS_DIR} ${DATASETS_DIR} ${DB_DIR} ${LOG_DIR} ${HF_HOME} ${CACHE_DIR}

# Copy non-writable source code into workdir
COPY --from=builder --chown=root:root --chmod=0755 /app /app

# Drop privileges
USER appuser

ENV PATH="/app/.venv/bin:/usr/bin:$PATH" \
    GIT_PYTHON_GIT_EXECUTABLE="/usr/bin/git"

EXPOSE 8000
STOPSIGNAL SIGINT

CMD ["gunicorn", "vectorize.app:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]