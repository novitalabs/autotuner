# Stage 1: Build frontend
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Copy frontend source
COPY frontend/package*.json ./
RUN npm ci --prefer-offline

COPY frontend/ ./
RUN npm run build


# Stage 2: Main application
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (Ubuntu 24.04 has Python 3.12 by default)
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-venv python3-pip \
    redis-server \
    git curl wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Create virtual environment
RUN python3 -m venv /app/env
ENV PATH="/app/env/bin:$PATH"
ENV VIRTUAL_ENV=/app/env

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install SGLang first (it will pull its compatible torch version)
# Use specific version pins to avoid dependency resolution issues
RUN pip install --no-cache-dir "sglang[all]>=0.4.0"

# Install vLLM separately - use a version compatible with SGLang's torch
# Force install to handle any remaining conflicts
RUN pip install --no-cache-dir vllm || pip install --no-cache-dir --no-deps vllm && \
    pip install --no-cache-dir ray>=2.9 pandas pyarrow prometheus-client py-cpuinfo

# Install autotuner dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install genai-bench from third_party
COPY third_party/genai-bench /app/third_party/genai-bench
RUN pip install --no-cache-dir /app/third_party/genai-bench

# Copy application code
COPY src/ /app/src/
COPY .env.example /app/.env

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Copy and setup entrypoint
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Environment configuration
ENV PYTHONPATH=/app/src
ENV REDIS_HOST=localhost
ENV REDIS_PORT=6379
ENV REDIS_DB=0
ENV DEPLOYMENT_MODE=local
ENV DATABASE_URL=sqlite+aiosqlite:////data/autotuner.db
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000

# Data directories
RUN mkdir -p /data/logs

VOLUME ["/data", "/models"]
EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
