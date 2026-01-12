#!/bin/bash
set -e

# Add torch library path to LD_LIBRARY_PATH for sgl_kernel native libraries
export LD_LIBRARY_PATH="/app/env/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"

# Create data directories if not exist
mkdir -p /data/logs

# Start Redis in background
echo "Starting Redis..."
redis-server --daemonize yes --dir /data

# Wait for Redis to be ready
until redis-cli ping > /dev/null 2>&1; do
    echo "Waiting for Redis..."
    sleep 1
done
echo "Redis is ready"

# Start ARQ worker in background
echo "Starting ARQ worker..."
cd /app/src
nohup arq web.workers.autotuner_worker.WorkerSettings > /data/logs/worker.log 2>&1 &
WORKER_PID=$!
echo "ARQ worker started (PID: $WORKER_PID)"

# Handle signals for graceful shutdown
cleanup() {
    echo "Shutting down..."
    kill $WORKER_PID 2>/dev/null || true
    redis-cli shutdown 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Start FastAPI server in foreground
echo "Starting FastAPI server on port ${SERVER_PORT}..."
cd /app/src
exec python web/server.py
