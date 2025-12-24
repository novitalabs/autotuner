#!/bin/bash
# Start ARQ worker with worker-specific configuration
# This script is for remote workers connecting to a central manager

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load environment variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Load worker-specific config (overrides .env)
if [ -f ".env.worker" ]; then
    set -a
    source .env.worker
    set +a
    echo "Loaded worker config from .env.worker"
fi

# Check if Redis is reachable
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

echo "Checking Redis connection at $REDIS_HOST:$REDIS_PORT..."
if ! nc -z "$REDIS_HOST" "$REDIS_PORT" 2>/dev/null; then
    echo "ERROR: Cannot connect to Redis at $REDIS_HOST:$REDIS_PORT"
    echo "Make sure the manager's Redis is accessible from this machine."
    echo ""
    echo "Options:"
    echo "  1. Expose Redis on manager: bind 0.0.0.0 in redis.conf"
    echo "  2. Use SSH tunnel: ssh -L 6379:localhost:6379 manager-host"
    exit 1
fi
echo "✓ Redis connection OK"

# Check for existing worker process
if [ -f "logs/worker.pid" ]; then
    OLD_PID=$(cat logs/worker.pid)
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Stopping existing worker (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
    rm -f logs/worker.pid
fi

# Create log directory
mkdir -p logs

# Activate virtual environment if it exists
if [ -f "env/bin/activate" ]; then
    source env/bin/activate
    echo "✓ Virtual environment activated"
fi

# Generate worker ID if not set
if [ -z "$WORKER_ID" ]; then
    WORKER_ID="$(hostname)-$(date +%s | tail -c 5)"
    export WORKER_ID
fi
echo "Worker ID: $WORKER_ID"

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPUs detected"
echo ""

# Start worker
echo "Starting ARQ worker..."
echo "  Redis: $REDIS_HOST:$REDIS_PORT"
echo "  Mode: ${DEPLOYMENT_MODE:-docker}"
echo "  Log: logs/worker.log"
echo ""

cd src

# Run in foreground (for debugging) or background
if [ "$1" == "--foreground" ] || [ "$1" == "-f" ]; then
    echo "Running in foreground (Ctrl+C to stop)..."
    python -m arq web.workers.autotuner_worker.WorkerSettings --verbose
else
    nohup python -m arq web.workers.autotuner_worker.WorkerSettings --verbose > ../logs/worker.log 2>&1 &
    WORKER_PID=$!
    echo $WORKER_PID > ../logs/worker.pid
    echo "Worker started (PID: $WORKER_PID)"
    echo ""
    echo "View logs: tail -f logs/worker.log"
    echo "Stop worker: kill \$(cat logs/worker.pid)"
fi
