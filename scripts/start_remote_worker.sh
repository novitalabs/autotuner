#!/bin/bash
# Start ARQ worker with worker-specific configuration
# This script is for remote workers connecting to a central manager

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Create log directory early (needed for SSH tunnel logs)
mkdir -p logs

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

# Default Redis settings
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

# Setup SSH tunnel if MANAGER_SSH is configured
if [ -n "$MANAGER_SSH" ]; then
    echo "Setting up SSH tunnel to manager..."

    # Kill any existing tunnel
    pkill -f "ssh.*-L.*:6379:localhost:6379" 2>/dev/null || true
    sleep 1

    # Parse SSH command for tunnel (e.g., "ssh -p 33773 user@host")
    # Start SSH tunnel in background
    TUNNEL_CMD="$MANAGER_SSH -N -L 16379:localhost:6379 -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -o ServerAliveCountMax=3"
    nohup $TUNNEL_CMD > logs/ssh_tunnel.log 2>&1 &
    TUNNEL_PID=$!
    echo $TUNNEL_PID > logs/tunnel.pid

    # Wait for tunnel to establish
    sleep 3

    # Check if tunnel is running
    if ! ps -p $TUNNEL_PID > /dev/null 2>&1; then
        echo "ERROR: SSH tunnel failed to start. Check logs/ssh_tunnel.log"
        cat logs/ssh_tunnel.log 2>/dev/null || true
        exit 1
    fi

    # Use localhost through tunnel
    REDIS_HOST="localhost"
    REDIS_PORT="16379"
    export REDIS_HOST REDIS_PORT
    echo "✓ SSH tunnel established (PID: $TUNNEL_PID)"
    echo "  Forwarding localhost:16379 -> manager:6379"
fi

# Check if Redis is reachable
echo "Checking Redis connection at $REDIS_HOST:$REDIS_PORT..."
RETRY=0
MAX_RETRY=5
while ! nc -z "$REDIS_HOST" "$REDIS_PORT" 2>/dev/null; do
    RETRY=$((RETRY + 1))
    if [ $RETRY -ge $MAX_RETRY ]; then
        echo "ERROR: Cannot connect to Redis at $REDIS_HOST:$REDIS_PORT"
        echo "Make sure the manager's Redis is accessible from this machine."
        echo ""
        echo "Options:"
        echo "  1. Expose Redis on manager: bind 0.0.0.0 in redis.conf"
        echo "  2. Use SSH tunnel: set MANAGER_SSH in .env.worker"
        exit 1
    fi
    echo "  Waiting for Redis... ($RETRY/$MAX_RETRY)"
    sleep 2
done
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
