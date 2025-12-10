#!/bin/bash
# Start both web server and ARQ worker for development

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Create logs directory if not exists
mkdir -p logs

# Activate virtual environment
source env/bin/activate

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Stop any existing server on port 8000
echo "🔍 Checking for existing server on port 8000..."
if fuser 8000/tcp > /dev/null 2>&1; then
    echo "   Killing existing server on port 8000..."
    fuser -k 8000/tcp > /dev/null 2>&1
    sleep 2
fi

# Stop any existing ARQ worker
echo "🔍 Checking for existing ARQ worker..."
EXISTING_WORKER=$(pgrep -f "arq.*autotuner_worker" 2>/dev/null)
if [ -n "$EXISTING_WORKER" ]; then
    echo "   Killing existing worker PIDs: $EXISTING_WORKER"
    kill $EXISTING_WORKER 2>/dev/null
    sleep 2
fi

# Load environment variables from .env if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "📋 Loading environment variables from .env..."
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Load environment variables from .env.local if it exists (overrides .env)
if [ -f "$PROJECT_ROOT/.env.local" ]; then
    echo "📋 Loading environment variables from .env.local..."
    set -a
    source "$PROJECT_ROOT/.env.local"
    set +a
fi

# Verify proxy configuration
if [ -n "$HTTP_PROXY" ]; then
    echo "   HTTP_PROXY configured: $HTTP_PROXY"
fi
if [ -n "$HTTPS_PROXY" ]; then
    echo "   HTTPS_PROXY configured: $HTTPS_PROXY"
fi

# Unset HF_HUB_OFFLINE to allow genai-bench to fetch tokenizer info from HuggingFace
unset HF_HUB_OFFLINE


echo "🚀 Starting Inference Autotuner Development Environment..."
echo ""

# Start ARQ worker in background with nohup
echo "📋 Starting ARQ worker with nohup..."
cd "$PROJECT_ROOT/src"
nohup arq web.workers.autotuner_worker.WorkerSettings --verbose > ../logs/worker.log 2>&1 &
cd "$PROJECT_ROOT"
sleep 1
WORKER_PID=$(pgrep -f "arq web.workers.autotuner_worker.WorkerSettings")
echo $WORKER_PID > logs/worker.pid
echo "   Worker PID: $WORKER_PID"
echo "   Logs: logs/worker.log"
echo ""

# Wait a moment for worker to start
sleep 2

# Start web server (foreground)
echo "🌐 Starting web server..."
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop web server (worker will continue running)"
echo "To stop worker: kill \$(cat logs/worker.pid)"
echo "---"

# Trap Ctrl+C to only kill web server (worker survives)
trap "echo ''; echo '🛑 Stopping web server...'; echo '   Worker still running with PID: $WORKER_PID'; exit" INT TERM

# Start web server in foreground (from src directory for proper imports)
cd "$PROJECT_ROOT/src"
python web/server.py

echo ""
echo "Web server stopped. Worker still running with PID: $WORKER_PID"
