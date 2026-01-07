#!/bin/bash
set -e
cd /home/claude/work/inference-autotuner

# Create logs directory
mkdir -p logs

echo "=== Stopping existing processes ==="
pkill -9 -f "server.py" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
pkill -9 -f "arq.*autotuner_worker" 2>/dev/null || true
pkill -9 -f "vite" 2>/dev/null || true
sleep 2

echo "=== Starting Backend Server ==="
source env/bin/activate
cd src
nohup python web/server.py > ../logs/server.log 2>&1 &
echo "Backend started"
cd ..
sleep 3

echo "=== Starting ARQ Worker ==="
cd src
export WORKER_ID=31999072c71de23f-917c98c4
export HTTP_PROXY=http://172.17.0.1:1081
export HTTPS_PROXY=http://172.17.0.1:1081
nohup ../env/bin/arq web.workers.autotuner_worker.WorkerSettings > ../logs/worker.log 2>&1 &
echo $! > ../logs/worker.pid
echo "Worker started"
cd ..
sleep 3

echo "=== Starting Frontend ==="
cd frontend
nohup npm run dev > ../logs/frontend.log 2>&1 &
echo "Frontend started"
cd ..
sleep 3

echo "=== Verifying Services ==="
curl -s http://localhost:9002/api/health > /dev/null && echo "Backend: OK" || echo "Backend: STARTING..."
curl -s http://localhost:9001 > /dev/null && echo "Frontend: OK" || echo "Frontend: STARTING..."
echo "Done!"
