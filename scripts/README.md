# Database Management Scripts

## reset_db.py

A comprehensive database management tool for resetting and managing the Autotuner database.

### Features

- **Drop and recreate tables** - Clear all data while keeping the database file
- **Delete database file** - Completely remove the database file
- **Reset task status** - Reset tasks to PENDING status
- **List tasks** - View all tasks in the database

### Usage

```bash
# Activate virtual environment first
source env/bin/activate

# List all tasks
python scripts/reset_db.py --list-tasks

# Reset specific task to PENDING
python scripts/reset_db.py --task-id 1

# Reset all tasks to PENDING
python scripts/reset_db.py --reset-tasks

# Drop all tables and recreate (clears all data)
python scripts/reset_db.py --drop-tables

# Delete database file completely
python scripts/reset_db.py --delete-db
```

---

# Worker Management Scripts

## Architecture Overview

The Autotuner uses **ARQ (Async Redis Queue)** for background job processing:

```
┌──────────────┐    Enqueue Job    ┌───────────┐
│  Web Server  │ ────────────────> │   Redis   │
│  (FastAPI)   │                   │   Queue   │
└──────────────┘                   └─────┬─────┘
                                         │
                                    Pick up job
                                         │
                                         ▼
                                  ┌──────────────┐
                                  │  ARQ Worker  │
                                  │   Process    │
                                  └──────────────┘
                                         │
                                    Execute
                                         │
                                         ▼
                              run_autotuning_task()
                              (Orchestrator runs experiments)
```

**Key Points:**
- Web server and worker are **separate processes**
- Redis acts as message broker (must be running)
- Worker picks up jobs and executes `run_autotuning_task()`
- Both can run independently or together

---

## Quick Start (Development)

### start_dev.sh

Start both web server and ARQ worker with one command:

```bash
./scripts/start_dev.sh
```

**What it does:**
- Starts ARQ worker in background
- Starts web server in foreground
- Logs worker output to `logs/worker.log`
- Stops both when you press Ctrl+C

**Example output:**
```
🚀 Starting Autotuner Development Environment...

📋 Starting ARQ worker...
   Worker PID: 12345
   Logs: logs/worker.log

🌐 Starting web server...
   API: http://localhost:8000
   Docs: http://localhost:8000/docs

Press Ctrl+C to stop both services
```

---

## Manual Control (Separate Processes)

### start_worker.sh

Start only the ARQ worker:

```bash
./scripts/start_worker.sh
```

**What it does:**
- Sets PYTHONPATH correctly
- Starts ARQ worker with verbose logging
- Runs in foreground (see real-time logs)

**Check worker status:**
```bash
# See if worker is running
ps aux | grep arq | grep -v grep

# Monitor logs
tail -f logs/worker.log
```

### Start Web Server Only

```bash
# Activate environment
source env/bin/activate

# Start server
python src/web/server.py
```

---

## Production Deployment (systemd)

### Install Services

```bash
# Copy service files
sudo cp scripts/autotuner-worker.service /etc/systemd/system/
sudo cp scripts/autotuner-web.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable autotuner-worker
sudo systemctl enable autotuner-web

# Start services
sudo systemctl start autotuner-worker
sudo systemctl start autotuner-web
```

### Manage Services

```bash
# Check status
sudo systemctl status autotuner-worker
sudo systemctl status autotuner-web

# View logs (real-time)
sudo journalctl -u autotuner-worker -f
sudo journalctl -u autotuner-web -f

# Restart services
sudo systemctl restart autotuner-worker
sudo systemctl restart autotuner-web

# Stop services
sudo systemctl stop autotuner-worker
sudo systemctl stop autotuner-web
```

---

## Troubleshooting

### Worker not processing jobs?

1. **Check if worker is running:**
   ```bash
   ps aux | grep arq
   ```

2. **Check Redis is running:**
   ```bash
   ps aux | grep redis
   ```

3. **Check worker logs:**
   ```bash
   # Development
   tail -f logs/worker.log

   # Production (systemd)
   sudo journalctl -u autotuner-worker -f
   ```

4. **Test Redis connection:**
   ```bash
   # Install redis-cli first: apt-get install redis-tools
   redis-cli ping
   # Should respond: PONG
   ```

### "ModuleNotFoundError: No module named 'web'"

**Cause:** PYTHONPATH not set correctly

**Fix:**
```bash
export PYTHONPATH="/path/to/autotuner/src:$PYTHONPATH"
```

Or use the provided scripts which set this automatically.

### Jobs stuck in queue?

**Check delayed jobs:**
```bash
# Start worker with verbose logging
./scripts/start_worker.sh
```

Look for lines like:
```
8314.22s → 39d110bc186c471d8aa669052f5b7fc6:run_autotuning_task(1) delayed=8314.22s
```

This means jobs were enqueued but worker wasn't running. They will process now.

---

## Architecture Decision: Separate vs Together?

### ✅ Recommended: Separate Processes

**Why separate is better:**

1. **Scalability:** Run multiple workers for parallel job processing
2. **Reliability:** Worker crash doesn't kill web server
3. **Deployment:** Restart workers without API downtime
4. **Resource Management:** Different memory/CPU limits per service
5. **Standard Pattern:** Industry best practice for job queues

**Production Setup:**
```bash
# Web server on port 8000
systemctl start autotuner-web

# Multiple workers (if needed)
systemctl start autotuner-worker
# Could run autotuner-worker@1, autotuner-worker@2, etc.
```

### Development Convenience

Use `start_dev.sh` for quick development - it handles both processes but keeps them architecturally separate.

---

## Files

- `start_dev.sh` - Start both services for development
- `start_worker.sh` - Start worker only
- `autotuner-worker.service` - Systemd service for worker
- `autotuner-web.service` - Systemd service for web server
- `reset_db.py` - Database management tool
- `README.md` - This file

---

## Requirements

- Redis server running on localhost:6379
- Python virtual environment activated
- All dependencies installed (`pip install -r requirements.txt`)

---

## Database Location

The database is stored at:
```
~/.local/share/autotuner/autotuner.db
```
