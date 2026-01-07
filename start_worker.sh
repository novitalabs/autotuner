#!/bin/bash
cd /home/claude/work/inference-autotuner/src
export WORKER_ID=31999072c71de23f-917c98c4
export HTTP_PROXY=http://172.17.0.1:1081
export HTTPS_PROXY=http://172.17.0.1:1081
exec ../env/bin/arq web.workers.autotuner_worker.WorkerSettings
