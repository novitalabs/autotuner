# Troubleshooting Guide

This document covers common issues and their solutions when using the LLM Inference Autotuner.

## Common Issues and Solutions

### 1. InferenceService Creation Fails: "cannot unmarshal number into Go struct field"

**Error:**
```
cannot unmarshal number into Go struct field ObjectMeta.labels of type string
```

**Cause:** Label values in Kubernetes must be strings, but numeric values were provided.

**Solution:** Already fixed in templates. Labels are now quoted:
```yaml
labels:
  autotuner.io/experiment-id: "{{ experiment_id }}"  # Quoted
```

### 2. Deployment Fails: "spec.template.spec.containers[0].name: Required value"

**Error:**
```
spec.template.spec.containers[0].name: Required value
```

**Cause:** Container name was missing in the runner specification.

**Solution:** Already fixed. Template now includes:
```yaml
runner:
  name: ome-container  # Required field
```

### 3. SGLang Fails: "Can't load the configuration of '$MODEL_PATH'"

**Error:**
```
OSError: Can't load the configuration of '$MODEL_PATH'
```

**Cause:** Environment variable not being expanded in the args list.

**Solution:** Already fixed. Template uses K8s env var syntax:
```yaml
args:
  - --model-path
  - $(MODEL_PATH)  # Proper K8s env var expansion
```

### 4. BenchmarkJob Creation Fails: "spec.outputLocation: Required value"

**Error:**
```
spec.outputLocation: Required value
```

**Cause:** OME BenchmarkJob CRD requires an output storage location.

**Solution:** Already fixed. Template includes:
```yaml
outputLocation:
  storageUri: "pvc://benchmark-results-pvc/{{ benchmark_name }}"
```

Make sure the PVC exists:
```bash
kubectl apply -f config/benchmark-pvc.yaml
```

### 5. BenchmarkJob Fails: "unknown storage type for URI: local:///"

**Error:**
```
unknown storage type for URI: local:///tmp/...
```

**Cause:** OME only supports `pvc://` (Persistent Volume Claims) and `oci://` (Object Storage).

**Solution:** Use PVC storage (already configured):
```bash
# Create PVC first
kubectl apply -f config/benchmark-pvc.yaml

# Template automatically uses pvc://benchmark-results-pvc/
```

### 6. InferenceService Not Becoming Ready

**Symptoms:**
- InferenceService shows `Ready=False`
- Status: "ComponentNotReady: Target service not ready for ingress creation"

**Debugging Steps:**
```bash
# Check pod status
kubectl get pods -n autotuner

# Check pod logs
kubectl logs <pod-name> -n autotuner --tail=50

# Check InferenceService events
kubectl describe inferenceservice <name> -n autotuner
```

**Common Causes:**
- Model not found or not ready
- Runtime mismatch with model
- Insufficient GPU resources
- Container image pull errors

**Typical Wait Time:** 60-90 seconds for model loading and CUDA graph capture

### 7. GPU Resource Issues in Minikube

**Problem:** Minikube with Docker driver cannot access host GPUs

**Symptoms:**
- Pods pending with: `Insufficient nvidia.com/gpu`
- NVIDIA device plugin shows: `"No devices found. Waiting indefinitely"`
- Even with `minikube start --gpus=all` flag

**Root Cause:**
Nested containerization architecture prevents GPU access:
```
Host (with GPUs) → Docker → Minikube Container → Inner Docker → K8s Pods
```
The inner Docker daemon cannot see host GPUs even when outer Docker has GPU access.

**Solutions:**

**Option A: Use Minikube with --driver=none** (Requires bare metal)
```bash
# CAUTION: This runs Kubernetes directly on host (no container isolation)
minikube start --driver=none
```

**Option B: Use proper Kubernetes cluster**
- Production K8s with NVIDIA GPU Operator
- Kind with GPU support
- K3s with proper GPU configuration

**Option C: Direct Docker deployment** (Development/Testing)
For quick testing without Kubernetes orchestration:
```bash
# Download model
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /tmp/llama-3.2-1b-instruct

# Run SGLang directly with Docker
docker run --gpus '"device=0"' -d --name sglang-llama \
  -p 8000:8080 \
  -v /tmp/llama-3.2-1b-instruct:/model \
  lmsysorg/sglang:v0.5.2-cu126 \
  python3 -m sglang.launch_server \
  --model-path /model \
  --host 0.0.0.0 \
  --port 8080 \
  --mem-frac 0.6

# Verify deployment
curl http://localhost:8000/health
```

**Important Notes:**
- Check GPU availability first: `nvidia-smi`
- Select a GPU with sufficient free memory
- Adjust `--mem-frac` based on available GPU memory
- Use `device=N` to select specific GPU (0-7)

### 8. SGLang CPU Backend Issues

**Problem:** SGLang CPU version crashes in containers

**Symptoms:**
- Pod logs stop at "Load weight end"
- Scheduler subprocess becomes defunct (zombie process)
- Server never starts or responds

**Root Cause:**
SGLang CPU backend (`lmsysorg/sglang:v0.5.3.post3-xeon`) has subprocess management issues in containerized environments.

**Solution:**
Use GPU-based deployment instead. CPU inference is not recommended for production or testing.

### 9. Model Download and Transfer Issues

#### Problem A: Gated Model Access Denied

**Error:**
```
401 Client Error: Unauthorized
Access to model meta-llama/Llama-3.2-1B-Instruct is restricted
```

**Solution:**
```bash
# 1. Accept license on HuggingFace website
# Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

# 2. Create access token on HuggingFace
# Visit: https://huggingface.co/settings/tokens

# 3. Create Kubernetes secret (for OME)
kubectl create secret generic hf-token \
  --from-literal=token=<your-token> \
  -n ome

# 4. Or login locally (for direct download)
huggingface-cli login --token <your-token>
```

#### Problem B: Transferring Large Model Files to Minikube

**Failed Methods:**
- `minikube cp /dir` → "Is a directory" error
- `minikube cp large.tar.gz` → "scp: Broken pipe" (files > 1GB)
- `cat file | minikube ssh` → Signal INT
- `rsync` → "protocol version mismatch"

**Working Solution:**
```bash
# Compress model files
tar czf /tmp/model.tar.gz -C /tmp llama-3.2-1b-instruct

# Transfer using SCP with Minikube SSH key
scp -i $(minikube ssh-key) /tmp/model.tar.gz \
  docker@$(minikube ip):~/

# Extract inside Minikube
minikube ssh "sudo mkdir -p /mnt/data/models && \
  sudo tar xzf ~/model.tar.gz -C /mnt/data/models/"

# Verify
minikube ssh "ls -lh /mnt/data/models/llama-3.2-1b-instruct"
```

**Size Reference:**
- Llama 3.2 1B: ~2.4GB uncompressed, ~887MB compressed
- Transfer time: ~30-60 seconds depending on disk speed

### 10. Docker GPU Out of Memory

**Symptoms:**
- Container starts but crashes during model loading
- Error: `torch.OutOfMemoryError: CUDA out of memory`
- CUDA graph capture fails

**Debugging:**
```bash
# Check GPU status and memory usage
nvidia-smi

# Look for existing workloads
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv
```

**Solutions:**

**A. Select a different GPU:**
```bash
# Use GPU 1 instead of GPU 0
docker run --gpus '"device=1"' ...
```

**B. Reduce memory allocation:**
```bash
# Reduce --mem-frac parameter
--mem-frac 0.6  # Instead of 0.8
```

**C. Stop competing workloads:**
```bash
# Identify process using GPU
ps aux | grep <pid-from-nvidia-smi>

# Kill if safe to do so
kill <pid>
```

**Memory Allocation Guide:**
- Small models (1-3B): `--mem-frac 0.6-0.7`
- Medium models (7-13B): `--mem-frac 0.8-0.85`
- Large models (70B+): `--mem-frac 0.9-0.95`

Always leave 10-20% GPU memory free for activations and temporary tensors.

### 11. Wrong Model or Runtime Name

**Symptoms:**
- InferenceService fails to create
- Error about model or runtime not found

**Solution:**
```bash
# List available models
kubectl get clusterbasemodels

# List available runtimes
kubectl get clusterservingruntimes

# Update examples/simple_task.json with correct names
```

### 12. Network and Proxy Configuration

**Problem:** Images can't be pulled, models can't be downloaded in Minikube

**Symptoms:**
- `ImagePullBackOff` errors
- OME model agent can't download models
- Connection timeouts

**Solution: Configure Docker proxy in Minikube**

```bash
# SSH into Minikube
minikube ssh

# Create proxy configuration
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=http://YOUR_PROXY:PORT"
Environment="HTTPS_PROXY=http://YOUR_PROXY:PORT"
Environment="NO_PROXY=localhost,127.0.0.1,10.96.0.0/12"
EOF

# Reload and restart Docker
sudo systemctl daemon-reload
sudo systemctl restart docker

# Exit Minikube SSH
exit

# Restart Minikube to apply changes
minikube stop && minikube start
```

**Configure OME Model Agent:**
```bash
# Patch model agent DaemonSet
kubectl set env daemonset/ome-model-agent \
  -n ome \
  HTTP_PROXY=http://YOUR_PROXY:PORT \
  HTTPS_PROXY=http://YOUR_PROXY:PORT \
  NO_PROXY=localhost,127.0.0.1,10.96.0.0/12

# Wait for pods to restart
kubectl rollout status daemonset/ome-model-agent -n ome
```

### 13. BenchmarkJob Stays in "Running" Status

**Symptoms:**
- BenchmarkJob doesn't complete
- No error messages

**Debugging:**
```bash
# Check benchmark pod logs
kubectl get pods -n autotuner | grep bench
kubectl logs <benchmark-pod> -n autotuner

# Check BenchmarkJob status
kubectl describe benchmarkjob <name> -n autotuner
```

**Common Causes:**
- InferenceService endpoint not reachable
- Traffic scenarios too demanding
- Timeout settings too low

## Monitoring Tips

### Watch Resources in Real-Time

```bash
# All resources in autotuner namespace
watch kubectl get inferenceservices,benchmarkjobs,pods -n autotuner

# Just InferenceServices
kubectl get inferenceservices -n autotuner -w

# Pod logs
kubectl logs -f <pod-name> -n autotuner
```

### Check OME Controller Logs

```bash
kubectl logs -n ome deployment/ome-controller-manager --tail=100
```

## Performance Tips

1. **Reduce timeout values** for faster iteration during development:
   ```json
   "optimization": {
     "timeout_per_iteration": 300  // 5 minutes instead of 10
   }
   ```

2. **Use smaller benchmark workloads** for testing:
   ```json
   "benchmark": {
     "traffic_scenarios": ["D(100,100)"],  // Lighter load
     "max_requests_per_iteration": 50      // Fewer requests
   }
   ```

3. **Limit parameter grid** for initial testing:
   ```json
   "parameters": {
     "mem_frac": {"type": "choice", "values": [0.85, 0.9]}  // Just 2 values
   }
   ```

---

## Viewing GenAI-Bench Logs

### Verbose Mode for Real-Time Output

Use the `--verbose` or `-v` flag to stream genai-bench output in real-time:

```bash
python src/run_autotuner.py examples/docker_task.json --mode docker --direct --verbose
```

**Benefits**:
- Real-time feedback during benchmarks
- See progress during long runs
- Useful for debugging connection/API issues
- Detect problems early

**Default mode** (no flag) shows output only after completion.

### Usage Examples

**Debugging connection issues:**
```bash
python src/run_autotuner.py examples/docker_task.json --mode docker --direct --verbose
```

**Long-running benchmarks with log file:**
```bash
python src/run_autotuner.py examples/docker_task.json --mode docker --direct --verbose 2>&1 | tee autotuner.log
```

### Manual Log Inspection

View benchmark results directly:
```bash
# List all benchmark results
ls -R benchmark_results/

# View specific experiment metadata
cat benchmark_results/docker-simple-tune-exp1/experiment_metadata.json
```

### When to Use Verbose Mode

**Use verbose for**:
- Initial testing of new configurations
- Debugging connection/API issues  
- Long-running benchmarks (>5 minutes)
- Monitoring progress

**Use default mode for**:
- Production runs
- CI/CD pipelines
- Multiple parallel experiments

