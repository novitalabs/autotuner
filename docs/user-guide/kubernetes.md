# OME Installation Guide

This guide provides step-by-step instructions for installing OME (Open Model Engine), which is a **required prerequisite** for the LLM Inference Autotuner.

## Table of Contents

1. [Why OME is Required](#why-ome-is-required)
2. [Prerequisites](#prerequisites)
3. [Quick Installation](#quick-installation)
4. [Manual Installation](#manual-installation)
5. [Verification](#verification)
6. [Post-Installation Setup](#post-installation-setup)
7. [Troubleshooting](#troubleshooting)

---

## Why OME is Required

The inference-autotuner depends on OME for core functionality:

- **InferenceService Deployment**: OME creates and manages SGLang inference servers with configurable parameters
- **Parameter Control**: Allows programmatic control of runtime parameters (tp_size, mem_frac, batch_size, etc.)
- **BenchmarkJob Execution**: Provides CRDs for running benchmarks against deployed services
- **Model Management**: Handles model loading, caching, and lifecycle
- **Resource Orchestration**: Manages GPU allocation and Kubernetes resources

**Without OME, the autotuner cannot:**
- Automatically deploy InferenceServices with different configurations
- Run parameter tuning experiments
- Execute the core autotuning workflow

---

## Prerequisites

Before installing OME, ensure you have:

### 1. Kubernetes Cluster
- **Version**: v1.28 or later
- **Platforms**: Minikube, Kind, EKS, GKE, AKS, or bare-metal
- **Resources**:
  - At least 4 CPU cores
  - 8GB+ RAM
  - GPU support (for inference workloads)

### 2. kubectl
```bash
kubectl version --client
# Should show v1.22+
```

### 3. Helm (v3+)
```bash
helm version
# Should show v3.x
```

If Helm is not installed:
```bash
# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### 4. Git Submodules
The OME repository must be initialized as a submodule:
```bash
git submodule update --init --recursive
```

---

## Quick Installation

**Recommended Method:** Use the autotuner installation script with the `--install-ome` flag:

```bash
cd /path/to/inference-autotuner
./install.sh --install-ome
```

This automatically installs:
1. cert-manager (required dependency)
2. OME CRDs from OCI registry
3. OME resources from local Helm charts
4. All necessary configurations

**Installation time:** ~3-5 minutes

---

## Manual Installation

If you prefer manual installation or the automatic method fails:

### Step 1: Install cert-manager (Required Dependency)

OME requires cert-manager for TLS certificate management:

```bash
# Add Helm repository
helm repo add jetstack https://charts.jetstack.io
helm repo update

# Install cert-manager
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set crds.enabled=true \
  --wait --timeout=5m
```

**Note:** If you encounter webhook validation issues:
```bash
# Delete webhook configurations
kubectl delete validatingwebhookconfiguration cert-manager-webhook
kubectl delete mutatingwebhookconfiguration cert-manager-webhook
```

### Step 2: Install KEDA (Required Dependency)

OME requires KEDA (Kubernetes Event-Driven Autoscaling) for InferenceService autoscaling:

```bash
# Add Helm repository
helm repo add kedacore https://kedacore.github.io/charts
helm repo update

# Install KEDA
helm install keda kedacore/keda \
  --namespace keda \
  --create-namespace \
  --wait --timeout=5m
```

**Verify KEDA:**
```bash
kubectl get pods -n keda
# Should show keda-operator and keda-metrics-apiserver pods running

kubectl get crd | grep keda
# Should show scaledobjects.keda.sh and related CRDs
```

### Step 3: Install OME CRDs

```bash
# Install from OCI registry (recommended)
helm upgrade --install ome-crd \
  oci://ghcr.io/moirai-internal/charts/ome-crd \
  --namespace ome \
  --create-namespace
```

**Verify CRDs:**
```bash
kubectl get crd | grep ome.io
# Should show: inferenceservices, benchmarkjobs, clusterbasemodels, clusterservingruntimes, etc.
```

### Step 4: Install OME Resources

```bash
# Install from local charts
cd third_party/ome
helm upgrade --install ome charts/ome-resources \
  --namespace ome \
  --wait --timeout=7m
```

**Verify installation:**
```bash
kubectl get pods -n ome
# Should show ome-controller-manager pods running
```

---

## Verification

After installation, verify OME is working correctly:

### 1. Check OME Namespace

```bash
kubectl get namespace ome
# Status should be: Active
```

### 2. Check OME Pods

```bash
kubectl get pods -n ome

# Expected pods (all Running):
# NAME                                       READY   STATUS
# ome-controller-manager-xxx                 2/2     Running
# ome-controller-manager-yyy                 2/2     Running
# ome-controller-manager-zzz                 2/2     Running
# ome-model-agent-xxx                        1/1     Running
```

### 3. Check CRDs

```bash
kubectl get crd | grep ome.io | wc -l
# Should show: 6 (or more)
```

### 4. Check OME Controller Logs

```bash
kubectl logs -n ome deployment/ome-controller-manager --tail=50

# Should show startup logs without errors
```

### 5. Verify API Resources

```bash
kubectl api-resources | grep ome.io

# Should list all OME resources:
# clusterbasemodels          ome.io/v1beta1          false   ClusterBaseModel
# clusterservingruntimes     ome.io/v1beta1          false   ClusterServingRuntime
# inferenceservices          ome.io/v1beta1          true    InferenceService
# benchmarkjobs              ome.io/v1beta1          true    BenchmarkJob
```

---

## Post-Installation Setup

After OME is installed, you need to create model and runtime resources.

### Using Pre-configured Examples

OME includes pre-configured model definitions in `third_party/ome/config/models/`:

```bash
# List available models
ls third_party/ome/config/models/meta/

# Apply Llama 3.2 1B model
kubectl apply -f third_party/ome/config/models/meta/Llama-3.2-1B-Instruct.yaml

# Verify model created
kubectl get clusterbasemodels
```

**Note:** The storage path in these YAMLs may need adjustment based on your setup.

### Creating Custom ClusterBaseModel

Create a custom model configuration:

```yaml
# my-model.yaml
apiVersion: ome.io/v1beta1
kind: ClusterBaseModel
metadata:
  name: my-model
spec:
  displayName: vendor.model-name
  vendor: vendor-name
  disabled: false
  version: "1.0.0"
  storage:
    storageUri: hf://vendor/model-name
    path: /path/to/models/my-model
    # Optional: HuggingFace token for gated models
    # key: "hf-token"
```

Apply:
```bash
kubectl apply -f my-model.yaml
```

### Creating ClusterServingRuntime

Create a runtime configuration for SGLang:

```yaml
# my-runtime.yaml
apiVersion: ome.io/v1beta1
kind: ClusterServingRuntime
metadata:
  name: my-runtime
spec:
  supportedModels:
    - my-model
  engine:
    type: sglang
    version: "v0.4.8.post1"
  image: "docker.io/lmsysorg/sglang:v0.4.8.post1-cu126"
  protocol: "openai"
  resources:
    limits:
      nvidia.com/gpu: 1
      memory: 16Gi
    requests:
      cpu: 2
      memory: 8Gi
  runner:
    command: ["python3", "-m", "sglang.launch_server"]
    args:
      - "--model-path"
      - "$(MODEL_PATH)"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8000"
```

Apply:
```bash
kubectl apply -f my-runtime.yaml
```

### Example Files

The autotuner includes example configurations:
- `config/examples/clusterbasemodel-llama-3.2-1b.yaml`
- `config/examples/clusterservingruntime-sglang.yaml`

**Important:** These are templates and require:
- Valid storage paths
- Model access permissions
- Appropriate resource limits

---

## Troubleshooting

### Issue 1: cert-manager Webhook Timeout

**Symptoms:**
```
Error: failed calling webhook "webhook.cert-manager.io": Post "https://cert-manager-webhook.cert-manager.svc:443/validate?timeout=30s": context deadline exceeded
```

**Cause:** The cert-manager webhook is not responding, often due to networking issues in the cluster.

**Solution:**
```bash
# Delete webhook configurations to bypass validation
kubectl delete validatingwebhookconfiguration cert-manager-webhook
kubectl delete mutatingwebhookconfiguration cert-manager-webhook

# Retry OME installation
helm upgrade --install ome third_party/ome/charts/ome-resources --namespace ome
```

### Issue 2: OME Pods Not Starting

**Symptoms:**
```bash
kubectl get pods -n ome
# Pods in CrashLoopBackOff or Error state
```

**Solutions:**
1. Check pod logs:
   ```bash
   kubectl logs -n ome <pod-name>
   ```

2. Check events:
   ```bash
   kubectl describe pod -n ome <pod-name>
   ```

3. Common issues:
   - Missing RBAC permissions → Reinstall with proper service account
   - Image pull errors → Check network/registry access
   - Resource constraints → Increase node resources

### Issue 2: CRDs Not Installed

**Symptoms:**
```bash
kubectl get crd | grep ome.io
# No output
```

**Solution:**
```bash
# Manually install CRDs
kubectl apply -f third_party/ome/config/crd/
```

### Issue 3: InferenceService Stays in Not Ready

**Symptoms:**
```bash
kubectl get inferenceservices
# Ready=False for extended period
```

**Debugging:**
```bash
# Check InferenceService status
kubectl describe inferenceservice <name>

# Check underlying pods
kubectl get pods -l serving.kserve.io/inferenceservice=<name>

# Check pod logs
kubectl logs <pod-name>
```

**Common causes:**
- Model not found or not downloaded
- Insufficient GPU resources
- Image pull failures
- Incorrect runtime configuration

### Issue 4: GPU Not Allocated

**Symptoms:**
InferenceService pods not getting GPU resources

**Solution:**
1. Verify GPU operator is installed:
   ```bash
   kubectl get nodes -o json | jq '.items[].status.allocatable'
   # Should show nvidia.com/gpu
   ```

2. Install NVIDIA device plugin if missing:
   ```bash
   kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml
   ```

### Issue 5: Model Download Failures

**Symptoms:**
Pods fail with "cannot download model" errors

**Solutions:**
1. Check internet connectivity from pods
2. Configure HuggingFace token if using gated models:
   ```yaml
   # Add to ClusterBaseModel
   source:
     huggingface:
       token:
         secretKeyRef:
           name: hf-token
           key: token
   ```

3. Use pre-downloaded models:
   ```yaml
   # Mount existing model directory
   modelPath: "/path/to/local/models"
   ```

---

## Additional Resources

- **OME GitHub Repository**: https://github.com/sgl-project/ome
- **OME Documentation**: Check `third_party/ome/docs/` for detailed docs
- **SGLang Documentation**: https://github.com/sgl-project/sglang
- **Example Configurations**: See `third_party/ome/examples/`

---

## Quick Verification Script

After installation, run this script to verify everything is working:

```bash
#!/bin/bash
# verify-ome.sh

echo "Checking OME installation..."

# Check namespace
if kubectl get namespace ome &> /dev/null; then
    echo "✅ OME namespace exists"
else
    echo "❌ OME namespace not found"
    exit 1
fi

# Check pods
POD_COUNT=$(kubectl get pods -n ome --no-headers 2>/dev/null | wc -l)
RUNNING_COUNT=$(kubectl get pods -n ome --no-headers 2>/dev/null | grep Running | wc -l)

echo "✅ OME pods: $RUNNING_COUNT/$POD_COUNT running"

if [ "$RUNNING_COUNT" -eq 0 ]; then
    echo "❌ No OME pods are running"
    exit 1
fi

# Check CRDs
CRD_COUNT=$(kubectl get crd | grep ome.io | wc -l)
echo "✅ OME CRDs installed: $CRD_COUNT"

if [ "$CRD_COUNT" -lt 4 ]; then
    echo "❌ Missing OME CRDs (expected at least 4)"
    exit 1
fi

# Check models
MODEL_COUNT=$(kubectl get clusterbasemodels --no-headers 2>/dev/null | wc -l)
echo "✅ ClusterBaseModels: $MODEL_COUNT"

# Check runtimes
RUNTIME_COUNT=$(kubectl get clusterservingruntimes --no-headers 2>/dev/null | wc -l)
echo "✅ ClusterServingRuntimes: $RUNTIME_COUNT"

echo ""
echo "OME installation verified successfully!"
echo "You can now run the autotuner installation: ./install.sh"
```

Save this as `verify-ome.sh`, make it executable, and run:
```bash
chmod +x verify-ome.sh
./verify-ome.sh
```

---

## Next Steps

Once OME is installed and verified:

1. **Run the autotuner installation**:
   ```bash
   cd /root/work/inference-autotuner
   ./install.sh
   ```

2. **Configure your first tuning task**:
   ```bash
   vi examples/simple_task.json
   ```

3. **Start tuning**:
   ```bash
   source env/bin/activate
   python src/run_autotuner.py examples/simple_task.json --direct
   ```

For more information, see the main [README.md](../README.md).
