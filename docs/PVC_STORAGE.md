# PVC Storage Implementation for OME Models

## Overview

This document describes the PVC (Persistent Volume Claim) storage implementation for the inference autotuner with OME. The implementation provides a way to use Kubernetes PVCs as storage for model weights instead of relying on hostPath volumes.

## Background

The OME (Open Model Engine) project has a design document (OEP-0004) for PVC storage support. However, based on testing with the current OME version, the full `pvc://` URI-based implementation is not yet complete in the controller.

The current OME InferenceService controller only supports hostPath volumes for model storage (see `third_party/ome/pkg/controller/v1beta1/inferenceservice/components/base.go:262-274`).

## What Was Implemented

### 1. Created PersistentVolume and PersistentVolumeClaim

Created a PV backed by local storage pointing to `/raid/models` on worker nodes where the model-agent has downloaded models:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-storage-pv
spec:
  capacity:
    storage: 500Gi
  volumeMode: Filesystem
  accessModes:
    - ReadOnlyMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: local-storage
  local:
    path: /raid/models
  nodeAffinity:
    required:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - host-10-97-65-155  # Node where models are downloaded
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage-pvc
  namespace: autotuner
spec:
  accessModes:
    - ReadOnlyMany
  volumeMode: Filesystem
  resources:
    requests:
      storage: 500Gi
  storageClassName: local-storage
```

**Location**: `/tmp/model-storage-pv-single-node.yaml`

### 2. Verified PVC Access

Created test pods to verify that the PVC correctly provides access to model files at `/raid/models/meta/llama-3-2-1b-instruct`.

**Test Result**: ✅ PVC successfully mounts and model files are accessible.

## Current Status

### What Works

1. ✅ **PV/PVC Creation**: Successfully created PV backed by local storage at `/raid/models`
2. ✅ **PVC Binding**: PVC successfully binds to PV
3. ✅ **File Access**: Test pods can successfully mount PVC and access model files
4. ✅ **Storage Backend**: Model-agent successfully downloads models to `/raid/models` on nodes

### What Doesn't Work (Yet)

1. ❌ **OEP-0004 pvc:// URI Format**: ClusterBaseModel with `storageUri: "pvc://namespace:pvc-name/path"` gets stuck in `In_Transit` state
2. ❌ **Automatic PVC Volume Injection**: InferenceService controller doesn't automatically use PVC volumes - only supports hostPath
3. ❌ **Metadata Extraction Jobs**: Controller doesn't create metadata extraction jobs for PVC storage types

## Workaround Approach

Since the `pvc://` URI format isn't fully implemented, a manual patching approach was attempted:

1. Create InferenceService with existing ClusterBaseModel (hostPath-based)
2. Manually patch the resulting Deployment to replace hostPath volume with PVC volume:

```bash
kubectl patch deployment <isvc-name>-engine -n autotuner --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/volumes/1",
    "value": {
      "name": "llama-3-2-1b-instruct",
      "persistentVolumeClaim": {
        "claimName": "model-storage-pvc",
        "readOnly": true
      }
    }
  },
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/volumeMounts/1",
    "value": {
      "name": "llama-3-2-1b-instruct",
      "mountPath": "/raid/models/meta/llama-3-2-1b-instruct",
      "subPath": "meta/llama-3-2-1b-instruct",
      "readOnly": true
    }
  }
]'
```

**Limitation**: This workaround encountered issues with SGLang not being able to load the model configuration, indicating possible path or permission problems.

## Implementation Gaps in Current OME Version

Based on code analysis, the following components would need to be implemented for full PVC support:

1. **BaseModel Controller** (`pkg/controller/v1beta1/basemodel/`):
   - Add PVC URI parsing and validation
   - Create metadata extraction jobs for PVC storage
   - Update BaseModel status based on job completion

2. **InferenceService Controller** (`pkg/controller/v1beta1/inferenceservice/components/base.go`):
   - Modify `UpdatePodSpecVolumes()` to detect PVC storage type
   - Generate PVC volume instead of hostPath volume
   - Remove node selector requirements for PVC storage

3. **Model Agent** (`pkg/modelagent/gopher.go`):
   - Skip PVC storage types (already partially implemented)

4. **Storage Utils** (`pkg/utils/storage/storage.go`):
   - `ParsePVCStorageURI()` function exists but not used by controllers

## Recommendations

### For Future Implementation

1. **Wait for OME OEP-0004 Completion**: Monitor the OME project for completion of PVC storage support
2. **Use NFS or Shared Storage**: For multi-node model access, consider using NFS-backed PVCs
3. **Stick with HostPath for Now**: The current hostPath approach works reliably when the model-agent is properly configured

### For Current Use

Continue using the hostPath storage approach with the model-agent:
- Models are downloaded to `/raid/models` on worker nodes
- InferenceService pods mount via hostPath
- Node selector ensures pods run on nodes with models

## Files Created

- `/tmp/model-storage-pv-single-node.yaml`: PV/PVC definitions
- `/tmp/clusterbasemodel-pvc.yaml`: Example ClusterBaseModel with pvc:// URI (non-functional in current version)
- `/tmp/pvc-test-pod.yaml`: Test pod for verifying PVC access
- `/tmp/pvc-debug-pod.yaml`: Debug pod for inspecting model files

## Key Learnings

1. **Model Storage Location**: OME model-agent stores models at `/raid/models`, not `/mnt/data/models`
2. **Node Affinity**: PVs with local storage must specify nodeAffinity matching where models exist
3. **SubPath Required**: When mounting PVC, use `subPath` to access specific model directories
4. **OEP-0004 Status**: The PVC storage feature design exists but implementation is incomplete in current OME version

## References

- OEP-0004 Document: `third_party/ome/oeps/0004-pvc-storage-support/README.md`
- OME InferenceService Components: `third_party/ome/pkg/controller/v1beta1/inferenceservice/components/base.go`
- Storage Utils: `third_party/ome/pkg/utils/storage/storage.go`
