#!/usr/bin/env bash
# Launch the Profile Photo Studio with CUDA support
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

# Point onnxruntime at all pip-installed NVIDIA CUDA libraries
NVIDIA_LIBS=$(python3 -c "
import importlib
libs = []
for pkg in ['nvidia.cublas.lib', 'nvidia.cudnn.lib', 'nvidia.cufft.lib', 'nvidia.curand.lib', 'nvidia.nvjitlink.lib', 'nvidia.cuda_runtime.lib', 'nvidia.cuda_nvrtc.lib']:
    try:
        m = importlib.import_module(pkg)
        libs.append(m.__path__[0])
    except: pass
print(':'.join(libs))
")
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"

python3 app.py "$@"
