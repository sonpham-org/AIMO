#!/bin/bash
# Install vLLM 0.8.0 on Kaggle (Python 3.12, CUDA 12.4)
# This script installs from pre-downloaded wheels in the dataset.
#
# Usage on Kaggle:
#   !bash /kaggle/input/vllm-wheels-cp312/install_vllm.sh
#
# Or if the dataset is mounted at a different path:
#   !bash /path/to/install_vllm.sh /path/to/wheels

WHEEL_DIR="${1:-/kaggle/input/vllm-wheels-cp312}"

echo "=== Installing vLLM 0.8.0 from wheels ==="
echo "Wheel directory: $WHEEL_DIR"
echo "Python version: $(python3 --version)"
echo "CUDA version: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"

# Uninstall packages that conflict with vLLM
echo "=== Removing conflicting packages ==="
pip uninstall -y tensorflow keras matplotlib scikit-learn 2>/dev/null

# Install vLLM + deps from wheels (use existing system torch)
echo "=== Installing vLLM wheels ==="
pip install --no-index --find-links "$WHEEL_DIR" vllm==0.8.0 2>&1

# Verify installation
echo "=== Verifying ==="
python3 -c "import vllm; print(f'vLLM {vllm.__version__} installed successfully')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

echo "=== Done ==="
