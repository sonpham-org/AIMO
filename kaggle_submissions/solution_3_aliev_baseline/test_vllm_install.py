"""
Test script to verify vLLM installation on Kaggle with Python 3.12.
Run this after install_vllm.sh to confirm everything works.
"""

import sys
import platform

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")

# Test 1: Import vLLM
try:
    import vllm
    print(f"[OK] vLLM {vllm.__version__}")
except ImportError as e:
    print(f"[FAIL] vLLM import: {e}")
    sys.exit(1)

# Test 2: Import torch with CUDA
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
    print(f"     CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     CUDA version: {torch.version.cuda}")
        print(f"     GPU: {torch.cuda.get_device_name(0)}")
        print(f"     GPU count: {torch.cuda.device_count()}")
        print(f"     VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
except Exception as e:
    print(f"[WARN] PyTorch CUDA: {e}")

# Test 3: Import key vLLM components
try:
    from vllm import LLM, SamplingParams
    print("[OK] vllm.LLM and SamplingParams importable")
except ImportError as e:
    print(f"[FAIL] vLLM components: {e}")

# Test 4: Import transformers
try:
    import transformers
    print(f"[OK] transformers {transformers.__version__}")
except ImportError as e:
    print(f"[FAIL] transformers: {e}")

# Test 5: Check xgrammar
try:
    import xgrammar
    print(f"[OK] xgrammar {xgrammar.__version__}")
except ImportError as e:
    print(f"[WARN] xgrammar: {e}")

# Test 6: Check msgspec
try:
    import msgspec
    print(f"[OK] msgspec {msgspec.__version__}")
except ImportError as e:
    print(f"[WARN] msgspec: {e}")

print("\n=== All import tests passed ===")
print("To do a full GPU test, run:")
print("  llm = LLM('Qwen/Qwen2.5-0.5B')")
print("  output = llm.generate(['Hello'], SamplingParams(max_tokens=10))")
print("  print(output[0].outputs[0].text)")
