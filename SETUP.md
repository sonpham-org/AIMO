# Setup Guide: AMD ROCm + Vulkan (Ubuntu 25.10 + Strix Halo / RDNA 3.5)

This guide documents how to get vLLM and llama.cpp (Vulkan) running natively on
AMD Strix Halo (gfx1151) with Ubuntu 25.10. This was hard-won knowledge — ROCm on
consumer RDNA GPUs with a bleeding-edge distro requires several workarounds.

All libraries install to **permanent locations** (`~/amd_libs/`, `~/llama.cpp/`, `~/models/`)
so nothing is lost on reboot.

---

## Why is this hard?

1. **Ubuntu 25.10 ships OpenMPI 5.x** which removed the C++ bindings (`libmpi_cxx.so`)
   that PyTorch ROCm wheels still expect.
2. **Strix Halo (gfx1151)** is RDNA 3.5 — very new silicon with limited ROCm support.
3. **No unified ROCm story** — PyTorch bundles ROCm 6.3, vLLM wheels target ROCm 7.0, system packages are ROCm 5.7. All ABI-incompatible.
4. **vLLM's ROCm wheels** only exist for Python 3.12 + ROCm 7.0.

---

## Part 1: vLLM (ROCm backend)

### Prerequisites

```bash
# C++ compiler (needed for stubs)
sudo apt-get install -y g++ libopenmpi40

# GPU driver — make sure /dev/kfd and /dev/dri exist
ls /dev/kfd /dev/dri/renderD128
```

### Step 1: Install uv and create Python 3.12 venv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv .venv --python 3.12
```

### Step 2: Install vLLM ROCm wheel

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/ --python .venv/bin/python
```

This pulls in `vllm-0.15.0+rocm700` along with a matching PyTorch (`2.9.1+git8907517`
built for ROCm 7.0).

### Step 3: Fix numpy version

```bash
uv pip install "numpy<2.3" --python .venv/bin/python
```

Numba (used by Triton) requires NumPy < 2.3.

### Step 4: Download TheRock SDK (ROCm 7.x runtime libraries)

The vLLM ROCm 7.0 wheel needs runtime libraries (`libroctx64.so.4`, `libhipblas.so.3`,
`libamd_smi.so.26`, etc.) that aren't in Ubuntu's repos. AMD's "TheRock" nightly SDK
provides them.

**Important:** We install to `~/amd_libs/` (permanent) instead of `/tmp` (wiped on reboot).

```bash
mkdir -p ~/amd_libs
cd ~/amd_libs

# Download the SDK for your GPU arch (gfx1151 for Strix Halo)
wget -q "https://github.com/ROCm/TheRock/releases/download/nightly-rocm-7.12/therock-dist-linux-gfx1151-7.12.0a20260129.tar.gz"

# Extract to permanent location
mkdir -p ~/amd_libs/therock-sdk
tar -xf therock-dist-linux-gfx1151-7.12.0a20260129.tar.gz -C ~/amd_libs/therock-sdk --strip-components=1
```

> Check https://github.com/ROCm/TheRock/releases for the latest nightly.
> For other GPUs: use `gfx1100` (RX 7900), `gfx1101` (RX 7800/7700), etc.

### Step 5: Install amdsmi from TheRock SDK

```bash
uv pip install ~/amd_libs/therock-sdk/share/amd_smi/ --python .venv/bin/python
```

Also create a focused lib directory so amdsmi finds the right `.so`:

```bash
mkdir -p ~/amd_libs/amdsmi-lib
ln -sf ~/amd_libs/therock-sdk/lib/libamd_smi.so* ~/amd_libs/amdsmi-lib/
```

### Step 6: Create OpenMPI C++ bindings stub

Ubuntu 25.10's OpenMPI 5.x removed the C++ bindings that PyTorch expects.
We create a tiny stub with the 3 symbols PyTorch actually needs:

```bash
cat > ~/amd_libs/mpi_cxx_stub.cpp << 'EOF'
namespace MPI {
    class Win {
    public:
        __attribute__((visibility("default"), noinline)) void Free();
    };
    void Win::Free() { volatile int x = 0; (void)x; }

    class Comm {
    public:
        __attribute__((visibility("default"), noinline)) Comm();
    };
    Comm::Comm() { volatile int x = 0; (void)x; }

    class Datatype {
    public:
        __attribute__((visibility("default"), noinline)) void Free();
    };
    void Datatype::Free() { volatile int x = 0; (void)x; }
}
extern "C" {
    void ompi_mpi_cxx_op_intercept(void* a, void* b, int* c, void* d) {
        (void)a; (void)b; (void)c; (void)d;
    }
    void ompi_op_set_cxx_callback(void* op, void* fn) {
        (void)op; (void)fn;
    }
}
EOF

g++ -shared -fPIC -O0 -fno-inline -o ~/amd_libs/libmpi_cxx.so.40 ~/amd_libs/mpi_cxx_stub.cpp
```

### Step 7: Run vLLM

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
LD_LIBRARY_PATH=~/amd_libs:~/amd_libs/amdsmi-lib:~/amd_libs/therock-sdk/lib \
.venv/bin/python example.py
```

Or for a quick sanity test with a tiny model:

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
LD_LIBRARY_PATH=~/amd_libs:~/amd_libs/amdsmi-lib:~/amd_libs/therock-sdk/lib \
.venv/bin/python example.py --model "facebook/opt-125m" --max-model-len 512 --gpu-memory-utilization 0.3
```

### Environment Variables Reference

| Variable | Value | Why |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION` | `11.5.1` | Tell ROCm this is gfx1151 (Strix Halo native arch) |
| `LD_LIBRARY_PATH` | `~/amd_libs:~/amd_libs/amdsmi-lib:~/amd_libs/therock-sdk/lib` | MPI stub + amdsmi + ROCm 7.x runtime (permanent) |
| `VLLM_TARGET_DEVICE` | `rocm` | Force vLLM to use ROCm backend |

---

## Part 2: llama.cpp (Vulkan backend) — Recommended

llama.cpp with Vulkan is **2-3x faster than vLLM ROCm** for single-user token generation
on Strix Halo.

### Build

```bash
# Prerequisites (one-time)
sudo apt-get install -y libvulkan-dev glslc cmake vulkan-tools

# Clone to permanent location
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp

# Build with Vulkan
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Verify Vulkan sees your GPU
~/llama.cpp/build/bin/llama-bench --list-devices
# Should show: Radeon 8060S Graphics (RADV GFX1151)
```

### Download GGUF models

```bash
mkdir -p ~/models ~/models/gpt-oss-120b

# Top pick: GPT-OSS-120B MoE (117B total, 5.1B active — 55 tok/s)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/gpt-oss-120b-GGUF', 'Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf', local_dir='$HOME/models/gpt-oss-120b')
hf_hub_download('unsloth/gpt-oss-120b-GGUF', 'Q4_K_M/gpt-oss-120b-Q4_K_M-00002-of-00002.gguf', local_dir='$HOME/models/gpt-oss-120b')
"

# GPT-OSS-20B MoE (20.9B total, 3.6B active — 79 tok/s)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/gpt-oss-20b-GGUF', 'gpt-oss-20b-Q4_K_M.gguf', local_dir='$HOME/models')
"

# Qwen3-30B-A3B MoE (30B total, 3B active — 89 tok/s)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3-30B-A3B-GGUF', 'Qwen3-30B-A3B-Q4_K_M.gguf', local_dir='$HOME/models')
"

# Other models
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3-4B-GGUF', 'Qwen3-4B-Q4_K_M.gguf', local_dir='$HOME/models')
hf_hub_download('unsloth/Qwen3-8B-GGUF', 'Qwen3-8B-Q4_K_M.gguf', local_dir='$HOME/models')
hf_hub_download('unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF', 'DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf', local_dir='$HOME/models')
hf_hub_download('unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF', 'DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf', local_dir='$HOME/models')
"
```

### Run inference

```bash
# GPT-OSS-120B (split GGUF — point to the first part, llama.cpp finds the rest)
~/llama.cpp/build/bin/llama-cli \
  -m ~/models/gpt-oss-120b/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  -ngl 99 -t 8 -b 256 -c 8192 -n 500 \
  -p "Your prompt here"

# Qwen3-30B-A3B (single file)
~/llama.cpp/build/bin/llama-cli \
  -m ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  -ngl 99 -t 8 -b 256 -n 500 \
  -p "Your prompt here"
```

### Run benchmarks

```bash
~/llama.cpp/build/bin/llama-bench \
  -m ~/models/gpt-oss-120b/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  -ngl 99 -t 8 -p 512 -n 128 -r 3
```

### Key llama.cpp flags for Strix Halo

- `-ngl 99` — offload all layers to GPU (Vulkan)
- `-b 256` — batch size 256, improves prompt processing
- `-t 8` — CPU threads for any non-GPU work
- `-c 32768` — context size (increase to use more VRAM for KV-cache)
- `ROCBLAS_USE_HIPBLASLT=1` — only if using HIP backend instead of Vulkan

---

## Troubleshooting

**`libmpi_cxx.so.40: cannot open shared object file`**
- Run Step 6 to build the stub, make sure `~/amd_libs` is in `LD_LIBRARY_PATH`.

**`libroctx64.so.4: cannot open shared object file`**
- TheRock SDK not extracted or `~/amd_libs/therock-sdk/lib` not in `LD_LIBRARY_PATH`.

**`Numba needs NumPy 2.2 or less`**
- Run `uv pip install "numpy<2.3" --python .venv/bin/python`.

**`TensileLibrary_lazy_gfx1100.dat: No such file or directory`**
- You're using `HSA_OVERRIDE_GFX_VERSION=11.0.0` but TheRock SDK has kernels for
  gfx1151. Use `HSA_OVERRIDE_GFX_VERSION=11.5.1` instead.

**`Cannot find ROCm device library`**
- Set `DEVICE_LIB_PATH=~/amd_libs/therock-sdk/lib/llvm/amdgcn/bitcode` if building from source.

**Docker alternative**
- If native setup is too painful, AMD provides container images:
  ```bash
  docker pull rocm/vllm-dev:rocm7.0.2_navi_ubuntu24.04_py3.12_pytorch_2.11_vllm_0.14.0
  ```
