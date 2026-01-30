# AIMO Solver

This repository contains a local inference script and environment setup for the AI Mathematical Olympiad (AIMO) competition. The system utilizes large language models to solve complex mathematical problems by iteratively generating and executing Python code in a sandboxed environment.

## Features
- Scalable inference using vLLM backend.
- Sandboxed Python execution via Jupyter kernels.
- Support for multiple solution attempts with early-stopping consensus.
- Native AMD ROCm support (tested on Strix Halo / Radeon 8060S).

## Quick Start (if already set up)

```bash
.venv/bin/python example.py
.venv/bin/python example.py --model "facebook/opt-125m" --max-model-len 512  # fast test
.venv/bin/python example.py --model "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"
```

---

## AMD ROCm Setup Guide (Ubuntu 25.10 + Strix Halo / RDNA 3.5)

This guide documents how to get vLLM running natively on AMD Strix Halo (gfx1151)
with Ubuntu 25.10. This was hard-won knowledge — ROCm on consumer RDNA GPUs with a
bleeding-edge distro requires several workarounds.

### Why is this hard?

1. **Ubuntu 25.10 ships OpenMPI 5.x** which removed the C++ bindings (`libmpi_cxx.so`)
   that PyTorch ROCm wheels still expect.
2. **Strix Halo (gfx1151)** is RDNA 3.5 — very new silicon with limited ROCm support.
3. **No unified ROCm story** — PyTorch bundles ROCm 6.3, vLLM wheels target ROCm 7.0,
   system packages are ROCm 5.7. All ABI-incompatible.
4. **vLLM's ROCm wheels** only exist for Python 3.12 + ROCm 7.0.

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

```bash
mkdir -p /tmp/therock-sdk
cd /tmp

# Download the SDK for your GPU arch (gfx1151 for Strix Halo)
wget -q "https://github.com/ROCm/TheRock/releases/download/nightly-rocm-7.12/therock-dist-linux-gfx1151-7.12.0a20260129.tar.gz"

# Extract
tar -xf therock-dist-linux-gfx1151-7.12.0a20260129.tar.gz -C /tmp/therock-sdk --strip-components=1
```

> Check https://github.com/ROCm/TheRock/releases for the latest nightly.
> For other GPUs: use `gfx1100` (RX 7900), `gfx1101` (RX 7800/7700), etc.

### Step 5: Install amdsmi from TheRock SDK

```bash
uv pip install /tmp/therock-sdk/share/amd_smi/ --python .venv/bin/python
```

Also create a focused lib directory so amdsmi finds the right `.so`:

```bash
mkdir -p /tmp/amdsmi-lib
ln -sf /tmp/therock-sdk/lib/libamd_smi.so* /tmp/amdsmi-lib/
```

### Step 6: Create OpenMPI C++ bindings stub

Ubuntu 25.10's OpenMPI 5.x removed the C++ bindings that PyTorch expects.
We create a tiny stub with the 3 symbols PyTorch actually needs:

```bash
cat > /tmp/mpi_cxx_stub.cpp << 'EOF'
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

g++ -shared -fPIC -O0 -fno-inline -o /tmp/libmpi_cxx.so.40 /tmp/mpi_cxx_stub.cpp
```

### Step 7: Run

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
LD_LIBRARY_PATH=/tmp:/tmp/amdsmi-lib:/tmp/therock-sdk/lib \
.venv/bin/python example.py
```

Or for a quick sanity test with a tiny model:

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
LD_LIBRARY_PATH=/tmp:/tmp/amdsmi-lib:/tmp/therock-sdk/lib \
.venv/bin/python example.py --model "facebook/opt-125m" --max-model-len 512 --gpu-memory-utilization 0.3
```

### Environment Variables Reference

| Variable | Value | Why |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION` | `11.5.1` | Tell ROCm this is gfx1151 (Strix Halo native arch) |
| `LD_LIBRARY_PATH` | `/tmp:/tmp/amdsmi-lib:/tmp/therock-sdk/lib` | MPI stub + amdsmi + ROCm 7.x runtime |
| `VLLM_TARGET_DEVICE` | `rocm` | Force vLLM to use ROCm backend |

### Performance (Strix Halo, 68.7 GB VRAM, DDR5-8000)

| Model | VRAM | Speed |
|---|---|---|
| OPT-125M (test) | 0.25 GiB | ~260 tok/s |
| DeepSeek-R1-70B-AWQ | 37.3 GiB | ~2.6 tok/s |

### Troubleshooting

**`libmpi_cxx.so.40: cannot open shared object file`**
- Run Step 6 to build the stub, make sure `/tmp` is in `LD_LIBRARY_PATH`.

**`libroctx64.so.4: cannot open shared object file`**
- TheRock SDK not extracted or `/tmp/therock-sdk/lib` not in `LD_LIBRARY_PATH`.

**`Numba needs NumPy 2.2 or less`**
- Run `uv pip install "numpy<2.3" --python .venv/bin/python`.

**`TensileLibrary_lazy_gfx1100.dat: No such file or directory`**
- You're using `HSA_OVERRIDE_GFX_VERSION=11.0.0` but TheRock SDK has kernels for
  gfx1151. Use `HSA_OVERRIDE_GFX_VERSION=11.5.1` instead.

**`Cannot find ROCm device library`**
- Set `DEVICE_LIB_PATH=/tmp/therock-sdk/lib/llvm/amdgcn/bitcode` if building from source.

**Docker alternative**
- If native setup is too painful, AMD provides container images:
  ```bash
  docker pull rocm/vllm-dev:rocm7.0.2_navi_ubuntu24.04_py3.12_pytorch_2.11_vllm_0.14.0
  ```

---

## Original Setup (NVIDIA / Kaggle)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your environment and run:
   ```bash
   python main.py --model_path /path/to/your/model
   ```

### Hardware Requirements
- High-VRAM GPU (recommended 24GB+)
- Support for FP8/BF16/FP16 precision
