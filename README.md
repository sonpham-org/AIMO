# AIMO Solver

Local inference system for the AI Mathematical Olympiad (AIMO) competition. Uses large language models to solve complex mathematical problems by iteratively generating and executing Python code in a sandboxed Jupyter environment.

## Features
- Scalable inference using vLLM or llama.cpp (Vulkan) backends.
- Sandboxed Python execution via Jupyter kernels.
- Support for multiple solution attempts with early-stopping consensus.
- Native AMD ROCm support (tested on Strix Halo / Radeon 8060S).

## Hardware

- **CPU:** AMD Ryzen AI MAX+ 395
- **GPU:** Radeon 8060S (RDNA 3.5 / gfx1151)
- **VRAM:** 96 GB (BIOS-configured unified memory allocation)
- **Memory bandwidth:** ~215 GB/s (LPDDR5X-8000)

## Quick Start

```bash
# llama.cpp Vulkan (recommended — fastest)
~/llama.cpp/build/bin/llama-cli \
  -m ~/models/gpt-oss-120b/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  -ngl 99 -t 8 -b 256 -c 8192 -n 500 \
  -p "Your prompt here"

# vLLM (for batched serving)
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
LD_LIBRARY_PATH=~/amd_libs:~/amd_libs/amdsmi-lib:~/amd_libs/therock-sdk/lib \
.venv/bin/python example.py
```

> For full installation instructions, see **[SETUP.md](SETUP.md)**.

---

## Benchmark Results (Strix Halo, 96 GB VRAM, ~215 GB/s)

### llama.cpp (Vulkan) — Recommended for single-user inference

Vulkan is **2-3x faster than vLLM ROCm** for token generation on this hardware.

| Model | Quant | Size | pp512 (tok/s) | tg128 (tok/s) |
|---|---|---|---|---|
| Qwen3-0.6B | Q8_0 | 604 MiB | 9,790 | **252** |
| Qwen3-1.7B | Q8_0 | 1.7 GiB | 4,512 | **106** |
| Qwen3-4B | Q4_K_M | 2.3 GiB | 1,726 | **75** |
| DeepSeek-R1-Distill-Qwen-7B | Q4_K_M | 4.4 GiB | 1,041 | **46** |
| Qwen3-8B | Q4_K_M | 4.7 GiB | 966 | **42** |
| Qwen3-14B | Q4_K_M | 8.4 GiB | 536 | **24** |
| GPT-OSS-20B (MoE, 3.6B active) | Q4_K_M | 10.8 GiB | 1,143 | **79** |
| Qwen3-30B-A3B (MoE, 3B active) | Q4_K_M | 17.3 GiB | 1,024 | **89** |
| Qwen3-32B | Q4_K_M | 18.4 GiB | 217 | **10.75** |
| DeepSeek-R1-Distill-Qwen-32B | Q4_K_M | 18.5 GiB | 228 | **10.9** |
| DeepSeek-R1-Distill-Llama-70B | Q4_K_M | 39.6 GiB | 97 | **5.0** |
| GPT-OSS-120B (MoE, 5.1B active) | Q4_K_M | 58.5 GiB | 498 | **55** |

### vLLM (0.15.0+rocm700)

PagedAttention KV-cache and batching, useful for multi-request serving.
Slower than Vulkan for single-user due to ROCm/HIP overhead on gfx1151.

| Model | Quant | Speed (tok/s) |
|---|---|---|
| OPT-125M (test) | fp16 | ~260 |
| Qwen3-1.7B | fp16 | ~46 |
| Qwen3-4B | fp16 | ~26 |
| Qwen3-8B | fp16 | ~13 |
| DeepSeek-R1-Distill-Qwen-7B | fp16 | ~16 |
| DeepSeek-R1-Distill-Llama-70B-AWQ | AWQ 4-bit | ~2.6 |

---

## Best Models for 50-100 tok/s (Serviceable Speed)

| Model | Backend | tok/s | Size | Why |
|---|---|---|---|---|
| **GPT-OSS-120B (MoE)** | Vulkan | **55** | 58.5 GiB | 117B total / 5.1B active. Near o4-mini reasoning quality. Uses 58.5 of 96 GiB, leaves 37 GiB for KV-cache. |
| **Qwen3-30B-A3B (MoE)** | Vulkan | **89** | 17.3 GiB | 30B total / 3B active. Best speed-to-quality. 79 GiB free for KV-cache. |
| **GPT-OSS-20B (MoE)** | Vulkan | **79** | 10.8 GiB | 20.9B total / 3.6B active. OpenAI quality at 79 tok/s. |
| **Qwen3-4B** | Vulkan | **75** | 2.3 GiB | Dense model, very fast. Great for code and math. |
| **Qwen3-1.7B Q8** | Vulkan | **106** | 1.7 GiB | Fastest serviceable model. Good for rapid iteration. |

> **Top recommendation:** GPT-OSS-120B at 55 tok/s — OpenAI's open-weight MoE model
> with near-o4-mini reasoning, running entirely on-device. Only 5.1B params active per
> token from 117B total, so it's fast despite its size. Fits in 96 GiB with room for
> large KV-cache contexts.
>
> **Runner-up:** Qwen3-30B-A3B at 89 tok/s if you want more speed and still excellent
> reasoning quality.

### Key Insights

- **MoE models dominate.** GPT-OSS-120B (55 tok/s) and Qwen3-30B-A3B (89 tok/s) are 5-11x faster than dense models of similar quality.
- **Vulkan >> vLLM ROCm** for single-user inference on RDNA 3.5 (2-3x faster).
- **96 GB VRAM** enables running GPT-OSS-120B (58.5 GiB) fully on-GPU with KV-cache headroom — impossible on typical 24 GB consumer GPUs.
- **Token generation is bandwidth-bound** at ~215 GB/s. Speed scales inversely with active model size: halve the active params, double the tok/s.

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
