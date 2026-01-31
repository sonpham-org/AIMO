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

### Column Definitions

- **pp512** — **Prompt processing** speed: how fast the model ingests a 512-token input prompt (tokens/sec). Higher is better. This measures prefill/batch throughput — relevant for long prompts and context.
- **tg128** — **Token generation** speed: how fast the model generates 128 new tokens one at a time (tokens/sec). Higher is better. **This is the number that determines interactive "feel"** — the speed at which you see the response stream in.
- **AIME'24** — Score on [AIME 2024](https://artofproblemsolving.com/wiki/index.php/AIME) competition math benchmark (pass@1, %). Higher = better mathematical reasoning.
- **MATH-500** — Score on the [MATH-500](https://arxiv.org/abs/2103.03874) benchmark subset (pass@1, %). Tests broad mathematical problem-solving.
- **GPQA** — Score on [GPQA Diamond](https://arxiv.org/abs/2311.12022) PhD-level science benchmark (pass@1, %). Tests expert-level reasoning.

### llama.cpp (Vulkan) — Recommended for single-user inference

Vulkan is **2-3x faster than vLLM ROCm** for token generation on this hardware.

| Model | Quant | Size | pp512 | tg128 | AIME'24 | MATH-500 | GPQA |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | Q8_0 | 604 MiB | 9,790 | **252** | — | — | — |
| Qwen3-1.7B | Q8_0 | 1.7 GiB | 4,512 | **106** | — | — | — |
| Qwen3-4B (thinking) | Q4_K_M | 2.3 GiB | 1,726 | **75** | 73.8% | ~90% | — |
| DeepSeek-R1-Distill-Qwen-7B | Q4_K_M | 4.4 GiB | 1,041 | **46** | 55.5% | 92.8% | 49.1% |
| Qwen3-8B (thinking) | Q4_K_M | 4.7 GiB | 966 | **42** | ~65% | ~92% | — |
| Qwen3-14B (thinking) | Q4_K_M | 8.4 GiB | 536 | **24** | ~72% | ~93% | — |
| GPT-OSS-20B (MoE, 3.6B active) | Q4_K_M | 10.8 GiB | 1,143 | **79** | 42.1% | — | 56.8% |
| Qwen3-30B-A3B (MoE, 3B active) | Q4_K_M | 17.3 GiB | 1,024 | **89** | 80.4% | — | 65.8% |
| Qwen3-32B (thinking) | Q4_K_M | 18.4 GiB | 217 | **10.75** | ~82% | ~95% | — |
| DeepSeek-R1-Distill-Qwen-32B | Q4_K_M | 18.5 GiB | 228 | **10.9** | 72.6% | 94.3% | — |
| DeepSeek-R1-Distill-Llama-70B | Q4_K_M | 39.6 GiB | 97 | **5.0** | 70.0% | 94.5% | 65.2% |
| GPT-OSS-120B (MoE, 5.1B active) | Q4_K_M | 58.5 GiB | 498 | **55** | 56.3% | — | 67.1% |

> **Notes on reasoning scores:**
> - Qwen3 scores are in **thinking mode** (chain-of-thought enabled). Scores marked with `~` are estimates interpolated from the [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).
> - DeepSeek-R1-Distill scores are from the [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) (pass@1).
> - GPT-OSS scores are **without tools** (no Python execution). With tools (code interpreter), GPT-OSS-120B reaches ~73% on AIME'24 and GPT-OSS-20B reaches ~61%. See [OpenAI's model card](https://arxiv.org/abs/2508.10925).
> - Qwen3 0.6B and 1.7B are primarily edge models; official AIME/MATH scores not published for these sizes.

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

| Model | tok/s | AIME'24 | GPQA | Size | Why |
|---|---|---|---|---|---|
| **Qwen3-30B-A3B (MoE)** | **89** | 80.4% | 65.8% | 17.3 GiB | Best speed-to-quality. 30B total / 3B active. 79 GiB free for KV-cache. |
| **GPT-OSS-20B (MoE)** | **79** | 42.1% | 56.8% | 10.8 GiB | OpenAI model, fast. Stronger with tool use (~61% AIME). |
| **Qwen3-4B (thinking)** | **75** | 73.8% | — | 2.3 GiB | Dense model. Surprisingly strong AIME for its size. |
| **GPT-OSS-120B (MoE)** | **55** | 56.3% | 67.1% | 58.5 GiB | Largest model that hits target. Best GPQA. With tools: ~73% AIME. |
| **Qwen3-1.7B Q8** | **106** | — | — | 1.7 GiB | Fastest serviceable model. Good for rapid iteration. |

> **Top recommendation for math competition:** **Qwen3-30B-A3B** at 89 tok/s — highest
> AIME'24 score (80.4%) among all models that hit the 50-100 tok/s target. MoE
> architecture means only 3B params active per token from 30B total. With 96 GB VRAM,
> the model (17 GiB) fits with 79 GiB free for massive KV-cache contexts.
>
> **If you need the best science/general reasoning:** GPT-OSS-120B at 55 tok/s has the
> highest GPQA score (67.1%) and reaches ~73% AIME with tool use (Python sandbox).
>
> **Surprise value pick:** Qwen3-4B at 75 tok/s scores 73.8% on AIME'24 in thinking
> mode — rivaling models 10x its size.

### Key Insights

- **MoE models dominate speed.** GPT-OSS-120B (55 tok/s) and Qwen3-30B-A3B (89 tok/s) are 5-11x faster than dense models of similar total parameter count.
- **Vulkan >> vLLM ROCm** for single-user inference on RDNA 3.5 (2-3x faster).
- **96 GB VRAM** enables running GPT-OSS-120B (58.5 GiB) fully on-GPU with KV-cache headroom — impossible on typical 24 GB consumer GPUs.
- **Token generation is bandwidth-bound** at ~215 GB/s. Speed scales inversely with active model size: halve the active params, double the tok/s.
- **Qwen3 thinking mode** dramatically boosts reasoning — Qwen3-4B with thinking rivals DeepSeek-R1-Distill-Qwen-32B on AIME without thinking.

### Benchmark Sources

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — Qwen3 family scores
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948) — DeepSeek-R1-Distill scores
- [GPT-OSS Model Card](https://arxiv.org/abs/2508.10925) — OpenAI GPT-OSS scores
- [OpenAI GPT-OSS Announcement](https://openai.com/index/introducing-gpt-oss/) — Architecture details

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
