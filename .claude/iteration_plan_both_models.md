# AIMO3 Iteration Plan: Dual-Machine Trace Generation

> Created: 2026-02-03
> Updated: 2026-02-03 — Switched to DeepSeek-R1-0528-Qwen3-8B, 12 samples
> Purpose: Generate large-scale traces for hyperparameter tuning using both machines

## Problem Statement

- Current reference set: 53 AIMO3 problems (too small for train/test split)
- Need more data to properly tune selection strategy hyperparameters
- Solution: Generate traces on AIME dataset (2,250 problems) using faster models

---

## Hardware Setup

### Machine 1: RTX 4090 (CUDA)
- **GPU**: NVIDIA RTX 4090, 24GB VRAM
- **Framework**: vLLM or llama.cpp CUDA
- **Expected speed**: ~150 tok/s with DeepSeek-R1-0528-Qwen3-8B

### Machine 2: AMD Strix Halo (Vulkan)
- **GPU**: Radeon 8060S, 96GB unified VRAM
- **Framework**: llama.cpp Vulkan
- **Expected speed**: ~70-80 tok/s with DeepSeek-R1-0528-Qwen3-8B Q4_K_M

---

## Model Selection

### Chosen: DeepSeek-R1-0528-Qwen3-8B

| Metric | Value |
|--------|-------|
| Parameters | 8B |
| VRAM (Q4) | ~5GB |
| AIME 2025 | **87.5%** (SOTA for 8B class) |
| MATH-500 | ~90% |
| Speed (4090) | ~150 tok/s |
| Speed (AMD) | ~70-80 tok/s |

**Why this model:**
- State-of-the-art math reasoning for its size (87.5% AIME 2025)
- Distilled from DeepSeek-R1-0528 — excellent reasoning chains
- Fast enough for bulk trace generation (~150 tok/s on 4090)
- Fits easily on both machines
- Architecture identical to Qwen3-8B, same tokenizer as DeepSeek-R1

### Model Comparison (with tok/s)

| Model | Size | VRAM | 4090 tok/s | AMD tok/s | AIME | MATH-500 |
|-------|------|------|------------|-----------|------|----------|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | ~1GB | ~280 | ~150 | 28.9% | 83.9% |
| Qwen3-4B | 4B | ~2.5GB | ~200 | 75 | ~40% | **91.4%** |
| DeepSeek-R1-Distill-Qwen-7B | 7B | ~4GB | ~150 | ~80 | 55.5% | 92.8% |
| **DeepSeek-R1-0528-Qwen3-8B** | 8B | ~5GB | **~150** | ~75 | **87.5%** | ~90% |
| Qwen3-8B | 8B | ~5GB | ~150 | ~70 | ~50% | 88.8% |
| Qwen3-30B-A3B | 30B/3B | ~17GB | ~110 | 89 | ~70% | ~85% |
| gpt-oss-120b | 120B | 60GB | N/A | 55 | ~80% | ~90% |

### Alternative: Two-Tier Approach

For faster iteration, consider:
1. **Tier 1 (Coarse filtering)**: Qwen3-4B at ~200 tok/s — filter 138 strategies down to top 10
2. **Tier 2 (Validation)**: DeepSeek-R1-0528-Qwen3-8B — validate top strategies on subset

---

## Dataset: AIME 1983-2024

### Source
- **HuggingFace**: [gneubig/aime-1983-2024](https://huggingface.co/datasets/gneubig/aime-1983-2024)
- **Kaggle**: [hemishveeraboina/aime-problem-set-1983-2024](https://www.kaggle.com/datasets/hemishveeraboina/aime-problem-set-1983-2024)
- **Format**: CSV with columns: Year, Problem Number, Problem, Answer
- **Total**: ~2,250 problems
- **Answer format**: Integer 0-999 (similar to AIMO's 0-99999)

### Train/Val/Test Split

| Split | Years | Problems | Purpose |
|-------|-------|----------|---------|
| **Train** | 1983-2018 | ~1,700 | Tune selection strategies |
| **Val** | 2019-2022 | ~180 | Early stopping / hyperparameter selection |
| **Test** | 2023-2024 + AIMO3 ref | ~120 + 53 | Final unbiased evaluation |

---

## Trace Schema (What to Save)

Each trace must include enough data to replay all 138+ selection strategies:

```python
trace = {
    # Per-attempt data (12 attempts per problem)
    "attempts": [
        {
            "answer": 42,
            "answer_source": "boxed",  # "boxed", "code_output", "fallback"
            "prompt_type": "reasoning",  # "reasoning", "code_first", "case_analysis"
            "prompt_text": "...",

            # Entropy data - FULL logprobs for custom calculations
            "token_logprobs": [...],  # List of per-token log probabilities
            "mean_entropy": 1.23,
            "min_entropy": 0.5,
            "max_entropy": 2.1,

            # All answers found across turns (not just final)
            "all_answers_extracted": [42, 42, 17, 42],
            "final_answer_turn": 3,

            # Code execution details
            "code_executions": [
                {"code": "...", "stdout": "42", "success": True, "turn": 2}
            ],

            # Generation metadata
            "token_count": 1523,
            "num_turns": 4,
            "temperature": 0.6,
            "seed": 12345,
        },
        # ... 11 more attempts
    ],

    # Problem metadata
    "problem_id": "aime_2020_I_1",
    "problem_text": "...",
    "ground_truth": 42,

    # Timing (for time-budget simulation)
    "generation_time_seconds": 45.2,
}
```

### Why Each Field Matters

| Field | Enables |
|-------|---------|
| `token_logprobs` | Custom entropy (Shannon, Rényi, top-k) |
| `all_answers_extracted` | Early-stopping simulation |
| `answer_source` | Source-aware weighting |
| `prompt_type` | Prompt-specific strategies |
| `code_executions` | Code success rate analysis |
| `num_turns` | Depth-based weighting |
| `generation_time_seconds` | Time-budget simulation |

---

## Work Distribution

### RTX 4090 (faster)
- **Model**: DeepSeek-R1-0528-Qwen3-8B Q4_K_M
- **Speed**: ~150 tok/s
- **Problems**: AIME 1983-2004 (~1,000 problems)
- **Samples per problem**: 12
- **Total samples**: 12,000
- **Estimated time**: ~18-20 hours

### AMD Strix (parallel)
- **Model**: DeepSeek-R1-0528-Qwen3-8B Q4_K_M
- **Speed**: ~75 tok/s
- **Problems**: AIME 2005-2022 (~700 problems)
- **Samples per problem**: 12
- **Total samples**: 8,400
- **Estimated time**: ~25-28 hours

### Combined Output
- **Total problems**: ~1,700 (train + val)
- **Total samples**: 20,400
- **Total time**: ~25-28 hours (limited by AMD)

---

## Time Estimates (12 Samples per Problem)

| Model | 4090 tok/s | Time/Sample | 400 problems (4090) | Both Machines (1700) |
|-------|------------|-------------|---------------------|----------------------|
| DS-R1-Distill-1.5B | 280 | ~25s | 33 hrs | ~17 hrs |
| Qwen3-4B | 200 | ~35s | 47 hrs | ~25 hrs |
| **DS-R1-0528-Qwen3-8B** | 150 | ~45s | 60 hrs | **~28 hrs** |
| Qwen3-30B-A3B | 110 | ~65s | 87 hrs | ~45 hrs |

---

## Implementation Steps

### Step 1: Download AIME Dataset
```bash
# Using HuggingFace datasets
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('gneubig/aime-1983-2024')
ds['train'].to_csv('data/aime/aime_1983_2024.csv')
"
```

### Step 2: Split Dataset
```bash
# Split by year into train/val/test
python scripts/split_aime.py \
  --input data/aime/aime_1983_2024.csv \
  --train-years 1983-2018 \
  --val-years 2019-2022 \
  --test-years 2023-2024
```

### Step 3: Download Model

```bash
# Download GGUF for both machines
huggingface-cli download \
  bartowski/DeepSeek-R1-0528-Qwen3-8B-GGUF \
  --include "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf" \
  --local-dir ~/models/DeepSeek-R1-0528-Qwen3-8B/

# Alternative: unsloth GGUF
huggingface-cli download \
  unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF \
  --include "*Q4_K_M.gguf" \
  --local-dir ~/models/DeepSeek-R1-0528-Qwen3-8B/
```

### Step 4: Setup RTX 4090
```bash
# Option A: llama.cpp with CUDA (recommended for simplicity)
~/llama.cpp/build/bin/llama-server \
  -m ~/models/DeepSeek-R1-0528-Qwen3-8B/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf \
  -ngl 99 -c 16384 --port 8080

# Option B: vLLM (if you prefer)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --max-model-len 16384 \
  --port 8080
```

### Step 5: Setup AMD Strix
```bash
# llama-server with Vulkan backend
~/llama.cpp/build/bin/llama-server \
  -m ~/models/DeepSeek-R1-0528-Qwen3-8B/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf \
  -ngl 99 -c 16384 --port 8080
```

### Step 6: Generate Traces (parallel on both machines)

**On RTX 4090:**
```bash
python scripts/generate_traces.py \
  --api-base http://localhost:8080/v1 \
  --model DeepSeek-R1-0528-Qwen3-8B \
  --problems-csv data/aime/aime_train_1983_2004.csv \
  --n-samples 12 \
  --max-turns 16 \
  --output-dir output/traces/aime_4090/
```

**On AMD Strix:**
```bash
python scripts/generate_traces.py \
  --api-base http://localhost:8080/v1 \
  --model DeepSeek-R1-0528-Qwen3-8B \
  --problems-csv data/aime/aime_train_2005_2022.csv \
  --n-samples 12 \
  --max-turns 16 \
  --output-dir output/traces/aime_amd/
```

### Step 7: Merge and Sweep
```bash
# Merge traces from both machines
mkdir -p output/traces/aime_combined/
cp output/traces/aime_4090/*.json output/traces/aime_combined/
cp output/traces/aime_amd/*.json output/traces/aime_combined/

# Run selection sweep on train set (138 strategies, <5 seconds)
python scripts/replay_selection.py sweep \
  --traces-dir output/traces/aime_combined/ \
  --split train

# Evaluate top-5 strategies on val set
python scripts/replay_selection.py evaluate \
  --traces-dir output/traces/aime_val/ \
  --strategies "hybrid_inv_k0.25,threshold_2.0_hybrid,majority_vote,normalized_alpha0.3,top_k_3"
```

---

## Expected Outcomes

1. **Train set traces**: ~1,700 problems × 12 samples = **20,400 traces**
2. **Selection strategy ranking**: 138 strategies ranked by accuracy on train
3. **Validation**: Top-5 strategies evaluated on held-out val set (~180 problems)
4. **Final test**: Best strategy evaluated on AIME 2023-2024 + AIMO3 reference

---

## Validation Strategy

After finding best hyperparameters on AIME train set:

1. **Validate on AIME val** (2019-2022): Confirm no overfitting
2. **Test on AIME test** (2023-2024): Unbiased performance estimate
3. **Final check on AIMO3 ref** (53 problems): Verify transfer to competition distribution
4. **Deploy winner** to Kaggle notebook with gpt-oss-120b

---

## Files to Create/Modify

- [x] `scripts/split_aime.py` — Dataset splitting utility (DONE)
- [x] `scripts/generate_traces.py` — Expanded trace schema with full logprobs, all_answers_extracted (DONE)
- [ ] `data/aime/` — Directory for AIME dataset files (run split_aime.py to create)
- [ ] `output/traces/aime_*/` — Trace output directories (created during generation)

---

## References

### Datasets
- [gneubig/aime-1983-2024](https://huggingface.co/datasets/gneubig/aime-1983-2024) — AIME dataset (2,250 problems)
- [opencompass/AIME2025](https://huggingface.co/datasets/opencompass/AIME2025) — AIME 2025 (30 problems)

### Models
- [DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B) — Primary model (87.5% AIME 2025)
- [bartowski/DeepSeek-R1-0528-Qwen3-8B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-0528-Qwen3-8B-GGUF) — GGUF quantizations
- [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388) — Model benchmarks

### Benchmarks
- [Qwen Speed Benchmarks](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)
- [RTX 4090 vLLM Benchmark](https://www.databasemart.com/blog/vllm-gpu-benchmark-rtx4090)
- [NVIDIA llama.cpp Blog](https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems)
- [Artificial Analysis - DS-R1-0528-Qwen3-8B](https://artificialanalysis.ai/models/deepseek-r1-qwen3-8b)
