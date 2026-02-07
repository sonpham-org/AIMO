# Tinker Fine-Tuning Plan for AIMO3

> Created: 2026-02-04
> Goal: RL fine-tune gpt-oss-120b (or proxy model) on math reasoning via Tinker API, then apply for $5K+ research grant

## Strategy

1. Sign up for Tinker (free, GA: https://auth.thinkingmachines.ai/sign-up)
2. Run small proof-of-concept with cheap model (Llama-3.2-1B or Qwen3-4B)
3. Scale to gpt-oss-120b or Qwen3-8B with math_rl recipe
4. Download LoRA weights, merge, quantize, deploy on Kaggle
5. Apply for Tinker Research Grant ($5K+) with initial results as evidence

## Tinker Overview

- **What**: Training API from Thinking Machines Lab (Mira Murati)
- **How**: Python script runs on CPU, 4 API primitives orchestrate GPU training
- **Primitives**: `forward_backward()`, `optim_step()`, `sample()`, `save_state()`
- **Method**: LoRA fine-tuning (full fine-tuning not yet supported)
- **Supports**: SFT, GRPO, DPO, RLHF
- **Docs**: https://tinker-docs.thinkingmachines.ai/
- **Cookbook**: https://github.com/thinking-machines-lab/tinker-cookbook

## Available Models on Tinker

### MoE (most relevant)
- **gpt-oss-120b** — our competition model
- **gpt-oss-20b** — smaller proxy for fast iteration
- Qwen3-235B-A22B-Instruct — massive, could distill from
- Qwen3-30B-A3B — fast MoE
- DeepSeek-V3.1 — top-tier reasoning
- Kimi-K2-Thinking — trillion-param reasoning

### Dense
- Qwen3-32B, Qwen3-8B, Qwen3-4B
- Llama-3.1-70B, Llama-3.1-8B, Llama-3.2-3B, Llama-3.2-1B

## Math RL Recipe (tinker-cookbook/recipes/math_rl/)

Files:
- `train.py` — CLI training wrapper (model, dataset, hyperparams)
- `math_env.py` — MATH dataset environment (reward = correct answer)
- `math_grading.py` — answer comparison/grading
- `arithmetic_env.py` — trivial arithmetic for smoke tests

Supported environments: `arithmetic`, `math`, `gsm8k`, `polaris`, `deepmath`

Benchmark results:
- Qwen3-8B on MATH → **76.7%** after 180 steps
- Llama-3.1-8B on GSM8K → **90.9%** after 220 steps

## Estimated Pricing (per million tokens)

| Model | Prefill | Sample | Train |
|---|---|---|---|
| Llama-3.2-1B | $0.03 | $0.09 | $0.09 |
| Qwen3-4B | $0.07 | $0.22 | $0.22 |
| Qwen3-8B | $0.13 | $0.40 | $0.40 |
| Qwen3-235B | $0.68 | $1.70 | $2.04 |

gpt-oss-120b pricing not listed — estimate ~$0.50-1.50 range (MoE, similar active params to Qwen3-30B)

## Execution Plan

### Phase 1: Smoke Test (cost: ~$1-5)
```bash
pip install tinker tinker-cookbook
python -m tinker_cookbook.recipes.math_rl.train \
  model=Llama-3.2-1B \
  env=arithmetic \
  group_size=4
```
- Validate the pipeline works end-to-end
- ~minutes to run, expect reward 0.66 → 1.0

### Phase 2: Real Training (cost: ~$20-100)
```bash
python -m tinker_cookbook.recipes.math_rl.train \
  model=Qwen3-8B \
  env=math \
  lr=4e-5 \
  group_size=4 \
  num_steps=200
```
- Target: replicate 76.7% on MATH benchmark
- This gives us a concrete result for the grant application

### Phase 3: Competition Model (cost: ~$100-500, use grant credits)
```bash
python -m tinker_cookbook.recipes.math_rl.train \
  model=gpt-oss-120b \
  env=math \
  lr=4e-5 \
  group_size=4 \
  num_steps=200
```
- Fine-tune the actual competition model
- Download LoRA weights → merge → quantize to MXFP4
- Test locally on AMD (llama.cpp) then deploy to Kaggle

### Phase 4: Custom Dataset (after Phase 3 works)
- Create AIME/olympiad-style environment using our 524 AIME problems
- Possibly add nvidia/OpenMathReasoning subset
- RL train on harder problems specifically

## Grant Application Strategy

**Apply to**: Tinker Research Grant (https://thinkingmachines.ai/blog/tinker-research-and-teaching-grants/)
- $5,000+ in credits
- Rolling applications, ~1 week response
- Frame as: "Open-source math reasoning research for AIMO competition"
- Include Phase 1-2 results as evidence of serious usage
- Commit to open-sourcing findings (required by AIMO anyway)

## Integration with Current Pipeline

After fine-tuning:
1. Download LoRA weights from Tinker
2. Merge with base gpt-oss-120b
3. Quantize (MXFP4 or Q4_K_M)
4. Upload to Kaggle as private dataset
5. Use in existing entropy-gated consensus notebook
6. The RL-trained model should produce better answers AND more confident (lower entropy) on correct answers

## Key Risks

- gpt-oss-120b pricing may be high for MoE
- LoRA may not improve much on already-strong reasoning model
- Quantization after LoRA merge may degrade quality
- Tinker doesn't support full fine-tuning yet (LoRA only)
- Need to figure out how to convert Tinker weights → GGUF for llama.cpp testing

## Alternative Approaches via Tinker

1. **Distillation**: Generate traces from Qwen3-235B, SFT into gpt-oss-120b
2. **Reward model**: Train Qwen3-8B as a verifier (replace entropy heuristic)
3. **DPO**: Collect (correct, incorrect) solution pairs from traces, train preference
