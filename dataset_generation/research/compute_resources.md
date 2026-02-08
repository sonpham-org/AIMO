# Compute Resources for Dataset Generation

> Last updated: 2026-02-07
> Author: resource-finder agent
> Purpose: Detailed cost analysis and compute planning for AIMO3 dataset generation

---

## 1. Google API Credits ($93 Available)

### 1.1 Gemini Pricing Summary

| Model | Input ($/M tok) | Output ($/M tok) | Batch Input | Batch Output | Best For |
|-------|-----------------|-------------------|-------------|--------------|----------|
| **Gemini 2.0 Flash** | $0.10 | $0.40 | $0.05 | $0.20 | Cheapest option. Classification, verification, scoring. |
| **Gemini 2.5 Flash-Lite** | $0.10 | $0.40 | $0.05 | $0.20 | Same price, newer architecture. |
| **Gemini 2.5 Flash** | $0.30 | $2.50 | $0.15 | $1.25 | Higher quality reasoning. Solution generation. |
| **Gemini 2.5 Pro** | $1.25 | $10.00 | $0.625 | $5.00 | Best quality. Use sparingly for hardest tasks. |

**Key cost optimization: Batch API gives 50% off all models.** Requires JSONL input, 24h turnaround (usually faster). Use for all non-interactive processing.

**Context caching**: 10% of input price for cached reads. Useful when scoring multiple solutions for the same problem (cache the problem text).

**Free tier**: 1,000 requests/day per model. Good for prototyping and small tests.

### 1.2 Token Estimation per Task

Typical token counts for math dataset processing:

| Content | Avg Tokens |
|---------|-----------|
| Math problem text (input) | ~200 tokens |
| System prompt + instructions (input) | ~500 tokens |
| Short classification output | ~50 tokens |
| Quality score + reasoning output | ~300 tokens |
| Full solution generation output | ~3,000-8,000 tokens |
| Full solution text (input for scoring) | ~3,000-5,000 tokens |

### 1.3 Cost per Task (Using Batch API Pricing)

| Task | Input Tokens | Output Tokens | Cost per Item | Cost per 10K Items |
|------|-------------|---------------|---------------|-------------------|
| **Topic classification** | ~700 | ~50 | $0.000045 | **$0.45** |
| **Answer extraction + verification** | ~4,000 | ~100 | $0.00022 | **$2.20** |
| **Quality scoring** (read solution, output score) | ~5,000 | ~300 | $0.000310 | **$3.10** |
| **Difficulty estimation** (solve attempt with 2.0 Flash) | ~700 | ~2,000 | $0.000435 | **$4.35** |
| **Solution shortening/rewriting** (2.5 Flash) | ~5,500 | ~3,000 | $0.004575 | **$45.75** |
| **Full solution generation** (2.5 Flash) | ~700 | ~5,000 | $0.006355 | **$63.55** |

### 1.4 What $93 Buys (Batch API Pricing)

**Scenario A: Maximum coverage curation (recommended)**

| Step | Items | Model | Est. Cost |
|------|-------|-------|-----------|
| Classify 141K AIMO3 TIR problems by topic | 141,277 | 2.0 Flash Batch | ~$6.35 |
| Quality-score 70K AIMO3 Hard traces | 70,000 | 2.0 Flash Batch | ~$21.70 |
| Extract + verify answers for 50K filtered traces | 50,000 | 2.0 Flash Batch | ~$11.00 |
| Difficulty estimation (4 solve attempts) on 5K problems | 20,000 | 2.0 Flash Batch | ~$8.70 |
| **Subtotal** | | | **~$47.75** |
| Reserve for iteration / retries | | | **~$45.25** |

**Scenario B: Quality + generation focus**

| Step | Items | Model | Est. Cost |
|------|-------|-------|-----------|
| Classify 141K problems by topic | 141,277 | 2.0 Flash Batch | ~$6.35 |
| Quality-score 30K best traces | 30,000 | 2.0 Flash Batch | ~$9.30 |
| Shorten/rewrite top 2K solutions | 2,000 | 2.5 Flash Batch | ~$9.15 |
| Generate new solutions for 1K hard problems | 1,000 | 2.5 Flash Batch | ~$6.36 |
| Cross-verify 5K answers | 5,000 | 2.0 Flash Batch | ~$1.10 |
| **Subtotal** | | | **~$32.26** |
| Reserve | | | **~$60.74** |

**Scenario C: Conservative (maximize reserve)**

| Step | Items | Model | Est. Cost |
|------|-------|-------|-----------|
| Classify 141K problems | 141,277 | 2.0 Flash Batch | ~$6.35 |
| Quality-score 20K top traces | 20,000 | 2.0 Flash Batch | ~$6.20 |
| Verify 10K answers | 10,000 | 2.0 Flash Batch | ~$2.20 |
| **Subtotal** | | | **~$14.75** |
| Reserve for future phases | | | **~$78.25** |

### 1.5 How to Use Batch API

```python
# 1. Prepare JSONL input file
# Each line: {"key": "unique_id", "request": {"model": "gemini-2.0-flash", "contents": [...]}}

# 2. Submit batch job
import google.generativeai as genai
genai.configure(api_key="...")

batch = genai.batches.create(
    model="gemini-2.0-flash",
    src="input.jsonl",  # up to 2GB
    config={"response_mime_type": "application/json"}
)

# 3. Poll for completion (usually <24h, often minutes)
# 4. Download results
```

Reference: https://ai.google.dev/gemini-api/docs/batch-api

---

## 2. Kaggle H100 Optimization

### 2.1 Specs & Constraints

- **GPU**: 1x NVIDIA H100 (80GB HBM3)
- **Time limit**: 9 hours per session
- **Network**: Offline (no internet) for competition submissions; online for dataset notebooks
- **Storage**: ~100GB scratch + dataset mounts
- **gpt-oss-120b**: Fits in MXFP4 (~60GB VRAM), leaving ~20GB for KV cache

### 2.2 Throughput Estimates (gpt-oss-120b on H100)

| Task | Throughput | Per 9h Session |
|------|-----------|----------------|
| TIR trace generation (128-turn, ~5K output tokens) | ~15-25 traces/hr (sequential) | ~135-225 traces |
| TIR trace generation (16 parallel workers, batch=256) | ~100-200 traces/hr | ~900-1,800 traces |
| RSR forward pass (scoring existing traces) | ~500-1K traces/hr | ~4,500-9,000 traces |
| QLoRA fine-tuning (Unsloth, 2K examples, 1 epoch) | ~700-1K examples/hr | Complete in ~2-3h |
| Inference validation (50 problems, 8 attempts each) | ~50-100 attempts/hr | Complete in ~4-8h |

### 2.3 Session Planning Templates

**Template A: Pure Trace Generation (maximize data)**
```
0:00 - 0:30  Setup: Install vLLM, start server, load model
0:30 - 8:30  Generate TIR traces (16 workers, 8 attempts per problem)
              Target: ~100-200 new problems traced = 800-1,600 traces
8:30 - 9:00  Save results, upload to Kaggle dataset
```
Best when: You need more training data and don't yet have enough curated traces.

**Template B: Trace + Fine-tune + Validate (full pipeline)**
```
0:00 - 0:30  Setup
0:30 - 3:30  Generate traces for 50-75 hard problems (400-600 traces)
3:30 - 4:00  Curate: filter + score traces (CPU while GPU free)
4:00 - 6:30  QLoRA fine-tuning with Unsloth (2K examples, 1-2 epochs)
6:30 - 7:00  Export LoRA adapter
7:00 - 9:00  Validation: run 50 test problems with fine-tuned model
```
Best when: You have curated data ready and want end-to-end testing.

**Template C: RSR Scoring + Fine-tune**
```
0:00 - 0:30  Setup
0:30 - 2:30  RSR scoring: forward pass on 5K candidate trajectories
2:30 - 3:00  Select top trajectories by RSR
3:00 - 5:30  QLoRA fine-tuning on selected data
5:30 - 9:00  Validation inference
```
Best when: You have candidate traces and want optimal selection before training.

**Template D: Competition Submission (with fine-tuned model)**
```
0:00 - 0:30  Setup: load base model + LoRA adapter
0:30 - 8:30  Run competition inference (50 problems)
8:30 - 9:00  Generate submission.parquet
```

### 2.4 Dual-Use Tips

- **Save intermediate results**: Upload generated traces as a Kaggle Dataset so they persist between sessions
- **Pre-process offline**: Do topic classification, quality scoring, answer verification via Gemini API beforehand. Kaggle H100 time is too expensive for tasks a $0.10/M API can do
- **Batch size**: Use `max_num_seqs=256` for maximum vLLM throughput
- **KV cache**: Use `kv_cache_dtype='fp8_e4m3'` to maximize context within 80GB
- **Early stopping**: Use `early_stop` parameter -- if 4+ attempts agree, skip remaining for that problem

---

## 3. Free Compute Options

### 3.1 Google Colab (Free Tier)

| Spec | Value |
|------|-------|
| GPU | NVIDIA T4 (16GB VRAM) |
| Weekly GPU hours | 15-30 hours |
| Session limit | 12 hours max |
| RAM | ~12GB |
| Storage | Ephemeral (mount Google Drive for persistence) |
| Internet | Yes |

**Best uses:**
- sentence-transformers dedup of 141K problems (~3h, fits in T4 16GB)
- PRM scoring with Qwen2.5-Math-7B-PRM800K (~6h for 10K traces)
- Data processing scripts (pandas, sympy verification)
- Small model inference (Qwen3-4B for topic classification as free Gemini alternative)

**Limitations:** No H100/A100 on free tier. T4 too small for gpt-oss-120b or even gpt-oss-20b.

### 3.2 Kaggle Notebooks (Free Tier)

| Spec | Value |
|------|-------|
| GPU | T4 (16GB) or P100 (16GB) |
| Weekly GPU hours | 30 hours |
| Session limit | 12 hours |
| RAM | ~16GB |
| Storage | ~70GB scratch + dataset mounts |
| Internet | Optional (on/off) |

**Best uses:** Same as Colab. Advantage: direct access to Kaggle datasets without download.

### 3.3 Lightning.ai Studios

| Spec | Value |
|------|-------|
| Free GPU | T4 (15 credits/month, ~7-35 hrs depending on usage) |
| Session limit | 4 hours per restart |
| Storage | 100GB persistent |
| Internet | Yes |

**Best uses:** Persistent dev environment. Good for iterative script development.

**How to get**: Sign up at lightning.ai, verify phone number. Use .edu email for instant verification.

### 3.4 Paperspace Gradient (Free Tier)

| Spec | Value |
|------|-------|
| Free GPU | Free GPU instances (varies, typically M4000 8GB) |
| Session limit | 6 hours per session (unlimited restarts) |
| Concurrent notebooks | 1 |
| Storage | 5GB persistent |
| RAM | 30GB system + 8 vCPUs minimum |
| Internet | Yes |

**Best uses:** Alternative to Colab for data processing. Unlimited 6h sessions.

**Limitation:** GPU is weaker than T4 on free tier. Low storage (5GB).

### 3.5 HuggingFace ZeroGPU

| Spec | Value |
|------|-------|
| GPU | NVIDIA H200 (~70GB VRAM) |
| Access | Free for all users (Gradio SDK only) |
| PRO ($9/mo) | 8x usage quota |
| Persistence | None (stateless function calls) |

**Best uses:** Quick inference tests on large models. Could potentially run gpt-oss-20b (14GB) for verification.

**Limitation:** Gradio-only. Not suitable for batch processing or long-running training. Stateless.

### 3.6 Summary: Free Compute Budget

| Resource | GPU | Hours/Week | Total hrs Available | Best Task |
|----------|-----|-----------|-------------------|-----------|
| Google Colab | T4 (16GB) | 15-30 | 15-30/week | Dedup, PRM scoring |
| Kaggle Notebooks | T4/P100 (16GB) | 30 | 30/week | Same, plus Kaggle dataset access |
| Lightning.ai | T4 (16GB) | ~7-35/month | 7-35/month | Persistent dev environment |
| Paperspace | ~M4000 (8GB) | Unlimited 6h sessions | Unlimited | Data processing fallback |
| HF ZeroGPU | H200 (70GB) | Quota-limited | Varies | Quick model inference tests |
| **Total free T4-equivalent** | | **~50-60 hrs/week** | | |

---

## 4. Paid Options (If Needed)

### 4.1 Tinker API ($150 Free Credits)

- Per-token pricing (~$0.40/M tokens for Qwen3-8B, higher for gpt-oss-120b)
- Supports SFT, GRPO, DPO
- Best for RL fine-tuning (GRPO on hard problems)
- No GPU management
- Estimated: $50-150 for full GRPO training run on gpt-oss-120b

### 4.2 Fields Model Initiative (Free, Application Required)

- Up to 128 H100 GPUs
- Through AIMO3 competition partnership
- Best option for serious multi-GPU fine-tuning
- Need to apply through Kaggle competition page

### 4.3 RunPod (Pay-per-Use)

- H100 spot instances: ~$2.49/hr
- A100 80GB spot: ~$1.64/hr
- $5-10 free credits for new users
- Good for: Quick fine-tuning runs if Kaggle H100 isn't enough

### 4.4 Google Colab Pro ($10/mo)

- A100 GPU (40GB or 80GB)
- 24h sessions
- More RAM (52GB)
- Worth it if we need A100 for training Qwen-7B PRM or embedding models

---

## 5. Recommended Resource Allocation Plan

### Phase 1: Data Curation (~$15 Google API + 10h free compute)

| Step | Resource | Time/Cost | Output |
|------|----------|-----------|--------|
| 1. Semantic dedup of 141K AIMO3 TIR | Colab T4 (free) | ~3h | Deduplicated problem set |
| 2. Topic classification (141K problems) | Gemini 2.0 Flash Batch | ~$6.35 | MSC domain labels |
| 3. Quality scoring (30K traces) | Gemini 2.0 Flash Batch | ~$9.30 | Quality scores per trace |
| 4. PRM step-level scoring (10K traces) | Kaggle T4 (free) | ~6h | Step-level quality labels |
| **Phase 1 total** | | **~$15.65 + 9h free** | **Curated candidate pool** |

### Phase 2: Trace Generation (1 Kaggle H100 session)

| Step | Resource | Time | Output |
|------|----------|------|--------|
| 5. Generate traces for hard problems | Kaggle H100 | 8h | 800-1,600 new TIR traces |
| 6. Save as Kaggle dataset | Kaggle H100 | 30min | Persistent trace dataset |
| **Phase 2 total** | | **1 Kaggle session** | **New traces for gaps** |

### Phase 3: Selection + Fine-Tuning (1 Kaggle H100 session + ~$15 API)

| Step | Resource | Time/Cost | Output |
|------|----------|-----------|--------|
| 7. Difficulty estimation (4 attempts, 5K problems) | Gemini 2.0 Flash Batch | ~$8.70 | Pass-rate estimates |
| 8. Final diversity selection (2K-4K examples) | Local CPU | Minutes | Training dataset |
| 9. QLoRA fine-tuning (Unsloth) | Kaggle H100 | ~3h | LoRA adapter |
| 10. Validation (50 problems) | Kaggle H100 | ~4h | Accuracy estimate |
| **Phase 3 total** | | **1 session + ~$8.70** | **Fine-tuned model** |

### Phase 4: Submission + Iteration (~$20 API reserve)

| Step | Resource | Time/Cost | Output |
|------|----------|-----------|--------|
| 11. Competition submission | Kaggle H100 | 9h | Submission with LoRA |
| 12. Analyze results, iterate | Gemini API | ~$20 reserve | Improved dataset |
| **Phase 4 total** | | **1 session + ~$20** | **Competition score** |

### Total Budget Summary

| Resource | Cost | Sessions/Hours |
|----------|------|---------------|
| Google API (Gemini) | ~$40-50 of $93 | Batch processing |
| Kaggle H100 | Free | 3 sessions (27h total) |
| Free compute (Colab/Kaggle T4) | Free | ~15h |
| **Total** | **~$40-50** | |
| **Remaining API budget** | **~$43-53** | For iteration |

---

## 6. Key References

- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini Batch API Docs](https://ai.google.dev/gemini-api/docs/batch-api)
- [Unsloth gpt-oss Fine-Tuning](https://docs.unsloth.ai/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss)
- [vLLM GPT-OSS Recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)
- [Lightning.ai Free Studios](https://lightning.ai/studio)
- [Paperspace Free GPU](https://www.paperspace.com/gradient/free-gpu)
- [HuggingFace ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu)
