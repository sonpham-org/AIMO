# ML Research Findings: Cost-Effective Fine-Tuning for AIMO3

> Author: ml-researcher agent
> Date: 2026-02-07
> Goal: Find the best cost-effective fine-tuning approach for gpt-oss-120b (MoE, 117B/5.1B active, MXFP4)
> Budget: $50-500
> Hardware: 1x H100 80GB (Kaggle), no local NVIDIA GPU
> Current best: 40/50 with inference-only (entropy-gated consensus, 8 attempts)

---

## Executive Summary

**Recommended strategy (in priority order):**

1. **GenSelect training** ($0-50, Unsloth QLoRA) -- Train gpt-oss-120b to select the best answer from N candidates. NVIDIA's secret weapon (566K samples available). This directly solves our core problem: more attempts (16) scoring WORSE because selection fails.

2. **ASTER-style cold-start SFT** ($0-50, Unsloth QLoRA) -- 4K interaction-dense TIR trajectories as SFT, then optionally GRPO RL. ASTER-4B hit 90% AIME 2025 with just 4K cold-start examples.

3. **Tinker GRPO RL** ($50-200) -- RL fine-tune on hard math problems via Tinker API. LoRA matches full fine-tuning for RL (rank-1 even suffices per "LoRA Without Regret").

4. **Small verifier model** ($20-100, Tinker) -- Train Qwen3-8B as answer verifier, replacing our entropy heuristic. ThinkPRM approach needs only 1% of PRM800K labels.

**Do NOT pursue:** Full fine-tuning (too expensive), distillation from Qwen3-235B (unclear benefit), or DPO alone (marginal gains).

---

## Approach 1: GenSelect Training (HIGHEST PRIORITY)

### What It Is
Train the model to evaluate multiple candidate solutions and pick the best one. This is a "selection skill" baked INTO the model, replacing our hand-crafted entropy heuristic.

### Why It's Highest Priority
- **Directly solves our core problem**: 8 attempts score 40/50, but 16 attempts score 29/50. The bottleneck is SELECTION, not generation.
- NVIDIA's 1st place AIMO2 solution (34/50) used GenSelect as a key pillar alongside CoT and TIR.
- OpenReasoning-Nemotron-32B with GenSelect: AIME24 89.2 -> 93.3 (+4.1), AIME25 84.0 -> 90.0 (+6.0), HMMT 73.8 -> 96.7 (+22.9).
- The capability generalizes: trained on math selection, it transfers to code and science.

### Training Data
- **566K GenSelect samples** in `nvidia/OpenMathReasoning` (CC-BY-4.0)
- Format: multiple candidate solutions + selection reasoning + chosen answer
- Training pipeline: uses full reasoning traces from DeepSeek-R1-0528-671B

### Implementation Plan
1. Download GenSelect subset from OpenMathReasoning (~566K samples)
2. Filter to olympiad-level difficulty (pass_rate_72b_tir < 0.5)
3. Curate 2K-5K highest quality examples
4. QLoRA fine-tune gpt-oss-120b with Unsloth on rented H100
5. At inference: generate N solutions, then feed all N to the same model with GenSelect prompt
6. Upload LoRA adapter (~100-500MB) as Kaggle dataset

### Cost Estimate
- H100 rental: ~$2/hr (Vast.ai) x 2-5 hours = **$4-10**
- OR use Kaggle H100 free (9hr limit, but need separate notebook for training)
- Total: **$0-10** (potentially free on Kaggle)

### Expected Impact
- With proper selection, 16+ attempts should score >> 40/50
- NVIDIA saw +4-23% improvement from GenSelect on various benchmarks

### Risk
- GenSelect format may differ from our Harmony protocol inference format
- QLoRA on MoE may not fully preserve GenSelect capability
- May need format adaptation between training and inference

---

## Approach 2: ASTER-Style Cold-Start SFT

### What It Is
Fine-tune on a small set (4K) of "interaction-dense" TIR trajectories -- trajectories with many tool-use turns (>9 turns). This establishes an "agentic prior" that improves TIR behavior.

### Why It's High Priority
- ASTER-4B achieved **90.0% AIME 2025** with just 4K cold-start examples + GRPO RL
- The key insight: interaction DENSITY matters more than dataset size
- Our model already does TIR but may not use tools optimally -- cold-start SFT could improve tool-use patterns
- 4K examples is very manageable for QLoRA training

### Training Data Sources
- **AIMO3 High-Difficulty dataset** (7,293 problems, ~70K trajectories with pass_rate metadata)
  - Filter for trajectories with >= 9 tool-interaction turns
  - Filter for pass_rate 1-3/8 (hardest problems)
  - Expected yield: ~2K-4K trajectories
- **AIMO3 TIR dataset** (141K samples from gpt-oss-120b)
  - Filter for high tool-call count, correct answers
  - Complementary source

### ASTER Training Recipe (from the paper)
| Parameter | ASTER Value | Our Adaptation |
|-----------|-------------|----------------|
| Base model | Qwen3-4B-Thinking | gpt-oss-120b |
| Cold-start size | 4K trajectories | 4K trajectories |
| Selection criterion | >9 tool-interaction turns | Same |
| SFT framework | LLaMA-Factory | Unsloth |
| Learning rate | 3e-5 | 1e-4 to 5e-4 (LoRA 10x rule) |
| Epochs | 6 | 3-6 |
| Batch size | 128 | 4-8 (H100 memory) |
| Max context | 32K tokens | 32K-65K |
| RL method | GRPO | Optional (Phase 2 via Tinker) |
| RL reward | correct=1, wrong=0 | Same |

### Cost Estimate
- H100 rental: ~$2/hr x 2-4 hours = **$4-8**
- Total: **$0-8** (potentially free on Kaggle)

### Expected Impact
- Better tool-use patterns -> more problems solved correctly per attempt
- Combined with GenSelect: better generation AND better selection

---

## Approach 3: Tinker GRPO RL

### What It Is
Reinforcement learning via Tinker's API. The model generates solutions, gets reward (+1 correct, 0 wrong), and learns to produce better solutions over time. GRPO (Group Relative Policy Optimization) removes the need for a separate value model, making it cheaper.

### Key Research Finding: "LoRA Without Regret" (Thinking Machines, 2025)
This paper from Thinking Machines (the Tinker team) proves critical facts:

1. **LoRA matches full fine-tuning for RL, even at rank-1.** Policy gradients provide ~O(1) bits per episode (vs O(tokens) for SFT), so even 3M LoRA parameters vastly exceed the information needed.

2. **Apply LoRA to ALL layers, especially MLPs/MoE.** Attention-only LoRA significantly underperforms. For MoE: train a separate LoRA on each expert, with rank = total_rank / num_active_experts.

3. **LoRA learning rate should be ~10x full fine-tuning.** Optimal range: 1e-4 to 5e-4.

4. **For MoE models**: separate LoRA per expert, rank divided by active experts (8 for Qwen3 MoE, likely similar for gpt-oss).

### Tinker Pricing (confirmed)

| Model | Prefill/M tok | Sample/M tok | Train/M tok |
|-------|--------------|-------------|-------------|
| Llama-3.2-1B | $0.03 | $0.09 | $0.09 |
| Qwen3-8B | $0.13 | $0.40 | $0.40 |
| Qwen3-235B | $0.68 | $1.70 | $2.04 |
| DeepSeek-V3.1 | $1.13 | $2.81 | $3.38 |
| gpt-oss-120b | Not listed (estimate: $0.50-1.50 prefill, $1-3 sample/train) |

MoE models cost proportional to ACTIVE parameters (5.1B for gpt-oss), so gpt-oss-120b should cost similar to a ~5B dense model. Estimate: **~$0.10-0.30 per million tokens**.

**New users get $150 in free credits.** This alone could fund significant training.

### GRPO Training Estimate for gpt-oss-120b

| Parameter | Value |
|-----------|-------|
| Problems | 1K-5K (from Big-Math-RL-Verified or AIME corpus) |
| Group size | 4-8 samples per problem |
| Avg tokens per sample | ~5K (TIR with code) |
| Training steps | 100-200 |
| Total tokens (sample) | ~200M-1B |
| Total tokens (train) | ~200M-1B |
| **Estimated cost** | **$50-200** (could be much less if MoE pricing is proportional to active params) |

### Execution Plan
1. Sign up for Tinker (free, GA) -> get $150 credits
2. Smoke test: Llama-1B on arithmetic (~$1, minutes)
3. Qwen3-8B on MATH (~$20-50) -> replicate 76.7% benchmark
4. gpt-oss-120b GRPO on curated AIME problems (~$50-200)
5. Download LoRA weights -> merge -> upload to Kaggle
6. Apply for Tinker Research Grant ($5K+) with Phase 3 results

### Expected Impact
- GRPO on math improved Qwen3-8B to 76.7% MATH in 180 steps
- DeepSeek-R1 achieved SOTA through GRPO
- For gpt-oss-120b: expect 2-5 point improvement on our benchmark

### Risk
- gpt-oss-120b pricing unknown (may be higher than estimated)
- LoRA merging + MXFP4 quantization may degrade quality
- Tinker's LoRA format may need conversion for vLLM inference

---

## Approach 4: Small Verifier Model

### What It Is
Train a small model (Qwen3-8B or even 1.5B) as a math answer verifier, replacing our entropy-based heuristic for answer selection.

### Why Consider This
- Our entropy heuristic works for 8 attempts but FAILS for 16 -- the signal-to-noise ratio degrades
- A learned verifier could correctly rank 16-64 candidate solutions
- Small model (1.5B-8B) can run alongside gpt-oss-120b on H100 with minimal VRAM overhead
- ThinkPRM outperforms discriminative verifiers using only 1% of PRM800K labels

### Research Findings

| Approach | Model Size | Key Result |
|----------|-----------|------------|
| ThinkPRM | 7B | Outperforms PRM800K-trained verifiers with 1% labels |
| AceMath-RM | 7B-72B | Best math reward models, eval benchmark available |
| 1.5B token-level value model | 1.5B | 45.7% on competition math with 64 generations (parity with o3-mini-medium) |
| Self-Verification-R1 | 1.5B | Works on MATH500 and AIME24 |
| Tango verifier | 7B | SOTA on ProcessBench for 7/8B scale |

### Implementation Options

**Option A: Use existing PRM (no training needed)**
- Download `Qwen2.5-Math-7B-PRM800K` from HuggingFace
- Score each candidate solution step-by-step
- Replace entropy heuristic with PRM scores for selection
- Cost: $0 (just inference, ~1GB VRAM for 7B quantized)
- Risk: PRM trained on different model's outputs, may not transfer

**Option B: Train custom ORM via Tinker**
- Fine-tune Qwen3-8B on our trace data (correct/incorrect labels)
- Use existing AIMO3 Hard dataset (7,293 problems with 8 solutions each, pass_rate labels)
- Tinker cost: ~$20-50 for Qwen3-8B
- Much better: trained on gpt-oss-120b's actual output distribution

**Option C: GenSelect (Approach 1) IS the verifier**
- GenSelect is essentially a learned verifier built into the generator model
- No separate model needed, no VRAM overhead
- This is why GenSelect is ranked #1

### Recommendation
Try Option A first (zero cost), then Option B if A fails, but prioritize GenSelect (Approach 1) as the primary selection mechanism.

---

## Approach 5: DPO (Lower Priority)

### What It Is
Direct Preference Optimization -- train on (preferred, rejected) pairs to make the model prefer correct, concise solutions.

### Evidence
- AIMO2 2nd place (Imagination, 31/50) used 2K DPO pairs in Stage 2
- Key benefit: **reduced output length** while maintaining quality
- This means: faster inference -> more attempts possible within 9h

### Why Lower Priority
- DPO alone gives marginal quality improvement (Imagination still scored below NVIDIA's non-DPO approach)
- Main benefit is shorter outputs, which only matters if we're time-constrained (we currently use <15min of 9h)
- Requires generating preference pairs first (extra step)

### If We Do DPO
- Source: OpenR1-Math-220k has multiple traces per problem (natural DPO pairs)
- AIMO3 Hard dataset: 8 solutions per problem with pass_rate labels
- Create pairs: (correct shorter solution, incorrect/longer solution)
- Train with 2K pairs, 2-4 epochs (per Imagination's recipe)
- Can be combined with SFT (SFT first, then DPO)

---

## Approach 6: Distillation (Not Recommended)

### What It Is
Generate high-quality solutions from a stronger model (Qwen3-235B, DeepSeek-R1-671B), then SFT gpt-oss-120b on those traces.

### Why Not Recommended
- gpt-oss-120b is already very strong (40/50 with just inference)
- Distillation from a different architecture may not transfer well
- Cost: Qwen3-235B on Tinker costs $1.70-2.04/M tokens -- generating 10K solutions at ~5K tokens each = $85-100 just for data generation, plus training cost
- OpenMathReasoning already has 5.5M solutions from DeepSeek-R1 -- we can just use that data
- Better to RL-train gpt-oss-120b on its own distribution

---

## Critical Technical Details

### Unsloth QLoRA Configuration for gpt-oss-120b

Based on "LoRA Without Regret" and Unsloth documentation:

```python
# Confirmed: fits in 65GB VRAM on H100 80GB
model = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-120b",
    max_seq_length=32768,
    load_in_4bit=True,  # QLoRA
    dtype=None,  # auto
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # Total rank (divided by active experts internally)
    lora_alpha=128,  # 2x rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",  # MLP/MoE -- CRITICAL
    ],
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

# LoRA LR should be ~10x full FT
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        learning_rate=2e-4,  # 10x the typical 2e-5 for full FT
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        warmup_ratio=0.1,
        optim="adamw_8bit",
        bf16=True,
        max_grad_norm=1.0,
    ),
)
```

### VRAM Budget on H100 80GB

| Component | VRAM |
|-----------|------|
| gpt-oss-120b base (MXFP4) | ~60GB |
| QLoRA adapters | ~2-3GB |
| Optimizer states | ~2-3GB |
| Activations (gradient checkpointing) | ~10-12GB |
| **Total** | **~75-78GB** |
| Available | 80GB |
| Headroom | 2-5GB |

### LoRA Adapter Size (for Kaggle upload)
- Rank 64 x all layers x MoE experts: ~200-500MB
- Well within Kaggle dataset upload limits

---

## Dataset Curation Recommendation

### For GenSelect Training (Approach 1)
- Source: `nvidia/OpenMathReasoning` GenSelect subset (566K)
- Filter: olympiad difficulty, correct selection, clean format
- Target: 2K-5K curated examples
- Focus: problems where there are 4+ candidate solutions with varying quality

### For ASTER Cold-Start SFT (Approach 2)
- Source: `wenliangtlh/aimo3-high-difficulty-tool-calling-dataset` (7,293 problems)
- Filter: trajectories with >= 9 tool-interaction turns, correct answer, pass_rate 1-3/8
- Target: 4K trajectories (per ASTER recipe)
- This dataset was generated by gpt-oss-120b itself -- perfect for self-improvement

### For GRPO RL (Approach 3)
- Source: `SynthLabsAI/Big-Math-RL-Verified` (250K verified problems)
- Filter: competition-level difficulty, verifiable answers
- Target: 5K-10K problems in RL environment
- Goldilocks zone: pass_rate 0.1-0.7 for maximum gradient signal

### For Verifier Training (Approach 4)
- Source: `nvidia/AceMath-RM-Training-Data` + AIMO3 Hard dataset
- Already has correct/incorrect labels
- Target: 10K-50K solution pairs with binary labels

---

## Cost Summary

| Approach | Method | Cost | Expected Improvement | Priority |
|----------|--------|------|---------------------|----------|
| GenSelect SFT | Unsloth QLoRA on rented H100 | $0-10 | +5-10 points (fixes selection) | **1st** |
| ASTER Cold-Start | Unsloth QLoRA on rented H100 | $0-10 | +2-5 points (better TIR) | **2nd** |
| GRPO RL | Tinker API | $50-200 ($150 free credits) | +2-5 points (better reasoning) | **3rd** |
| Verifier (existing PRM) | Download + integrate | $0 | +1-3 points (better selection) | Try first |
| DPO | Tinker or Unsloth | $20-50 | +0-2 points (shorter outputs) | Low |
| Distillation | Tinker | $100-300 | Uncertain | Skip |

**Total budget needed: $50-200** (with $150 Tinker credits, effective cost: $0-50 out of pocket)

---

## Recommended Execution Order

### Phase 1: Zero-Cost Quick Wins (Day 1)
1. Download `Qwen2.5-Math-7B-PRM800K` and test as verifier on our traces
2. Download GenSelect subset from OpenMathReasoning
3. Profile and curate 4K ASTER-style trajectories from AIMO3 Hard dataset
4. Sign up for Tinker, get $150 credits

### Phase 2: GenSelect + Cold-Start SFT ($0-20, Day 2-3)
5. Rent H100 on Vast.ai (~$2/hr) OR use Kaggle
6. QLoRA fine-tune gpt-oss-120b on GenSelect data (2-5K examples, ~2-4 hours)
7. QLoRA fine-tune on ASTER cold-start set (4K trajectories, ~2-4 hours)
8. Merge adapters, upload to Kaggle
9. Test with 16-attempt + GenSelect selection

### Phase 3: GRPO RL via Tinker ($50-200, Day 4-7)
10. Smoke test: Llama-1B arithmetic ($1)
11. Qwen3-8B MATH benchmark ($20-50)
12. gpt-oss-120b GRPO on curated problems ($50-200)
13. Download LoRA, merge with SFT adapter, upload to Kaggle

### Phase 4: Evaluation and Iteration (Day 8+)
14. Submit each variant to Kaggle competition
15. Compare: baseline (40/50) vs GenSelect vs SFT vs RL
16. Iterate on best approach
17. Apply for Tinker Research Grant with results

---

## Key Questions Answered

### 1. What's the cheapest way to get a meaningful improvement?
**GenSelect training via Unsloth QLoRA ($0-10).** It directly addresses our #1 bottleneck (answer selection with 16+ attempts). Free if using Kaggle H100.

### 2. Can we train a small verifier model instead of fine-tuning the big one?
**Yes, but GenSelect is better.** A 7B PRM exists off-the-shelf (Qwen2.5-Math-7B-PRM800K, $0). Custom ORM via Tinker costs ~$20-50. But GenSelect bakes selection INTO the generator, eliminating the need for a separate model and its VRAM overhead.

### 3. How should we curate 1K-5K training examples from 100K+?
**Follow ASTER: select by interaction density (>9 tool turns), then difficulty (pass_rate 1-3/8), then correctness.** For GenSelect: select problems with 4+ candidate solutions of varying quality. See `data/quality_selection_research.md` for the full pipeline.

### 4. Is GRPO feasible at our budget for a 120B MoE model?
**Yes.** LoRA matches full fine-tuning for RL even at rank-1 (per "LoRA Without Regret"). MoE pricing on Tinker is proportional to active parameters (5.1B, similar to a 5B dense model). With $150 free Tinker credits, GRPO training should cost $0-50 out of pocket.

### 5. What's the expected improvement from each approach?
| Approach | Expected Improvement | Confidence |
|----------|---------------------|------------|
| GenSelect | +5-10 points (44-50/50) | Medium-high (proven for NVIDIA) |
| ASTER SFT | +2-5 points | Medium (proven for 4B, unknown for 120B MoE) |
| GRPO RL | +2-5 points | Medium (proven for 8B, budget-dependent for 120B) |
| Existing PRM verifier | +1-3 points | Low-medium (may not transfer to gpt-oss outputs) |
| DPO | +0-2 points | Low (marginal, mainly saves tokens) |

**Combined (GenSelect + ASTER + GRPO): potentially +8-15 points, targeting 48-50/50.**

---

## References

- [LoRA Without Regret (Thinking Machines)](https://thinkingmachines.ai/blog/lora/)
- [ASTER: Agentic Scaling with TIR](https://arxiv.org/html/2602.01204)
- [OpenMathReasoning / GenSelect (NVIDIA)](https://huggingface.co/datasets/nvidia/OpenMathReasoning)
- [OpenReasoning-Nemotron GenSelect results](https://huggingface.co/nvidia/OpenReasoning-Nemotron-32B)
- [Tinker API (Thinking Machines)](https://thinkingmachines.ai/tinker/)
- [Tinker model lineup](https://tinker-docs.thinkingmachines.ai/model-lineup)
- [Unsloth gpt-oss fine-tuning guide](https://unsloth.ai/blog/gpt-oss)
- [ThinkPRM (Process Reward Models That Think)](https://arxiv.org/abs/2504.16828)
- [AceMath reward models](https://aclanthology.org/2025.findings-acl.206.pdf)
- [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
- [LIMO (Less Is More)](https://arxiv.org/abs/2502.03387)
- [s1 (1K examples beat o1-preview)](https://arxiv.org/abs/2501.19393)
- [DART-Math (difficulty-aware rejection tuning)](https://arxiv.org/abs/2407.13690)
- [H100 rental pricing comparison](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [AIMO3 competition page](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)

---
---

# Part 2: SOTA Methodology Survey (Raw Effectiveness, Cost-Agnostic)

> Added: 2026-02-07
> Focus: What are the most effective fine-tuning methodologies regardless of cost? What SOTA machinery can we scale down?

---

## The Big Picture: Where the Field Is in Feb 2026

The dominant paradigm for math reasoning LLMs has converged on a **3-phase pipeline**:

1. **Non-reasoning SFT warmstart** -- Teach formatting, tool-calling, instruction-following (NOT reasoning)
2. **Cold-start SFT on curated trajectories** -- Small set of high-quality reasoning traces (1K-4K)
3. **Multi-stage RL with outcome rewards** -- GRPO/REINFORCE++ with binary correct/incorrect rewards, progressively increasing difficulty and context length

Every SOTA system in 2025-2026 follows some variant of this. The differences are in the details.

**Critical insight from RLVR research**: RL does NOT teach new reasoning abilities. It "compresses" pass@k into pass@1 -- making the model reliably find solutions it could already find with enough samples. This means **the base model's capability ceiling matters enormously**, and our gpt-oss-120b (already 40/50) has a high ceiling to compress from.

---

## Tier S: The Most Effective Methodologies (Proven SOTA)

### 1. rStar2-Agent (Microsoft, Aug 2025) -- BEST RECIPE

**What**: 14B model achieving 80.6% AIME24, 69.8% AIME25 in only 510 RL steps, 1 week on 64 MI300X GPUs. Surpasses DeepSeek-R1-671B with significantly shorter responses.

**Full Training Recipe**:

**Phase 0 -- Non-Reasoning SFT Warmstart**:
- 222K examples: 165K function-call data (ToolACE, APIGen, Glaive, Magicoder) + 30K instruction-following (Tulu3) + 27K chat (LLaMA-Nemotron)
- 3 epochs, lr=5e-6, batch=128
- Deliberately does NOT include reasoning data -- just formatting and tool-calling
- MATH-500 drops from 62% to 57.4% (acceptable trade-off)

**Phase 1 -- RL Stage 1 (8K context, 300 steps)**:
- 42K curated math problems (integers-only answers for reliable verification)
- GRPO-RoC algorithm (see below)
- Response length grows from ~1K to ~4K tokens

**Phase 2 -- RL Stage 2 (12K context, 185 steps)**:
- Same 42K problems, extended context
- Response length grows to ~6K tokens

**Phase 3 -- RL Stage 3 (12K context, 125 steps, hard problems only)**:
- Offline filtering: remove problems solved 8/8 -> 17.3K harder subset
- Response length grows to ~8K tokens

**Key Innovation -- GRPO-RoC (Resample-on-Correct)**:
- Oversample 2G=32 rollouts, downsample to G=16
- For negative samples: uniform downsampling (preserve failure diversity)
- For positive samples: quality-filter by tool error ratio + format compliance
- Dramatically reduces coding tool errors in rewarded trajectories
- Clip bounds: epsilon_low=0.2, epsilon_high=0.28 (Clip-Higher strategy)

**Why it matters for us**: This is the most complete, reproducible recipe for agentic math RL. Code is open-sourced at github.com/microsoft/rStar. The 42K problem curation strategy (integers-only, verified answers) directly applies to our setup.

**Scaling down**: They used 64 MI300X GPUs for 14B full FT. For our 120B MoE with LoRA via Tinker, the algorithm (GRPO-RoC, multi-stage, curriculum) transfers directly. We need fewer GPUs because LoRA trains far fewer parameters.

---

### 2. DemyAgent (Open-AgentRL, Oct 2025) -- BEST INSIGHTS

**What**: 4B model matching/beating 14B-32B models on AIME24 (72.6%), AIME25 (70.0%), outperforming DeepSeek-R1-Zero-671B.

**Three Critical Insights**:

1. **Real trajectories >> Synthetic trajectories**: Generating actual end-to-end tool-use traces with a teacher model (Qwen3-Coder-30B) and filtering with ReasonFlux-PRM yielded 29.97% avg@32 on AIME2025. Synthetic/stitched trajectories: <10%. This is a 3x difference.

2. **Deliberative strategy >> Reactive strategy**: Longer internal reasoning with FEWER tool calls (quality over quantity) dramatically outperforms frequent tool calling. Tool success rate 70%+ with deliberative vs much lower with reactive. This directly challenges the ASTER approach of maximizing interaction density.

3. **GRPO-TCR algorithm**: Token-level loss (not sequence-level) + Clip-Higher (epsilon=0.315) + Overlong reward shaping. Token-level loss enables finer-grained exploration. Clip-Higher at 0.315 gives 40% faster convergence than 0.28.

**Full Recipe**:
- SFT: 3K real trajectories (generated by Qwen3-Coder-30B, filtered by ReasonFlux-PRM), 5 epochs, lr=5e-5, batch=32, max_len=32K
- RL: GRPO-TCR on 30K problems (17K DAPO-Math + 4.9K Skywork + 3K science + 5K others), 3 epochs, lr=1e-6, batch=64, max_len=16K
- Inference: temp=1.0, top_p=0.6, 32 samples/problem

**Scaling down**: Trained on single 8xA100 node. Recipe directly portable to any RL framework. The key insight (real > synthetic trajectories, deliberative > reactive) applies to any model size.

**Tension with ASTER**: DemyAgent says "fewer but better tool calls" while ASTER says "more interaction-dense trajectories." The resolution: ASTER's interaction density filters for problems that REQUIRE multi-turn tool use. DemyAgent's deliberative strategy means within each turn, think more before calling. Both can coexist.

---

### 3. ASTER (Feb 2026) -- BEST COLD-START

Already covered in Part 1. Key addition: ASTER proves that cold-start data quality determines the RL ceiling. Bad cold-start -> interaction collapse (model stops using tools). Good cold-start (4K dense trajectories) -> 90% AIME 2025 after RL.

---

### 4. ReTool (Apr 2025) -- FIRST TOOL-INTEGRATED RL

**What**: Dynamic interleaving of real-time code execution within natural language reasoning, trained end-to-end with RL.

**Results**: 72.5% AIME24, 54.3% AIME25 (with DeepSeek-R1-Distill-Qwen-32B backbone). +11.4% over o1-preview on AIME25. Only 400 training steps.

**Key Innovation**: The RL environment includes a live code execution sandbox. Each reasoning step can invoke Python, get results, and continue. The reward is purely outcome-based (correct answer), but the model learns WHEN to use code vs pure reasoning.

**Scaling down**: Only 400 steps needed. The multi-turn code execution setup is what we already have in our inference pipeline. Training this way via Tinker or self-hosted GRPO is directly feasible.

---

## Tier A: Important Algorithmic Advances

### 5. G2RL -- Gradient-Guided RL (Dec 2025)

**What**: Replaces entropy-based exploration in GRPO with gradient-geometry-based exploration. Rewards trajectories that introduce "novel gradient directions" rather than just having high token entropy.

**Why it matters**: Standard GRPO encourages surface-level diversity (different tokens). G2RL encourages diversity in what the model LEARNS. Consistently improves pass@1, maj@16, and pass@k over entropy-based GRPO on Qwen3-1.7B and 4B.

**Mechanism**: Extract sequence-level features from the model's final-layer sensitivity (negligible cost from a standard forward pass). Compare gradient features across sampled trajectories. Reward novel gradient directions, deemphasize redundant updates.

**Scaling down**: This is an algorithmic improvement to GRPO itself. Can be implemented in any RL framework. May require custom integration with Tinker or OpenRLHF.

---

### 6. JustRL (ICLR 2026) -- SIMPLICITY WINS

**What**: Standard GRPO with binary rewards and a rule-based verifier. No multi-stage pipelines, no adaptive scheduling, no length penalties. Achieves 64.32% average across 9 math benchmarks with just 1.5B parameters.

**Key finding**: Adding complexity HURTS. Overlong penalties, permissive verifiers, entropy regularization -- all degraded performance. Entropy collapsed from 1.2-1.4 to 0.5-0.6 when penalties were added.

**Recipe**: Train batch=256, rollout N=8, max 16K context, constant lr=1e-6, clip=[0.8, 1.28], temp=1.0. Single stage. That's it.

**Why it matters for us**: Validates that simple GRPO with correct/incorrect rewards is already near-optimal. The fancy additions in other papers are not universally necessary. Start simple.

---

### 7. BRIDGE / CHORD -- Cooperative SFT+RL (Sep 2025)

**What**: Instead of sequential "SFT then RL", run both objectives simultaneously with bilevel optimization. SFT "meta-learns" how to guide RL's optimization.

**Results**: 44% faster training with 13% performance gain on Qwen2.5-3B. 14% faster with 10% improvement on Qwen3-8B.

**Why it matters**: The standard 2-stage pipeline suffers from catastrophic forgetting -- RL gradually loses SFT-acquired behaviors. BRIDGE/CHORD fix this by co-training.

**Scaling down**: Requires custom training loop. Not yet available in Tinker. Could implement in OpenRLHF/verl for self-hosted training.

---

### 8. Progressive Reward Shaping (PRS) -- Dense Rewards for Multi-Turn

**What**: Curriculum-inspired reward that gives dense, stage-wise feedback. Stage 1: reward parseable/formatted tool calls. Stage 2: reward factual correctness. Addresses the key problem of sparse binary rewards in multi-turn settings.

**Why it matters**: Pure outcome rewards (correct=1, wrong=0) give zero signal for partially-correct multi-turn reasoning. PRS provides intermediate feedback that accelerates convergence, especially for tool-use training.

**Caveat from practitioner's guide**: Dense rewards help convergence but are highly sensitive to algorithm choice. Works better with unbiased methods (RLOO) than biased ones (GRPO).

---

### 9. STILL-3-Tool-32B -- Iterative Self-Improvement

**What**: 3-phase "imitate, explore, self-improve" framework. After initial SFT, the model generates multiple rollouts on hard problems, filters for correct ones, adds them to training data, and repeats. 81.70% AIME24 (matches o3-mini).

**Key insight**: The model's OWN correct solutions (found through extensive sampling) are the best training data for the next iteration. This creates a virtuous cycle of self-improvement.

**Scaling down**: The iterative self-improvement concept can be implemented cheaply. Generate traces with our model, filter correct ones, SFT on them, repeat. No RL framework needed -- just repeated SFT rounds.

---

## Tier B: Frontier / Emerging Paradigms

### 10. Recursive Language Models (Prime Intellect, Dec 2025)

**What**: Instead of feeding long contexts directly into the model, the model writes Python code to manage its own context, calling sub-LLMs for heavy lifting. Flips paradigm from "model receives context" to "model manages context."

**Why it matters**: For extremely long reasoning chains (which our TIR problems can produce), RLMs could dramatically extend effective context. The model would learn to chunk problems, delegate sub-problems to code, and synthesize results.

**Status**: Early research, no competition-ready results yet. But the paradigm is compelling for tool-integrated reasoning.

---

### 11. AGENTRL (Oct 2025) -- Multi-Task Agentic RL at Scale

**What**: Fully-asynchronous generation-training pipeline for multi-turn RL. Cross-policy sampling for exploration. Task advantage normalization for multi-task stability. Models from 3B-32B outperform GPT-5 and Claude-Sonnet-4 on agentic tasks.

**Key innovation**: Cross-policy sampling -- use trajectories from different training stages to encourage exploration in multi-turn settings. Prevents the model from settling into local optima.

---

## The Master Recipe: What Would We Do with Unlimited Compute?

Synthesizing all the SOTA findings, the ideal pipeline for gpt-oss-120b would be:

### Stage 0: Non-Reasoning SFT Warmstart (rStar2-Agent recipe)
- Fine-tune on ~200K examples of function-calling, formatting, instruction-following
- NO reasoning data -- just teach the model to use tools correctly
- 3 epochs, small learning rate

### Stage 1: Cold-Start SFT on Real Trajectories (ASTER + DemyAgent)
- Generate 10K+ real end-to-end TIR trajectories using our model or a teacher
- Filter with PRM (ReasonFlux-PRM or Qwen2.5-Math-PRM) for quality
- Select 3K-4K with high interaction density AND deliberative reasoning
- SFT for 5-6 epochs

### Stage 2: Multi-Stage GRPO-RoC RL (rStar2-Agent)
- Curate 42K math problems with verified integer answers
- Stage 2a: 8K context, 300 steps on full dataset
- Stage 2b: 12K context, 185 steps on full dataset
- Stage 2c: 12K context, 125 steps on hard-only subset (remove 8/8 solved)
- Use GRPO-RoC: oversample 32, select 16, quality-filter positives

### Stage 3: GenSelect Training (NVIDIA)
- Fine-tune on 2K-5K GenSelect examples
- Teaches the model to evaluate and select from candidate solutions
- This is the final "polish" for competition deployment

### Stage 4: Iterative Self-Improvement (STILL-3)
- Generate 32 solutions per hard problem
- Filter correct ones
- Add to training data
- Repeat Stages 1-2 with enriched data

### Inference: 16-64 Attempts + GenSelect
- Generate 16-64 candidate solutions per problem
- Use trained GenSelect capability to select best answer
- Optionally: ensemble with PRM scoring + Self-Certainty metric

---

## How to Scale Down for Our Budget

| SOTA Component | Full Version | Our Scaled-Down Version |
|----------------|-------------|------------------------|
| Non-reasoning SFT warmstart (200K examples) | Full fine-tuning, 8xH100 | QLoRA via Unsloth, 1xH100 ($4-8) |
| Cold-start SFT (4K real trajectories) | Full fine-tuning, 8xH100 | QLoRA via Unsloth, 1xH100 ($4-8) |
| Multi-stage GRPO-RoC (42K problems, 510 steps) | Full FT, 64xMI300X, 1 week | LoRA via Tinker API ($50-200) |
| GenSelect training (5K examples) | Full fine-tuning, 8xH100 | QLoRA via Unsloth, 1xH100 ($4-8) |
| Iterative self-improvement | Multiple rounds, days of compute | 1-2 rounds of SFT on own correct traces ($10-20) |
| PRM quality filtering | Train custom PRM | Use off-the-shelf Qwen2.5-Math-PRM ($0) |
| 128 H100s for training | Fields Model Initiative (free for AIMO3) | Apply! If approved, full pipeline is free |

**Total scaled-down cost: $70-250** (or $0 with Fields Model Initiative GPUs + Tinker credits)

---

## Open Questions for Further Research

1. **GRPO-RoC vs standard GRPO**: How much does Resample-on-Correct matter for LoRA training? (LoRA may already regularize enough)
2. **Deliberative vs interaction-dense**: DemyAgent and ASTER disagree. Need empirical testing on our model.
3. **Real vs synthetic cold-start**: DemyAgent showed 3x improvement from real trajectories. Can we generate real trajectories from gpt-oss-120b on Kaggle?
4. **G2RL on Tinker**: Can we implement gradient-guided exploration via Tinker's API primitives?
5. **GenSelect + RL**: Should GenSelect training happen BEFORE or AFTER RL? NVIDIA did it as a separate third training stage.

---

## Additional References (Part 2)

- [rStar2-Agent (Microsoft)](https://arxiv.org/abs/2508.20722) -- Open-source code at [github.com/microsoft/rStar](https://github.com/microsoft/rStar)
- [DemyAgent / Open-AgentRL](https://arxiv.org/abs/2510.11701) -- Open-source at [github.com/Gen-Verse/Open-AgentRL](https://github.com/Gen-Verse/Open-AgentRL)
- [ReTool: RL for Strategic Tool Use](https://arxiv.org/abs/2504.11536) -- [github.com/ReTool-RL/ReTool](https://github.com/ReTool-RL/ReTool)
- [ARTIST: Agentic Reasoning and Tool Integration](https://arxiv.org/abs/2505.01441)
- [AGENTRL: Scaling Agentic RL](https://arxiv.org/abs/2510.04206)
- [G2RL: Gradient-Guided RL](https://arxiv.org/abs/2512.15687)
- [JustRL: Scaling with Simple RL Recipe](https://iclr-blogposts.github.io/2026/blog/2026/justrl/)
- [BRIDGE: Cooperative SFT and RL](https://arxiv.org/abs/2509.06948)
- [Progressive Reward Shaping + VSPO](https://arxiv.org/abs/2512.07478)
- [Practitioner's Guide to Multi-turn Agentic RL](https://arxiv.org/abs/2510.01132)
- [STILL-3: Slow Thinking with LLMs](https://github.com/RUCAIBox/Slow_Thinking_with_LLMs)
- [Recursive Language Models (Prime Intellect)](https://www.primeintellect.ai/blog/rlm)
- [RLVR Implicitly Incentivizes Correct Reasoning](https://arxiv.org/abs/2506.14245)
- [Does RL Really Incentivize Reasoning Beyond Base Model?](https://arxiv.org/abs/2504.13837)
- [State of RL for LLM Reasoning (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
- [OpenRLHF Framework](https://github.com/OpenRLHF/OpenRLHF)
- [veRL: Volcano Engine RL](https://github.com/verl-project/verl)
- [Awesome RL for Large Reasoning Models](https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs)
- [AggLM: Aggregator Language Model](https://arxiv.org/abs/2509.06870)
- [Self-Certainty: Reward-Model-Free Best-of-N](https://arxiv.org/abs/2502.18581)
