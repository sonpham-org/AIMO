# AIMO3 Research Report — February 7, 2026

Comprehensive research findings for the AI Mathematical Olympiad Progress Prize 3 competition.

## Table of Contents
1. [Competition Status](#competition-status)
2. [Selection Strategies](#selection-strategies)
3. [Fine-Tuning Platforms](#fine-tuning-platforms)
4. [Datasets](#datasets)
5. [Key Papers](#key-papers)
6. [Free Compute Resources](#free-compute-resources)
7. [Actionable Recommendations](#actionable-recommendations)

---

## 1. Competition Status

### AIMO3 Overview
- **Prize pool**: $2,207,152 main + $110,000 extra prizes
- **Problems**: 110 original problems (50 public, 50+ private), National Olympiad to IMO level
- **Domains**: Algebra, combinatorics, geometry, number theory
- **Answers**: 5 digits (0-99999), designed to prevent guessing
- **Hardware**: 1x H100 80GB, 9-hour limit, offline (no internet)
- **Deadline**: April 15, 2026
- **Extra prizes**: Longest Leader Prize, Write-up Prizes, MathCorpus Prize, Hardest Problem Prize

### AIMO2 Winner Reference
- **1st place (NemoSkills/NVIDIA)**: 34/50, custom OpenMath-Nemotron-14B-Kaggle
- **Our best**: 40/50 (feb3, entropy-gated consensus) — already above AIMO2 winner
- **Our worst**: 29/50 (feb6, 16 attempts) — selection strategy degrades with more samples

### Current Leaderboard
- Could not access real-time leaderboard data (Kaggle uses JS rendering)
- Competition is active with teams submitting regularly
- Leaderboard: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/leaderboard
- Leaderboard tracker notebook: https://www.kaggle.com/code/hvanphucs112/aimo3-leaderboard-evolution-updated-daily

### Notable Public Notebooks
- "AIMO 3 - GPT OSS 120B [~3hours wow H100]" by seshurajup
- "AIMO 3 | GPT-OSS-120B (with tools)" by andreasbis (our deps source)
- "AIMO 3 - GPT OSS 120B + Agentic Solver" by seshurajup
- "AIMO 3 Baseline - GPT OSS 120B" by takuji

---

## 2. Selection Strategies

### Problem Statement
Our entropy-gated consensus works great at 8 attempts (40/50) but degrades at 16 attempts (29/50). We need selection strategies that SCALE with more samples.

### 2.1 AggLM — Learned Aggregation via RL (MOST PROMISING)
- **Paper**: "The Majority is not always right: RL training for solution aggregation" (arXiv:2509.06870, Sep 2025)
- **Method**: Train a small aggregator model (1.7B params) with RL to review, reconcile, and synthesize answers from multiple candidate solutions
- **Key Results**:
  - AggLM-1.7B outperforms majority voting by 3-7 points across AIME24, AIME25, HMMT24, HMMT25
  - Outperforms 72B reward models despite being only 1.7B
  - Evaluated with 128 solutions per problem, partitioned into 16 sets of 8
  - **Generalizes to solutions from stronger models** — can aggregate gpt-oss-120b outputs
- **Training**: Careful balancing of easy/hard examples, learns to recover minority-but-correct answers
- **Relevance**: Directly addresses our problem. Could use a small model to aggregate gpt-oss-120b's solutions.
- **Status**: Paper published, model/code availability unclear — need to check HuggingFace

### 2.2 Self-Certainty — KL-Divergence-Based Selection (DIRECTLY IMPLEMENTABLE)
- **Paper**: "Scalable Best-of-N Selection for LLMs via Self-Certainty" (arXiv:2502.18581, Feb 2025, ICLR workshop)
- **Method**: Uses KL divergence from uniform distribution on the model's output logprobs as a confidence measure
- **Formula**: Self-Certainty = -1/(nV) * sum(log(V * p(j|x, y<=i)))
- **Key Properties**:
  - Scales effectively with increasing N (unlike majority voting)
  - No external reward model needed — uses logprobs we already collect
  - Combined with Borda voting, outperforms self-consistency on LiveBench-Math, GSM8K, MATH
  - KL divergence is the ONLY measure that consistently improves as N increases to 32 and 64
- **Implementation**: GitHub at backprop07/Self-Certainty
- **Relevance**: We already collect top-5 logprobs. This can be computed directly from our trace data. Should be the FIRST thing we try.

### 2.3 REBASE — Compute-Optimal Tree Search (ICLR 2025)
- **Paper**: "Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference" (ICLR 2025)
- **Method**: Tree search using process reward model to control node expansion
- **Key Results**:
  - Compute-optimal strategy at ALL budgets
  - 7x more efficient than best-of-N
  - 7B model is typically optimal size
- **Limitation**: Requires tree search (not just parallel sampling), needs architectural changes to our pipeline

### 2.4 GenSelect — NVIDIA's Answer Picker (AIMO2 Winner)
- **Paper**: AIMO-2 Winning Solution (arXiv:2504.16891)
- **Method**: Train a model to select the most promising solution from candidates
- **Training Data**: 566K GenSelect examples in OpenMathReasoning dataset
  - 2-16 candidate solution summaries per problem
  - At least one correct and one incorrect
  - QwQ-32B selects the best
- **Dataset available**: `load_dataset("nvidia/OpenMathReasoning", split="genselect")`
- **Relevance**: We could fine-tune a small model on this data for answer selection

### 2.5 Process Reward Models (PRMs)
- **Qwen2.5-Math-PRM-7B/72B**: Open-source PRMs that score each reasoning step
  - Insert `<extra_0>` token after each step, extract probability score (0-1)
  - Outperform majority voting at best-of-N reranking
  - 7B model outperforms maj@8 by 1.4% average across 7 tasks
- **ThinkPRM**: Builds data-efficient PRMs using verification chain-of-thought
  - Outperforms LLM-as-Judge using only 1% of process labels
- **R-PRM**: Reasoning-driven PRM, +13.9 F1 over Qwen2.5-Math-7B baseline
- **Limitation**: Running a separate PRM alongside gpt-oss-120b on single H100 may be infeasible

### 2.6 Inference Scaling Laws
- **Key finding**: No single test-time compute strategy universally dominates
- **Prompt-dependent**: Effectiveness varies by problem difficulty
- **Optimal**: Adaptive, prompt-dependent strategy allocation
- **Implication**: We should use different selection strategies for easy vs hard problems

---

## 3. Fine-Tuning Platforms

### 3.1 Tinker (Thinking Machines Lab) — RECOMMENDED
- **Website**: https://thinkingmachines.ai/tinker/
- **Docs**: https://tinker-docs.thinkingmachines.ai/
- **Status**: General Availability (no waitlist)
- **Supports gpt-oss-120b**: YES (listed as "Medium, MoE" in model lineup)
- **Training methods**: SFT (LoRA), RL (REINFORCE, PPO, GRPO-style)
- **Cookbook**: https://github.com/thinking-machines-lab/tinker-cookbook
  - Math RL recipe included (reward = correct answer verification)
  - Ready-to-use SL and RL training loops
- **Pricing** (per million tokens):
  | Model | Prefill | Sample | Train |
  |-------|---------|--------|-------|
  | Llama-3.2-1B | $0.03 | $0.09 | $0.09 |
  | Qwen3-4B | $0.07 | $0.22 | $0.22 |
  | Llama-3.1-8B / Qwen3-8B | $0.13 | $0.40 | $0.40 |
  | Llama-3.1-70B | $1.05 | $3.16 | $3.16 |
  | Qwen3-235B (MoE) | $0.68 | $1.70 | $2.04 |
  | gpt-oss-120b (MoE) | ~TBD | ~TBD | ~TBD |
  - MoE models priced by active params, so gpt-oss-120b (5.1B active) should be ~$0.13-0.40/M tokens (comparable to 8B dense)
- **Research Grants**: Starting at $5,000 credits, rolling applications, ~1 week response
  - Apply: https://thinkingmachines.ai/blog/tinker-research-and-teaching-grants/
  - Eligible: "research projects and open-source software that uses Tinker"
- **Estimate**: 5K training examples × ~4K tokens/example × 3 epochs = ~60M tokens → ~$24-80 total for gpt-oss-120b SFT

### 3.2 Unsloth — Local QLoRA on H100 (ALTERNATIVE)
- **Website**: https://unsloth.ai/
- **gpt-oss-120b support**: YES
- **VRAM for QLoRA**: 65GB (fits H100 80GB)
- **VRAM for GRPO RL**: Claims 120GB needed, may not fit single H100 80GB
- **Features**: 1.5x faster, 70% less VRAM, 10x longer context
- **Context with QLoRA**: Up to 81K on 80GB H100
- **Notebooks**: Google Colab notebooks available for both SFT and GRPO
- **4-bit RL**: "Unsloth is the only framework to support 4-bit RL for gpt-oss"
- **Approach**: Run on Kaggle H100 or rent an H100
- **Tutorial**: https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune

### 3.3 Cloud GPU Rentals
- **RunPod**: H100 at $1.99/hr (community) to $2.39/hr (secure)
- **Vast.ai**: H100 from $1.87/hr
- **Lambda Labs**: H100 cluster access, enterprise pricing
- **Jarvislabs**: H100 at $2.99/hr, per-minute billing
- **Estimate for RL training**: 10-50 hours on 1x H100 = $20-120

### 3.4 Fields Model Initiative — FREE COMPUTE
- Up to 128 H100 GPUs available for AIMO3 participants
- Partnership with National Institute of Informatics (Tokyo) and Benchmarks+Baselines (Vienna)
- Tinker credits also available through AIMO3 partnership
- **Must apply through Kaggle competition page**

---

## 4. Datasets

### 4.1 OpenMathReasoning (NVIDIA) — PRIMARY DATASET
- **HuggingFace**: nvidia/OpenMathReasoning
- **License**: CC-BY-4.0
- **Source**: 306K unique problems from AoPS forums
- **Total**: 5.68M rows across splits
- **Splits**:
  | Split | Size | Description |
  |-------|------|-------------|
  | cot | 3.2M | Chain-of-thought solutions |
  | tir | 1.72M | Tool-integrated reasoning |
  | genselect | 566K | Solution selection training |
  | additional_problems | 193K | Problems without solutions |
- **Fields**: problem, expected_answer, generated_solution, problem_type, problem_source, generation_model, inference_mode, pass_rate_72b_tir, used_in_kaggle
- **Generation models**: DeepSeek-R1, Qwen2.5-32B-Instruct, QwQ-32B
- **Won AIMO2**: Foundation of NVIDIA's winning submission

### 4.2 NuminaMath-TIR
- **Size**: ~70K problems with TIR solutions
- **Source**: Subset of 860K NuminaMath-CoT, filtered for numerical outputs
- **Won AIMO1**: Foundation of Numina's winning submission
- **Usage**: SFT training for tool-integrated reasoning

### 4.3 DeepScaleR Dataset
- **Size**: ~40K unique math problem-answer pairs
- **Sources**: AIME (1984-2023), AMC (pre-2023), Omni-MATH, Still
- **Processing**: Answer extraction via Gemini, dedup via RAG, SymPy filtering
- **Result**: DeepScaleR-1.5B achieves 43.1% on AIME 2024 (vs 28.8% base)

### 4.4 AoPS-Instruct
- **Size**: 600K+ QA pairs from Art of Problem Solving forum
- **Features**: Community-driven, olympiad-level, step-by-step solutions
- **LiveAoPSBench**: Contamination-resistant evaluation benchmark

### 4.5 S1.1K Dataset
- **Size**: 1,000 high-quality reasoning traces
- **Key finding**: 1K curated examples with iw-SFT achieve 66.7% on AIME
- **Lesson**: Quality >>> quantity for fine-tuning

### 4.6 AIMO3-TIR (Kaggle)
- **Size**: ~141K problems
- **Location**: Kaggle dataset (competition-specific)
- **Format**: TIR solutions for AIMO3 problems

---

## 5. Key Papers (2025-2026)

### 5.1 ASTER — Agentic Scaling with Tool-integrated Extended Reasoning
- **arXiv**: 2602.01204 (February 2026 — 1 WEEK OLD)
- **Result**: ASTER-4B achieves **90.0% on AIME 2025** — SOTA
- **Method**: Cold-start SFT (4K interaction-dense trajectories) + RL
- **Key insight**: Interaction density > trajectory quantity
  - Small expert cold-start set of 4K trajectories yields strongest performance
  - Avoids "interaction collapse" where models stop using tools
- **Outperforms**: DeepSeek-V3.2-Exp, Qwen3-235B-A22B-Thinking despite being only 4B
- **Base model**: Not specified (likely Qwen-series based)
- **Release status**: Code/model referenced but availability unclear

### 5.2 SimpleTIR — End-to-End RL for Multi-Turn TIR
- **arXiv**: 2509.02479 (September 2025)
- **Result**: 50.5% on AIME24 (7B), 59.9% (32B)
- **Method**: Zero RL from Qwen2.5 base models (no instruction tuning needed)
- **Key insight**: Fixes training instability from "distributional drift" via void turn filtering
- **Base models**: Qwen2.5-7B-Base and Qwen2.5-32B-Base
- **Outperforms**: ToRL (50.5 vs 40.2 on AIME24 at 7B scale)
- **Training data**: Math3-5 from SimpleRL and DeepScaleR
- **Code & Model**: Released on GitHub and HuggingFace

### 5.3 ToRL — Scaling Tool-Integrated RL
- **arXiv**: 2503.23383 (March 2025)
- **Result**: ToRL-7B reaches 43.3% on AIME '24
- **Method**: RL for autonomous tool use, models explore optimal strategies
- **Emergent behaviors**: Strategic tool invocation, self-regulation of bad code
- **Training data**: 75K verifiable questions from NuminaMATH, MATH, DeepScaleR

### 5.4 THOR — Tool-Integrated Hierarchical Optimization via RL
- **Method**: TIRGen pipeline for constructing high-quality TIR training data
- **Features**: Actor-critic pipeline + hierarchical RL for joint trajectory/code optimization
- **Self-correction**: Uses tool feedback to fix reasoning errors at inference

### 5.5 AggLM — Learned Solution Aggregation
- (See Section 2.1 above)

### 5.6 Self-Certainty
- (See Section 2.2 above)

---

## 6. Free Compute Resources

### Fields Model Initiative (through AIMO3)
- **What**: Up to 128 H100 GPUs for fine-tuning
- **Partners**: National Institute of Informatics (Tokyo), Benchmarks+Baselines (Vienna)
- **How to apply**: Through Kaggle competition page
- **Cost**: FREE

### Tinker Research Grants
- **What**: $5,000+ in credits for fine-tuning via Tinker API
- **Eligibility**: Research projects using Tinker, open-source software
- **How to apply**: Typeform application, rolling basis, ~1 week response
- **URL**: https://thinkingmachines.ai/blog/tinker-research-and-teaching-grants/

### Tinker Credits via AIMO3 Partnership
- **What**: Tinker credits available to AIMO3 participants
- **How to apply**: Through competition page

---

## 7. Actionable Recommendations

### Immediate (This Week)
1. **Implement Self-Certainty selection** — Uses logprobs we already collect, no extra model needed. Compute KL divergence from uniform distribution instead of raw entropy. Test with our existing 16-attempt trace data.
2. **Apply for free compute**: Fields Model Initiative (128 H100s) and Tinker Research Grant ($5K+ credits)
3. **Download OpenMathReasoning GenSelect split** (566K examples) for answer selection training

### Short-term (Next 2 Weeks)
4. **Tinker smoke test**: Run math RL recipe on gpt-oss-120b with 1K examples, estimate cost
5. **Try Unsloth QLoRA SFT**: Fine-tune gpt-oss-120b on curated TIR data (start with 1-5K high-quality examples)
6. **Explore AggLM approach**: Check if model is released; if not, train small aggregator on OpenMathReasoning GenSelect data

### Medium-term (Next Month)
7. **Full RL training**: Use Tinker or Unsloth for GRPO on gpt-oss-120b with math rewards
8. **ASTER-style cold start**: Curate 4K interaction-dense TIR trajectories for cold-start SFT before RL
9. **Test 32-48 samples per problem** with improved selection strategy (self-certainty + weighted voting)

### Key Insight
The gap between our 40/50 and a potential 45+/50 is likely **selection strategy + fine-tuning**, not more raw samples. The ASTER paper proves that even a 4B model can reach 90% on AIME with the right training recipe. Our gpt-oss-120b is already strong — we need to:
1. Fix selection to scale with N (self-certainty, AggLM)
2. Fine-tune for tool-use density (ASTER-style cold start)
3. Use enough samples (32-48, not 8 or 16)

---

## Sources

### Papers
- ASTER: https://arxiv.org/abs/2602.01204
- SimpleTIR: https://arxiv.org/abs/2509.02479
- ToRL: https://arxiv.org/abs/2503.23383
- AggLM: https://arxiv.org/abs/2509.06870
- Self-Certainty: https://arxiv.org/abs/2502.18581
- REBASE/Inference Scaling: https://arxiv.org/abs/2408.00724
- AIMO2 Winning Solution: https://arxiv.org/abs/2504.16891
- ThinkPRM: https://arxiv.org/abs/2504.16828
- R-PRM: https://arxiv.org/abs/2503.21295

### Datasets
- OpenMathReasoning: https://huggingface.co/datasets/nvidia/OpenMathReasoning
- NuminaMath: https://huggingface.co/AI-MO/NuminaMath-7B-TIR
- Qwen2.5-Math-PRM-7B: https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B
- DeepScaleR: https://github.com/tongyao-zhu/deepscaler
- Self-Certainty impl: https://github.com/backprop07/Self-Certainty

### Platforms
- Tinker: https://thinkingmachines.ai/tinker/
- Tinker Cookbook: https://github.com/thinking-machines-lab/tinker-cookbook
- Tinker Grants: https://thinkingmachines.ai/blog/tinker-research-and-teaching-grants/
- Unsloth gpt-oss: https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune
- Unsloth GRPO: https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/gpt-oss-reinforcement-learning

### Competition
- AIMO3 Kaggle: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/
- AIMO Prize updates: https://aimoprize.com/updates/
- Fields Model Initiative: via Kaggle competition page
