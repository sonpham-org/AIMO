# AIMO3 Dataset Generation: Comprehensive Research Report

> Date: 2026-02-07
> Team: aimo-dataset research team
> Status: FINAL
> Urgency: **MathCorpus Prize deadline is February 9, 2026 -- 2 days away**

---

## 1. Executive Summary

This report synthesizes all research conducted by our team across five parallel investigation tracks: competition rules, token efficiency, answer verification, existing datasets, and compute resources. The findings inform a dataset generation strategy for two overlapping goals:

1. **MathCorpus Prize (deadline: Feb 9)**: Publish a novel, high-quality dataset on Kaggle/HuggingFace
2. **Fine-tuning gpt-oss-120b (deadline: Apr 15)**: Create curated training data to push our score beyond 40/50

### Key Findings

- **Our baseline (40/50) already beats the AIMO2 winner (34/50)** -- but the competition model (gpt-oss-120b) is shared, so others will catch up fast.
- **The bottleneck is answer selection, not generation.** 8 attempts score 40/50, but 16 attempts score 29/50. Selection degrades with more samples.
- **Quality crushes quantity.** LIMO (817 examples) = 57.1% AIME. 100K random SFT = 32.3%. ASTER (4K TIR) = 90% AIME 2025.
- **GenSelect (learned answer selection)** is the single most impactful technique: +4 to +23 points on benchmarks (NVIDIA, AIMO2 winner).
- **Token efficiency gains of 30-40%** are achievable through difficulty-aware trace curation + DPO, enabling 12-16 attempts within the same time budget.
- **Verification must go beyond answer checking.** "Right answer, wrong reasoning" is a toxic training signal. ThinkPRM-14B + cross-model agreement catches it.
- **$93 Google API + free Kaggle H100 + $150 Tinker credits** provide sufficient compute for a competitive dataset and fine-tuning pipeline.

### Recommendations for the 2-Day Sprint (Feb 7-9)

| Priority | Action | Time | Output |
|----------|--------|------|--------|
| 1 | Curate 3K-4K TIR traces from AIMO3 Hard (7.3K problems) | 6-8h | Track A SFT dataset |
| 2 | Package as Kaggle Dataset + HuggingFace with documentation | 3-4h | MathCorpus submission |
| 3 | Download + curate GenSelect subset from OpenMathReasoning | 4-6h | Track B selection dataset |
| 4 | Run verification pipeline on curated data | 4-6h | Quality-assured final dataset |

---

## 2. Competition Context

### 2.1 AIMO Progress Prize 3

The AI Mathematical Olympiad Progress Prize 3 (AIMO3) is a Kaggle competition with a total prize pool of **$2.2M** plus **$110K in Extra Prizes**. Key parameters:

| Parameter | Value |
|-----------|-------|
| Problems | 110 original math problems |
| Difficulty | National Olympiad to IMO |
| Hardware | 1x NVIDIA H100 (80GB), 9h limit, offline |
| Answers | Integer 0-99999 |
| Main deadline | April 15, 2026 |
| Public sharing | **Mandatory** -- all code and datasets must be public |

### 2.2 The MathCorpus Prize

The MathCorpus Prize is one of four Extra Prizes (EP), allocated from the $110K EP pool (estimated $25-30K). It replaces the "Early Sharing Prize" from previous iterations and explicitly rewards **publishing novel datasets that help the wider community.**

**Criteria (inferred from rules + precedent)**:
1. **Novelty** -- not repackaging existing datasets
2. **Community benefit** -- demonstrably helps train better math models
3. **Public availability** -- freely downloadable on Kaggle/HuggingFace
4. **Mathematical depth** -- targets olympiad-level reasoning

**Precedent**: AIMO1's Early Sharing Prize went to NuminaMath (860K problems). AIMO2's went to the first public notebook reaching 20/50. The shift to "MathCorpus" signals the organizers want **datasets specifically**, not just shared notebooks.

### 2.3 What a Winning MathCorpus Submission Looks Like

Based on analysis of AIMO1/AIMO2 winning datasets and current state of the art:

**Must-haves:**
- Novel data (not just repackaged NuminaMath or OpenMathReasoning)
- Verified correctness (both answer and reasoning)
- Public availability with permissive license
- Olympiad-level difficulty

**Differentiators:**
- TIR format with real code execution (ASTER proves this is most effective)
- gpt-oss-120b traces (model-matched to competition hardware)
- Difficulty metadata (pass_rate) for targeted training
- Multi-solution data with correct/incorrect labels (enables GenSelect-style training)
- Diverse mathematical domains (algebra, combinatorics, geometry, number theory)

### 2.4 Existing AIMO3-Specific Datasets

| Dataset | Size | Generator | Key Feature |
|---------|------|-----------|-------------|
| AIMO3 TIR | 141K traces | gpt-oss-120b | Harmony format, avg 21K chars |
| AIMO3 Hard | 7.3K problems / 70K traces | gpt-oss-120b | Has pass_rate metadata |
| AIMO External | 4.5MB | Various | Most popular (78 votes) |

Our novelty angle: **curated, quality-verified, difficulty-stratified TIR traces with selection metadata** -- no existing dataset provides this combination.

---

## 3. Current State Analysis

### 3.1 Our Baseline: 40/50

Our best submission (feb3) uses entropy-gated consensus with 8 parallel attempts per problem:
- Model: gpt-oss-120b (117B total / 5.1B active MoE, MXFP4, ~60GB VRAM)
- Inference: vLLM + OpenAI API + 16 persistent Jupyter kernels (TIR)
- Selection: filter by entropy threshold, require consensus, entropy-weight remaining answers
- Runtime: ~15 minutes (massive headroom in the 9h budget)

### 3.2 The Selection Bottleneck

| Attempts | Score | Lesson |
|----------|-------|--------|
| 8 | **40/50** | Sweet spot for our heuristic |
| 12 | 37/50 | Selection starts degrading |
| 16 | 29/50 | Selection collapses |

**Root cause**: Our entropy-weighted consensus is a heuristic that breaks down with more samples. The signal-to-noise ratio degrades as N increases. NVIDIA solved this with GenSelect -- a learned selection model that improves with more candidates.

### 3.3 Four Improvement Goals

| Goal | Why | How |
|------|-----|-----|
| (a) Shorter solutions | More attempts within time budget | SFT on concise traces + DPO |
| (b) Higher accuracy | Fewer wrong attempts = less selection noise | SFT on verified correct solutions |
| (c) Better verification | "Right answer, wrong reasoning" hurts | PRM + cross-model checking |
| (d) Better selection | Core bottleneck -- more attempts should help | GenSelect training |

---

## 4. Research Findings

### 4.1 Token Efficiency (Researcher-2)

Every token saved per solution translates to more attempts per problem. Research identifies multiple proven techniques for 30-60% token reduction without accuracy loss.

**Ranked by practicality for our setup:**

| Priority | Technique | Reduction | Accuracy Impact | Source |
|----------|-----------|-----------|-----------------|--------|
| 1 | SFT + DPO length optimization | 30% | Neutral to +2% | AIMO2 2nd place |
| 2 | Difficulty-aware trace curation | 30% | Neutral | DA-CoTD (arxiv:2509.05226) |
| 3 | Length reward in GRPO | 33-40% | Neutral to +14% | L1, e1, STILL-2 |
| 4 | TokenSkip compression | 40% | -0.4% | EMNLP 2025 |
| 5 | MCoT (Markov Chain of Thought) | 47% | Slight drop on hard | NAACL 2025 |

**Key mechanism: Difficulty-aware trace selection.** For each problem:
- Easy (pass_rate > 0.5): pick the **shortest** correct solution
- Medium (0.15 < pass_rate <= 0.5): pick the **median-length** correct solution
- Hard (pass_rate <= 0.15): pick the **highest-quality** solution (allow longer)

This teaches the model to allocate reasoning effort proportionally to difficulty. Expected impact: ~30% token reduction across 50 problems = ~35% more attempts possible.

**DPO for length reduction** (AIMO2 2nd place recipe): After SFT, train on (shorter_correct, longer_correct) pairs with ratio_threshold=0.7 and min_length=500. The model learns to prefer concise correct solutions. Additional 10-20% token reduction.

**Token budget impact:**

| Configuration | Tokens/Solution | Attempts Possible |
|--------------|----------------|-------------------|
| Current (no FT) | ~10K | 8 |
| SFT only | ~8K | 10 |
| SFT + DPO | ~7K | 12-14 |
| SFT + DPO + Length RL | ~6K | 14-16 |

**What NOT to do:**
- Do not try latent reasoning (Coconut/CODI) -- incompatible with vLLM
- Do not aggressively compress hard problems -- they need full reasoning depth
- Do not use length penalties without correctness gating -- produces short wrong answers

### 4.2 Answer and Logic Verification (Researcher-3)

Verification requires two distinct layers: outcome verification (is the answer correct?) and process verification (is the reasoning sound?).

**The critical insight: training on solutions with correct answers but flawed reasoning ("right answer, wrong reasoning") actively degrades model performance.**

#### Recommended 5-Stage Verification Pipeline

| Stage | Method | Filters | Cost |
|-------|--------|---------|------|
| 1. Outcome | math_verify + SymPy | ~30-50% of solutions | Near-zero |
| 2. TIR Integrity | Structural checks | ~10-20% of correct | Near-zero |
| 3. Cross-Model | Second model agreement | ~20-40% more | Low |
| 4. Process (PRM) | ThinkPRM-14B step-level | Score remaining | ~1h for 10K |
| 5. Quality Ranking | Combined scoring | Select best per problem | Near-zero |

**Available Process Reward Models (zero cost):**

| Model | Size | Type | Strength |
|-------|------|------|----------|
| ThinkPRM-14B | 14B | Generative (recommended) | Interpretable, generalizes across models |
| ThinkPRM-1.5B | 1.5B | Generative | Lightweight screening |
| Qwen2.5-Math-PRM-7B | 7B | Discriminative | General math |
| ReasonFlux-PRM-7B | 7B | Trajectory-aware | Long CoT / TIR traces |

**PRM Mismatch Warning**: No PRM exists for gpt-oss-120b. Use generative PRMs (ThinkPRM) which generalize better than discriminative ones, and use scores as **soft weights, not hard filters**.

**Cross-model agreement** (AceMath method) is the strongest filter for catching "lucky correct" solutions. Generate answers with two architecturally different models; keep only where both agree with ground truth. AceMath reduced 2.3M to 800K samples with no loss in benchmark performance.

**GenSelect** (NVIDIA, AIMO2 winner) addresses both verification problems simultaneously by training the model to evaluate and select the best solution from N candidates. Results: AIME24 +4.1, HMMT +22.9. The 566K GenSelect training samples are publicly available in nvidia/OpenMathReasoning (CC-BY-4.0).

### 4.3 Existing Datasets and Quality Selection (Resource-Finder + Researcher-4)

We cataloged 29+ datasets across 7 tiers. The most relevant for our pipeline:

**Tier 1: Competition-Winning**

| Dataset | Size | Key Feature | License |
|---------|------|-------------|---------|
| OpenMathReasoning (NVIDIA) | 306K problems / 5.5M solutions | Won AIMO2, has TIR + GenSelect | CC-BY-4.0 |
| NuminaMath-1.5 | 896K problems | Won AIMO1 | Apache 2.0 |

**Tier 1.5: AIMO3-Specific (already downloaded)**

| Dataset | Size | Key Feature |
|---------|------|-------------|
| AIMO3 TIR | 141K traces | gpt-oss-120b, Harmony format |
| AIMO3 Hard | 7.3K problems / 70K traces | pass_rate metadata, difficulty-filtered |

**Tier 3: Curated High-Quality Small (proven quality > quantity)**

| Dataset | Size | Result |
|---------|------|--------|
| LIMO | 817 | 57.1% AIME, 94.8% MATH |
| s1K | 1,000 | Beat o1-preview |
| ASTER cold-start | 4,000 TIR | 90% AIME 2025 |

**Quality Selection Evidence:**

The research is unambiguous -- quality crushes quantity:
- LIMO: 817 curated >> 100K random (57.1% vs 32.3% AIME)
- s1: 1,000 examples beat o1-preview, trained in 26 minutes
- ASTER: 4K interaction-dense TIR >> 45K mixed
- NVIDIA Front-Loading: doubling mixed-quality data hurt by -5%
- Skill-Aware Selection: full 100K corpus DEGRADES performance vs base model

**The 3 essential selection criteria** (all three matter, any single criterion alone is significantly worse):
1. **Difficulty**: pass_rate 0.03-0.15 for SFT, 0.10-0.70 for RL (LIMO, DART-Math, s1)
2. **Quality**: LIMO's 4-dimension scoring (reasoning depth, self-verification, exploration, logical flow)
3. **Diversity**: topic classification + balanced sampling across mathematical domains (s1's MSC codes)

**ASTER's key finding for TIR**: interaction density is the critical quality metric. Trajectories with >= 9 tool calls dramatically outperform those with fewer. 4K dense trajectories beat 45K mixed.

### 4.4 Novelty Strategy

To win the MathCorpus Prize, our dataset must be genuinely novel. Existing datasets already cover:
- AoPS problems with CoT solutions (OpenMathReasoning)
- Large-scale TIR traces (NuminaMath-TIR, AIMO3 TIR)
- Curated small datasets (LIMO, s1K)

**Our unique angle:**
1. **gpt-oss-120b TIR traces** -- the competition model itself, giving perfectly matched training data
2. **Quality-verified with multi-stage pipeline** -- not just answer-correct, but reasoning-verified
3. **Difficulty-stratified** -- pass_rate metadata enables targeted training
4. **Selection metadata** -- correct/incorrect labels + entropy scores enable GenSelect-style training
5. **Difficulty-aware curation** -- traces selected with length proportional to problem difficulty

This combination does not exist in any publicly available dataset.

---

## 5. Dataset Generation Plan

### 5.1 Three-Track Architecture

| Track | Purpose | Target Size | Source | Priority |
|-------|---------|-------------|--------|----------|
| **A: TIR Cold-Start SFT** | Improve per-attempt accuracy | 3K-4K | AIMO3 Hard + TIR | 1st |
| **B: GenSelect** | Fix selection for 16+ attempts | 2K-5K | OpenMathReasoning | 1st (tied) |
| **C: RL Problems** | Self-improvement via GRPO | 5K-10K | Big-Math-RL-Verified | 2nd |

### 5.2 Track A: 8-Stage Curation Pipeline

```
Stage 0: Load & normalize → ~150K raw samples
Stage 1: Hard filters (length, tool calls, answer marker, repetition) → ~105K (70%)
Stage 2: Outcome verification (math_verify + SymPy) → ~50-70K (50%)
Stage 3: Decontamination (9-gram + embedding similarity) → ~48-65K (95%)
Stage 4: Difficulty calibration (pass_rate 0.03-0.30) → ~15-25K (30%)
Stage 5: Process verification (TIR integrity + cross-model + PRM) → ~8-15K (50%)
Stage 6: Quality scoring (difficulty + solution quality + interaction + structure) → ranked
Stage 7: Difficulty-aware trace selection (shortest for easy, best for hard) → ~1/problem
Stage 8: Diversity selection (balanced topics, min coverage per domain) → 3K-4K final
```

### 5.3 Track B: GenSelect

Download the 566K GenSelect samples from nvidia/OpenMathReasoning, filter to olympiad difficulty, select 2K-5K with multiple candidate solutions of varying quality. At inference: generate N solutions, feed all to the model with GenSelect prompt, model selects the best answer.

### 5.4 Track C: RL Problem Set

Filter SynthLabsAI/Big-Math-RL-Verified (250K) to competition-level difficulty with verifiable integer answers. Goldilocks zone: pass_rate 0.10-0.70. Train via Tinker GRPO with binary rewards + cosine length bonus.

### 5.5 Training Strategy

| Stage | Method | Data | Expected Impact |
|-------|--------|------|-----------------|
| 1 | QLoRA SFT (Unsloth) | 3K-4K TIR (Track A) | Better per-attempt accuracy |
| 2 | QLoRA SFT (Unsloth) | 2K-5K GenSelect (Track B) | Fixes selection for N>8 |
| 3 | DPO | 2K-5K length-preference pairs | 10-20% shorter outputs |
| 4 | GRPO RL (Tinker) | 5K-10K RL problems (Track C) | Further reasoning improvement |

**Advanced technique: importance-weighted SFT (iw-SFT)** can be applied during Stage 1 for +10% AIME improvement at zero extra data cost (arxiv:2507.12856). Maintains a frozen reference model, computes per-token log-probability differences, and applies smoothed importance weights to the loss.

---

## 6. 2-Day Sprint Plan (Feb 7-9)

**The MathCorpus Prize deadline is February 9. This is 2 days away.** The sprint focuses on what is achievable in this timeframe: curating and publishing a novel, high-quality dataset.

### Day 1 (Feb 7): Data Curation

| Time | Action | Output |
|------|--------|--------|
| 0-2h | Profile AIMO3 Hard + AIMO3 TIR datasets | Statistics, distributions |
| 2-4h | Run hard filters + outcome verification (math_verify) | ~50K verified correct traces |
| 4-6h | Run difficulty calibration + quality scoring | ~15K ranked candidates |
| 6-8h | Apply diversity selection + final curation | 3K-4K curated dataset |

### Day 2 (Feb 8-9): Verification, Packaging, Submission

| Time | Action | Output |
|------|--------|--------|
| 0-3h | Run TIR integrity checks + decontamination | Cleaned dataset |
| 3-5h | Generate dataset documentation (README, data card) | Publication-ready docs |
| 5-7h | Upload to Kaggle Datasets + HuggingFace | Public dataset |
| 7-8h | Write MathCorpus Prize submission description | Competition submission |

### What Gets Cut (Not Feasible in 2 Days)

- Full PRM scoring with ThinkPRM-14B (requires H100 GPU time)
- Cross-model agreement checking (requires running Qwen3-8B on all solutions)
- GenSelect data curation from OpenMathReasoning (50GB download)
- DPO pair construction
- Any fine-tuning or training

### What IS Feasible in 2 Days

- 8-stage curation pipeline using AIMO3 Hard metadata (pass_rate already computed)
- Outcome verification using math_verify (fast, CPU-only)
- Hard filtering + quality scoring (CPU-only, scriptable)
- Difficulty-aware trace selection (novel, uses existing pass_rate data)
- Topic classification using keyword heuristics
- Dataset packaging with comprehensive metadata
- Publication on Kaggle + HuggingFace

### Minimum Viable Dataset for MathCorpus

If time is extremely tight, the minimum viable submission is:

1. Take AIMO3 Hard dataset (7,293 problems, 70K traces, already has pass_rate)
2. Filter to correct answers + clean TIR traces (hard filters only)
3. Apply difficulty-aware trace selection (1 trace per problem)
4. Add topic labels + quality scores as metadata
5. Package with documentation explaining the curation methodology
6. Publish as "AIMO3-Curated: Quality-Verified, Difficulty-Stratified TIR Traces for gpt-oss-120b"

This can be done in **4-6 hours** and provides genuine novelty through the curation methodology and metadata enrichment.

---

## 7. Budget and Resources

### 7.1 Available Resources

| Resource | Amount | Best Use |
|----------|--------|----------|
| Google Gemini API | ~$93 credits | Topic classification, quality scoring, verification |
| Kaggle H100 | 9h sessions (free) | Trace generation, fine-tuning, PRM inference |
| Tinker API | $150 free credits | GRPO RL training for gpt-oss-120b |
| Google Colab T4 | 15-30 GPU hrs/week | Embedding dedup, small model inference |
| HuggingFace ZeroGPU | H200, free | Quick inference tests |
| Kaggle T4 | 30 hrs/week | Data processing |

### 7.2 Google API Budget Allocation ($93)

| Task | Model | Cost | Volume |
|------|-------|------|--------|
| Topic classification of 141K problems | Gemini 2.0 Flash (Batch) | ~$3.50 | All AIMO3 TIR problems |
| Quality scoring of 70K traces | Gemini 2.0 Flash (Batch) | ~$17.50 | AIMO3 Hard traces |
| Answer verification of 20K candidates | Gemini 2.0 Flash | ~$3 | Cross-check answers |
| Difficulty estimation (4 attempts/problem) | Gemini 2.5 Flash | ~$30 | ~2K problems |
| Reserve for iteration | -- | ~$39 | Buffer |

### 7.3 Kaggle H100 Time Budget (per 9h session)

| Task | Time | Output |
|------|------|--------|
| TIR trace generation (1K problems x 8 attempts) | 3-4h | 8K traces with logprobs |
| RSR scoring of 10K trajectories | 30 min | Trajectory selection scores |
| Unsloth QLoRA SFT (2K examples, 1 epoch) | 2-3h | LoRA adapter |
| Validation inference (50 problems) | 1-2h | Pre-submission accuracy check |

### 7.4 Total Estimated Cost

| Item | Cost |
|------|------|
| Dataset curation (Google API + free compute) | $45-63 |
| QLoRA SFT (Kaggle H100 or Vast.ai) | $0-16 |
| GRPO RL (Tinker, $150 free credits) | $0-50 out of pocket |
| Dataset downloads | $0 |
| PRM inference | $0 |
| **Total** | **$45-129** |
| **Effective (with free credits)** | **$0-50** |

### 7.5 Fields Model Initiative (Aspirational)

128 H100 GPUs available for AIMO3 participants through the Fields Model Initiative partnership. This would enable full GRPO RL runs on gpt-oss-120b. Application through the Kaggle competition page. **Priority: HIGHEST** -- worth applying even if the 2-day sprint doesn't depend on it.

---

## 8. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **MathCorpus deadline missed (Feb 9)** | Lose EP prize | Medium | Focus on minimum viable dataset; 4-6h fast path |
| **QLoRA on MoE degrades base quality** | Wasted training | Medium | Test on small subset first; compare to 40/50 baseline |
| **GenSelect format mismatch with Harmony** | Selection fails | Medium | Adapt format during curation; test before training |
| **PRM mismatch (Qwen PRM on gpt-oss outputs)** | Bad quality filtering | Medium | Use generative PRMs + soft weights, not hard filters |
| **Contamination with AIMO3 test problems** | Overfitting / DQ | Low | 9-gram + embedding decontamination |
| **More attempts still hurt selection** | No improvement | Medium | GenSelect is primary fix; fall back to 8 attempts |
| **Dataset not novel enough for MathCorpus** | No prize | Medium | Emphasize curation methodology + selection metadata |
| **Pipeline implementation time exceeds 2 days** | Missed deadline | Medium | Pre-built scripts exist; focus on simplest pipeline |
| **Training time exceeds H100 budget** | Incomplete training | Low | 4K SFT takes ~2-4h, well within limits |
| **DPO causes quality degradation** | Worse model | Medium | min_length=500, verify accuracy after DPO |

### Critical Risk: Time

The 2-day timeline for MathCorpus is the biggest risk. Mitigations:
1. **Parallelize**: Data profiling + hard filtering can run while documentation is written
2. **Simplify**: Use keyword-based topic classification instead of LLM-based
3. **Leverage existing metadata**: AIMO3 Hard already has pass_rate -- no need to compute
4. **Minimum viable path**: 4-6 hours for a basic but novel curated dataset

---

## 9. References

### Competition and Rules
- [AIMO3 Kaggle Competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- [Third Progress Prize Announcement](https://aimoprize.com/updates/2025-11-19-third-progress-prize-launched)
- [Early Sharing Prize Awarded (AIMO2)](https://aimoprize.com/updates/2024-12-12-sharing-prize-awarded)

### Winning Solutions
- [AIMO2 1st Place (NVIDIA) Paper](https://arxiv.org/abs/2504.16891) -- OpenMathReasoning, GenSelect
- [AIMO1 Winner (Numina) Blog](https://huggingface.co/blog/winning-aimo-progress-prize) -- NuminaMath
- [AIMO2 2nd Place (Imagination)](https://github.com/imagination-research/aimo2) -- SFT + DPO

### Dataset Curation and Quality
- [LIMO: Less Is More (arxiv:2502.03387)](https://arxiv.org/abs/2502.03387) -- 817 curated >> 100K random
- [s1: Simple Scaling (arxiv:2501.19393)](https://arxiv.org/abs/2501.19393) -- 1K examples beat o1-preview
- [ASTER: Agentic Scaling with TIR (arxiv:2602.01204)](https://arxiv.org/html/2602.01204) -- 4K TIR = 90% AIME 2025
- [DART-Math (arxiv:2407.13690)](https://arxiv.org/abs/2407.13690) -- Difficulty-aware rejection tuning
- [Front-Loading Reasoning (arxiv:2510.03264)](https://arxiv.org/abs/2510.03264) -- SFT quality > quantity
- [RSR: Rank-Surprisal Ratio (arxiv:2601.14249)](https://arxiv.org/abs/2601.14249) -- 0.86 correlation trajectory metric
- [iw-SFT (arxiv:2507.12856)](https://arxiv.org/abs/2507.12856) -- Importance-weighted SFT = +10% AIME

### Token Efficiency
- [DA-CoTD: Difficulty-Aware Distillation (arxiv:2509.05226)](https://arxiv.org/abs/2509.05226) -- 30% reduction
- [TokenSkip (arxiv:2502.12067)](https://arxiv.org/abs/2502.12067) -- 40% reduction, -0.4% accuracy
- [L1: Length-Controlled RL (arxiv:2503.04697)](https://arxiv.org/abs/2503.04697) -- Controllable length
- [e1: Adaptive Effort (arxiv:2510.27042)](https://arxiv.org/abs/2510.27042) -- 3x reduction
- [MCoT: Markov Chain of Thought (arxiv:2410.17635)](https://arxiv.org/abs/2410.17635) -- 1.9x speed

### Verification and Process Supervision
- [ThinkPRM (arxiv:2504.16828)](https://arxiv.org/abs/2504.16828) -- Generative PRM, 1% of PRM800K labels
- [Math-Shepherd (arxiv:2312.08935)](https://arxiv.org/abs/2312.08935) -- Automated step-level labels
- [OmegaPRM (arxiv:2406.06592)](https://arxiv.org/abs/2406.06592) -- MCTS process supervision
- [Dyve (arxiv:2502.11157)](https://arxiv.org/abs/2502.11157) -- Fast/slow dynamic verification
- [AceMath (arxiv:2412.15084)](https://arxiv.org/abs/2412.15084) -- Cross-model verification
- [math_verify (PyPI)](https://libraries.io/pypi/math-verify) -- Answer verification library
- [MATH-VF (arxiv:2505.20869)](https://arxiv.org/abs/2505.20869) -- Formal step verification
- [FANS (arxiv:2503.03238)](https://arxiv.org/abs/2503.03238) -- Lean4 formal verification

### Training Methods
- [rStar2-Agent (arxiv:2508.20722)](https://arxiv.org/abs/2508.20722) -- Best RL recipe, 80.6% AIME24
- [DemyAgent / Open-AgentRL (arxiv:2510.11701)](https://arxiv.org/abs/2510.11701) -- Real > synthetic trajectories
- [LoRA Without Regret (Thinking Machines)](https://thinkingmachines.ai/blog/lora/) -- LoRA = full FT for RL
- [ReTool (arxiv:2504.11536)](https://arxiv.org/abs/2504.11536) -- Tool-integrated RL
- [JustRL (ICLR 2026)](https://iclr-blogposts.github.io/2026/blog/2026/justrl/) -- Simple GRPO wins
- [Light-R1 (arxiv:2503.10460)](https://arxiv.org/abs/2503.10460) -- Curriculum SFT + DPO + RL
- [AdaRFT (arxiv:2504.05520)](https://arxiv.org/abs/2504.05520) -- Dynamic difficulty curriculum
- [GRPO-LEAD (arxiv:2504.09696)](https://arxiv.org/abs/2504.09696) -- Difficulty-aware advantage reweighting

### Datasets
- [OpenMathReasoning (NVIDIA)](https://huggingface.co/datasets/nvidia/OpenMathReasoning) -- 306K problems, 5.5M solutions
- [AIMO3 TIR (Kaggle)](https://kaggle.com/datasets/jeannkouagou/aimo3-tool-integrated-reasoning) -- 141K traces
- [AIMO3 Hard (Kaggle)](https://kaggle.com/datasets/wenliangtlh/aimo3-high-difficulty-tool-calling-dataset) -- 7.3K problems
- [Big-Math-RL-Verified](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified) -- 250K RL problems
- [NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) -- 70K TIR traces

### Tools and Frameworks
- [Unsloth (gpt-oss fine-tuning)](https://docs.unsloth.ai/models/gpt-oss-how-to-run-and-fine-tune)
- [TRL (HuggingFace)](https://github.com/huggingface/trl) -- SFT, GRPO, DPO
- [Tinker API](https://tinker-docs.thinkingmachines.ai/) -- API-based fine-tuning
- [vLLM (inference)](https://docs.vllm.ai/)
- [OpenAI Harmony SDK](https://github.com/openai/harmony)

---

## Appendix A: Research File Index

| File | Author | Topic |
|------|--------|-------|
| `dataset_generation/research/competition_rules.md` | researcher-1 | MathCorpus Prize rules, precedent, winning criteria |
| `dataset_generation/research/token_efficiency.md` | researcher-2 | SFT+DPO, difficulty-aware curation, length RL |
| `dataset_generation/research/verification_methods.md` | researcher-3 | PRMs, cross-model, formal verification |
| `dataset_generation/research/resources.md` | resource-finder | Tools, datasets, compute, notebooks |
| `dataset_generation/PLAN.md` | principal-investigator | 3-track dataset plan, training config |
| `data/research/quality_selection_research.md` | -- | Evidence-based quality pipeline |
| `data/research/sample_quality_research.md` | -- | Scoring functions, RSR, iw-SFT, winner pipelines |
| `data/research/ml_research_findings.md` | -- | SOTA methods: rStar2, DemyAgent, ASTER, GenSelect |
| `data/research/dataset.md` | -- | 29+ datasets catalog |
| `data/research/grants_and_free_compute.md` | -- | Fields Model Initiative, Tinker grants, free GPUs |

## Appendix B: Key Metrics Summary

| Metric | Our Current | AIMO2 Winner | Target |
|--------|------------|--------------|--------|
| Score | 40/50 | 34/50 | 45-50/50 |
| Attempts per problem | 8 | 48-64 | 16-32 |
| Selection method | Entropy heuristic | GenSelect (learned) | GenSelect |
| Fine-tuning | None | Custom SFT + RL | QLoRA SFT + DPO + RL |
| Training data | 0 | 306K problems / 5.5M solutions | 3K-10K curated |
| Token efficiency | ~10K/solution | Unknown | ~7K/solution (30% reduction) |
| Verification | None | Cross-model | 5-stage pipeline |

## Appendix C: Anti-Patterns (What NOT to Do)

1. Do NOT use all correct traces -- quality >> quantity
2. Do NOT bias toward easy problems -- DART showed 90% of standard rejection tuning is easy
3. Do NOT keep "lucky correct" solutions -- use cross-model + PRM
4. Do NOT ignore topic balance -- geometry and probability are underrepresented
5. Do NOT train on brute-force solutions -- prefer analytical with code verification
6. Do NOT skip decontamination -- must exclude AIME 2023-2025 and AIMO3 problems
7. Do NOT use excessively long solutions -- >80K chars = stuck/looping
8. Do NOT add complexity to the selection heuristic -- simple > complex (40/50 > 32/50). GenSelect replaces the heuristic entirely
9. Do NOT use latent reasoning (Coconut/CODI) -- incompatible with vLLM
10. Do NOT aggressively compress hard problems -- they need full reasoning depth
