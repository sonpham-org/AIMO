# AIMO3 Master Plan: February 7 - April 10, 2026

> **Goal**: Score 45+/50 on AIMO3 (up from 40/50)
> **Deadline**: April 15 competition close; plan through April 10 for buffer
> **Budget**: $50-500 out of pocket + $150 Tinker free credits + free compute applications
> **Hardware**: 1x H100 80GB (Kaggle, 9h limit), AMD Radeon 8060S (local dev, no CUDA)

---

## The Problem

We score 40/50 with 8 attempts and entropy-gated consensus. But:
- 12 attempts: 37/50 (-3)
- 16 attempts: 29/50 (-11)

**Selection degrades with more samples.** The bottleneck is not generation quality -- it is answer selection. AIMO2 winners used 48-64 samples per problem. We have 8h45m of unused compute. Fixing selection unlocks everything.

---

## Strategy Summary

| Phase | Weeks | Focus | Cost | Expected Gain |
|-------|-------|-------|------|---------------|
| **1: Free inference fixes** | Feb 7-21 | Self-Certainty, adversarial verification, cascading difficulty | $0 | +2-5 pts |
| **2: GenSelect + ASTER SFT** | Feb 22 - Mar 7 | Train selection + generation LoRAs | $0-20 | +5-10 pts |
| **3: GRPO RL** | Mar 8-22 | RL fine-tune via Tinker | $50-150 (free credits) | +2-5 pts |
| **4: Integration + submission** | Mar 23 - Apr 10 | Merge, test, iterate, final submissions | $0-50 | Consolidation |

**Principle: Exhaust $0 strategies before spending. Each phase has a clear go/no-go decision.**

---

## Phase 1: Free Inference-Time Improvements (Feb 7-21)

### Week 1 (Feb 7-14): Fix Selection for N>8

**1a. Self-Certainty (HIGHEST PRIORITY, $0)**
- Replace raw entropy with KL-divergence from uniform distribution as confidence measure
- Self-Certainty is the ONLY metric proven to scale consistently from N=8 to N=64 (arXiv 2502.18581)
- We already collect top-5 logprobs -- this is a formula change, not an architecture change
- Implementation: GitHub at backprop07/Self-Certainty
- Combine with Borda voting across attempts
- **Action**: Modify `replay_selection.py` to add Self-Certainty strategies, test on existing Qwen3-8B traces
- **Go/no-go**: If Self-Certainty + 16 attempts > 40/50 on offline traces, proceed to submission

**1b. Trace Fingerprinting ($0)**
- Current selection only looks at final answer + entropy
- New: score solutions by intermediate code execution values, not just final answer
- If 12/16 attempts produce the same intermediate result at step 3, those are more likely correct
- **Action**: Extract intermediate values from TIR traces, use as additional consensus signal
- **Implementation**: Parse Jupyter kernel outputs from each TIR turn; cluster intermediate results

**1c. Adversarial Self-Debate ($0)**
- After generating N solutions, prompt the model with: "Here are N candidate answers: [list]. Identify which is correct and why."
- The model uses its own reasoning to evaluate candidates (like GenSelect but without training)
- Uses 1 additional inference call per problem (~2 min extra, well within budget)
- **Action**: Add a debate round after generation in the submission notebook
- **Risk**: Mixed prompts hurt before (entropy-plus: 33/50). Keep the debate prompt simple and structured.

### Week 2 (Feb 15-21): Cascading Difficulty + Trace Generation

**1d. Cascading Difficulty ($0)**
- Easy problems: 4 attempts, tight early stop (save time)
- Medium problems: 8-12 attempts
- Hard problems: 16-32 attempts with enhanced selection
- Difficulty estimated from first-pass entropy + consensus of first 4 attempts
- **Expected benefit**: Better allocation of the 9h budget; more attempts on hard problems where they matter
- **Action**: Build `AdaptiveRouter` class, integrate into solver

**1e. Generate gpt-oss-120b Traces on H100 ($0)**
- Use Kaggle H100 (free) to generate traces on AIME problems
- Config: 16 attempts per problem, full logprobs, 524 AIME problems
- Output: ~8,400 traced solutions with entropy, logprobs, intermediate values
- **Purpose**: These traces feed the 138-strategy sweep AND Phase 2 data curation
- **Action**: Push trace-generation notebook to Kaggle, run overnight
- **Timeline**: Submit notebook by Feb 15, results by Feb 17

**1f. Run Full Strategy Sweep ($0)**
- Once gpt-oss-120b traces are available: run `replay_selection.py` with all 138 strategies
- Add Self-Certainty, Borda voting, trace fingerprinting as new strategies
- Identify the optimal selection strategy for N=8, N=16, N=32
- **Deliverable**: Ranked strategy table with accuracy by N, ready for submission

### Phase 1 Submissions

| Submission | Date | Config | Expected |
|------------|------|--------|----------|
| P1-A: Self-Certainty 8-attempt | Feb 14 | 8 attempts + Self-Certainty selection | 40-43/50 |
| P1-B: Self-Certainty 16-attempt | Feb 17 | 16 attempts + Self-Certainty selection | 40-44/50 |
| P1-C: Cascading difficulty | Feb 21 | Adaptive 4-32 attempts + best selection | 41-45/50 |

### Phase 1 Go/No-Go (Feb 21)
- **If 16 attempts + Self-Certainty > 42**: Selection fix is working, proceed to Phase 2 with confidence
- **If 16 attempts + Self-Certainty <= 40**: Selection needs a learned model (GenSelect), accelerate Phase 2
- **Either way**: Phase 2 proceeds, but results inform priority of GenSelect vs ASTER

---

## Phase 2: GenSelect + ASTER SFT (Feb 22 - Mar 7)

### Week 3 (Feb 22-28): Data Curation + GenSelect Training

**2a. Download and Curate GenSelect Training Data ($0)**
- Source: `nvidia/OpenMathReasoning` GenSelect split (566K samples)
- Filter pipeline:
  1. Keep only olympiad difficulty (pass_rate_72b_tir < 0.5)
  2. Keep problems with 4+ candidate solutions of varying quality
  3. Decontaminate: 9-gram match vs AIME 2023-2025, AIMO3 test set
  4. Deduplicate: 8-gram Jaccard > 0.5 = remove
  5. Quality score using the combined scorer (Section 8 of sample_quality_research.md)
  6. Select top 3K-5K examples balanced across topics
- **Deliverable**: `data/genselect_curated.jsonl` (3K-5K examples)

**2b. Curate ASTER Cold-Start SFT Data ($0)**
- Source: `wenliangtlh/aimo3-high-difficulty-tool-calling-dataset` (7,293 problems, ~70K traces)
- Filter pipeline:
  1. Hard filters: correct answer, length 2K-80K, 1-15 tool calls, no repetition, has \boxed{}
  2. Interaction density: keep only trajectories with >= 9 tool-interaction turns (ASTER's magic number)
  3. Difficulty: pass_rate 0.05-0.25 (hardest solvable problems)
  4. Quality score + LIMO-style top-1 per problem
  5. Topic diversity: balanced sampling across algebra, NT, combo, geometry
  6. Decontamination check
- **Expected yield**: ~2K-4K trajectories from 70K
- **Deliverable**: `data/aster_coldstart_curated.jsonl` (2K-4K examples)

**2c. Train GenSelect LoRA ($0-10)**
- Framework: Unsloth QLoRA on Kaggle H100 (free) or rented H100 ($2/hr)
- Config:
  ```
  rank=64, lora_alpha=128, targets=q/k/v/o/gate/up/down_proj
  lr=2e-4, epochs=3, batch=4, grad_accum=4, max_seq=32768
  optim=adamw_8bit, bf16=True, gradient_checkpointing="unsloth"
  ```
- Training time: ~2-4 hours on H100
- VRAM: ~75-78GB (fits H100 80GB)
- Output: `genselect_lora/` adapter (~200-500MB)
- **Upload**: Kaggle private dataset

### Week 4 (Mar 1-7): ASTER SFT + Multi-LoRA Integration

**2d. Train ASTER Cold-Start SFT LoRA ($0-10)**
- Same Unsloth QLoRA setup as 2c
- Data: 2K-4K interaction-dense TIR trajectories from 2b
- If using iw-SFT (importance-weighted SFT): additional +10% AIME expected
  - Requires maintaining reference model + per-token importance weights
  - Implementation: modify Unsloth training loop with weight = q(traj) / pi_ref(traj)
- Training time: ~2-4 hours on H100
- Output: `aster_sft_lora/` adapter (~200-500MB)

**2e. Multi-LoRA Inference Setup ($0)**
- vLLM supports serving multiple LoRA adapters on the same base model natively
- Config:
  ```bash
  vllm serve gpt-oss-120b \
    --enable-lora --max-loras 3 --max-lora-rank 64 --max-cpu-loras 8 \
    --lora-modules sft-lora=/path/sft genselect-lora=/path/genselect
  ```
- Pipeline per problem:
  1. Generate 16-32 solutions using `sft-lora` (better generation)
  2. Score each solution using `genselect-lora` (learned selection)
  3. Entropy-gated consensus on GenSelect scores
- VRAM overhead: ~200-400MB per adapter (negligible with 15-18GB free)
- **This mirrors NVIDIA's AIMO2 winning architecture**

**2f. Merge vs Multi-LoRA Decision**
- If SFT + GenSelect serve different roles (generation vs selection): **keep separate, use multi-LoRA**
- If we also produce a GRPO adapter for the same task: merge SFT + GRPO using **TIES or DARE**
- Only merge adapters that serve the same purpose
- Validate merged adapter on held-out problems before deploying

### Phase 2 Submissions

| Submission | Date | Config | Expected |
|------------|------|--------|----------|
| P2-A: GenSelect only | Mar 3 | 8 attempts + GenSelect selection | 42-46/50 |
| P2-B: SFT + GenSelect | Mar 7 | 16 attempts + SFT generation + GenSelect selection | 44-48/50 |

### Phase 2 Go/No-Go (Mar 7)
- **If P2-B > 44**: Strong trajectory, proceed to Phase 3 for marginal gains
- **If P2-B = 41-44**: Working but not enough, Phase 3 GRPO is critical
- **If P2-B <= 40**: Something is wrong. Debug LoRA quality, check for format mismatch, consider different approach

---

## Phase 3: GRPO RL via Tinker (Mar 8-22)

### Week 5 (Mar 8-14): Tinker Setup + Smoke Tests

**3a. Sign Up for Tinker + Apply for Credits ($0)**
- Sign up: https://auth.thinkingmachines.ai/sign-up (free, GA)
- Get $150 free new-user credits
- Apply for Research Grant ($5K+): https://thinkingmachines.ai/blog/tinker-research-and-teaching-grants/
  - Use Phase 1-2 results as evidence of serious research
  - Mention AIMO3 competition, open-source fine-tuning for math
  - Expected response: ~1 week

**3b. Apply for Fields Model Initiative ($0)**
- Up to 128 H100 GPUs free for AIMO3 participants
- Apply through Kaggle competition page
- Partnership with National Institute of Informatics (Tokyo) + Benchmarks+Baselines (Vienna)
- Also provides Tinker credits via AIMO3 partnership
- **Do this immediately (Feb 7), don't wait for Phase 3**

**3c. Smoke Test: Llama-1B ($1)**
- Run Tinker math RL recipe on Llama-3.2-1B
- Purpose: validate pipeline end-to-end (data format, reward function, LoRA download)
- Expected: minutes to run, confirm everything works

**3d. Medium Test: Qwen3-8B ($20-50)**
- GRPO on curated math problems (5K problems from Big-Math-RL-Verified)
- Replicate the 76.7% MATH benchmark from Tinker cookbook
- Validate: LoRA download, format conversion, vLLM compatibility
- **Critical test**: Does Tinker's LoRA format work with vLLM's `--enable-lora`?

### Week 6 (Mar 15-22): gpt-oss-120b GRPO

**3e. Curate RL Training Data ($0)**
- Source: `SynthLabsAI/Big-Math-RL-Verified` (250K verified problems)
- Filter for competition-level difficulty: pass_rate 0.1-0.7 (Goldilocks zone for gradient signal)
- Use AdaRFT dynamic difficulty: start at ~50% success, auto-adjust harder as model improves
- Target: 5K-10K problems in RL environment
- Reward: correct answer = +1, wrong = 0 (binary, verified by SymPy)

**3f. GRPO Training on gpt-oss-120b ($50-200)**
- Use Tinker API (or $150 free credits)
- MoE pricing: proportional to active params (5.1B), so ~$0.10-0.40/M tokens
- LoRA config per "LoRA Without Regret":
  - Apply to ALL layers including MLPs/MoE experts
  - Separate LoRA per expert, rank = total_rank / num_active_experts
  - LR = 1e-4 to 5e-4 (10x full FT)
- Estimated: 200M-1B total tokens, 100-200 training steps
- Cost estimate: $50-200 (covered by $150 free credits)
- Output: `grpo_lora/` adapter

**3g. Adapter Strategy**
- Merge GRPO + SFT adapters using TIES-Merging (both serve generation):
  ```python
  model.add_weighted_adapter(
      adapters=["sft", "grpo"], weights=[1.0, 0.8],
      adapter_name="merged_gen", combination_type="ties", density=0.5
  )
  ```
- Keep GenSelect adapter separate (different role)
- Final inference: `merged_gen` for generation, `genselect` for selection via multi-LoRA

### Phase 3 Submissions

| Submission | Date | Config | Expected |
|------------|------|--------|----------|
| P3-A: GRPO generation + GenSelect | Mar 19 | Merged SFT+GRPO gen, GenSelect select, 16 attempts | 45-49/50 |
| P3-B: Full pipeline | Mar 22 | Cascading difficulty + merged gen + GenSelect + best selection | 46-50/50 |

### Phase 3 Go/No-Go (Mar 22)
- **If P3-B > 46**: Excellent. Move to consolidation and polishing.
- **If P3-B = 42-46**: Good progress. Focus Phase 4 on edge cases and reliability.
- **If P3-B <= 42**: Fine-tuning not working as expected. Fall back to best Phase 1-2 result and polish that.

---

## Phase 4: Integration, Testing & Final Submissions (Mar 23 - Apr 10)

### Week 7-8 (Mar 23 - Apr 5): Iteration and Hardening

**4a. Error Analysis ($0)**
- Download full competition traces (from best submission)
- Identify the 5-10 problems we consistently get wrong
- Categorize: geometry? combinatorics? tricky number theory?
- Targeted intervention: more attempts, different prompts, or special handling

**4b. A/B Testing at Scale ($0)**
- Submit variants systematically:
  - Vary attempts: 8, 16, 24, 32
  - Vary selection: entropy-gated, Self-Certainty, GenSelect, combined
  - Vary adapters: base, SFT, SFT+GRPO, SFT+GRPO+GenSelect
- Maximum: 2 submissions per day on Kaggle
- Track results in a structured table

**4c. Reliability Engineering ($0)**
- Run each submission config 3x to measure variance
- Identify configs that are consistently good (not just lucky once)
- Add fallback logic: if GenSelect fails, fall back to entropy-gated consensus
- Harden timeout handling, error recovery, notebook restart logic

**4d. Optional: Small Verifier Model ($0-50)**
- If GenSelect underperforms: try off-the-shelf `Qwen2.5-Math-7B-PRM800K` as verifier
- Score each candidate solution step-by-step (process reward model)
- Can run alongside gpt-oss-120b via vLLM Sleep Mode (~5s switch time)
- Or quantize to 4-bit (~2GB), run in remaining VRAM
- Cost: $0 (download existing model), or $20-50 to fine-tune custom ORM via Tinker

### Week 9 (Apr 6-10): Final Submissions

**4e. Final Submission Strategy**
- Select the 2 best-performing configs from A/B testing
- Run each 3x on competition rerun to confirm consistency
- Submit final version by April 10 (5-day buffer before deadline)
- Write submission documentation (for write-up prizes: $5K for top 3 write-ups)

**4f. Competition Extras**
- Consider entering for:
  - **Longest Leader Prize**: if we can get on top and stay there
  - **Write-up Prize**: $5K for top 3 write-ups, requires clear documentation
  - **Hardest Problem Prize**: which problems we uniquely solve

---

## Budget Allocation

| Item | Cost | Phase | Notes |
|------|------|-------|-------|
| Kaggle H100 (trace gen) | $0 | 1 | Free Kaggle compute |
| Kaggle H100 (training) | $0 | 2 | 9h limit per run, may need multiple |
| H100 rental (if Kaggle insufficient) | $10-20 | 2 | Vast.ai $1.87/hr x 5-10hr |
| Tinker smoke test | $1 | 3 | Llama-1B |
| Tinker Qwen3-8B test | $20-50 | 3 | Covered by $150 free credits |
| Tinker gpt-oss-120b GRPO | $50-200 | 3 | Covered by $150 free credits + small top-up |
| Optional verifier training | $20-50 | 4 | Only if GenSelect fails |
| **Total out-of-pocket** | **$0-100** | | Most covered by free credits |
| **Total with credits** | **$100-320** | | $150 Tinker + possible grant |

### Free Compute Applications (DO IMMEDIATELY)
1. **Fields Model Initiative**: up to 128 H100s free. Apply through Kaggle competition page.
2. **Tinker Research Grant**: $5K+ credits. Apply at https://thinkingmachines.ai/blog/tinker-research-and-teaching-grants/
3. **Tinker AIMO3 Partnership Credits**: Available to AIMO3 participants via competition page.

---

## Risk Assessment

### Risk 1: GenSelect Format Mismatch (Medium)
- **Problem**: GenSelect training data format (from OpenMathReasoning) may differ from our Harmony protocol inference format
- **Mitigation**: Inspect GenSelect data format before training. Adapt our inference prompt to match, or convert training data to match our format.
- **Fallback**: If GenSelect doesn't work, Self-Certainty + better entropy calibration as selection strategy

### Risk 2: QLoRA on MoE Quality Degradation (Medium)
- **Problem**: QLoRA on gpt-oss-120b (MoE, MXFP4) may degrade generation quality
- **Mitigation**: Test adapter on held-out problems before deploying. Use rank-64 to maximize capacity.
- **Fallback**: Lower LoRA rank (16-32), or skip SFT and rely on GenSelect + base model

### Risk 3: vLLM Multi-LoRA Compatibility (Low-Medium)
- **Problem**: `--enable-lora` may conflict with MXFP4 quantization or other vLLM optimization flags
- **Mitigation**: Test multi-LoRA on Kaggle H100 before competition run. Check vLLM docs + issues.
- **Fallback**: Merge all adapters into one (TIES/DARE), serve as single model

### Risk 4: Tinker gpt-oss-120b Pricing (Low)
- **Problem**: gpt-oss-120b pricing not yet confirmed; may be higher than estimated
- **Mitigation**: Smoke test with cheaper models first. Apply for research grant early.
- **Fallback**: Use Unsloth QLoRA for all training (skip Tinker RL), rent H100 at $2/hr

### Risk 5: Competition Variance (Always Present)
- **Problem**: Same config can score differently across competition reruns (different 50 problems from 110 pool)
- **Mitigation**: Submit best configs multiple times, track variance. Aim for reliable 44+ rather than volatile 48.
- **Fallback**: Stick with proven 40/50 config if new approaches are less reliable

### Risk 6: Time Overrun
- **Problem**: Fine-tuning debugging takes longer than expected
- **Mitigation**: Hard go/no-go dates at end of each phase. Phase 1 results are usable standalone. Never abandon a working 40/50 for an unproven approach.
- **Fallback**: Submit best Phase 1 result + best Phase 2 result. Only deploy Phase 3 if tested.

---

## Week-by-Week Calendar

### Week 1: Feb 7-14 -- Selection Fix
- [ ] Implement Self-Certainty in `replay_selection.py`
- [ ] Test Self-Certainty on Qwen3-8B traces (offline)
- [ ] Build trace fingerprinting (intermediate value consensus)
- [ ] Submit P1-A: Self-Certainty 8-attempt to Kaggle
- [ ] Apply for Fields Model Initiative (FREE, do now)
- [ ] Apply for Tinker Research Grant (FREE, do now)
- [ ] Sign up for Tinker, get $150 credits

### Week 2: Feb 15-21 -- Cascading + Traces
- [ ] Push trace generation notebook to Kaggle (gpt-oss-120b, 16 attempts, AIME)
- [ ] Implement cascading difficulty router
- [ ] Run 138+ strategy sweep on gpt-oss-120b traces
- [ ] Submit P1-B: Self-Certainty 16-attempt
- [ ] Submit P1-C: Cascading difficulty
- [ ] **Phase 1 Go/No-Go decision**

### Week 3: Feb 22-28 -- Data Curation + GenSelect
- [ ] Download OpenMathReasoning GenSelect split (566K)
- [ ] Download AIMO3-Hard dataset (70K traces)
- [ ] Run 6-stage curation pipeline on both datasets
- [ ] Train GenSelect LoRA on Kaggle H100 (or rented H100)
- [ ] Validate GenSelect adapter on held-out problems

### Week 4: Mar 1-7 -- ASTER SFT + Integration
- [ ] Train ASTER cold-start SFT LoRA (2K-4K interaction-dense traces)
- [ ] Set up multi-LoRA serving in submission notebook
- [ ] Test multi-LoRA: SFT generation + GenSelect selection
- [ ] Submit P2-A: GenSelect only
- [ ] Submit P2-B: SFT + GenSelect
- [ ] **Phase 2 Go/No-Go decision**

### Week 5: Mar 8-14 -- Tinker Setup + Smoke Tests
- [ ] Tinker smoke test: Llama-1B ($1)
- [ ] Tinker medium test: Qwen3-8B GRPO ($20-50)
- [ ] Validate LoRA download + vLLM compatibility
- [ ] Curate RL training data (5K-10K problems)
- [ ] Check on Fields Initiative / Tinker Grant applications

### Week 6: Mar 15-22 -- GRPO RL
- [ ] GRPO training on gpt-oss-120b via Tinker ($50-200)
- [ ] Merge GRPO + SFT adapters (TIES)
- [ ] Test full pipeline: merged gen + GenSelect select
- [ ] Submit P3-A: GRPO gen + GenSelect
- [ ] Submit P3-B: Full pipeline
- [ ] **Phase 3 Go/No-Go decision**

### Week 7-8: Mar 23 - Apr 5 -- Iteration
- [ ] Error analysis on wrong problems
- [ ] A/B test: attempts, selection, adapters
- [ ] Reliability testing (3x runs per config)
- [ ] Optional: small verifier model
- [ ] Submit variants, track results

### Week 9: Apr 6-10 -- Final
- [ ] Select 2 best configs from A/B testing
- [ ] Final reliability runs (3x each)
- [ ] Submit final versions by April 10
- [ ] Write submission documentation (for write-up prize)

---

## Decision Tree

```
Start (40/50)
  |
  v
Phase 1: Self-Certainty + 16 attempts
  |
  +--> > 42/50: Selection fix works!
  |      |
  |      v
  |    Phase 2: GenSelect + SFT (amplify gains)
  |      |
  |      +--> > 44/50: Strong trajectory
  |      |      -> Phase 3: GRPO for marginal gains
  |      |      -> Target: 46-50/50
  |      |
  |      +--> 41-44/50: Working but not enough
  |      |      -> Phase 3: GRPO is critical
  |      |      -> Target: 45-48/50
  |      |
  |      +--> <= 40/50: Format/quality issue
  |             -> Debug LoRA, try different data
  |             -> Fallback: submit Phase 1 best
  |
  +--> <= 40/50: Self-Certainty not enough
         |
         v
       Phase 2: GenSelect is CRITICAL (learned selection)
         |
         +--> GenSelect > 42: Proceed to Phase 3
         |
         +--> GenSelect <= 40: Try AggLM or PRM verifier
                -> If nothing works: submit 8-attempt 40/50
                -> Investigate if the 40/50 was lucky variance
```

---

## Key Metrics to Track

| Metric | Current | Target Phase 1 | Target Phase 2 | Target Phase 4 |
|--------|---------|----------------|----------------|----------------|
| Score (8 attempts) | 40/50 | 42/50 | 44/50 | 45/50 |
| Score (16 attempts) | 29/50 | 40/50 | 44/50 | 46/50 |
| Score (32 attempts) | N/A | N/A | 45/50 | 47/50 |
| Selection accuracy (trace data) | ~80% | ~85% | ~90% | ~92% |
| Per-attempt solve rate | ~60% | ~60% | ~65% | ~68% |
| Time per 50 problems | ~15min | ~20min | ~30min | ~45min |

---

## Files and Resources

### Code
- `kaggle_submissions/feb3_entropy_gated/` -- Best submission (40/50), base for all new variants
- `scripts/generate_traces.py` -- TIR trace generation with logprobs
- `scripts/replay_selection.py` -- 138+ offline selection strategies
- `scripts/ablation_test.py` -- Ablation configs for answer selection

### Research
- `data/ml_research_findings.md` -- Fine-tuning methods (GenSelect, ASTER, GRPO, verifiers)
- `data/crawler_research.md` -- Datasets, platforms, papers, competition updates
- `data/sample_quality_research.md` -- Sample quality metrics, 6-stage curation pipeline
- `data/model_merging_research.md` -- LoRA merging, multi-LoRA, ensembling strategies

### External
- OpenMathReasoning GenSelect: `nvidia/OpenMathReasoning` (566K)
- AIMO3-Hard: `wenliangtlh/aimo3-high-difficulty-tool-calling-dataset` (70K)
- Big-Math-RL-Verified: `SynthLabsAI/Big-Math-RL-Verified` (250K)
- Self-Certainty: https://github.com/backprop07/Self-Certainty
- Tinker: https://thinkingmachines.ai/tinker/
- Tinker Cookbook: https://github.com/thinking-machines-lab/tinker-cookbook

### Key Papers
- Self-Certainty (arXiv 2502.18581) -- KL-divergence scaling for best-of-N
- GenSelect / AIMO2 Winner (arXiv 2504.16891) -- Learned answer selection
- ASTER (arXiv 2602.01204) -- 4K cold-start SFT, 90% AIME 2025
- LoRA Without Regret (Thinking Machines) -- LoRA rank-1 matches full FT for RL
- AggLM (arXiv 2509.06870) -- Learned aggregation beats majority vote + 72B reward models
- RSR (arXiv 2601.14249) -- 0.86 correlation metric for trajectory quality

---

## Immediate Actions (Today, Feb 7)

1. **Apply for Fields Model Initiative** -- free 128 H100s, do it NOW
2. **Apply for Tinker AIMO3 partnership credits** -- through competition page
3. **Sign up for Tinker** -- get $150 free credits immediately
4. **Clone Self-Certainty repo** -- start implementing in replay_selection.py
5. **Start gpt-oss-120b trace generation notebook** -- push to Kaggle for overnight run
