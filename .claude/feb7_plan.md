# Feb 7 Plan — AIMO3 Next Steps

> Created: 2026-02-07
> Focus: (3) Fine-tuning dataset quality + (4) Fine-tuning strategy
> Also tracking: (1) Strategy exploration + (2) 16-attempt scoring

---

## Current State Summary

| Item | Status | Key Finding |
|------|--------|-------------|
| **feb6 (16 attempts)** | COMPLETE | 16 kernels + early_stop=6 works! Solved 3 test problems in 0.4 min. **16 attempts is feasible** within 9h budget. |
| **feb5 (adaptive)** | COMPLETE | Ran successfully, but test problems are trivial (0x10, 4+x=4, 1-1). Need competition rerun for true score. |
| **Local traces** | 526 files | Qwen3-8B on 524 AIME, 8 samples each. entropy<0.3 + consensus>=4 = **95.2% accuracy** |
| **Fine-tuning research** | Documented | 5 datasets identified, Tinker + Unsloth paths, quality metrics defined |
| **Best Kaggle score** | **40/50** | feb3 entropy-gated consensus, gpt-oss-120b, 8 attempts |

---

## (1) Strategy Exploration from Local Traces

### What We Know
- 524 AIME traces with Qwen3-8B (526 files in `output/traces/aime_qwen3_8b_nothink_20260204_170730/`)
- `replay_selection.py` has 138 strategies ready to sweep
- Key findings from `output/traces/analysis_insights.md`:

| Entropy | Consensus | Coverage | Accuracy |
|---------|-----------|----------|----------|
| < 0.3   | >= 4      | 8.0%     | **95.2%** |
| < 0.3   | >= 3      | 10.1%    | **90.6%** |
| < 0.5   | >= 4      | 16.2%    | 72.9%    |

- Correct answers have **28% lower entropy** than wrong ones
- Qwen3-8B only solves 22% of AIME (too weak for definitive strategy testing)

### What's Left
- [ ] Run full 138-strategy sweep on **gpt-oss-120b** traces (need H100 trace data)
- [ ] Download traces from `traces-h100-gptoss-aime` Kaggle notebook (status: ran on Feb 7)
- [ ] Compare gpt-oss-120b vs Qwen3-8B strategy rankings — do optimal strategies differ by model strength?

---

## (2) 16-Attempt Scoring

### What We Know
- feb6 notebook (`aimo3-16attempts-feb6`) ran successfully on Kaggle H100
- Config: `attempts=16, workers=16, early_stop=6, entropy_threshold=5.0, min_consensus=2`
- Initialized 16 Jupyter kernels in 15 seconds
- Solved 3 test problems in 0.4 min total, estimated ~15 min for 50 problems
- **Massive compute headroom** — we're using <15 min out of 9 hours available

### What's Left
- [ ] Submit feb6 to competition for real scoring (current run was local gateway with 3 trivial problems)
- [ ] If score > 40: 16 attempts is strictly better, adopt as new baseline
- [ ] If score <= 40: more attempts doesn't help without better selection → focus on fine-tuning
- [ ] Consider: with 15 min runtime, we could go to **32 or even 48 attempts** (like AIMO2 winner's 48)

### Key Question
Winners use 48-64 samples. We now know 16 is feasible and fast. Can we push to 32+?

---

## (3) Fine-Tuning Dataset Quality — FOCUS AREA

### Datasets Available

| # | Dataset | Samples | Source Model | Difficulty | Format |
|---|---------|---------|-------------|------------|--------|
| 1 | `jeannkouagou/aimo3-tool-integrated-reasoning` | 141,277 | gpt-oss-120b | Olympiad | Harmony CSV |
| 2 | `wenliangtlh/aimo3-high-difficulty-tool-calling` | ~70,000 | gpt-oss-120b | High (pass<=7/8) | Harmony JSONL |
| 3 | `alejopaullier/openr1-math-220k` | 220,000 | Various | Mixed | Chat |
| 4 | `jorgeplazas/numinamath-tir` | ~70,000 | NuminaMath | Medium | Standard |
| 5 | `alejopaullier/aimo-external-dataset` | Small (4.5MB) | - | Mixed | - |

### Quality Criteria for "Worthy" Traces

A trace deserves inclusion in fine-tuning if:
1. **Correct answer** (required) — final answer matches ground truth
2. **Appropriate difficulty** — model gets it right 1-3 out of 8 times (not 0/8 trivial-fail or 8/8 trivial-pass)
3. **Clean tool usage** — 2-8 Python calls, no infinite loops or errors
4. **Structured reasoning** — step-by-step with \boxed{} answer
5. **Mathematical rigor** — uses sympy/symbolic math when appropriate
6. **Topic diversity** — balanced across number_theory, algebra, geometry, combinatorics, probability
7. **No contamination** — AIME 2023-2025 excluded (13-gram matching + LCS ratio > 0.6)

### Quality Scoring (from `data/problem_quality_metrics.md`)
```
score = difficulty(0-25) + quality(0-35) + diversity(0-20) + type(0-20)
Filter: quality_score >= 50, tool_calls 1-15, length 2k-80k chars
Expected retention: ~60-70%
```

### Action Items
- [ ] Download top 2 datasets (141K TIR + 70K high-difficulty)
- [ ] Profile: answer distributions, difficulty spread, topic coverage, trace lengths
- [ ] Build quality scoring script (turn the spec into actual code)
- [ ] Run filtering → estimate curated dataset size
- [ ] Contamination check (exclude AIME 2023-2025)
- [ ] Produce final output: "X high-quality traces for SFT, Y hard problems for RL"

---

## (4) Fine-Tuning Strategy — FOCUS AREA

### Three Viable Paths

| Path | Method | Cost | Risk | Upside |
|------|--------|------|------|--------|
| **A: Tinker RL** | GRPO on Qwen3-8B → gpt-oss-120b via API | $50-500 | Medium (LoRA only, untested on MoE) | Proven recipe (76.7% MATH), no GPU needed |
| **B: Unsloth QLoRA** | SFT on H100 (Kaggle or rented) | $0-50 | Low (well-tested framework) | gpt-oss-120b fits in 65GB, full control |
| **C: Distillation** | Generate from Qwen3-235B, SFT into gpt-oss | $100-300 | High (expensive, unclear benefit) | Could unlock new capabilities |

### Recommended Strategy: B first, then A

**Phase 1: SFT with Unsloth QLoRA** (~$0-50)
- Fine-tune gpt-oss-120b on curated TIR traces
- QLoRA 4-bit: r=16, lora_alpha=32, targets=q/k/v/o/gate/up/down_proj
- ~1,000-5,000 high-quality examples (s1 paper showed 1K can beat o1-preview)
- Run on rented H100 or Kaggle (fits in 65GB)
- Output: LoRA adapter (~100-500MB)

**Phase 2: RL with Tinker GRPO** (~$50-200)
- RL fine-tune on hard problems (pass rate 1-3/8)
- Reward: correct answer = +1, wrong = -1
- Curriculum learning: start with 50% pass rate, progress to harder
- Use Tinker API (no GPU needed, pay per token)

**Phase 3: Deploy**
- Merge LoRA weights with base model
- Quantize to MXFP4
- Upload to Kaggle as private dataset
- Use in existing entropy-gated consensus notebook
- Expected: better answers AND lower entropy on correct answers

### Alternative Approaches (via Tinker)
- **Reward model**: Train Qwen3-8B as a verifier (replace entropy heuristic with learned scoring)
- **DPO**: Collect (correct, incorrect) solution pairs, train preference
- **GenSelect**: Train model to pick best answer from candidates (NVIDIA's secret weapon, 566K samples available)

### Key Decisions Needed
- [ ] Pick path: Tinker vs Unsloth vs both?
- [ ] Size the training: how many examples, how many epochs?
- [ ] Design RL reward function (if doing GRPO)
- [ ] Deployment plan: LoRA merge → quantize → upload to Kaggle
- [ ] Budget: how much are we willing to spend?

### Cost Estimates (Tinker)
| Model | Prefill/M tok | Sample/M tok | Train/M tok |
|-------|--------------|-------------|-------------|
| Qwen3-8B | $0.13 | $0.40 | $0.40 |
| gpt-oss-120b | ~$0.50-1.50 | ~$1.50-3.00 | ~$1.50-3.00 |

### Execution Timeline
1. Smoke test: Llama-1B on arithmetic (~$1, minutes) — validate pipeline
2. Real test: Qwen3-8B on MATH (~$20-50) — replicate 76.7% benchmark
3. Competition: gpt-oss-120b on curated data (~$100-500) — actual improvement
4. Grant application: use Phase 2 results as evidence for $5K+ Tinker credits

---

## Proposed Execution Order

### Today (Feb 7) — Focus on (3) and (4)
1. Download and profile top 2 fine-tuning datasets
2. Build quality scoring/filtering script
3. Decide on fine-tuning path (Tinker vs Unsloth)
4. Write concrete training script (not just plan)
5. Run smoke test if possible (~$1-5)

### Parallel — Items (1) and (2)
6. Submit feb6 (16 attempts) to competition for real score
7. Download H100 gpt-oss-120b traces and run strategy sweep
8. If 16 attempts >> 8 attempts: consider pushing to 32+

### Next Session
- Based on feb6 score + dataset profiling: finalize strategy
- Start Phase 1 fine-tuning (SFT with curated data)
- If budget allows: Phase 2 RL

---

## Reference Files
- **`data/dataset.md`** — **MASTER DATASET CATALOG** (29+ datasets with URLs and download commands)
- **`data/quality_selection_research.md`** — **HOW TO SELECT TRAINING SAMPLES** (evidence-based pipeline)
- `data/fine_tuning_datasets.md` — older dataset summary (superseded by dataset.md)
- `data/fine_tuning_resources.md` — Tinker, GRPO, Unsloth details
- `data/problem_quality_metrics.md` — quality scoring spec (original, simpler version)
- `tinker_plan.md` — Tinker API execution plan
- `fine_tuning.md` — QLoRA/Unsloth approach + what worked in AIMO2
- `output/traces/analysis_insights.md` — Qwen3-8B trace analysis
- `scripts/replay_selection.py` — 138 offline selection strategies
- `kaggle_submissions/feb6_16attempts/` — 16-attempt notebook (COMPLETE)
- `kaggle_submissions/feb7_adaptive_entropy/` — two-phase adaptive (NOT PUSHED)
