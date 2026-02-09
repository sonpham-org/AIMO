# AIMO Project - Conversation Log

> Sessions 1-17 (Jan 31 – Feb 7) archived into MEMORY.md.
> Key decisions and findings are preserved there. Below is the rolling log for recent work.

---

## Session History (Archived Summary)

| Session | Date | Topic |
|---------|------|-------|
| 1-2 | Jan 31 | AIMO2 top-3 solution deep dive (NVIDIA/Imagination/Aliev) |
| 3 | Feb 1 | Kaggle submission notebook, predict signature fix |
| 4-6 | Feb 1 | vLLM wheels for Python 3.12/CUDA 12.9 (resolved) |
| 7 | Feb 2 | Weighted entropy (38/50) analysis, entropy-plus (33/50 regression) |
| 8 | Feb 2 | Built trace generation + 138-strategy replay system |
| 9 | Feb 3 | Kaggle trace generator notebook for H100 |
| 10-11 | Feb 3 | **Feb3 submission (40/50 best)**, local trace jobs, Kaggle GPU issues |
| 12 | Feb 4 | Feb4 verified consensus (32/50 regression), ablation testing setup |
| 13 | Feb 4 | Qwen3 /no_think fix (13x speedup), AIME trace restart |
| 14-15 | Feb 4 | Analyzed jonathanchan (42) and kishanvavdara (44) — both variance, no real improvement |
| 16-17 | Feb 4 | Fine-tuning dataset research (30+ datasets), Kaggle hardware analysis, Eagle3 notebooks |
| 17+ | Feb 7 | Trace analysis (95.2% at entropy<0.3), Tinker plan, dataset catalog, feb7 master plan |

---

## Feb 7, 2026 - Session 18: Memory Cleanup + Critical Score Findings

- Deleted `.claude/iteration_plan_both_models.md` (superseded by feb7_plan.md)
- Consolidated 75KB conversation log into concise MEMORY.md

### Critical Finding: More Attempts = Worse Score
| Attempts | Early Stop | Score |
|----------|-----------|-------|
| 8 | 4 | **40/50** |
| 12 | 5 | 37/50 |
| 16 | 6 | **29/50** |

- 16 attempts confirmed feasible (~15min runtime, massive headroom)
- But current entropy-gated selection breaks with more samples
- Hypothesis: more attempts → more noise → diluted consensus → wrong answer wins
- **Action**: Redesign selection strategy for N=16 using H100 trace data

### Team Spawned: crawler + ml-researcher + planner
- **Crawler**: Research all resources (datasets, methods, tools)
- **ML Researcher**: Find cost-effective fine-tuning methods for gpt-oss-120b
- **Planner**: Synthesize into actionable plan for April 10 deadline

---

## Feb 7, 2026 - Session 19: Comprehensive Dataset Search

**Objective**: 100% confidence we haven't missed any AIMO3-relevant datasets

### Search Coverage
- **Kaggle searches**: "aimo", "aimo3", "tool-integrated-reasoning", "gpt-oss-120b", "math olympiad", "numinamath-tir", "openr1-math", "deepseek-math", "metamath", "mathinstruct", "GRPO", user datasets (andreasbis, wenliangtlh, jeannkouagou, alejopaullier, jorgeplazas, seshurajup, kishanvavdara)
- **HuggingFace searches**: "gpt-oss-120b AIMO", "TIR math dataset", "openr1-math", "aime olympiad", "numina CoT", "qwen2.5-math", "deepseek-math"
- **Total datasets cataloged**: 29 in `data/dataset.md`

### Key Findings

#### Top Curated Datasets (Quality >> Quantity)
1. **LIMO** (GAIR): 817 samples → **57.1% AIME, 94.8% MATH** (already downloaded)
2. **s1K** (SimpleScaling): 1K samples → **beat o1-preview on MATH/AIME24 by 27%** (already downloaded)
3. **ASTER cold-start**: 4K TIR trajectories → **90% AIME 2025, 73.3% HMMT 2025**

#### Best TIR Datasets
1. **AIMO3 TIR** (jeannkouagou): 141,277 samples from gpt-oss-120b, olympiad level (downloaded)
2. **AIMO3 Hard** (wenliangtlh): 7,293 problems × 8 traces = 70K, **has pass_rate metadata** (downloaded)
3. **NuminaMath-TIR**: 70K samples, GPT-4 generated, medium difficulty

#### RL-Specific Datasets
1. **Big-Math-RL-Verified** (SynthLabsAI): 250K verified problems for RL
2. **PRM800K** (OpenAI): 800K step-level labels, process reward model training
3. **AceMath-RM-Training-Data** (NVIDIA): Reward model training data

#### Largest Datasets
1. **OpenMathReasoning** (NVIDIA): 306K problems, 5.68M solutions (3.2M CoT, 1.7M TIR, 566K GenSelect)
2. **OpenMathInstruct-2** (NVIDIA): 14M problem-solutions (600K unique)
3. **OpenThoughts3-1.2M**: 1.2M reasoning traces (850K math)

### Downloaded Dataset Profiles

| Dataset | Rows | Avg Length | Key Insight |
|---------|------|-----------|-------------|
| LIMO | 817 | 18,659 chars | 817 curated → 57.1% AIME |
| s1K | 1,000 | 1,946 chars | 50% AIME problems, beat o1 |
| AIMO3 TIR | 141,277 | 21,820 chars | All gpt-oss-120b, olympiad |
| AIMO3 Hard | 7,293 | 67,350 chars | **Has pass_rate metadata!** |

### AIMO3 Hard Pass Rate Distribution (Critical for Difficulty Filtering)
| Pass Rate | Count | % | Use Case |
|-----------|-------|---|----------|
| 1/8 (12.5%) | 662 | 9.1% | Hardest — RL gold |
| 2/8 (25%) | 1,050 | 14.4% | SFT sweet spot |
| 3/8 (37.5%) | 1,794 | 24.6% | Goldilocks zone |
| 4/8 (50%) | 612 | 8.4% | Medium |
| 5/8 (62.5%) | 752 | 10.3% | Medium-easy |
| 6/8 (75%) | 950 | 13.0% | Easy |
| 7/8 (87.5%) | 1,473 | 20.2% | Too easy for RL |

**Sources**: AoPS 4,817 (66%) + StackOverflow 2,476 (34%)

### Not Yet Downloaded (But Available)
- **Big-Math-RL-Verified**: Gated, need HF token + agreement
- **OpenMathReasoning**: Very large (5.5M solutions), download when needed
- **OpenThoughts3-1.2M**: Very large, download when needed

### Verification
- Cross-referenced all datasets from `data/fine_tuning_datasets.md`, `data/dataset.md`, and feb7_plan
- Fetched direct metadata from Kaggle and HuggingFace for top 6 datasets
- **Conclusion**: All major AIMO3-relevant datasets are cataloged in `data/dataset.md`
- No significant datasets were missed

---

## Feb 7, 2026 - Session 20: Dataset Curation Pipeline + Downloads

### New Downloads
- **OpenMathReasoning GenSelect**: 565,620 samples (2.3 GB, 14 parquet shards) — answer selection training
- **Big-Math-RL-Verified**: 251,122 problems (80 MB, gated access approved)
- HF login as `sonphamorg`, access granted for gated datasets

### Built: `scripts/curate_dataset.py`
6-stage pipeline: Load → Hard Filter → Decontaminate → Difficulty → Quality Score → Diversity Select

**Key discovery**: Tool call marker in Harmony format is `<|start|>python`, NOT `channel|>python`

### Pipeline Results (300K loaded, 3K target)
| Stage | Passed | Key rejection |
|-------|--------|--------------|
| Hard Filter | 285,393 (95.1%) | 9,574 no_answer_marker |
| Decontamination | 100,876 | 184,517 overlap with eval |
| Difficulty (3-50%) | 98,873 | 835 too hard, 1,168 too easy |
| Quality Scoring | 98,873 scored | LIMO=0.713, Hard=0.655, TIR=0.542 |

### 4 Diversity Modes Compared (3K samples each)
| Mode | Best for | Source mix | Topic coverage |
|------|----------|-----------|----------------|
| topic | SFT training | 68% TIR | 19 categories, min 78 each |
| source | Balanced training | 63% TIR, 8% BigMath | Weaker topics |
| difficulty | RL / hard problems | 59% TIR, 33% GenSelect | Good spread |
| multi | Needs tuning | 60% BigMath (too much) | Balanced topics |

### Topic Taxonomy: 19 categories
4 number theory + 4 algebra + 4 combinatorics + 5 geometry + 1 analysis
(Still 31% "other" — keyword classifier needs improvement)

### Outputs
- `data/curated_3000_topic.jsonl` — recommended for first SFT experiment
- `data/curated_3000_source.jsonl`, `data/curated_3000_difficulty.jsonl`, `data/curated_3000_multi.jsonl`
- `docs/data_selection_strategy.md` — full report with methodology and next steps

### Bug Fixes
- Fixed catastrophic regex backtracking in repetition check (`.{80,}\1` → hash-based O(n))
- Fixed tool call detection: `<|start|>python` not `channel|>python`
- Relaxed hard filter for AIMO3 TIR (doesn't require tool calls)

---

## Feb 7, 2026 - Session 21: Deep Research — AIMO Winner Trace Metadata

### What NVIDIA NeMo-Skills Stores Per Solution (from source code)

**Core generation output fields** (from `nemo_skills/inference/model/base.py`):
- `generation` (string): Full LLM output text
- `num_generated_tokens` (int): Completion token count
- `generation_start_time` (float): Unix timestamp
- `generation_end_time` (float): Unix timestamp
- `generation_time` (float): Wall-clock seconds
- `finish_reason` (string): "stop", "length", etc.
- `logprobs` (list[float]): Per-token log probabilities (when `top_logprobs > 0`)
- `tokens` (list[string]): Corresponding token strings
- `top_logprobs` (list[dict]): Top-K alternatives per position {token: logprob}
- `reasoning_content` (string): Separate thinking content (reasoning models)
- `num_reasoning_tokens` (int): Token count in thinking block
- `num_answer_tokens` (int): Token count outside thinking
- `num_input_tokens` (int): Prompt token count (when `count_prompt_tokens=True`)

**Evaluation fields** (added by `nemo_skills/evaluation/evaluator/math.py`):
- `predicted_answer` (string): Extracted from `\boxed{}` in generation
- `expected_answer` (string): Ground truth
- `symbolic_correct` (bool): Whether predicted matches expected via symbolic comparison
- `judgement` (string): LLM judge output ("Judgement: Yes/No")

**Aggregation fields** (from `aggregate_answers.py`):
- `reward_model_score` (float): RM score per solution (optional)
- `majority_votes` (int): Number of votes for winning answer
- `total_votes` (int): Total solutions considered
- `fill_mode` (string): "majority" or "majority_rm" or "highest_rm"

### OpenMathReasoning HuggingFace Dataset Schema (ALL splits share same 9 columns)
| Column | Type | Description |
|--------|------|-------------|
| `problem` | string | Problem statement (AoPS/MATH) |
| `generated_solution` | string | Full solution text |
| `expected_answer` | string | Ground truth or majority answer |
| `problem_type` | string | "has_answer_extracted" / "no_answer_extracted" / "converted_proof" |
| `problem_source` | string | AoPS forum or "MATH_training_set" |
| `generation_model` | string | "DeepSeek-R1" or "QwQ-32B" |
| `pass_rate_72b_tir` | string | Pass rate (0-32) for Qwen2.5-Math-72B TIR |
| `inference_mode` | string | "cot", "tir", or "genselect" |
| `used_in_kaggle` | bool | Whether used in AIMO-2 Kaggle training |

### GenSelect — How It Works (from NeMo-Skills source code)
1. **Input per candidate**: Full solution text (or summary after `</think>` tag, max 3000 chars)
2. **Per-candidate fields in training data**:
   - `solution_{i}` (string): The candidate solution text
   - `predicted_answer_{i}` (string): Extracted answer from that solution
   - `label_{i}` (string): "Correct" or "Incorrect"
3. **GenSelect prompt format**: Problem + N solutions (2-16, typically 8) → model outputs `Judgment: [IDX]`
4. **Training data construction**: 8 random comparison groups per problem, each with at least 1 correct + 1 incorrect
5. **Key constraint**: Unstable with >32 candidates (won't fit in context)
6. **At inference**: Run GenSelect 8 times on 16-solution subsets, then majority@8 over selected answers

### Imagination Team (2nd Place AIMO-2) — Per-Solution Metadata
From `imagination_aimo2/local_eval.py` source code:

**Per-sample data structures** (OrderedDict keyed by token count):
- `cot_answers[i]` — Dict[token_count → boxed_answer]: Answer history as generation progresses
- `code_answers[i]` — Dict[token_count → code_exec_answer]: Code execution results over time
- `python_code_map_list[i]` — Dict[token_count → code_string]: Extracted Python code
- `code_exec_error_map_list[i]` — Dict[token_count → error_string]: Execution errors
- `token_counts[i]` (int): Total tokens generated per sample
- `outputs[i]` (string): Complete generation text

**Per-question stored in results.json**:
- `id`, `question`, `correct_answer`
- `answer` (int): Final aggregated answer (mod 1000)
- `cot_answers` (list[OrderedDict]): All boxed answers per sample
- `code_answers` (list[OrderedDict]): All code execution answers per sample
- `out_lens` (list[int]): Token counts per sample
- `python_code_map_list`: All extracted code per sample
- `code_exec_error_map_list`: All errors per sample
- `question_duration` (float): Wall-clock seconds

**Dual prompt strategy**: 7 CoT + 8 Code prompts = 15 samples per question
**Aggregation**: Weighted majority voting with priority (code > CoT by default)

**Statistics tracked per-problem** (from `_report_statistics`):
- `no_code_ratio`: Fraction with no Python code
- `code_exec_error_ratio`: Fraction with execution errors
- `answer_wrong_fail_parseint_ratio`: Integer parse failures
- `answer_wrong_number_ratio`: Wrong integer answers
- `correct_cot_ratio`, `correct_code_ratio`: Per-method accuracy
- `avg_out_len`: Average output length

### NuminaMath (1st Place AIMO-1) — SC-TIR
- N=48 candidates, depth M=4 (self-correction rounds)
- Per candidate: problem, interleaved (rationale + code + execution output), final answer
- Selection: Simple majority voting on extracted integer answers
- **No logprobs, no reward model, no confidence scores**

### CMU-MATH (2nd Place AIMO-1) — Reward Model
- 42 candidates per question, DeepSeek-Math-7B-RL policy model
- Reward model: Fine-tuned DeepSeek-Math-7B-RL, scores 0-1 per solution
- **Input to RM**: Just the (problem, solution) text pair — no structured features
- Selection: Weighted majority voting (geometric mean of RM scores)
- Training data: 7,000 problems x ~5.4 solutions each = 37,880 labeled pairs

### Key Takeaway: Nobody Uses Logprobs for Selection
- NVIDIA GenSelect: Text-only comparison (model reads N solutions, picks best)
- Imagination: Priority-weighted majority voting (code answer > CoT answer)
- NuminaMath: Pure frequency majority voting
- CMU-MATH: RM score weighting (but RM is a separate model, not logprobs)
- **Entropy/logprobs are our unique signal** — not validated by any winner

---

## Feb 8, 2026 - Session 22: Judge Architecture + H100 Traces

### 7-7-2 Judge Architecture (Inspired by ARC-AGI beetree)
- **Phase 1**: 7 broad attempts (5 TIR + 2 text-only), early stop if 4+ agree
- **Phase 2**: 7 deep attempts (only if Phase 1 didn't converge)
- **Phase 3**: 2 judges evaluate all solutions, pick top 2 answers each
- **Judge-only scoring**: 1st pick = 2pts, 2nd pick = 1pt per judge. Solvers DON'T vote — judges have full authority.
- **Fallback**: Entropy-gated consensus if both judges fail

### Local Testing Results (Qwen3-8B-Q4_K_M on AMD iGPU)
- **15-problem comparison**:
  - Ensemble-8 (majority vote): 10/15 = 67%
  - 7-7-2 Judge: 13/15 = **87%** (+20%, +3 problems, broke 0)
  - **CAVEAT**: This was on a weak 8B model. Unclear if judge helps with gpt-oss-120b which is already much stronger. Kaggle run will tell.
- **50-problem AIME test** (started, partially completed):
  - Judge-772: 4/17 (23.5%) — Qwen3-8B too weak for AIME
  - Ensemble-8: 2/4 (50%) — only 4 problems completed before process died
  - Low accuracy expected with 8B model on competition-level problems

### H100 Trace Data Downloaded and Analyzed
- **74 problems × 16 samples = 1,184 traces** from gpt-oss-120b on H100
- Sources: 35 aimo3_hard, 25 bigmath, 13 limo, 1 genselect
- **48.2% sample accuracy** (limo: 89%, bigmath: 57%, aimo3_hard: 28%)
- **Oracle upper bound: 78%** (58/74 problems have ≥1 correct in 16 samples)
- **Majority vote: 57%** (42/74) — 21-point gap = room for better selection
- All 3 selection strategies (majority, entropy-gated, entropy-weighted) tie at 42/74

### Key Signal Analysis (Correct vs Incorrect)
| Signal | Correct | Incorrect | Insight |
|--------|---------|-----------|---------|
| Entropy | 0.751 | 0.909 | Lower = better |
| Tokens | 3,783 | 6,166 | Shorter = better |
| Code calls | 2.0 | 4.4 | Fewer = better |
| ngram_rep_4 | 0.070 | 0.112 | Less repetition = better |
| Time | 19.8s | 32.3s | Faster = better |

### Data Quality Issues (saved in `docs/math_problems.md`)
- 2 problems have answers leaked in problem text (remove from training)
- ~7 problems have suspicious ground truths (need re-validation)
- 11.1% of samples have answer=None (turn limit exhaustion)

### Notebooks Created/Updated
- `submissions/feb8_judge_772/` — **Pushed to Kaggle** (v3, RUNNING)
  - `judge_max_tokens = 16384` (was 4096), `trace_max_chars = 6000` (was 3000)
  - Judge-only scoring, 2 judges, 7+7+2 architecture
- `submissions/feb8_judge/` — Updated with multi-judge support (num_judges=1 default)
- `scripts/run_judge_local_amd.py` — Local AMD testing with multi-judge
- `scripts/run_judge_local.py` — Local vLLM testing with multi-judge

### Math Corpus Prize
- Discussion post drafted: `docs/math_corpus_discussion.md`
- Dataset: 1,184 traces with 24 per-sample fields including per-token entropy/logprobs
- Unique angle: nobody else publishes logprob signals for answer selection research

### Files Created/Modified
- `docs/math_corpus_discussion.md` — Kaggle discussion post for Math Corpus submission
- `docs/math_problems.md` — Data quality issues log (leaked answers, suspicious GTs)
- `submissions/feb8_judge_772/` — New 7-7-2 Kaggle notebook (pushed)
- `scripts/run_judge_local_amd.py` — Multi-judge, judge-only scoring
- `scripts/run_judge_local.py` — Same changes for vLLM version
