# Data Selection Strategy for AIMO3 Fine-Tuning

## Overview

This document describes the dataset curation pipeline built in `scripts/curate_dataset.py` for selecting high-quality training samples to fine-tune gpt-oss-120b for the AIMO Progress Prize 3 competition (50 math olympiad problems, H100 GPU, 9h limit).

## Available Datasets

| Dataset | Samples | Size on Disk | Format | Has Pass Rate | Has Tool Calls |
|---------|---------|-------------|--------|--------------|----------------|
| AIMO3 Hard | 7,293 | ~3.5 GB | JSONL (Harmony) | Yes (1/8 to 7/8) | Yes (median 6) |
| AIMO3 TIR | 141,277 | ~3 GB | CSV (Harmony) | No | Rare |
| LIMO | 817 | ~15 MB | JSONL | Fixed 0.05 | No (CoT) |
| s1K | 1,000 | ~2 MB | Parquet | Fixed 0.10 | No (CoT) |
| GenSelect | 565,620 | 2.3 GB | Parquet (14 shards) | Partial | No (answer selection) |
| Big-Math-RL | 251,122 | ~80 MB | Parquet | No | No (problems only) |
| **Total** | **~967K** | **~9 GB** | | | |

### Dataset Descriptions

- **AIMO3 Hard**: 7,293 olympiad problems with 8 solution traces each, from gpt-oss-120b. Includes pass rate metadata (pass/count) for difficulty filtering. Tool-integrated reasoning with `<|start|>python` segments.
- **AIMO3 TIR**: 141K traces from gpt-oss-120b on olympiad problems. No pass_rate. Mixed CoT and tool-calling.
- **LIMO**: 817 curated problems from GAIR. Achieves 57.1% AIME, 94.8% MATH with just 817 samples. Demonstrates that quality >> quantity.
- **s1K**: 1,000 curated hard problems (877 math, 108 science). Beat o1-preview on MATH/AIME24 by 27%.
- **GenSelect** (NVIDIA OpenMathReasoning): 566K answer selection training samples. Each sample presents 3 candidate solutions and asks the model to identify the correct one. This trains a "learned selection" capability. NVIDIA's AIMO2-winning GenSelect model improved AIME25 from 84% to 90%.
- **Big-Math-RL-Verified** (SynthLabsAI): 251K verified math problems for RL training. Problems only, no solutions.

## 6-Stage Curation Pipeline

### Stage 0: Load & Normalize
All datasets loaded into unified `Sample` format with: uid, source_dataset, problem_text, solution_text, answer, pass_rate, n_tool_calls, topic.

### Stage 1: Hard Filters
Binary pass/fail filters:
- Length: 500 - 100,000 chars (AIMO3 Hard solutions range 1.7K - 277K)
- Tool calls: AIMO3 Hard requires 1-20 tool calls
- Answer marker: must contain `\boxed{}`, "final answer", or "the answer is"
- Repetition: hash-based sliding window check for 80+ char repeated blocks

**Results** (300K loaded, 20K TIR + 20K GenSelect subsampled):
- Rejected: 9,574 no_answer_marker, 2,012 repetitive, 1,745 too_long, 757 too_short, 405 no_tool_calls, 346 too_many_tool_calls
- **285,393 / 300,232 passed (95.1%)**

### Stage 2: Decontamination
9-gram overlap check against evaluation problems (75 AIME problems from 2005-2024 + IMO benchmark).
- Normalized text: lowercase, strip LaTeX, keep alphanumeric
- Threshold: >10% n-gram overlap = contaminated
- **100,876 / 285,393 passed (184,517 removed)**
- High removal rate is mostly Big-Math-RL problems overlapping with AIME eval sets

### Stage 3: Difficulty Calibration
Filter by pass_rate range (default: 0.03 - 0.50):
- Too hard (< 3%): 835 removed — model can't learn from these
- Too easy (> 50%): 1,168 removed — no signal for hard problems
- Unknown pass_rate: kept (96,845 samples)
- **98,873 / 100,876 passed**

### Stage 4: Quality Scoring
LIMO-style multi-dimensional scoring (weights in parentheses):

- **Difficulty (25%)**: Score based on pass_rate (sweet spot: 3-15% = 1.0, 15-30% = 0.7)
- **Solution Quality (35%)**: Length + verification words + exploration words + logical connectives
- **Interaction Density (20%)**: Number of tool calls (sweet spot: 9-15 = 1.0, following ASTER paper)
- **Structure (20%)**: Has steps, has `\boxed{}`, uses SymPy, no repetition

**Score distribution by dataset:**
| Dataset | Mean Score | N samples |
|---------|-----------|-----------|
| LIMO | 0.713 | 35 |
| AIMO3 Hard | 0.655 | 218 |
| AIMO3 TIR | 0.542 | 5,839 |
| s1K | 0.505 | 35 |
| GenSelect | 0.423 | 8,363 |
| Big-Math-RL | 0.208 | 84,383 |

### Stage 5: Diversity Selection
Four diversity modes were tested (all targeting 3,000 samples):

#### Topic Diversity (`--diversity topic`)
- 19 fine-grained math categories with balanced round-robin
- Categories: 4 number theory (primes, divisibility, modular, diophantine), 4 algebra (polynomial, inequality, equations, sequences), 4 combinatorics (counting, probability, graph, pigeonhole), 5 geometry (triangle, circle, polygon, coordinate, 3D), 1 analysis
- Ensures minimum 78 samples per category
- **Result**: Balanced across 19 topics, but 31% in "other" and 68% AIMO3 TIR

#### Source Diversity (`--diversity source`)
- Balance across source datasets using `sqrt(n)` proportional allocation
- **Result**: More balanced sources (63% TIR, 19% GenSelect, 8% BigMath, 7% Hard) but weaker topic coverage

#### Difficulty Diversity (`--diversity difficulty`)
- Balance across 5 difficulty bands: very_hard (≤10%), hard (10-25%), medium (25-40%), easy (40-50%), unknown
- **Result**: Equal 10% allocation to very_hard/hard/medium bands, pulls more GenSelect (33%)

#### Multi-Axis Diversity (`--diversity multi`)
- Two-stage: allocate per source (sqrt-proportional), then balance by topic within each source
- **Result**: Over-represents BigMath (60%) due to sqrt(251K) domination — needs tuning

### Stage 6: Importance Weighting (Optional)
When `--iw-sft` is enabled:
- Prop2Diff-style weighting: `weight = (1 - pass_rate) * (0.5 + 0.5 * quality_score)`
- Normalized to mean=1.0
- Weight range: 0.62 - 2.11
- Hard problems with high quality get up to 2x training weight

## Comparison of Diversity Modes

| Metric | topic | source | difficulty | multi |
|--------|-------|--------|-----------|-------|
| AIMO3 Hard % | 5.5% | 7.3% | 7.3% | 3.2% |
| AIMO3 TIR % | 68.4% | 63.2% | 58.7% | 15.7% |
| GenSelect % | 23.5% | 18.8% | 32.7% | 18.8% |
| BigMath % | 1.2% | 8.3% | 0% | 59.9% |
| LIMO % | 1.2% | 1.2% | 1.2% | 1.2% |
| "other" topic % | 31.4% | 38.5% | 29.4% | 38.8% |
| Min topic count | 78 | 6 | 5 | 74 |
| Mean pass_rate | 0.164 | 0.188 | 0.256 | 0.132 |

**Recommendation**: `topic` mode gives the most balanced coverage. For SFT training, combine with `--iw-sft` for importance weighting. For RL, the `difficulty` mode is better as it ensures coverage of the hardest problems.

## Usage

```bash
# Profile all datasets
python3 scripts/curate_dataset.py profile

# Select 3000 samples with topic diversity + importance weighting
python3 scripts/curate_dataset.py select --target 3000 --diversity topic --iw-sft

# Select 5000 samples focused on hard problems
python3 scripts/curate_dataset.py select --target 5000 --diversity difficulty --min-pass-rate 0.03 --max-pass-rate 0.30

# Select with full datasets (slow, ~2 min)
python3 scripts/curate_dataset.py select --target 4000 --max-tir-rows 0 --diversity multi

# Stop early to see intermediate stats
python3 scripts/curate_dataset.py select --target 3000 --stop-after scoring
```

Output: `data/curated_N.jsonl` with fields: uid, source_dataset, problem_text, solution_text, answer, pass_rate, n_tool_calls, topic, combined_score, difficulty_score, quality_score, interaction_score, structure_score, iw_weight (if --iw-sft).

## Known Issues & Limitations

1. **"other" category is large (30-39%)**: The keyword classifier misses many problem types. Could improve with:
   - Embedding-based clustering (requires model inference)
   - Sub-classification of "other" using more keywords (calculus, linear algebra, etc.)
   - Using the `problem_source` field from GenSelect/Big-Math as ground truth labels

2. **Decontamination removes 65% of Big-Math**: Only 75 eval problems are loaded, but 9-gram overlap catches many related problems. May be over-filtering — consider raising the 10% threshold to 20%.

3. **No pass_rate for 97% of samples**: Only AIMO3 Hard and GenSelect have real pass_rate. Others use fixed estimates (LIMO=0.05, s1K=0.10). Could compute pass_rate by running samples through a smaller model.

4. **Multi-axis diversity over-represents Big-Math**: The `sqrt(n)` proportional allocation makes Big-Math dominant (251K >> others). Fix: cap per-source at e.g. 25% of target.

5. **GenSelect format is different**: These aren't problem+solution pairs — they're "pick the best solution" training data. Should be used to train a selector model, not a solver.

## Next Steps

1. **Fix multi-axis diversity**: Cap source allocation at 25% to prevent Big-Math domination
2. **Separate GenSelect for selector training**: Don't mix with solver SFT data
3. **Compute pass_rate for AIMO3 TIR**: Run a quick eval on a sample of TIR problems to get difficulty estimates
4. **Add embedding-based topic clustering**: Use a small model to compute embeddings and cluster into finer topics
5. **Train with curated data**: Use the `topic` 3K dataset as a first SFT experiment:
   - Unsloth QLoRA on gpt-oss-120b (fits in 65GB on H100)
   - Compare with/without importance weighting
   - Target: +2-3 points on Kaggle (42-43/50)
6. **Train GenSelect adapter**: Separate LoRA for answer selection, serve alongside solver via vLLM multi-LoRA
7. **GRPO RL via Tinker**: After SFT, apply GRPO on the hardest problems (Big-Math-RL subset)
