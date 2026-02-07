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
