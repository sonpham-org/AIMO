# Math Dataset Catalog for AIMO3

> Last updated: 2026-02-07
> Purpose: Comprehensive list of all math datasets relevant to fine-tuning for AIMO3 competition
> Downloaded to: `data/downloads/`

---

## Downloaded Dataset Profiles (Feb 7)

| Dataset | Location | Rows | Avg Length | Format | Key Insight |
|---------|----------|------|-----------|--------|-------------|
| **LIMO** | `data/downloads/LIMO/limo.jsonl` | 817 | 18,659 chars | question/solution/answer | 817 curated → 57.1% AIME |
| **s1K** | `data/downloads/s1K/data/*.parquet` | 1,000 | 1,946 chars | question/solution/thinking_trajectories | 50% AIME problems, beat o1 |
| **AIMO3 TIR** | `data/downloads/aimo3_tir/data.csv` | 141,277 | 21,820 chars | Harmony CSV (prompt/completion) | All gpt-oss-120b, olympiad |
| **AIMO3 Hard** | `data/downloads/aimo3_hard/*.jsonl` | 7,293 | 67,350 chars | Harmony JSONL (full tool traces) | Has pass_rate metadata! |

### AIMO3 TIR Length Distribution
| Range | Count | % |
|-------|-------|---|
| 0-5K | 43,000 | 30.4% |
| 5K-10K | 26,670 | 18.9% |
| 10K-20K | 26,550 | 18.8% |
| 20K-30K | 12,824 | 9.1% |
| 30K-50K | 12,456 | 8.8% |
| 50K-100K | 16,611 | 11.8% |
| 100K+ | 3,166 | 2.2% |

### AIMO3 Hard — Pass Rate Distribution (KEY for difficulty filtering)
| Pass Rate | Count | % | Use Case |
|-----------|-------|---|----------|
| 1/8 (12.5%) | 662 | 9.1% | Hardest — RL gold |
| 2/8 (25%) | 1,050 | 14.4% | SFT sweet spot |
| 3/8 (37.5%) | 1,794 | 24.6% | Goldilocks zone |
| 4/8 (50%) | 612 | 8.4% | Medium |
| 5/8 (62.5%) | 752 | 10.3% | Medium-easy |
| 6/8 (75%) | 950 | 13.0% | Easy |
| 7/8 (87.5%) | 1,473 | 20.2% | Too easy for RL |

Sources: AoPS 4,817 (66%) + StackOverflow 2,476 (34%)

### s1K Source Distribution
- AIME 1983-2024: 287 (29%)
- NuminaMath/AoPS: 271 (27%)
- GPQA: 88 (9%)
- OlympiadBench Physics: 74 (7%)
- Omni-MATH: 71 (7%)
- OpenAI Math: 73 (7%)
- Other (TheoremQA, JEEBench, crosswords): 136 (14%)

### NOT YET Downloaded (Need HF Login or Large)
- **Big-Math-RL-Verified** — gated, need HF token + agreement
- **OpenMathReasoning** — very large (5.5M solutions), download when needed
- **OpenThoughts3-1.2M** — very large, download when needed

---

## Tier 1: Competition-Winning Datasets (Proven in AIMO)

### 1. nvidia/OpenMathReasoning
- **URL**: https://huggingface.co/datasets/nvidia/OpenMathReasoning
- **Size**: 306K unique problems, 3.2M CoT solutions, 1.7M TIR solutions, 566K GenSelect samples
- **Generator**: DeepSeek-R1, QwQ-32B
- **Problem source**: AoPS (Art of Problem Solving) forums
- **Difficulty**: Competition-level (AMC through IMO)
- **Format**: CoT, TIR (with Python execution), GenSelect (solution selection)
- **License**: CC-BY-4.0
- **Tool use**: Yes (1.7M TIR solutions)
- **Result**: Foundation of **AIMO-2 1st place (34/50)**
- **Paper**: https://arxiv.org/abs/2504.16891
- **Download**: `huggingface-cli download nvidia/OpenMathReasoning`
- **Key fields**: `pass_rate_72b_tir` (difficulty metric), problem/solution/answer

### 2. AI-MO/NuminaMath-1.5
- **URL**: https://huggingface.co/datasets/AI-MO/NuminaMath-1.5
- **Size**: ~896K competition-level math problems with CoT solutions
- **Sources**: 11 sources — Chinese high school exams, US/international olympiads, forums
- **Difficulty**: Mixed (high school through olympiad)
- **Format**: CoT
- **License**: Apache 2.0
- **Tool use**: No
- **Result**: Foundation of AIMO-1 winner (NuminaMath-7B-TIR scored 29/50)
- **Download**: `huggingface-cli download AI-MO/NuminaMath-1.5`

### 3. AI-MO/NuminaMath-TIR
- **URL**: https://huggingface.co/datasets/AI-MO/NuminaMath-TIR
- **Kaggle**: `jorgeplazas/numinamath-tir`
- **Size**: ~70K problems with TIR solutions
- **Generator**: GPT-4 (TORA-format)
- **Difficulty**: Medium-high (competition level)
- **Format**: TIR (interleaved NL + Python + execution output)
- **License**: Apache 2.0
- **Tool use**: Yes
- **Result**: Stage 2 training data for AIMO-1 winner; MATH 56.3% → 68.2%
- **Download**: `kaggle datasets download jorgeplazas/numinamath-tir`

### 4. AI-MO/NuminaMath-CoT
- **URL**: https://huggingface.co/datasets/AI-MO/NuminaMath-CoT
- **Size**: ~860K problem-solution pairs
- **Format**: Chain of thought
- **License**: Apache 2.0
- **Download**: `huggingface-cli download AI-MO/NuminaMath-CoT`

---

## Tier 1.5: AIMO3-Specific Datasets (Kaggle)

### 5. AIMO3 Tool-Integrated Reasoning Dataset ⭐
- **Kaggle**: `jeannkouagou/aimo3-tool-integrated-reasoning`
- **Size**: 141,277 samples (902MB)
- **Generator**: GPT-OSS-120b
- **Format**: Harmony protocol CSV (with conversion scripts)
- **License**: Apache 2.0
- **Tool use**: Yes — real Python execution traces
- **Key features**: Solution hint methodology, avg 21,825 chars, includes format converters
- **Best for**: Direct TIR fine-tuning for AIMO3
- **Download**: `kaggle datasets download jeannkouagou/aimo3-tool-integrated-reasoning`

### 6. AIMO3 High-Difficulty Tool-Calling Dataset ⭐
- **Kaggle**: `wenliangtlh/aimo3-high-difficulty-tool-calling-dataset`
- **Size**: ~70,000 trajectories from 7,293 problems
- **Generator**: GPT-OSS-120b (8 samples per problem)
- **Format**: Harmony JSONL
- **License**: Apache 2.0
- **Tool use**: Yes
- **Key features**: High difficulty (pass rate ≤7/8), Eagle3 trained on this, IMO 50%→60%
- **Download**: `kaggle datasets download wenliangtlh/aimo3-high-difficulty-tool-calling-dataset`

### 7. AIMO External Dataset
- **Kaggle**: `alejopaullier/aimo-external-dataset`
- **Size**: 4.5MB (smaller, curated)
- **Votes**: 78 (most popular)
- **Best for**: Quick experiments, baseline
- **Download**: `kaggle datasets download alejopaullier/aimo-external-dataset`

---

## Tier 2: Large-Scale Training Datasets

### 8. nvidia/OpenMathInstruct-2
- **URL**: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
- **Size**: 14M problem-solution pairs (~600K unique questions)
- **Generator**: Llama-3.1-405B-Instruct
- **Difficulty**: Grade school through competition
- **Format**: CoT
- **License**: Commercially permissive
- **Result**: Llama-3.1-8B MATH 51.9% → 67.8%
- **Insight**: Question diversity > solution count
- **Paper**: https://arxiv.org/abs/2410.01560
- **Download**: `huggingface-cli download nvidia/OpenMathInstruct-2`

### 9. a-m-team/AM-DeepSeek-R1-Distilled-1.4M
- **URL**: https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M
- **Size**: 1.4M entries (0.5M open-source + 0.9M distilled from DeepSeek-R1-671B)
- **Format**: Long reasoning traces
- **License**: Open-source
- **Result**: AM-Distill-Qwen-32B outperforms DeepSeek-R1-Distill-Qwen-32B
- **Paper**: https://arxiv.org/abs/2503.19633
- **Download**: `huggingface-cli download a-m-team/AM-DeepSeek-R1-Distilled-1.4M`

### 10. open-thoughts/OpenThoughts3-1.2M
- **URL**: https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M
- **Size**: 1.2M examples (850K math, 250K code, 100K science)
- **Generator**: QwQ-32B
- **Format**: Long reasoning traces
- **License**: Open
- **Result**: OpenThoughts3-7B: 53% AIME 2025, 51% LiveCodeBench
- **Paper**: https://arxiv.org/abs/2506.04178
- **Download**: `huggingface-cli download open-thoughts/OpenThoughts3-1.2M`

### 11. open-thoughts/OpenThoughts-114k
- **URL**: https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k
- **Size**: 114K examples (math, science, code, puzzles)
- **Generator**: DeepSeek-R1
- **License**: Open
- **Download**: `huggingface-cli download open-thoughts/OpenThoughts-114k`

### 12. open-r1/OpenR1-Math-220k
- **URL**: https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
- **Kaggle**: `alejopaullier/openr1-math-220k`
- **Size**: 220K problems, 2-4 reasoning traces each
- **Source**: NuminaMath-1.5 problems + DeepSeek-R1 solutions
- **Verification**: Math Verify + Llama-3.3-70B judge
- **License**: Open
- **Good for**: Rejection sampling and DPO pairs
- **Paper**: https://huggingface.co/blog/open-r1
- **Download**: `kaggle datasets download alejopaullier/openr1-math-220k`

---

## Tier 3: Curated High-Quality Small Datasets (Quality > Quantity)

### 13. GAIR/LIMO (Less Is More for Reasoning) ⭐⭐
- **URL**: https://huggingface.co/datasets/GAIR/LIMO
- **Size**: 817 curated samples (also GAIR/LIMO-v2 available)
- **Curation**: Tens of millions → filtered with Qwen2.5-Math-7B → 32 attempts with DeepSeek-R1-Distill-Qwen-32B → kept only consistently hard problems → manual eval
- **Difficulty**: Very high (competition/olympiad)
- **Format**: (question, reasoning chain, answer) triplets
- **License**: Open
- **Result**: **57.1% AIME, 94.8% MATH with only 817 samples**
- **Key insight**: 800 curated examples >> 100K random examples
- **Paper**: https://arxiv.org/abs/2502.03387
- **Download**: `huggingface-cli download GAIR/LIMO`

### 14. simplescaling/s1K ⭐⭐
- **URL**: https://huggingface.co/datasets/simplescaling/s1K
- **Size**: 1,000 questions with reasoning traces
- **Sources**: NuminaMath, AIME, OmniMath, AGIEval, OlympicArena (from 59K candidates)
- **Generator**: Gemini Thinking (for reasoning traces)
- **Curation**: Difficulty + diversity + quality filtering
- **Difficulty**: Very high
- **License**: Open
- **Result**: **s1-32B exceeds o1-preview on MATH and AIME24 by up to 27%**, trained in 26 minutes on 16 H100s
- **Paper**: https://arxiv.org/abs/2501.19393
- **Download**: `huggingface-cli download simplescaling/s1K`

### 15. Light-R1 Training Data
- **URL**: https://github.com/Qihoo360/Light-R1
- **Size**: 76K (stage 1) + 3K (stage 2)
- **Sources**: OpenR1-Math, OpenThoughts, LIMO, OpenMathInstruct-2, s1K, AIME
- **Generator**: DeepSeek-R1, filtered by verification + difficulty
- **Result**: Light-R1-14B: AIME24 74.0%, AIME25 60.2% (SOTA 14B)
- **Paper**: https://arxiv.org/abs/2503.10460

### 16. ASTER Cold-Start Set (Feb 2026)
- **Size**: 4K interaction-dense TIR trajectories
- **Result**: **ASTER-4B: 90.0% AIME 2025, 73.3% HMMT 2025**
- **Key insight**: 4K expert TIR cold-start set → strongest downstream performance
- **Paper**: https://arxiv.org/html/2602.01204

---

## Tier 4: RL-Specific Datasets

### 17. SynthLabsAI/Big-Math-RL-Verified ⭐
- **URL**: https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified
- **Size**: 250K+ verified math questions with verifiable answers
- **Purpose**: Specifically designed for RL training
- **Curation**: Rigorously filtered — requires uniquely verifiable, open-ended, closed-form answers
- **License**: Open
- **Paper**: https://arxiv.org/abs/2502.17387
- **Download**: `huggingface-cli download SynthLabsAI/Big-Math-RL-Verified`

### 18. openai/PRM800K (Process Reward Model)
- **URL**: https://github.com/openai/prm800k
- **Size**: 800K step-level correctness labels over 75K solutions
- **Source**: MATH dataset
- **Purpose**: PRM training — labels individual reasoning steps correct/incorrect
- **Result**: Process supervision >> outcome supervision; 78% on MATH subset
- **Paper**: https://arxiv.org/abs/2305.20050
- **Available PRM**: `Qwen2.5-Math-7B-PRM800K` on HuggingFace

### 19. nvidia/AceMath-RM-Training-Data
- **URL**: https://huggingface.co/datasets/nvidia/AceMath-RM-Training-Data
- **Purpose**: Reward model training for math evaluation
- **Models**: AceMath-7B-RM, AceMath-72B-RM (outcome reward models)
- **Eval**: https://huggingface.co/datasets/nvidia/AceMath-RewardBench
- **Result**: AceMath-RL-Nemotron-7B: 69.0% AIME24, 53.6% AIME25

### 20. nvidia/AceMath-Instruct-Training-Data
- **URL**: https://huggingface.co/datasets/nvidia/AceMath-Instruct-Training-Data
- **Purpose**: SFT training for math instruction
- **Result**: AceMath-72B-Instruct outperforms Qwen2.5-Math-72B (71.8 vs 68.2 MATH)

---

## Tier 5: TIR-Specific Resources

### 21. ToRA-Corpus
- **URL**: https://github.com/microsoft/ToRA
- **Size**: 16K annotations
- **Generator**: GPT-4
- **Format**: Interactive tool-use trajectories (GSM8K + MATH)
- **Result**: 13-19% absolute improvement over baselines
- **Paper**: https://arxiv.org/abs/2309.17452

### 22. hkust-nlp/dart-math-hard (DART-Math)
- **URL**: https://huggingface.co/datasets/hkust-nlp/dart-math-hard
- **Size**: ~590K examples
- **Innovation**: Difficulty-Aware Rejection Tuning — allocates more budget to harder problems
- **Paper**: https://arxiv.org/abs/2407.13690 (NeurIPS 2024)
- **Download**: `huggingface-cli download hkust-nlp/dart-math-hard`

---

## Tier 6: Evaluation / Benchmark Datasets

### 23. KbsdJames/Omni-MATH
- **URL**: https://huggingface.co/datasets/KbsdJames/Omni-MATH
- **Size**: 4,428 competition-level problems, 33+ sub-domains
- **Difficulty**: Olympiad level (o1-mini only 60.5%)
- **Paper**: https://arxiv.org/abs/2410.07985

### 24. hendrycks/competition_math (MATH)
- **URL**: https://huggingface.co/datasets/hendrycks/competition_math
- **Size**: 12.5K problems (AMC/AIME), 7 subjects, 5 difficulty levels

### 25. MathArena (AIME 2025, HMMT)
- **URL**: https://huggingface.co/datasets/MathArena/aime_2025
- **Size**: 30 AIME problems
- **Purpose**: Contamination-resistant evaluation

### 26. AIME Problem Sets (1983-2024)
- **Kaggle**: `hemishveeraboina/aime-problem-set-1983-2024`
- **Local**: `data/aime_train_2005_2022.csv` (524), `data/aime_test_2023_2024.csv` (43)

---

## Tier 7: Other Potentially Useful

### 27. AoPS-Instruct
- **Size**: 650K+ QA pairs from AoPS forums
- **Paper**: https://arxiv.org/abs/2501.14275

### 28. meta-math/MetaMathQA
- **URL**: https://huggingface.co/datasets/meta-math/MetaMathQA
- **Method**: Bootstrap question augmentation from GSM8K + MATH
- **Paper**: https://arxiv.org/abs/2309.12284

### 29. TIGER-Lab/MathInstruct
- **URL**: https://huggingface.co/datasets/TIGER-Lab/MathInstruct
- **Size**: 13 math rationale datasets compiled; hybrid CoT + PoT

---

## Math Corpus Prize (AIMO3)

The AIMO3 competition includes a **Math Corpus Prize** as one of 4 Extra Prizes ($110K total pool). Details:
- Awarded for "publishing novel datasets that will help the wider community"
- All entrants must make code and datasets publicly available
- Competition deadline: April 15, 2026
- Judging criteria, specific prize amount, and submission process not publicly documented
- Datasets #5 and #6 above (Kaggle AIMO3-specific) are likely early candidates
- Our own curated dataset could potentially qualify

---

## Quick Download Reference

```bash
# Top priority datasets
kaggle datasets download jeannkouagou/aimo3-tool-integrated-reasoning          # 141K TIR traces
kaggle datasets download wenliangtlh/aimo3-high-difficulty-tool-calling-dataset # 70K hard TIR
huggingface-cli download nvidia/OpenMathReasoning                              # 306K problems, 5.5M solutions
huggingface-cli download GAIR/LIMO                                             # 817 curated (quality>quantity)
huggingface-cli download simplescaling/s1K                                     # 1K curated (beat o1-preview)

# RL-specific
huggingface-cli download SynthLabsAI/Big-Math-RL-Verified                      # 250K RL-ready problems

# Large-scale
huggingface-cli download open-thoughts/OpenThoughts3-1.2M                      # 1.2M reasoning traces
huggingface-cli download open-r1/OpenR1-Math-220k                              # 220K with DPO potential
huggingface-cli download nvidia/OpenMathInstruct-2                             # 14M problem-solutions

# TIR-specific
kaggle datasets download jorgeplazas/numinamath-tir                            # 70K TIR (medium difficulty)
huggingface-cli download hkust-nlp/dart-math-hard                              # 590K difficulty-aware
```
