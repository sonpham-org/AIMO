# AIMO3 Dataset Creation Competition: Rules, Requirements, and Strategy

> Last updated: 2026-02-07
> Researcher: researcher-1

---

## 1. Competition Overview

The **AI Mathematical Olympiad Progress Prize 3 (AIMO3)** is a Kaggle competition with a total prize pool of **$2,207,152** plus **$110,000 in Extra Prizes (EPs)**. It runs from November 20, 2025 to April 15, 2026.

- **Problems**: 110 original math problems (algebra, combinatorics, geometry, number theory)
- **Difficulty**: National Olympiad to IMO standard
- **Hardware**: 1x NVIDIA H100, 9h limit (GPU: 5h), offline (no internet)
- **Answers**: Integer 0-99999 (5 digits, guessing virtually impossible)
- **Submission**: Python evaluation API via Kaggle Notebooks

**Key rule**: All entrants must make their code AND datasets publicly available to qualify for ANY prizes.

---

## 2. The MathCorpus Prize (Dataset Creation Track)

### What It Is
The **MathCorpus Prize** is one of four Extra Prizes (EPs) in AIMO3, replacing the "Early Sharing Prize" from AIMO1/AIMO2. It is awarded for **"publishing novel datasets that will help the wider community."**

### Prize Structure
- Extra Prizes total: **$110,000** shared across 4 categories:
  1. **Longest Leader Prize** - team whose model stays on top of public leaderboard longest
  2. **Hard Problem Prize** - best model solving the least-solved problem
  3. **MathCorpus Prize** - publishing novel datasets for the community
  4. **Write-up Prizes** - best technical explanations of approach

- Individual EP amounts are NOT publicly specified (likely $25K-30K each, but unconfirmed)

### What We Know About Requirements
- **Publicly stated criteria**: "Novel datasets that will help the wider community"
- **No formal submission format** has been publicly documented (as of Feb 2026)
- **No specific evaluation rubric** has been published
- The prize description says "novel" -- implying the dataset must be new/original, not just a re-release of existing data
- Full rules are on Kaggle's AIMO3 overview and rules pages (behind Kaggle auth wall)

### What "Novel" Likely Means (Inferred from Precedent)
Based on the Early Sharing Prize precedent and the AIMO ethos:
1. **New data** -- not just repackaging NuminaMath or OpenMathReasoning
2. **Community benefit** -- the dataset should demonstrably help others train better models
3. **Public availability** -- must be freely accessible (HuggingFace, Kaggle Datasets, etc.)
4. **Mathematical depth** -- should target olympiad-level reasoning, not basic math

---

## 3. Precedent: The Early Sharing Prize (AIMO1/AIMO2)

The MathCorpus Prize replaces the Early Sharing Prize. Understanding that precedent:

### AIMO1 Early Sharing Prize
- **Winner**: Project Numina (same team that won 1st place)
- **What they shared**: NuminaMath dataset (860K problems + solutions) + trained models
- **Impact**: Became the most-used open math dataset in the field
- **Format**: NuminaMath-CoT (860K) + NuminaMath-TIR (70K)

### AIMO2 Early Sharing Prize ($20,000)
- **Winner**: Md Boktiar Mahbub Murad
- **What they shared**: Public notebook solving 20/50 competition problems
- **Criteria**: First public notebook to reach the 20-problem threshold

### Key Lesson
The evolution from "Early Sharing" to "MathCorpus" signals AIMO organizers want **datasets specifically**, not just shared notebooks. This is a dedicated dataset creation incentive.

---

## 4. What Winning Datasets Look Like

### AIMO2 Winner (NVIDIA, 34/50): OpenMathReasoning
- **306K unique problems**, 3.2M CoT solutions, 1.7M TIR solutions, 566K GenSelect samples
- **Problem source**: AoPS (Art of Problem Solving) forums, preprocessed with Qwen2.5-32B-Instruct
- **Solution generators**: DeepSeek-R1 and QwQ-32B (dual model diversity)
- **Innovation**: GenSelect -- a learned answer selection mechanism
- **Verification**: Cross-model agreement + answer extraction
- **License**: CC-BY-4.0

### AIMO1 Winner (Numina, 29/50): NuminaMath
- **860K problems** + solutions
- **Sources**: Chinese high school exams, US/international olympiads, forums
- **Pipeline**: OCR from PDFs -> segmentation -> translation -> CoT reformatting
- **Key contribution**: First large-scale public math reasoning dataset

### LIMO (817 samples, 57.1% AIME)
- Proved quality >> quantity
- Filtered millions of problems down to 817 through multi-stage difficulty filtering
- Each problem must be too hard for small models but solvable by large ones

### s1K (1,000 samples, beat o1-preview)
- Curated from 59K candidates across 50 mathematical domains
- Used difficulty + diversity + quality scoring
- 26 minutes training on 16 H100s

### ASTER (4K TIR, 90% AIME 2025)
- **Interaction-dense TIR trajectories** (multiple tool calls per solution)
- Proved that TIR-format data with real code execution is extremely effective
- Only 4K examples needed

---

## 5. Existing AIMO3-Specific Datasets on Kaggle (Competition)

These are datasets already published by AIMO3 participants:

| Dataset | Size | Format | Key Feature |
|---------|------|--------|-------------|
| [AIMO3 TIR](https://kaggle.com/datasets/jeannkouagou/aimo3-tool-integrated-reasoning) | 141K | Harmony CSV | gpt-oss-120b traces, avg 21K chars |
| [AIMO3 Hard](https://kaggle.com/datasets/wenliangtlh/aimo3-high-difficulty-tool-calling-dataset) | 7.3K problems (70K traces) | Harmony JSONL | Has pass_rate metadata, difficulty filtered |
| [AIMO External](https://kaggle.com/datasets/alejopaullier/aimo-external-dataset) | 4.5MB | Various | Most popular (78 votes) |
| [AIMO3 Dependencies](https://kaggle.com/datasets/ermecan/aimo3-dependency-dataset) | ? | ? | Dependency/utility dataset |
| [AIMO3 Math Olympiad](https://kaggle.com/datasets/ariadneannetsambali/aimo-3-math-olympiad) | ? | ? | Olympiad problems |

---

## 6. What Would Make a Winning MathCorpus Submission

Based on all evidence, a competitive MathCorpus Prize submission should have:

### Must-Haves
1. **Novelty** -- cannot be a repackaging of existing public datasets
2. **Olympiad-level difficulty** -- problems at national/international olympiad standard
3. **Verified correctness** -- solutions must be mathematically correct
4. **Public availability** -- freely downloadable (Kaggle Datasets or HuggingFace)
5. **Practical utility** -- should demonstrably help train better math reasoning models

### Differentiators (What Would Win)
1. **TIR format with real execution** -- ASTER proves this is the most effective format
2. **Quality over quantity** -- LIMO (817) and s1K (1000) beat massive datasets
3. **Difficulty-aware curation** -- pass_rate metadata, targeting sweet spot (3-15% for SFT)
4. **Multi-model verification** -- cross-checking answers across different models
5. **Novel problem sources** -- not just AoPS (already covered by NVIDIA)
6. **gpt-oss-120b traces** -- the competition model itself, giving perfectly matched training data
7. **GenSelect-style selection data** -- pairs of correct/incorrect solutions for training answer selectors
8. **Diverse mathematical domains** -- algebra, combinatorics, geometry, number theory, balanced

### Format Recommendations
Based on what works in the ecosystem:
- **Primary**: Harmony protocol (matches AIMO3 evaluation format)
- **Alternative**: Standard JSONL with problem/solution/answer fields
- **Include metadata**: difficulty level, topic, source, pass_rate, solution length
- **Include both CoT and TIR** formats if possible

---

## 7. Compute Resources Available

For eligible participants:
- **128 H100 GPUs** via Fields Model Initiative partnership (for fine-tuning)
- **Tinker credits** via Thinking Machines partnership (API-based training)
- Standard Kaggle H100 for submission evaluation

---

## 8. Timeline

| Date | Event |
|------|-------|
| Nov 20, 2025 | Competition start |
| Apr 8, 2026 | Entry deadline / team merger deadline |
| Apr 15, 2026 | **Final submission deadline** |

---

## 9. Key Sources

- [AIMO3 Kaggle Competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- [AIMO3 Rules](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/rules)
- [Third Progress Prize Announcement](https://aimoprize.com/updates/2025-11-19-third-progress-prize-launched)
- [AIMO2 Winner (NVIDIA) Paper](https://arxiv.org/pdf/2504.16891)
- [OpenMathReasoning Dataset](https://huggingface.co/datasets/nvidia/OpenMathReasoning)
- [NuminaMath (AIMO1 Winner)](https://huggingface.co/blog/winning-aimo-progress-prize)
- [Early Sharing Prize Awarded](https://aimoprize.com/updates/2024-12-12-sharing-prize-awarded)
- [CompeteHub AIMO3 Overview](https://www.competehub.dev/en/competitions/kaggleai-mathematical-olympiad-progress-prize-3)

---

## 10. Recommendations for Our Team

### Strategy A: Quality-First Small Dataset (Highest ROI)
Create a curated dataset of 1K-5K olympiad-level problems with:
- gpt-oss-120b TIR traces (model-matched to competition)
- Verified correct solutions (cross-model + symbolic checking)
- Difficulty metadata (pass_rate from multiple attempts)
- Diverse problem sources (not just AoPS)

**Why**: LIMO (817), s1K (1K), and ASTER (4K) all prove small curated >> large random. This is feasible within our compute budget and timeline.

### Strategy B: Novel Problem Generation
Generate genuinely new math problems (not sourced from existing databases) using:
- Problem transformation (modify existing problems to create novel variants)
- Template-based generation with mathematical constraints
- Verification that solutions are correct and unique

**Why**: "Novel" is the key criterion. Existing datasets already cover AoPS, MATH, GSM8K, etc. New problems have clear novelty value.

### Strategy C: Selection/Ranking Dataset
Create a dataset specifically for training answer selection models (like NVIDIA's GenSelect):
- Multiple solutions per problem (correct and incorrect)
- Entropy/confidence metadata per solution
- Labeled selection pairs for training verifiers

**Why**: This fills a gap -- most public datasets have only correct solutions. Selection data is what differentiated NVIDIA's winning approach.

### Recommended: Combine A + C
A curated TIR dataset with both correct and incorrect solutions, plus selection metadata. This is novel, practical, and directly useful for the competition format.

---

## 11. Community Insights & Prior Art

> Added: 2026-02-07 (deep dive research)

### 11.1 Kaggle AIMO3 Dataset Landscape (as of Feb 7, 2026)

A comprehensive search of Kaggle datasets tagged for AIMO3 reveals a growing ecosystem. Here are all currently published datasets:

| Dataset | Author | Size | Downloads | Votes | Description |
|---------|--------|------|-----------|-------|-------------|
| **AIMO3 TIR** | jeannkouagou | 903MB | 34 | 8 | 141K gpt-oss-120b TIR traces, Harmony CSV format |
| **AIMO3 High-Difficulty Tool-Calling** | wenliangtlh | 137MB | 16 | 2 | 7.3K hard problems with pass_rate, Harmony JSONL |
| **Math Corpus GRPO 2.6K** | sangrampatil5150 | 609KB | 53 | 12 | GRPO-format dataset, 2.6K samples (most votes!) |
| **AIMO3 Math Bank / Math Corpus** | ngarai | 43MB | 4 | 0 | RAG dataset for model training |
| **AIMO3 Math Olympiad 4.5M** | archange3553 | 135MB | 19 | 2 | 4.5M problems (large-scale generation) |
| **Large-Scale Math Reasoning 520K** | dineshkumar0705 | 51MB | 18 | 0 | 520K reasoning dataset |
| **AIMO3 Dependency Dataset** | ermecan | 5.1GB | 23 | 7 | Utility/dependency wheels |
| **AIMO3 CoT** | srcxiaoyang | 203KB | 0 | 0 | Chain-of-thought traces |
| **AIMO3 To Hard** | srcxiaoyang | 33KB | 0 | 0 | Hard problem subset |
| **AIMO3 Omni-MATH** | andreasbis | 584KB | 1 | 1 | Omni-MATH benchmark for AIMO3 |
| **AIMO3 Benchmark** | ritwikakancharla | 54KB | 3 | 0 | Benchmark dataset |
| **AIMO3 Qwen LoRA** | sachchidanandadaki | 153MB | 9 | 6 | Pre-trained LoRA adapter |
| **AIMO3 Quickstart Wheels** | philipvonderlind | 4.9GB | 0 | 1 | Utility package wheels |
| **AnswerBench GPT-OSS** | yiyangzheng | 108MB | 5 | 1 | gpt-oss-120b answer benchmarking |

**Key observations:**
- The **Math Corpus GRPO** dataset has the most community votes (12) despite being small (2.6K samples), suggesting the community values GRPO-format data
- Multiple teams are targeting dataset creation, indicating competition for the MathCorpus Prize
- Most datasets are relatively small -- nobody has published a massive novel dataset yet
- The field is wide open for a well-curated, high-quality submission

### 11.2 AIMO2 Winner & Runner-Up Dataset Strategies (Prior Art)

#### 1st Place: NVIDIA NemoSkills (34/50)
**Paper**: [AIMO-2 Winning Solution](https://arxiv.org/abs/2504.16891)

Dataset creation pipeline:
1. **Problem sourcing**: Scraped 540K problems from AoPS (Art of Problem Solving) forums
2. **Preprocessing**: Used Qwen2.5-32B-Instruct to clean and standardize problem statements
3. **Solution generation**: Dual-model approach -- DeepSeek-R1 AND QwQ-32B generated solutions independently (model diversity is key)
4. **Output**: 3.2M CoT solutions + 1.7M TIR solutions
5. **GenSelect innovation**: 566K labeled selection pairs -- trained a model to pick the best solution from N candidates (outperformed majority voting)
6. **Verification**: Cross-model answer agreement + answer extraction validation

**What made it win**: Not just scale (3.2M solutions), but the GenSelect selection data that no one else had. The ability to train a learned answer selector was the differentiator.

#### 2nd Place: Imagination Research (31/50)
**GitHub**: [imagination-research/aimo2](https://github.com/imagination-research/aimo2)
**HuggingFace**: [imagination-research/aimo2 collection](https://huggingface.co/collections/imagination-research/aimo2)

Training data approach:
1. **SFT Stage**: Combined Light-R1 stage-2 data + LIMO dataset (deduplicated), trained 8 epochs on 8xA800
2. **DPO Stage**: Created 2,000 DPO pairs from OpenR1-Math-220k with 4 selection criteria:
   - Correctness (chosen must be correct)
   - Minimum length threshold
   - Length ratio (prefer concise: `len(chosen) < ratio * len(rejected)`)
   - Similarity filtering (sentence transformers)
3. **Dual prompt**: 7 CoT + 8 Code prompts per problem (15 samples total)
4. **Key insight**: DPO for shorter outputs -- their model learned to be concise, saving inference time

**Published datasets**: `imagination-research/aimo2-datasets` on HuggingFace

#### 3rd Place: Aliev (30/50)
- **Zero fine-tuning** -- just used DeepSeek-R1-Distill-Qwen-14B-AWQ off the shelf
- Pure majority voting across multiple samples
- Proves that a strong base model + good selection can compete without any custom dataset

### 11.3 Prior Dataset Competitions & Benchmarks (Lessons Learned)

#### NeurIPS Datasets & Benchmarks Track
From the [State of Data Curation at NeurIPS](https://arxiv.org/abs/2410.22473) study (60 datasets, 2021-2023):

**What makes a winning dataset paper:**
- Clear documentation of data sourcing and collection methodology
- Ethical considerations and bias analysis
- Provenance and accessibility information
- Reproducibility of data pipeline
- Evidence of downstream utility (train a model, show improvement)

**Common pitfalls to avoid:**
- Missing environmental impact documentation
- Insufficient ethical considerations
- Opaque data management practices
- No train/test split rationale

#### Notable Math Benchmarks at NeurIPS
- **MATH Dataset** (NeurIPS 2021): 12.5K problems, 5 difficulty levels -- became THE benchmark. Success = clear difficulty taxonomy + expert-written solutions
- **MATH-Vision** (NeurIPS 2024): 3,040 problems with visual context from real competitions -- novelty was multimodal
- **WAMP** (NeurIPS 2023): Competition-level problems annotated with "knowledge pieces" (general facts) and "hints" (problem-specific tricks) in a graph structure -- novelty was rich annotation beyond just problem+solution
- **Easy2Hard-Bench** (NeurIPS 2024): Continuous difficulty estimation from human statistics -- novelty was calibrated difficulty labels

#### OlymMATH (March 2025)
- 200 problems, manually sourced from **printed** publications (not web-scraped) to prevent contamination
- Bilingual (Chinese + English), expert-verified
- Even o3-mini only gets 30.3% on hard split
- Key lesson: **manual sourcing from physical books** is a strong contamination prevention strategy

#### IMO-AnswerBench (Google DeepMind, 2025)
- 400 short-answer IMO problems
- Released alongside DeepMind's 2025 IMO gold medal achievement
- Answers verified by IMO medalists and mathematicians
- Format: problem + short answer + category + subcategory + source
- Licensed CC-BY-4.0

### 11.4 Insights from Harvard CMSA Seminar (Feb 2025)

Simon Frieder (Oxford) gave a talk at Harvard's Center of Mathematical Sciences and Applications: [Datasets for Math: From AIMO Competitions to Math Copilots](https://www.math.harvard.edu/event/datasets-for-math-from-aimo-competitions-to-math-copilots-for-research/)

Key arguments:
1. **Current math datasets have structural problems**: "Binary evaluation or constrained sets of use cases" don't reflect real mathematical workflows
2. **Need fundamental shift**: Dataset structure should map to how mathematicians actually work, not just olympiad Q&A
3. **Documentation standards**: Proposed "mathematical adaptations of dataset documentation (datasheets)" for math-specific datasets
4. **Evaluation rethink**: New thinking LLMs require new evaluation approaches

**Implication for us**: A dataset that includes rich metadata (difficulty, topic taxonomy, solution strategies, knowledge prerequisites) would be more novel and useful than just more problem-solution pairs.

### 11.5 gpt-oss-120b Fine-Tuning Community Knowledge

From [Unsloth](https://unsloth.ai/blog/gpt-oss) and [NVIDIA QAT blog](https://developer.nvidia.com/blog/fine-tuning-gpt-oss-for-accuracy-and-performance-with-quantization-aware-training/):

- **Hardware**: gpt-oss-120b fits in 65GB VRAM for QLoRA training (feasible on H100)
- **Method**: QLoRA on all linear layers (q, k, v, o, gate, up, down), rank=32
- **MXFP4 challenge**: Native weights are MXFP4, requiring custom upcasting for training
- **NVIDIA QAT approach**: Upcast to BF16 -> SFT -> QAT back to FP4 (3-step pipeline)
- **Key warning**: Fine-tuning with non-reasoning data may hurt gpt-oss's reasoning ability. Use reasoning-format data to maintain capabilities.

### 11.6 Key Gaps in the Current Landscape (Opportunities for Us)

Based on all research, these are the **underserved areas** where a novel dataset would have the most impact:

1. **Selection/Verification Data**: NVIDIA's GenSelect is the only public selection dataset. There's huge demand for labeled correct/incorrect solution pairs for training verifiers and answer selectors.

2. **gpt-oss-120b Native Traces**: Very few datasets contain traces from gpt-oss-120b specifically. The existing AIMO3 TIR dataset (141K) is the main one, but it lacks selection metadata.

3. **Rich Metadata**: No current dataset includes entropy scores, confidence metrics, per-step verification labels, or knowledge prerequisite annotations. WAMP-style rich annotation is unexplored for training data.

4. **Interaction-Dense TIR**: ASTER proved that 4K trajectories with 9+ tool calls each are extremely effective (90% AIME 2025). No public dataset specifically curates for interaction density.

5. **DPO Pairs for Math**: Only imagination-research published DPO pairs (2K). High-quality preference pairs for math (correct vs incorrect, concise vs verbose) are scarce.

6. **Contamination-Free Problems**: OlymMATH showed that manually sourced problems from printed books prevent contamination. A dataset of novel, non-web-scraped problems would be highly valued.

7. **GRPO-Format Data**: The Math Corpus GRPO dataset's high vote count (12 votes, most of any AIMO3 dataset) suggests strong community demand for RL-ready data formats.

### 11.7 Revised Recommendation: What to Build

Based on community insights and gaps analysis, the strongest MathCorpus Prize submission would be:

**A curated dataset of 2K-5K olympiad-level problems with:**
1. **Multiple gpt-oss-120b TIR solutions per problem** (correct and incorrect)
2. **Selection labels**: which solution is best and why (GenSelect-style)
3. **DPO pairs**: correct/concise vs incorrect/verbose for preference training
4. **Rich metadata per solution**: entropy score, token count, number of tool calls, verification status
5. **Rich metadata per problem**: difficulty (pass_rate), topic taxonomy, required knowledge
6. **Interaction-dense curation**: prioritize problems requiring 5+ tool calls (ASTER insight)
7. **GRPO-ready format**: verifiable answers for RL training

This fills multiple gaps simultaneously and is differentiated from everything currently available.

### 11.8 Sources for This Section

- [NVIDIA AIMO-2 Winning Solution](https://arxiv.org/abs/2504.16891)
- [Imagination Research AIMO2 2nd Place](https://github.com/imagination-research/aimo2)
- [Imagination Research HuggingFace Collection](https://huggingface.co/collections/imagination-research/aimo2)
- [State of Data Curation at NeurIPS](https://arxiv.org/abs/2410.22473)
- [OlymMATH Dataset](https://github.com/RUCAIBox/OlymMATH)
- [IMO-AnswerBench](https://huggingface.co/datasets/OpenEvals/IMO-AnswerBench)
- [Harvard CMSA: Datasets for Math](https://www.math.harvard.edu/event/datasets-for-math-from-aimo-competitions-to-math-copilots-for-research/)
- [ASTER Paper](https://arxiv.org/html/2602.01204)
- [LIMO-v2 Dataset](https://huggingface.co/datasets/GAIR/LIMO-v2)
- [Unsloth gpt-oss Fine-Tuning](https://unsloth.ai/blog/gpt-oss)
- [NVIDIA QAT for gpt-oss](https://developer.nvidia.com/blog/fine-tuning-gpt-oss-for-accuracy-and-performance-with-quantization-aware-training/)
- [AIMO3 Kaggle Competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- [AIMO3 MathCorpus Prize Announcement](https://aimoprize.com/updates/2025-11-19-third-progress-prize-launched)
