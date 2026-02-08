# MathCorpus Prize: Novelty & Community Utility Strategy

> Last updated: 2026-02-07
> Researcher: researcher-1

---

## 1. The "4 MathCorpus Categories" — Clarification

The AIMO3 MathCorpus Prize description is deliberately open-ended: "For publishing novel datasets that will help the wider community." There are no formally defined "4 MathCorpus categories" in the official rules. However, looking at what datasets the math AI community actually needs, four natural submission archetypes emerge:

### Category A: Training Data (SFT/RL)
Problem-solution pairs for supervised fine-tuning or RL. Examples: NuminaMath (860K CoT), OpenMathReasoning (3.2M CoT + 1.7M TIR), LIMO (817 curated).

**Gap**: Plenty of large-scale SFT data exists. What's missing is **difficulty-calibrated, quality-scored, model-matched** data for gpt-oss-120b.

### Category B: Selection/Verification Data
Labeled solution groups (correct vs incorrect) for training answer selectors, verifiers, and reward models. Examples: GenSelect (566K), PRM800K (step-level labels).

**Gap**: Only NVIDIA has GenSelect, and PRM800K is from 2023. **No public selection/verification dataset exists for gpt-oss-120b or AIMO3-era models.** This is the biggest gap.

### Category C: Benchmark/Evaluation Data
New problem sets for evaluation. Examples: OlymMATH (200), IMO-AnswerBench (400), Omni-MATH (4,428).

**Gap**: Evaluation sets are well-covered. Adding another benchmark is low novelty.

### Category D: Rich-Annotation Data
Problems with metadata beyond just problem+solution: knowledge prerequisites, step-level labels, difficulty taxonomy, solution strategies. Examples: WAMP (knowledge pieces + hints), PRM800K (step-level correctness).

**Gap**: **Nobody has combined rich annotation with TIR traces.** WAMP has annotations but only CoT. PRM800K has step labels but no tool integration. This is highly novel territory.

### Best Fit for Us: **B + D hybrid**

A **selection + rich-annotation dataset** built on gpt-oss-120b TIR traces is the most novel and useful approach. It combines:
- GenSelect-style selection labels (Category B)
- Rich per-solution metadata: entropy, confidence, interaction density (Category D)
- TIR format with real code execution (unlike any existing rich-annotation dataset)

---

## 2. Gap Analysis: What's Missing in Public Math Datasets

### 2.1 Comprehensive Gap Table

| Feature | OpenMathReasoning | AIMO3 TIR | LIMO | PRM800K | WAMP | **Ours (proposed)** |
|---------|------------------|-----------|------|---------|------|---------------------|
| Model | DeepSeek-R1, QwQ | gpt-oss-120b | DeepSeek-R1 | GPT-4 | N/A | **gpt-oss-120b** |
| Format | CoT + TIR | TIR (Harmony) | CoT | CoT | Q+A | **TIR (Harmony)** |
| Selection labels | GenSelect (566K) | None | None | None | None | **Yes** |
| DPO pairs | None | None | None | None | None | **Yes** |
| Step-level labels | None | None | None | +1/0/-1 per step | None | **Possible** |
| Difficulty score | pass_rate_72b_tir | None | Implicit | None | None | **pass_rate + entropy** |
| Entropy/confidence | None | None | None | None | None | **Yes (per-token logprobs)** |
| Interaction density | Not tracked | Not tracked | N/A | N/A | N/A | **# tool calls per trace** |
| Knowledge prereqs | None | None | None | None | Yes (knowledge pieces) | **Topic taxonomy** |
| Correct + incorrect | GenSelect only | Correct only | Correct only | Both (labeled) | N/A | **Both (labeled)** |
| Verification method | Answer match | Answer match | Multi-stage | Human | N/A | **SymPy + cross-model** |
| Size | 5.68M rows | 141K rows | 817 | 800K labels | Small | **2K-5K (curated)** |
| GRPO-ready | No | No | No | No | No | **Yes** |

### 2.2 The 7 Critical Gaps (Ranked by Impact)

**Gap 1: Selection/Verification Data for gpt-oss-120b**
- NVIDIA's GenSelect is the ONLY public selection dataset. It uses DeepSeek-R1 and QwQ, not gpt-oss-120b
- No one has published labeled correct/incorrect solution groups for the AIMO3 competition model
- **Impact**: Directly enables training better answer selectors for AIMO3 submissions
- **Novelty**: High -- model-matched selection data doesn't exist

**Gap 2: Per-Solution Confidence/Entropy Metadata**
- DeepConf (2025) proved that token-entropy filtering achieves 99.9% on AIME 2025 while reducing tokens by 84.7%
- No public dataset includes entropy or confidence scores per solution
- **Impact**: Enables entropy-based filtering at training time, not just inference time
- **Novelty**: High -- this metadata doesn't exist in any published dataset

**Gap 3: Interaction-Dense TIR Curation**
- ASTER proved 4K trajectories with 9+ tool calls yield 90% AIME 2025
- No existing dataset curates by interaction density
- The AIMO3 TIR dataset (141K) includes everything -- no density filtering
- **Impact**: Directly trains models to use tools effectively, which is the key skill for AIMO3
- **Novelty**: Medium-high -- ASTER showed the concept but didn't publish the dataset

**Gap 4: DPO Pairs for Math**
- Imagination Research published 2K DPO pairs (only public math DPO data)
- Their pairs focused on conciseness (short vs long). No pairs for correctness or solution strategy
- **Impact**: Enables preference learning for math reasoning
- **Novelty**: Medium -- concept exists, but high-quality TIR DPO pairs don't

**Gap 5: GRPO-Ready Format**
- The Math Corpus GRPO dataset on Kaggle (2.6K samples) has 12 votes -- the highest of any AIMO3 dataset
- Community demand for RL-ready data with verifiable answers is clearly strong
- **Impact**: Enables RL training without answer verification pipeline
- **Novelty**: Medium -- format exists, but quality and scale matter

**Gap 6: Trajectory Quality Scores**
- AutoTraj (2026) showed trajectory scoring across 4 dimensions improves selection by 7.5%:
  - Confidence score (uncertainty keywords: "maybe", "unsure", "guess")
  - Length score (Gaussian fit to optimal length per difficulty)
  - Repetition score (N-gram repetition rate)
  - Answer correctness (binary)
- No public dataset includes these trajectory-level quality scores
- **Impact**: Enables quality-aware sampling and curriculum learning
- **Novelty**: High -- novel metadata type

**Gap 7: Contamination-Free Novel Problems**
- Most datasets source from AoPS, MATH, GSM8K (all potentially contaminated in LLM training data)
- OlymMATH showed manual sourcing from printed books prevents contamination
- **Impact**: Provides clean evaluation signal
- **Novelty**: High but requires significant manual effort

---

## 3. Selection Metadata: What It Should Look Like

Based on analysis of GenSelect, PRM800K, DART-Math, LIMO, DeepConf, and AutoTraj, here's a comprehensive metadata schema for our dataset:

### 3.1 Per-Problem Metadata

```json
{
  "problem_id": "olympiad-algebra-042",
  "problem": "Let $f(x)$ be a polynomial...",
  "expected_answer": 42,
  "source": "AoPS / printed book / synthetic",
  "topic": "algebra",
  "subtopic": "polynomial",
  "difficulty": {
    "pass_rate_gptoss": 0.125,
    "pass_rate_8_attempts": 0.25,
    "difficulty_tier": "hard",
    "estimated_imo_level": 3
  },
  "n_solutions_generated": 8,
  "n_correct": 2,
  "n_incorrect": 6,
  "consensus_answer": 42,
  "consensus_strength": 2
}
```

### 3.2 Per-Solution Metadata

```json
{
  "solution_id": "olympiad-algebra-042-sol-3",
  "problem_id": "olympiad-algebra-042",
  "generated_solution": "...(full TIR trace)...",
  "extracted_answer": 42,
  "is_correct": true,
  "generation_model": "gpt-oss-120b",
  "format": "TIR",
  "quality_metrics": {
    "answer_entropy": 0.23,
    "mean_token_entropy": 1.45,
    "max_step_entropy": 3.21,
    "confidence_score": 0.95,
    "has_uncertainty_cues": false,
    "n_tool_calls": 7,
    "n_successful_executions": 7,
    "n_failed_executions": 0,
    "total_tokens": 4521,
    "reasoning_tokens": 3200,
    "code_tokens": 1321,
    "repetition_score": 0.98,
    "has_self_verification": true,
    "has_exploration": true,
    "length_score": 0.82
  },
  "selection_label": {
    "is_best_solution": true,
    "rank_among_correct": 1,
    "selection_reason": "concise, correct, well-verified"
  }
}
```

### 3.3 Per-Problem Selection Group (GenSelect-style)

```json
{
  "problem_id": "olympiad-algebra-042",
  "candidate_summaries": [
    {"solution_id": "sol-1", "answer": 42, "is_correct": true, "summary": "Used substitution..."},
    {"solution_id": "sol-2", "answer": 37, "is_correct": false, "summary": "Applied AM-GM..."},
    {"solution_id": "sol-3", "answer": 42, "is_correct": true, "summary": "Computed via sympy..."}
  ],
  "best_solution_id": "sol-3",
  "selection_rationale": "Most concise correct solution with code verification"
}
```

### 3.4 DPO Pair Format

```json
{
  "problem_id": "olympiad-algebra-042",
  "chosen": {
    "solution_id": "sol-3",
    "text": "...(correct, concise TIR trace)...",
    "answer": 42,
    "is_correct": true,
    "token_count": 3200
  },
  "rejected": {
    "solution_id": "sol-5",
    "text": "...(incorrect or verbose TIR trace)...",
    "answer": 37,
    "is_correct": false,
    "token_count": 8500
  },
  "pair_type": "correct_vs_incorrect"
}
```

### 3.5 Key Metadata Fields Explained

| Field | Source/Method | Why Novel | Community Value |
|-------|-------------|-----------|-----------------|
| `answer_entropy` | Top-5 logprobs at answer token | No dataset has this | Enables entropy-gated filtering during inference |
| `mean_token_entropy` | Average entropy across all tokens | DeepConf showed this correlates with correctness | Enables confidence-aware training |
| `n_tool_calls` | Count tool invocations in trace | ASTER proved 9+ calls optimal | Enables interaction-density curation |
| `confidence_score` | 1.0 if no uncertainty cues, else 0.0 | AutoTraj scoring dimension | Quality signal for selection |
| `repetition_score` | 1 - max(N-gram repetition rates) | AutoTraj scoring dimension | Filters degenerate traces |
| `length_score` | Gaussian fit to optimal length | AutoTraj scoring dimension | Identifies right-sized solutions |
| `pass_rate_gptoss` | N correct / N total per problem | DART-Math's core metric | Difficulty calibration |
| `is_best_solution` | Ranked by quality score | GenSelect innovation | Selection model training |
| `has_self_verification` | Keywords: "verify", "check", "confirm" | LIMO quality dimension | Identifies thorough solutions |
| `selection_rationale` | Free text explaining why best | Novel -- no dataset has this | Interpretable selection training |

---

## 4. Novelty Argument: Why This Dataset Wins the MathCorpus Prize

### What No One Else Has Done

1. **First model-matched selection dataset for AIMO3**: gpt-oss-120b traces with correct/incorrect labels. NVIDIA's GenSelect uses DeepSeek-R1 -- different model, different behavior patterns.

2. **First entropy-annotated math dataset**: Per-solution and per-token entropy metadata. DeepConf proved this is the single most impactful quality signal, but no dataset includes it.

3. **First interaction-density-curated TIR dataset**: ASTER proved interaction-dense trajectories (9+ tool calls) are optimal for cold-start training, but didn't publish the curated subset. We publish it.

4. **First combined SFT + DPO + GenSelect + GRPO dataset**: Existing datasets serve ONE purpose. Ours serves four training paradigms from a single source.

5. **First trajectory-quality-scored dataset**: AutoTraj's scoring dimensions (confidence, length, repetition) applied at dataset level, not just runtime.

### Community Utility Argument

| Use Case | How Our Dataset Helps |
|----------|----------------------|
| SFT fine-tuning | High-quality correct TIR solutions with difficulty calibration |
| DPO preference learning | Labeled correct/incorrect pairs with quality metrics |
| GenSelect training | Solution groups with selection labels and rationale |
| GRPO/RL training | Verified answers + difficulty metadata for reward shaping |
| Answer selection at inference | Entropy thresholds directly usable without recomputing |
| Curriculum learning | Pass rate + difficulty tier enables easy-to-hard scheduling |
| Research on solution quality | Rich metadata enables studying what makes solutions good |

### Comparison to Existing AIMO3 Datasets on Kaggle

| Dataset | Their Approach | Our Advantage |
|---------|---------------|---------------|
| AIMO3 TIR (141K) | Raw traces, no quality filtering | Curated + quality scored + selection labels |
| AIMO3 Hard (7.3K) | Difficulty filtered, no selection | Selection labels + DPO pairs + entropy |
| Math Corpus GRPO (2.6K) | GRPO format only | Multi-format (SFT + DPO + GenSelect + GRPO) |
| AIMO3 Math Bank (43MB) | RAG-focused | Training-focused with verified solutions |
| AIMO3 4.5M Problems | Quantity over quality | Quality over quantity (proven superior by LIMO/s1K) |

---

## 5. Practical Dataset Specification

### 5.1 Target Size and Composition

Based on LIMO (817), s1K (1K), and ASTER (4K) evidence, optimal size is **2K-5K problems** with **8-16 solutions each**:

| Component | Size | Purpose |
|-----------|------|---------|
| Problems | 2,000-5,000 | Unique olympiad-level math problems |
| Solutions per problem | 8-16 | Multiple gpt-oss-120b TIR traces |
| Total traces | 16K-80K | Full TIR trajectories with metadata |
| GenSelect groups | 2K-5K | One selection group per problem |
| DPO pairs | 2K-5K | One correct/incorrect pair per problem |
| GRPO entries | 2K-5K | Verified answer + problem for RL |

### 5.2 Problem Sourcing (Ranked by Novelty)

1. **AIMO3-Hard dataset problems** (7.3K available, already have pass_rate)
2. **AoPS forum problems** (huge supply, but common in other datasets)
3. **OpenMathReasoning additional_problems** (193K unsolved problems)
4. **NuminaMath problems** (860K, various difficulties)
5. **Printed book problems** (highest novelty but requires manual effort)

### 5.3 Generation Pipeline

```
Problem Pool (5K+ problems with known answers)
    |
    v
Generate 8-16 TIR solutions per problem via gpt-oss-120b
    |-- Collect top-5 logprobs per token
    |-- Compute answer entropy, token entropy, confidence
    |
    v
Verify correctness (SymPy + exact match)
    |-- Label each solution: correct / incorrect
    |-- Compute pass_rate per problem
    |
    v
Score each trajectory (AutoTraj-style)
    |-- Confidence score (uncertainty cues)
    |-- Length score (Gaussian fit)
    |-- Repetition score (N-gram)
    |-- Interaction density (# tool calls)
    |
    v
Curate and package:
    |-- SFT split: best correct solution per problem
    |-- DPO split: best correct vs worst incorrect per problem
    |-- GenSelect split: all solutions with selection labels
    |-- GRPO split: problem + verified answer
    |-- Full metadata attached to every trace
```

### 5.4 Format and Packaging

- **Primary format**: Parquet (HuggingFace standard) + JSONL (Kaggle standard)
- **Splits**: `train_sft`, `train_dpo`, `train_genselect`, `train_grpo`, `metadata`
- **Compatibility**: Harmony protocol format for direct AIMO3 submission use
- **License**: CC-BY-4.0 (same as OpenMathReasoning)
- **Hosting**: Kaggle Datasets + HuggingFace Hub (dual publication)

### 5.5 Documentation (NeurIPS Standard)

Following the NeurIPS Datasets & Benchmarks Track best practices:
- Datasheet for datasets (Gebru et al.)
- Clear methodology documentation
- Reproducible generation pipeline (code published)
- Downstream evaluation (show model improvement from training on our data)
- Ethical considerations (no personally identifiable info, open license)

---

## 6. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Someone publishes similar dataset first | Medium | Speed -- generate now on H100, publish before April 8 |
| gpt-oss-120b traces not available offline | Low | We already have the model locally, and it runs on H100 |
| Quality not high enough to win | Medium | Apply LIMO-style multi-stage filtering, demonstrate downstream improvement |
| Dataset too small to be useful | Low | LIMO (817), s1K (1K), ASTER (4K) all prove small curated > large random |
| MathCorpus Prize criteria unclear | High | Cover multiple use cases (SFT/DPO/GenSelect/GRPO) to maximize chances |

---

## 7. Sources

- [NVIDIA OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) -- GenSelect format, pass_rate_72b_tir
- [GenSelect: A Generative Approach to Best-of-N](https://openreview.net/forum?id=8LhnmNmUDb) -- Selection methodology
- [OpenAI PRM800K](https://github.com/openai/prm800k) -- Step-level correctness labels
- [DART-Math](https://github.com/hkust-nlp/dart-math) -- Difficulty-aware rejection tuning
- [LIMO](https://github.com/GAIR-NLP/LIMO) -- Quality scoring system (L1-L5 rating)
- [ASTER](https://arxiv.org/html/2602.01204) -- Interaction-dense cold-start (4K trajectories)
- [AutoTraj](https://arxiv.org/html/2601.23032) -- Trajectory quality scoring (confidence + length + repetition)
- [DeepConf](https://arxiv.org/pdf/2508.15260) -- Token-entropy confidence filtering
- [Big-Math-RL-Verified](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified) -- RL-ready format
- [AIMO3 Kaggle Datasets](#) -- Landscape analysis (14 datasets cataloged)
- [NeurIPS Data Curation](https://arxiv.org/abs/2410.22473) -- Documentation best practices
- [Imagination Research AIMO2](https://github.com/imagination-research/aimo2) -- DPO pair construction
