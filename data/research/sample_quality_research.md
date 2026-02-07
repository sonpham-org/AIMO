# Deep Research: Training Sample Quality for Math LLM Fine-Tuning

> Last updated: 2026-02-07
> Author: ml-researcher2
> Purpose: Comprehensive, actionable guide to selecting high-quality training samples for fine-tuning gpt-oss-120b on olympiad math
> Companion doc: `data/quality_selection_research.md` (summary version)

---

## Executive Summary

We need to select 1K-5K high-quality examples from 100K+ candidates. The research is unambiguous: **quality crushes quantity**.

| System | Examples | AIME Score | Key Technique |
|--------|----------|------------|---------------|
| LIMO | 817 | 63.3% | Pass-rate 3-9%, quality scoring |
| s1 | 1,000 | Beat o1-preview | Difficulty + diversity + quality |
| ASTER | 4,000 | 90.0% AIME 2025 | Interaction-dense TIR trajectories |
| 100K random SFT | 100,000 | 32.3% | No filtering (anti-pattern) |
| Full 100K skill-aware | 100,000 | Worse than base | Proved: more data can HURT |

**Our recommendation**: 2K-4K examples, filtered through a 6-stage pipeline, with importance-weighted SFT loss. Details below.

---

## 1. Difficulty Scoring (Most Important After Correctness)

### 1.1 Why Difficulty Is the #1 Signal

Every top-performing small-data approach gates on difficulty:
- **DART-Math** (NeurIPS 2024): Standard rejection tuning produces datasets that are 90% easy problems. Their Prop2Diff strategy allocates samples proportional to difficulty, gaining +4.5 points.
- **LIMO** (COLM 2025): Kept only problems with pass_rate 3-9% (1-3 correct out of 32 attempts). This alone explains most of their performance.
- **s1**: Removed ALL problems solvable by either Qwen-7B or Qwen-32B-Instruct. Dual-model gating ensures true difficulty.
- **Online Difficulty Filtering**: Theoretical proof that intermediate difficulty maximizes gradient signal.
- **Skill-Aware Selection** (2025): Training on full 100K corpus *degrades* performance vs base model. 1K selected examples beat 100K random.

### 1.2 How to Compute Difficulty

**Method A: Pass-rate estimation (recommended)**
```
For each problem:
    Generate N solutions (N=8 for speed, N=32 for accuracy)
    pass_rate = n_correct / N
    difficulty = 1 - pass_rate
```

**Method B: Dual-model gating (s1 approach)**
```
For each problem:
    Test with weak model (e.g., Qwen3-8B)
    Test with strong model (e.g., gpt-oss-120b)
    Keep only if: weak_correct == False AND strong_pass_rate in [0.03, 0.30]
```

**Method C: Proxy from existing metadata**
Some datasets (e.g., AIMO3-Hard) already include pass_rate fields. Use directly.

### 1.3 Optimal Difficulty Ranges

| Use Case | Pass Rate Range | Justification |
|----------|----------------|---------------|
| SFT (quality) | 0.03-0.15 | LIMO: 1-3/32, maximizes reasoning depth |
| SFT (broader) | 0.03-0.30 | s1: dual-model filter, more coverage |
| RL (GRPO/PPO) | 0.10-0.70 | AdaRFT: targets ~50% success rate, adapts dynamically |
| Cold-start SFT | 0.05-0.25 | ASTER: hard enough to need tools, solvable enough to learn from |

### 1.4 DART-Math Prop2Diff Sampling

Instead of uniform sampling from correct solutions, allocate more budget to harder problems:

```python
# Prop2Diff: probability of selecting a solution from problem i
# is proportional to its difficulty
weight_i = (1 - pass_rate_i)  # harder = higher weight
sample_prob_i = weight_i / sum(all_weights)
```

This simple reweighting yielded +4.5% on MATH benchmark. Apply at the sampling stage after quality filtering.

### 1.5 AdaRFT: Dynamic Difficulty for RL

For RL fine-tuning specifically, AdaRFT dynamically adjusts difficulty during training:

```
Target difficulty: T (starts at medium)
Update rule: T' = clip(T + eta * tanh(alpha * (R_avg - beta)), d_min, d_max)
Where:
    R_avg = average reward over recent batch
    beta = 0.5 (target ~50% success)
    eta = learning rate for curriculum
    tanh provides smooth, bounded updates
```

Key insight: as the model improves, it automatically trains on harder problems. This reduces training time by up to 2x vs fixed difficulty.

---

## 2. Trace Quality Metrics

### 2.1 LIMO Quality Scoring (Best Published Formula)

LIMO's rule-based scoring system with four dimensions, all normalized by text length:

| Dimension | Weight | Measurement | Rationale |
|-----------|--------|-------------|-----------|
| Elaborated Reasoning | 30% | `min(len(solution) / 30000, 1.0)` | Harder problems need more steps |
| Self-Verification | 20% | Count of: "check", "verify", "confirm", "let's test" | Good solutions validate intermediate results |
| Exploratory Approach | 25% | Count of: "perhaps", "might", "alternatively", "another approach" | Exploring multiple paths = deeper reasoning |
| Adaptive Granularity | 25% | Count of: "therefore", "since", "hence", "thus", "because" | Logical flow indicators |

**LIMO pipeline**: Score all correct solutions per problem, rank by quality, keep top-1.

### 2.2 TIR-Specific Quality (ASTER's Key Insight)

For Tool-Integrated Reasoning traces, ASTER discovered that **interaction density** is the critical quality metric, not just correctness or length:

| Metric | Threshold | Evidence |
|--------|-----------|----------|
| Tool calls per trajectory | >= 9 | 4K dense trajectories beat 45K mixed |
| Tool execution success | 100% | No failed code = clean training signal |
| Multi-turn tool coordination | Required | Single verify-at-end = "interaction collapse" |

**Interaction collapse** is a pathological failure where the model degenerates into long internal reasoning followed by a single trivial code check. Preventing this requires cold-start SFT on interaction-dense trajectories.

**TIR quality checklist**:
- [ ] Uses Python/SymPy for symbolic computation (not just arithmetic)
- [ ] Multiple tool calls with iterative refinement (>= 3 calls ideal)
- [ ] Code actually performs computation (not just print statements)
- [ ] No infinite loops or timeouts
- [ ] Each code block builds on results of previous blocks
- [ ] Final answer derived from code output, not hallucinated

### 2.3 Anti-Signals (What Indicates Bad Quality)

| Signal | Threshold | Action |
|--------|-----------|--------|
| Solution length > 80K chars | Hard cutoff | Remove (stuck in loops) |
| Solution length < 2K chars | Soft filter | Remove unless problem is trivially short |
| Repetitive 100+ char substrings | Pattern match | Remove (degenerate generation) |
| > 15 tool calls | Soft filter | Likely spinning on failed approaches |
| 0 tool calls (in TIR dataset) | Hard cutoff | Not actually TIR |
| Mixed languages | Pattern match | Remove (DeepSeek-R1 artifact) |
| No \boxed{} answer | Hard cutoff | Can't verify correctness |

### 2.4 Process Reward Models (Step-Level Verification)

PRMs score individual reasoning steps, catching solutions that reach the right answer through flawed reasoning.

**Math-Shepherd** (ACL 2024):
- Automated step-level labels via Monte Carlo Tree Search
- No human annotation needed
- Mistral-7B: 77.9% -> 84.1% on GSM8K with PRM-guided PPO
- Can be used as a *quality filter*: reject solutions where any step scores < threshold

**OmegaPRM** (DeepMind, 2024):
- Divide-and-conquer MCTS to find first error in CoT
- 1.5M process supervision annotations
- Outperforms human annotations from PRM800K
- Gemini Pro: 51% -> 69.4% on MATH500

**Available PRMs for our use**:
- `Qwen2.5-Math-7B-PRM800K` on HuggingFace (step-level scorer)
- Math-Shepherd models (need to verify compatibility with our trace format)

**Caveat**: No PRM exists for gpt-oss-120b specifically. Using a mismatched PRM (e.g., Qwen-based) may introduce bias. Best used as a *soft signal* combined with other metrics, not as a hard filter.

---

## 3. Rank-Surprisal Ratio (RSR) -- Novel Metric

The RSR metric (Jan 2026) is the most promising new approach for trajectory selection. It achieved 0.86 Spearman correlation with post-training performance across 5 student models and 11 teachers.

### 3.1 Formula

```
RSR(trajectory) = sum(min(Rank(t_k), r_max)) / sum(Surprisal(t_k))

Where:
    Rank(t_k) = number of tokens with strictly higher probability under student model
    Surprisal(t_k) = -log p_student(t_k | context_k)
    r_max = 100 (clipping threshold)
```

### 3.2 Intuition

Good training trajectories are "informatively surprising":
- **Low absolute probability** (high surprisal): the student doesn't already know this
- **High relative rank**: but the tokens aren't completely alien to the student

This means: the trajectory teaches something new but still within the student's distribution. Compare to difficulty: RSR captures *trajectory-level* teachability, not just *problem-level* difficulty.

### 3.3 Comparison to Baselines

| Metric | Avg Spearman Correlation |
|--------|-------------------------|
| **RSR** | **0.86** |
| Token Length | 0.53 |
| Avg Surprisal (perplexity) | 0.49 |
| Random | ~0 |

### 3.4 Practical Usage

1. Take candidate trajectories (e.g., 10 per problem from different generators)
2. Run single forward pass through student model (gpt-oss-120b) on each trajectory
3. Compute RSR for each
4. Select trajectory with **lowest RSR** per problem (lower = more teachable)
5. Code available: https://github.com/UmeanNever/RankSurprisalRatio

**Cost**: One forward pass per trajectory through the student model. With vLLM batching on H100, this is fast (minutes for 10K trajectories).

**Critical advantage**: Works without teacher model access. Purely student-centric.

### 3.5 How to Integrate with Our Pipeline

RSR can replace or supplement LIMO's quality score at Stage 4:
```python
# Instead of rule-based quality scoring:
rsr_score = compute_rsr(trajectory, student_model)
# Select per-problem: trajectory with lowest RSR among correct solutions
```

This is more principled than keyword counting but requires a forward pass through the model.

---

## 4. Diversity Metrics

### 4.1 s1's Approach: MSC Classification + Balanced Sampling

s1 used Mathematics Subject Classification (MSC) codes from the American Mathematical Society:
1. Classify each problem into one of 50 mathematical domains using Claude 3.5 Sonnet
2. Iterative balanced sampling:
   - Randomly select a domain
   - Sample a problem weighted by reasoning trace length (longer = harder)
   - Repeat until 1,000 samples reached across all 50 domains
3. Ensure every domain has at least some representation

### 4.2 Skill-Aware Selection (2025)

More sophisticated than topic-level diversity:
1. Build a hierarchical skill tree (e.g., Mathematics -> Probability -> Bayes' theorem)
2. Map each problem to its required skill chain
3. Evaluate student model accuracy per leaf-level skill
4. Sample with probability inversely proportional to skill accuracy:
   ```
   P(skill) = clip(1/accuracy, 0, w_max) / sum(clipped)
   ```
5. Embed skill chain in training data: prepend "Skills: [Math -> Algebra -> Polynomial roots]" before solution

**Result**: +1.6% on Qwen3-4B, +1.4% on Qwen3-8B with just 1K examples.
**Implementation cost**: ~200 GPU-hours for labeling 100K problems (one-time).

### 4.3 Embedding-Based Diversity

For additional deduplication and diversity:
1. Compute embeddings of problem texts (sentence-transformers or similar)
2. Cluster with k-means or HDBSCAN
3. Within each cluster, keep highest-quality representative
4. Remove near-duplicates (cosine similarity > 0.95)
5. Ensure balanced sampling across clusters

### 4.4 D3: Unified Diversity-Difficulty-Dependability

D3 (IJCAI 2025) combines all three dimensions into a single scoring framework:
- **Diversity**: Measured by embedding distance from already-selected samples
- **Difficulty**: Measured by model performance on the problem
- **Dependability**: Measured by solution quality and reliability
- Selection: Greedy iterative process, picking highest-utility sample at each step

### 4.5 Minimum Coverage Requirements

Based on AIMO3 competition topics:

| Topic | Min Examples | Priority |
|-------|-------------|----------|
| Number Theory | 150+ | High (common in olympiad) |
| Algebra | 150+ | High |
| Combinatorics | 100+ | High |
| Geometry | 100+ | Medium (underrepresented in datasets) |
| Probability | 75+ | Medium |
| Calculus/Analysis | 50+ | Lower (less common in AIMO) |
| Other (logic, sets) | 50+ | Lower |

---

## 5. Contamination Detection

### 5.1 Standard N-Gram Approach

Following established practice (s1, Qwen):

```python
def decontaminate(train_problems, eval_problems, n=9):
    """Remove training problems that overlap with evaluation sets."""
    eval_ngrams = set()
    for p in eval_problems:
        text = normalize(p)  # lowercase, strip LaTeX, remove whitespace
        for i in range(len(text) - n + 1):
            eval_ngrams.add(text[i:i+n])

    clean = []
    for p in train_problems:
        text = normalize(p)
        train_ngrams = set(text[i:i+n] for i in range(len(text) - n + 1))
        overlap = len(train_ngrams & eval_ngrams) / max(len(train_ngrams), 1)
        if overlap < 0.1:  # less than 10% overlap
            clean.append(p)
    return clean
```

### 5.2 Evaluation Sets to Exclude

Must decontaminate against:
- **AIME 2023, 2024, 2025** (likely in AIMO3 test set)
- **MATH500** (standard evaluation)
- **GPQA Diamond** (if used for eval)
- **AIMO3 competition problems** (from the actual Kaggle competition)
- **HMMT 2025** (emerging benchmark)

### 5.3 Beyond N-Grams

Recent research (2025-2026) shows n-gram matching is insufficient:
- Paraphrased problems bypass string matching
- Translated problems (Chinese -> English) retain same mathematical content
- **Recommendation**: Supplement n-grams with embedding similarity check (cosine > 0.90 = suspicious)

### 5.4 Deduplication

Following s1's 8-gram deduplication:
```python
# Within training data: remove near-duplicate problems
# 1. Normalize text (lowercase, strip LaTeX/whitespace)
# 2. Compute 8-gram set for each problem
# 3. Jaccard similarity > 0.5 = probable duplicate
# 4. Keep higher-quality instance of each duplicate pair
```

---

## 6. Concrete Curation Pipelines (How Winners Did It)

### 6.1 s1: From 59K to 1K (26 minutes training -> beat o1-preview)

```
Stage 1: Quality filter
    59,029 -> 54,116 (remove API errors)
    -> 51,581 (remove formatting issues: ASCII art, broken images, bad numbering)
    384 high-confidence samples set aside

Stage 2: Difficulty filter (dual-model gating)
    Test with Qwen-7B and Qwen-32B-Instruct
    Remove if EITHER model solves correctly (too easy)
    51,581 -> 24,496

Stage 3: 8-gram deduplication + decontamination
    Remove overlap with MATH500, GPQA Diamond, AIME24

Stage 4: Diversity selection (MSC classification)
    Classify into 50 math domains via Claude 3.5 Sonnet
    Iterative balanced sampling favoring longer traces
    24,496 -> 1,000

Key detail: 53.6% of final samples were rated correct by their grader.
They explicitly accepted incorrect traces, prioritizing "capturing the
reasoning process rather than entirely correct solutions."
```

**Ablation results (Table 2 of s1 paper)**:
| Selection | AIME24 |
|-----------|--------|
| s1K (full pipeline) | Best |
| 1K-random (quality only) | -26.7% to -3.3% |
| 1K-diverse (diversity only) | -40.0% to -10.0% |
| 1K-longest (difficulty only) | -36.7% to 0% |
| Full 59K | Comparable to s1K with 56x more data |

**Lesson**: All three criteria (difficulty, diversity, quality) matter. Any single criterion alone is significantly worse.

### 6.2 LIMO: From Millions to 817 (57.1% AIME, 94.8% MATH)

```
Stage 1: Coarse difficulty filter
    Use Qwen2.5-Math-7B-Instruct
    Remove problems solved in <= 4 attempts
    Millions -> tens of thousands

Stage 2: Fine-grained difficulty
    Use DeepSeek-R1-Distill-Qwen-32B
    Generate 32 attempts per problem
    Keep only: 1-3 correct out of 32 (pass_rate 3-9%)
    -> 2,125 problems (LIMO-Pool)

Stage 3: Solution quality scoring
    For each problem in LIMO-Pool:
        Score all correct solutions using 4-dim quality metric
        Keep top-1 solution per problem
    Rank by quality score
    Take top 800

Stage 4: Knowledge diversification
    Ensure coverage across: NuminaMath-CoT, DeepScaleR, AIME, MATH, Chinese materials
```

**Ablation on dataset size**:
| Size | AIME24 | MATH500 |
|------|--------|---------|
| 400 | 57.5% | 94.8% |
| 800 | 63.3% | 95.6% |
| 1,200 | 64.2% | 95.8% |
| 1,600 | Diminishing | Diminishing |
| 2,000 | Marginal | Marginal |

**Lesson**: Diminishing returns after ~800 for 32B dense model. MoE models (like gpt-oss-120b) may benefit from slightly more data (2K-4K).

### 6.3 ASTER: 4K TIR Cold-Start (90% AIME 2025)

```
Stage 1: Trajectory synthesis
    Model: GPT-OSS-20B-high
    Problems: Skywork-OR1-RL-Data + 93K from AoPS (OpenMathReasoning)
    Filter: English, positive integer answers, correct final answer
    -> 45K initial pool

Stage 2: Quality gate
    Require: 100% tool execution success rate (no code errors)

Stage 3: Interaction density stratification
    Bin by number of tool calls per trajectory:
    - <= 1 call: 22K trajectories (weakest)
    - <= 5 calls: 37K (intermediate)
    - >= 9 calls: 4K (strongest!)

Stage 4: Select >= 9 tool calls subset
    4K interaction-dense trajectories
    This 4K subset OUTPERFORMS the full 45K dataset

Stage 5: Two-stage RL (GRPO)
    Stage 1: 18K context, max 50 tool calls
    Stage 2: 32K context, remove solved problems
```

**Key result**: 4K interaction-dense trajectories > 45K mixed > 22K sparse. Interaction density is the critical quality metric for TIR.

### 6.4 NVIDIA OpenMathReasoning: 3-Stage Industrial Pipeline

```
Stage 1: Problem curation
    540K unique problems from AoPS forums
    Preprocessed with Qwen2.5-32B-Instruct

Stage 2: Solution generation
    3.2M CoT solutions (DeepSeek-R1 + QwQ-32B)
    1.7M TIR solutions (iterative generation + quality filtering)
    Iterative: generate -> filter correct -> retrain -> repeat

Stage 3: GenSelect training
    566K samples for training a solution-selection model
    Model learns to pick the best answer from candidates
    Significantly outperforms majority voting

Won AIMO-2 with 34/50. The key was iterative curation: generate, filter, retrain.
```

---

## 7. Importance-Weighted SFT (iw-SFT)

### 7.1 Theoretical Foundation

The paper "SFT on Curated Data is RL" (arxiv 2507.12856) proves that:
- Standard SFT on filtered data optimizes a *lower bound* on the RL objective
- This bound becomes looser as the model diverges from the reference policy
- **iw-SFT tightens this bound** by reweighting the loss

### 7.2 Formula

```python
# Standard SFT loss:
loss = -log p(trajectory | theta)

# iw-SFT loss:
w = q(trajectory) / pi_ref(trajectory)  # importance weight
loss = -w * log p(trajectory | theta)

# Where:
#   q = auxiliary distribution (lagged training policy)
#   pi_ref = reference/base model policy
```

### 7.3 Practical Implementation

Three minimal changes to standard SFT:
1. Maintain a reference model `pi_ref` (frozen copy of base model)
2. Compute per-token log-probability differences: `rho_i = log q(a_i|s_i) - log pi_ref(a_i|s_i)`
3. Apply smoothing: `weight = 0.1 * clip(exp(rho), 0.2, 1.8)` to prevent weight explosion
4. Multiply loss by weight

### 7.4 Results

| Benchmark | Standard SFT | iw-SFT | Improvement |
|-----------|-------------|--------|-------------|
| AIME 2024 | 56.7% | **66.7%** | +10.0% |
| MATH500 | 94.4% | 94.8% | +0.4% |
| GPQA Diamond | 60.6% | 64.1% | +3.5% |

**+10% on AIME is massive** with zero additional data. This is essentially free improvement.

### 7.5 Recommendation for Our Pipeline

If using quality scores (from Stage 4), integrate them into iw-SFT:
- Bin quality scores into 2-3 ordinal classes (low/medium/high)
- Sample proportional to quality during training
- Apply importance weighting on top
- Use sequence-level weighting (not per-token) for best results

---

## 8. Automated Quality Scoring: Proposed Implementation

### 8.1 Combined Scoring Function

Synthesizing all research into a concrete, implementable scorer:

```python
import re
import math

def score_trajectory(solution_text, n_tool_calls, pass_rate, problem_topic=None):
    """
    Score a math reasoning trajectory for training quality.
    Returns score in [0, 1] where higher = better.

    Inputs:
        solution_text: full solution string
        n_tool_calls: number of code execution blocks
        pass_rate: fraction of attempts that solved this problem correctly
        problem_topic: optional topic label for diversity tracking
    """
    scores = {}

    # --- Difficulty component (25%) ---
    # Sweet spot: pass_rate 0.03-0.15 for SFT
    if pass_rate <= 0.0:
        scores['difficulty'] = 0.0  # Impossible = useless
    elif pass_rate <= 0.03:
        scores['difficulty'] = 0.3  # Very hard, risky
    elif pass_rate <= 0.15:
        scores['difficulty'] = 1.0  # Sweet spot (LIMO range)
    elif pass_rate <= 0.30:
        scores['difficulty'] = 0.7  # Good (s1 range)
    elif pass_rate <= 0.50:
        scores['difficulty'] = 0.4  # Medium
    else:
        scores['difficulty'] = 0.1  # Too easy

    # --- Solution quality component (35%) ---
    length = len(solution_text)

    # Length sub-score (8%)
    length_score = min(length / 30000, 1.0)
    if length > 80000:
        length_score = 0.0  # Stuck in loops

    # Verification keywords (9%)
    verify_words = ['check', 'verify', 'confirm', "let's test", 'validate', 'double-check']
    verify_count = sum(solution_text.lower().count(w) for w in verify_words)
    verify_score = min(verify_count / max(length / 5000, 1), 1.0)

    # Exploration keywords (9%)
    explore_words = ['perhaps', 'might', 'alternatively', 'another approach', 'let me try', 'consider']
    explore_count = sum(solution_text.lower().count(w) for w in explore_words)
    explore_score = min(explore_count / max(length / 5000, 1), 1.0)

    # Logical connectives (9%)
    logic_words = ['therefore', 'since', 'hence', 'thus', 'because', 'it follows', 'we conclude']
    logic_count = sum(solution_text.lower().count(w) for w in logic_words)
    logic_score = min(logic_count / max(length / 3000, 1), 1.0)

    scores['quality'] = (
        0.23 * length_score +
        0.26 * verify_score +
        0.26 * explore_score +
        0.25 * logic_score
    )

    # --- Interaction density component (20%) ---
    # ASTER finding: >= 9 tool calls is optimal
    if n_tool_calls == 0:
        scores['interaction'] = 0.0
    elif n_tool_calls <= 2:
        scores['interaction'] = 0.3
    elif n_tool_calls <= 5:
        scores['interaction'] = 0.6
    elif n_tool_calls <= 9:
        scores['interaction'] = 0.9
    elif n_tool_calls <= 15:
        scores['interaction'] = 1.0
    else:
        scores['interaction'] = 0.4  # Too many = spinning

    # --- Structural quality component (20%) ---
    # Step numbering
    has_steps = bool(re.search(r'step\s*\d|^\d+[\.\)]', solution_text, re.MULTILINE | re.IGNORECASE))
    # Boxed answer
    has_boxed = '\\boxed{' in solution_text
    # Code quality: uses sympy
    uses_sympy = 'sympy' in solution_text.lower() or 'from sympy' in solution_text
    # No repetition
    has_repetition = bool(re.search(r'(.{100,})\1', solution_text))

    scores['structure'] = (
        0.20 * float(has_steps) +
        0.30 * float(has_boxed) +
        0.30 * float(uses_sympy) +
        0.20 * float(not has_repetition)
    )

    # --- Final weighted score ---
    final = (
        0.25 * scores['difficulty'] +
        0.35 * scores['quality'] +
        0.20 * scores['interaction'] +
        0.20 * scores['structure']
    )

    return final, scores
```

### 8.2 Hard Filters (Apply Before Scoring)

```python
def passes_hard_filters(solution_text, n_tool_calls, answer_correct):
    """Binary filters - must pass ALL to be considered."""
    if not answer_correct:
        return False
    if len(solution_text) < 2000:
        return False  # Too short
    if len(solution_text) > 80000:
        return False  # Stuck in loops
    if '\\boxed{' not in solution_text:
        return False  # No extractable answer
    if n_tool_calls < 1:
        return False  # Not TIR
    if n_tool_calls > 15:
        return False  # Spinning
    # Repetition check
    if re.search(r'(.{100,})\1', solution_text):
        return False
    return True
```

### 8.3 Expected Yield Estimates

Starting from our available datasets:

| Dataset | Raw | After Hard Filters | After Scoring (top 50%) | Final Target |
|---------|-----|-------------------|------------------------|--------------|
| AIMO3 TIR (141K) | 141,277 | ~100K (70%) | ~50K | 2K-3K |
| AIMO3 Hard (7.3K unique problems, ~70K traces) | 70,000 | ~50K (70%) | ~25K | 1K-2K |
| Combined | 211,277 | ~150K | ~75K | 2K-4K |

After difficulty filtering (pass_rate 0.03-0.30), expect ~20-40% retention.
After quality scoring + diversity selection, target 2K-4K final examples.

---

## 9. Complete Recommended Pipeline

### Step 1: Download and Merge Datasets
```bash
kaggle datasets download jeannkouagou/aimo3-tool-integrated-reasoning
kaggle datasets download wenliangtlh/aimo3-high-difficulty-tool-calling-dataset
```

### Step 2: Hard Filtering (~70% retention)
- Correct answer verified
- Length 2K-80K chars
- 1-15 tool calls
- No repetitive patterns
- Has \boxed{} answer

### Step 3: Decontamination (<5% removal)
- 9-gram overlap vs AIME 2023-2025, MATH500, AIMO3 problems
- Embedding similarity check (cosine > 0.90 = remove)
- 8-gram deduplication within training set

### Step 4: Difficulty Calibration (~30-40% retention)
- Use pass_rate from AIMO3-Hard metadata where available
- For AIMO3-TIR: estimate from answer distribution across solutions for same problem
- Keep: pass_rate 0.03-0.30 for SFT
- Apply DART Prop2Diff weighting: oversample harder problems

### Step 5: Quality Scoring
- Apply combined scoring function (Section 8.1)
- Keep top solution per problem (LIMO approach)
- OR keep top-3 diverse solutions per problem (if dataset is too small)

### Step 6: Diversity Selection
- Classify problems by topic (use LLM or existing labels)
- Ensure minimum coverage per topic (see Section 4.5)
- Iterative balanced sampling weighted by difficulty * quality
- Target: 2K-4K final examples

### Step 7: Training Enhancement (Optional but Recommended)
- **iw-SFT**: Apply importance weighting during training (+10% AIME for free)
- **RSR**: If compute available, score with RSR and select lowest-RSR trajectories
- **Skill embedding**: Prepend skill chain to each training example

---

## 10. Open Questions and Recommendations

### Answered Questions
1. **How many examples?** 2K-4K for gpt-oss-120b (MoE may need slightly more than 32B dense)
2. **What difficulty?** Pass rate 0.03-0.30 for SFT, 0.10-0.70 for RL
3. **What quality metric?** Combined scorer (Section 8.1), or RSR if compute permits
4. **How to ensure diversity?** Topic classification + balanced sampling + embedding dedup
5. **How to decontaminate?** 9-gram + embedding similarity

### Remaining Questions
1. **MoE data appetite**: LIMO/s1 proven for 32B dense. gpt-oss-120b is 117B MoE with 5.1B active params. May need more data for the routing mechanism, or may be fine with 2K since only 5.1B params are active.
2. **PRM mismatch**: No PRM for gpt-oss-120b. Using Qwen-based PRM as a soft filter is worth trying but may introduce bias.
3. **RSR compute cost**: Needs forward pass through gpt-oss-120b for each candidate trajectory. With vLLM on H100, ~10K trajectories in minutes -- worth trying if we have H100 access.
4. **iw-SFT with Unsloth**: Need to verify importance weighting is compatible with QLoRA. May need custom training loop.
5. **Interaction density threshold for gpt-oss-120b**: ASTER found >= 9 optimal for GPT-OSS-20B. Larger model may have different sweet spot.

---

## 11. Key Papers Reference

| Paper | Key Contribution | Year | Link |
|-------|-----------------|------|------|
| LIMO | 817 curated >> 100K random; quality scoring formula | 2025 | https://arxiv.org/abs/2502.03387 |
| s1 | 1K examples beat o1; difficulty+diversity+quality | 2025 | https://arxiv.org/abs/2501.19393 |
| DART-Math | Difficulty-aware rejection tuning; Prop2Diff | 2024 | https://arxiv.org/abs/2407.13690 |
| ASTER | 4K interaction-dense TIR -> 90% AIME 2025 | 2026 | https://arxiv.org/abs/2602.01204 |
| RSR | 0.86 correlation metric for trajectory selection | 2026 | https://arxiv.org/abs/2601.14249 |
| AdaRFT | Dynamic difficulty curriculum for RL | 2025 | https://arxiv.org/abs/2504.05520 |
| iw-SFT | Importance-weighted SFT = +10% AIME | 2025 | https://arxiv.org/abs/2507.12856 |
| Math-Shepherd | Automated step-level PRM | 2024 | https://arxiv.org/abs/2312.08935 |
| OmegaPRM | MCTS-based process supervision | 2024 | https://arxiv.org/abs/2406.06592 |
| Skill-Aware | Hierarchical skill tree + weak-skill oversampling | 2025 | https://arxiv.org/abs/2601.10109 |
| D3 | Unified diversity-difficulty-dependability selection | 2025 | https://arxiv.org/abs/2503.11441 |
| Front-Loading | SFT quality >> quantity; -5% from doubling data | 2025 | https://arxiv.org/abs/2510.03264 |
| OpenMathReasoning | 3-stage pipeline won AIMO-2 (34/50) | 2025 | https://arxiv.org/abs/2504.16891 |
| AceMath | Cross-model verification + reward models | 2024 | https://arxiv.org/abs/2412.15084 |
