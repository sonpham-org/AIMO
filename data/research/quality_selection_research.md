# How to Select High-Quality Training Samples for Math LLMs

> Last updated: 2026-02-07
> Purpose: Evidence-based guide to data selection for fine-tuning on math reasoning

---

## The Core Insight

**Quality >> Quantity**. This is proven across multiple papers:
- LIMO: **817 examples → 57.1% AIME** (vs NuminaMath-100K → 32.3%)
- s1: **1,000 examples → beat o1-preview** on MATH and AIME24
- NVIDIA Front-Loading: "doubling mixed-quality SFT data hurt math by -5%"
- ASTER: **4K TIR trajectories → 90% AIME 2025**

The question is not "how much data" but "which data."

---

## Top Quality Metrics (Ranked by Evidence Strength)

### Tier 1: Strongest Evidence (Must Use)

#### 1. Correctness Verification
Every approach starts here. Filter to solutions where the final answer matches ground truth.

**Best method**: Cross-model verification
- Generate answer with Model A and Model B independently
- Keep only where both agree with ground truth
- AceMath reduced 2.3M → 800K samples, same benchmark performance

**Minimum method**: Direct answer check via SymPy

#### 2. Difficulty (Pass Rate)
**The single most impactful metric after correctness.**

| Paper | Evidence |
|-------|----------|
| DART-Math (NeurIPS 2024) | Standard rejection tuning biased 90% toward easy problems; Prop2Diff +4.5 pts |
| LIMO (COLM 2025) | Kept only pass_rate 3-9% → 63.3% AIME with 800 examples |
| s1 (2025) | Removed all problems solvable by weaker model → beat o1-preview |
| Online Difficulty Filtering (2025) | Theoretical proof: intermediate difficulty maximizes learning |

**How to compute**: Generate N solutions per problem, count correct ones.
```
pass_rate = n_correct / n_total
```

**Sweet spots by use case**:
- SFT: pass_rate 0.03-0.15 (very hard, LIMO-style)
- RL: pass_rate 0.1-0.7 (Goldilocks zone, maximizes gradient signal)
- General: Oversample hard, undersample easy (DART Prop2Diff)

#### 3. Diversity (Topic + Method Coverage)
Every high-performing small-data approach optimizes for diversity.

| Paper | Method |
|-------|--------|
| s1 | Classified into 50 domains via MSC codes (Claude 3.5 Sonnet), iterative balanced sampling |
| LIMO | Multiple sources (NuminaMath, DeepScaleR, AIME, MATH, Chinese) |
| Skill-Aware Selection (2025) | Hierarchical skill tree, oversample weakest skills (+1.6% with 1K examples) |

---

### Tier 2: Strong Evidence

#### 4. Solution Quality Score
LIMO's scoring system (most detailed published):

| Dimension | Weight | What to Look For |
|-----------|--------|-----------------|
| Elaborated Reasoning | 30% | Solution length (normalized to 30K max) |
| Self-Verification | 20% | Keywords: "check", "verify", "confirm", "let's test" |
| Exploratory Approach | 25% | Keywords: "perhaps", "might", "alternatively", "another approach" |
| Adaptive Granularity | 25% | Connectives: "therefore", "since", "hence", "thus" |

**Anti-signals**:
- Solutions >2,500 words (for CoT) usually verbose/incorrect (AceMath)
- Repetitive token patterns = always low quality
- Mixed languages in reasoning (DeepSeek-R1 lesson)

#### 5. Cross-Model Answer Agreement
AceMath's strongest filter. Generates solutions with two independent models, keeps only where both agree. Catches "lucky correct" solutions where reasoning is wrong but answer is right.

#### 6. Reasoning Trace Length (as Difficulty Proxy)
s1 weighted toward longer traces during selection. LIMO used length as 30% of quality score. Harder problems genuinely need more steps. But beware: >80K chars usually means the model got stuck in loops.

---

### Tier 3: Emerging / Promising

#### 7. Influence Functions
Most influential tokens in math CoTs are logical connectors: "Wait", "However", "Verify", "Hence", "First", "Therefore", "Alternatively". High-difficulty math examples improve both math AND code reasoning.
- Paper: https://arxiv.org/abs/2510.06108

#### 8. Rank-Surprisal Ratio (RSR)
Effective training trajectories combine low absolute probability with high relative rank under the student model. RSR achieved **0.86 Spearman correlation** with post-training performance. Good samples are "informatively surprising."
- Paper: https://arxiv.org/abs/2601.14249

#### 9. Entropy/Confidence of Generator
For training data: medium entropy may be ideal. Low entropy = model already knows how to solve it (too easy). Very high entropy = model is guessing.

#### 10. Process Reward Model (PRM) Scores
Step-level quality scores catch solutions that reach right answer through flawed reasoning.
- Math-Shepherd: automated step-level labels without human annotation
- OmegaPRM: outperforms human annotations (PRM800K)
- Available: Qwen2.5-Math-7B-PRM800K on HuggingFace

---

## Practical Filtering Pipeline

### Stage 0: Decontamination
```
- 9-gram overlap check against AIME 2023-2025, MATH500, GPQA Diamond
- 8-gram deduplication (following s1)
- Normalize text (lowercase, strip LaTeX), MD5 hash, remove near-dupes
```

### Stage 1: Basic Quality Filters (Cheap)
```python
# Remove obvious garbage
filtered = dataset.filter(
    has_boxed_answer == True,               # Must have extractable answer
    completion_length >= 2000,              # Not trivially short
    completion_length <= 80000,             # Not stuck in loops
    no_repetitive_patterns,                 # No 100+ char repeated substrings
    n_tool_calls >= 1 and n_tool_calls <= 15,  # For TIR: uses tools but not spinning
)
# Expected retention: ~70-80%
```

### Stage 2: Correctness Verification (Most Impactful)
```python
# Keep only solutions matching ground truth
# Best: cross-model verification
# Minimum: direct answer check via SymPy / math_verify
verified = filtered.filter(extracted_answer == ground_truth)
# Expected retention: ~30-50% of Stage 1
```

### Stage 3: Difficulty Calibration (Key Differentiator)
```python
# For each problem, compute pass_rate from N attempts
# Then apply difficulty-aware sampling:

# Option A (LIMO-style, extreme): keep only pass_rate 0.03-0.15
# Option B (s1-style): remove anything solvable by weaker model
# Option C (DART Prop2Diff): keep all, oversample proportional to difficulty
# Option D (RL Goldilocks): keep pass_rate 0.1-0.7

# Recommended for SFT: Option C
# Recommended for RL: Option D
```

### Stage 4: Quality Scoring (Select Best Per Problem)
```python
def quality_score(solution):
    score = 0.0
    score += 0.25 * min(len(solution) / 30000, 1.0)    # Elaborated reasoning
    score += 0.20 * verification_keywords(solution)      # "check", "verify", "confirm"
    score += 0.20 * exploration_keywords(solution)        # "alternatively", "another approach"
    score += 0.15 * logical_connectives(solution)         # "therefore", "hence", "since"
    score += 0.10 * code_quality(solution)                # For TIR: clean sympy usage
    score += 0.10 * step_structure(solution)              # Numbered steps, clear structure
    return score

# Keep top-1 solution per problem (LIMO approach)
# OR top-K diverse solutions (NVIDIA approach)
```

### Stage 5: Diversity Selection (Final Curation)
```python
# 1. Classify each problem by topic (number_theory, algebra, geometry, etc.)
# 2. Build skill taxonomy (hierarchical)
# 3. Iterative balanced sampling:
#    - Pick topic uniformly at random
#    - Sample problem weighted by difficulty × quality × length
#    - Repeat until target size reached
# 4. Ensure minimum representation per topic (≥50 problems each)
```

### Stage 6: Final Deduplication
```
- Compute embeddings of problem texts
- Cluster (HDBSCAN or k-means)
- Within each cluster, keep highest-quality representative
- Remove remaining near-duplicates (cosine similarity > 0.95)
```

---

## Target Numbers

| Metric | Target | Justification |
|--------|--------|---------------|
| SFT dataset size | 1K-5K curated examples | LIMO (800), s1 (1K), ASTER (4K) |
| Difficulty distribution | 60% hard, 30% medium, 10% easy | DART-Math Prop2Diff |
| Topic coverage | ≥50 problems per topic | s1 (50 domains) |
| Max solution length | 30K tokens | AceMath finding |
| Quality score threshold | Top 50% of correct solutions | LIMO approach |

---

## Critical Anti-Patterns (What NOT to Do)

1. **Don't use all correct traces** — Quality >> quantity. Doubling mixed-quality data hurts.
2. **Don't bias toward easy problems** — DART showed standard rejection tuning is 90% easy problems.
3. **Don't keep "lucky correct" solutions** — Use cross-model verification or PRM to catch wrong reasoning.
4. **Don't ignore topic balance** — Geometry and probability are underrepresented, need explicit oversampling.
5. **Don't train on brute-force solutions** — Prefer algebraic/analytical approaches with code verification.
6. **Don't skip decontamination** — Must exclude AIME 2023-2025 evaluation problems.
7. **Don't use excessively long solutions** — >80K chars usually means stuck/looping.

---

## Key Papers (Read These)

| Paper | Key Insight | Link |
|-------|-------------|------|
| LIMO | 800 curated >> 100K random | https://arxiv.org/abs/2502.03387 |
| s1 | 1K examples + budget forcing beats o1 | https://arxiv.org/abs/2501.19393 |
| DART-Math | Difficulty-aware rejection tuning | https://arxiv.org/abs/2407.13690 |
| OpenMathReasoning | 3-stage training won AIMO-2 | https://arxiv.org/abs/2504.16891 |
| AceMath | Cross-model verification + reward models | https://arxiv.org/abs/2412.15084 |
| SFT on Curated Data is RL | Theoretical connection | https://arxiv.org/abs/2507.12856 |
| Front-Loading Reasoning | SFT quality > quantity (+15% vs -5%) | https://arxiv.org/abs/2510.03264 |
| Online Difficulty Filtering | Intermediate difficulty maximizes learning | https://arxiv.org/abs/2504.03380 |
| GRPO-LEAD | Difficulty-aware advantage reweighting | https://arxiv.org/abs/2504.09696 |
| Which Trajectories Teach Better | RSR metric, 0.86 correlation | https://arxiv.org/abs/2601.14249 |
| ASTER | 4K TIR cold-start → 90% AIME 2025 | https://arxiv.org/html/2602.01204 |

---

## Open Questions

1. **Optimal dataset size for 120B MoE** — LIMO/s1 proven for 32B dense, unknown for MoE
2. **TIR-specific quality metrics** — Most papers focus on CoT; ideal #tool_calls unknown
3. **PRM for our model** — No PRM trained for gpt-oss-120b; using mismatched PRM may hurt
4. **Transfer across difficulty levels** — Does training on AIME improve IMO performance?
5. **SFT-RL interaction** — How does SFT data quality affect subsequent RL training?
