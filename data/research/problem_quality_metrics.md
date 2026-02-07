# Problem Quality Metrics for Fine-Tuning

## Overview

When curating math problems for fine-tuning, we evaluate them on 4 dimensions:

---

## 1. Difficulty Metrics

| Metric | Description | Ideal Range |
|--------|-------------|-------------|
| `n_tool_calls` | Number of Python tool calls in solution | 2-8 (too few = trivial, too many = stuck) |
| `completion_length` | Total solution length in characters | 5k-30k (substantial but not excessive) |
| `math_complexity` | Count of advanced math keywords | 1-5 |
| `answer_magnitude` | Number of digits in answer | 1-5 (AIME-style) |

**Reasoning:** Problems with 2-8 tool calls tend to be appropriately challenging. Too few means the model can solve it without computation; too many may indicate the model is spinning.

---

## 2. Diversity Metrics

### Topic Classification
| Topic | Keywords |
|-------|----------|
| `number_theory` | prime, divisor, modulo, gcd, remainder |
| `algebra` | equation, polynomial, root, coefficient |
| `geometry` | triangle, circle, angle, area, perimeter |
| `combinatorics` | count, permutation, combination, ways |
| `probability` | probability, random, expected, dice |
| `calculus` | integral, derivative, limit, continuous |

### Problem Type
| Type | Keywords |
|------|----------|
| `computation` | find, calculate, determine, evaluate |
| `counting` | how many, number of |
| `proof` | prove, show that, demonstrate |
| `construction` | construct, give an example |

**Goal:** Balance topic coverage. Underrepresented topics (geometry, probability) get bonus points.

---

## 3. Quality Metrics

| Metric | Description | Weight |
|--------|-------------|--------|
| `has_boxed_answer` | Solution ends with \boxed{} | Required |
| `uses_sympy` | Uses symbolic math | +10 pts |
| `uses_numpy` | Uses numerical methods | +5 pts |
| `has_step_structure` | Clear step-by-step reasoning | +10 pts |
| `n_reasoning_sentences` | Count of explanation sentences | Higher is better |

**Key Insight:** Solutions using `sympy` tend to be more mathematically rigorous. Step-by-step structure aids learning.

---

## 4. Novelty Metrics

| Metric | Description |
|--------|-------------|
| `problem_hash` | MD5 of normalized problem text |
| `near_duplicates` | Problems with same hash |

**Deduplication:** Normalize text (lowercase, remove LaTeX formatting), then hash. Keep highest-quality version of duplicates.

---

## Composite Quality Score (0-100)

```python
score = 0

# Difficulty (0-25 pts)
score += min(n_tool_calls / 6, 1) * 15      # Sweet spot ~6 calls
score += min(completion_length / 30000, 1) * 10

# Quality (0-35 pts)
score += 15 if has_boxed_answer else 0
score += 10 if uses_sympy else 0
score += 10 if has_step_structure else 0

# Topic diversity (0-20 pts)
score += 15 if topic in ['geometry', 'probability', 'calculus'] else 10

# Problem type (0-20 pts)
score += 20 if problem_type == 'computation' else 10
```

---

## Recommended Filter Criteria

For high-quality fine-tuning data:

```python
filtered = df[
    (has_boxed_answer == True) &      # Must have proper answer
    (n_tool_calls >= 1) &              # Must use tools
    (n_tool_calls <= 15) &             # Not stuck in loop
    (completion_length >= 2000) &      # Substantial reasoning
    (completion_length <= 80000) &     # Not excessively long
    (quality_score >= 50)              # Minimum quality bar
]
```

Expected retention: ~60-70% of original dataset.

---

## Stratified Sampling

For topic balance:
```python
balanced = []
for topic in topics:
    balanced.append(df[df.topic == topic].nlargest(N, 'quality_score'))
```

This ensures underrepresented topics aren't drowned out.

---

## Files

- `kaggle_submissions/dataset_curation/kaggle_submission.ipynb` - Full curation notebook
- `data/fine_tuning_datasets.md` - Dataset sources and download commands
