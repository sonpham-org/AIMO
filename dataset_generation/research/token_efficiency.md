# Token-Efficient Math Reasoning: Formats, Distillation & Compression Techniques

> Last updated: 2026-02-07
> Author: researcher-2
> Purpose: Comprehensive guide to reducing reasoning token counts while maintaining (or improving) math accuracy for gpt-oss-120b fine-tuning

---

## Executive Summary

Our gpt-oss-120b model runs on 1x H100 with a 9-hour limit for 50 problems. Every token saved per solution translates directly into more attempts per problem, which our entropy-gated consensus system can leverage for better accuracy. The research is clear: **shorter correct solutions consistently outperform longer ones**, and multiple techniques exist to achieve 30-60% token reduction without accuracy loss.

### Key Recommendations (Ranked by Practicality for Our Setup)

| Priority | Technique | Token Reduction | Effort | Accuracy Impact |
|----------|-----------|-----------------|--------|-----------------|
| 1 | SFT + DPO length optimization | 30% | Medium | Neutral to +2% |
| 2 | Difficulty-aware trace curation | 30% | Low | Neutral |
| 3 | Length reward in RL (GRPO) | 33-40% | Medium | Neutral to +14% |
| 4 | TokenSkip compression training | 40% | Medium | -0.4% |
| 5 | MCoT (Markov Chain of Thought) | 47% (1.9x speed) | High | Slight drop on hard problems |
| 6 | Latent reasoning (Coconut/CODI) | 60%+ | Very High | Experimental |

---

## 1. SFT + DPO for Shorter Outputs (AIMO2 2nd Place Approach)

### 1.1 How Imagination Team Did It

The AIMO2 2nd place team (imagination-research, 31/50) used a two-stage approach:

**Stage 1: SFT** on high-quality reasoning traces
- Dataset: Light-R1 stage2 data + LIMO dataset
- Both contain hard math problems with DeepSeek-R1 reasoning trajectories
- Training: 8 epochs, 8x A800 GPUs, 11 hours

**Stage 2: DPO** specifically to reduce output length
- Source: OpenR1-Math-220k (default subset)
- DPO pair construction criteria:
  1. **Correctness**: Chosen response must be correct
  2. **Min Length**: Enforce minimum threshold (avoid degenerate short answers)
  3. **Length Ratio**: `len(chosen) < ratio_threshold * len(rejected)`
  4. **Similarity filter**: Sentence transformer embeddings to remove near-duplicates
- Result: Shorter outputs with preserved accuracy

**Inference**: lmdeploy with W4KV8 quantization (4-bit weights, 8-bit KV cache)
- W4KV8 decreases time per output token by ~20% vs W4KV16, ~55% vs FP16
- 15 samples per question with dynamic speed adjustment

### 1.2 Constructing DPO Pairs for Length Reduction

The key insight: DPO can teach the model to prefer shorter correct solutions over longer ones.

```python
def construct_length_dpo_pairs(solutions_per_problem, ratio_threshold=0.7, min_length=500):
    """
    Build DPO pairs where chosen = shorter correct, rejected = longer correct.
    """
    pairs = []
    for problem_id, solutions in solutions_per_problem.items():
        correct = [s for s in solutions if s['is_correct']]
        correct.sort(key=lambda s: len(s['text']))

        for i, short in enumerate(correct):
            for long in correct[i+1:]:
                if len(short['text']) >= min_length and \
                   len(short['text']) < ratio_threshold * len(long['text']):
                    pairs.append({
                        'prompt': solutions[0]['question'],
                        'chosen': short['text'],
                        'rejected': long['text']
                    })
    return pairs
```

### 1.3 Why This Works

- SFT captures reasoning patterns and format
- DPO then optimizes for conciseness within that format
- The model learns: "if two solutions are both correct, prefer the shorter one"
- This does NOT teach the model to skip steps -- it teaches efficient expression

**Reference**: [Imagination AIMO2 solution](https://github.com/imagination-research/aimo2)

---

## 2. Difficulty-Aware Trace Curation

### 2.1 "Less is More Tokens" (DA-CoTD)

Paper: "Less is More Tokens: Efficient Math Reasoning via Difficulty-Aware Chain-of-Thought Distillation" (Sep 2025, arxiv:2509.05226)

Core idea: Train on traces whose length is **proportional to problem difficulty**. Easy problems get short traces, hard problems get long traces. The model learns "think proportionally."

**Method**:
1. Classify problems by difficulty (e.g., using pass rate)
2. For easy problems: select the shortest correct solution
3. For hard problems: select solutions with appropriate depth
4. SFT on this difficulty-calibrated dataset
5. Optional DPO stage to further reinforce length-difficulty alignment

**Results**: Up to 30% reduction in reasoning length without sacrificing accuracy across math benchmarks.

### 2.2 Practical Implementation for Our Pipeline

```python
def select_difficulty_aware_traces(problems_with_solutions):
    """
    Select traces with length proportional to difficulty.

    For each problem:
    - Easy (pass_rate > 0.5): pick shortest correct solution
    - Medium (0.15 < pass_rate <= 0.5): pick median-length correct solution
    - Hard (pass_rate <= 0.15): pick solution with best quality score (allow longer)
    """
    selected = []
    for problem in problems_with_solutions:
        correct = sorted(
            [s for s in problem['solutions'] if s['correct']],
            key=lambda s: len(s['text'])
        )
        if not correct:
            continue

        if problem['pass_rate'] > 0.5:
            selected.append(correct[0])  # shortest
        elif problem['pass_rate'] > 0.15:
            selected.append(correct[len(correct)//2])  # median
        else:
            # For hard problems, pick by quality, not length
            best = max(correct, key=lambda s: s.get('quality_score', 0))
            selected.append(best)
    return selected
```

### 2.3 Why Difficulty-Awareness Matters for Us

Our model currently produces ~8K-15K tokens per solution regardless of difficulty. A well-calibrated model would:
- Solve easy problems in ~2K tokens (saving ~10K per problem)
- Spend full token budget only on hard problems
- Net effect: ~30% fewer total tokens across 50 problems = ~35% more attempts possible

---

## 3. Length Rewards in Reinforcement Learning

### 3.1 L1: Length Controlled Policy Optimization (LCPO)

Paper: "L1: Controlling How Long A Reasoning Model Thinks" (Mar 2025, arxiv:2503.04697)

**Key innovation**: Train the model to satisfy user-specified length constraints via RL.

**Method (LCPO)**:
- Reward = accuracy_reward + lambda * length_penalty
- Length penalty activates when output exceeds a target length specified in the prompt
- The model learns to produce reasoning chains of controllable length
- At inference time, you can specify "solve this in ~2000 tokens" vs "use up to 10000 tokens"

**Results**:
- 1.5B L1 model surpasses GPT-4o at equal reasoning lengths
- Smooth trade-off between compute cost and accuracy
- Short Reasoning Models (SRMs) achieve reasoning-model quality at non-reasoning-model lengths

### 3.2 e1: Adaptive Effort Control

Paper: "e1: Learning Adaptive Control of Reasoning Effort" (Oct 2025, arxiv:2510.27042)

**Key finding**: ~3x reduction in CoT length while maintaining or improving performance, from 1.5B to 32B scale.

**Method**:
- User specifies fraction of tokens relative to current average
- Model automatically allocates resources proportionally to task difficulty
- No architectural changes needed

### 3.3 Length Reward Shaping Patterns

Common reward designs for length-efficient RL:

```python
# Pattern 1: Simple length penalty (most common)
def reward_with_length_penalty(correct, length, max_length, lambda_len=0.001):
    accuracy_reward = 1.0 if correct else 0.0
    length_penalty = -lambda_len * max(0, length - max_length * 0.5)
    return accuracy_reward + length_penalty

# Pattern 2: Cosine-shaped (better stability, from recent research)
import math
def cosine_length_reward(correct, length, target_length):
    accuracy_reward = 1.0 if correct else 0.0
    ratio = min(length / target_length, 2.0)
    length_bonus = 0.5 * (1 + math.cos(math.pi * ratio))  # 1 at 0, 0 at target, negative beyond
    return accuracy_reward + 0.1 * length_bonus * accuracy_reward  # only reward short if correct

# Pattern 3: Step-level (highest quality but hardest to implement)
def step_level_reward(correct, step_lengths, step_importances):
    if not correct:
        return -0.5
    # Penalize long steps that aren't important
    total = 0
    for length, importance in zip(step_lengths, step_importances):
        if importance < 0.3 and length > 200:  # low importance, high length
            total -= 0.01 * length
    return 1.0 + total
```

**For our GRPO training**: Pattern 2 (cosine) is recommended. It provides smoother gradients than hard penalties and naturally encourages proportional reasoning.

### 3.4 Empirical Results from Length-Reward RL

| Paper | Method | Length Reduction | Accuracy Change |
|-------|--------|-----------------|-----------------|
| L1 (LCPO) | RL with length constraint | Controllable (up to 5x) | Smooth trade-off |
| e1 | Adaptive effort | 3x reduction | Maintained |
| STILL-2 | Step-level length reward | 33% | Maintained |
| Ada-R1 | Difficulty-adaptive | Variable | +2% on hard problems |

**References**:
- [L1 paper](https://arxiv.org/abs/2503.04697)
- [e1 paper](https://arxiv.org/abs/2510.27042)

---

## 4. TokenSkip: Controllable CoT Compression

Paper: "TokenSkip: Controllable Chain-of-Thought Compression in LLMs" (EMNLP 2025, arxiv:2502.12067)

### 4.1 Method

1. Generate full CoT trajectories from the target model
2. Score token importance (semantic importance via attention/gradient)
3. Remove less important tokens at a specified compression ratio
4. Fine-tune the model on compressed CoTs
5. At inference, the model generates naturally compressed reasoning

### 4.2 Results

Applied to Qwen2.5-14B-Instruct:
- **40% token reduction** (313 -> 181 tokens on GSM8K)
- **Less than 0.4% accuracy drop**
- Training: ~2.5 hours on 2x 3090 GPUs

### 4.3 Applicability to Our Setup

- Could be applied to gpt-oss-120b traces before SFT
- Generate full traces -> compress -> train on compressed versions
- Risk: MoE models may have different compression characteristics
- The 40% reduction is attractive but needs validation with olympiad-level problems

**Code**: [github.com/hemingkx/TokenSkip](https://github.com/hemingkx/TokenSkip)

---

## 5. Markov Chain of Thought (MCoT)

Paper: "Markov Chain of Thought for Efficient Mathematical Reasoning" (NAACL 2025, arxiv:2410.17635)

### 5.1 Core Idea

Standard CoT maintains the full history of all previous steps. MCoT applies a "derive, then reduce" approach:
1. Complete one reasoning step
2. Compress all previous context into a simplified question
3. Proceed with the next step using only the compressed question
4. Repeat until solution

This implements a "memoryless" Markov property -- each step only sees the current compressed state, not the full history.

### 5.2 Results

- **1.9x speed increase** compared to traditional multi-step reasoning
- Works well with code interpreter interactions (self-correction compensates for information loss)
- Best suited for problems with clear step-by-step decomposition

### 5.3 Applicability

- High synergy with our TIR (Tool-Integrated Reasoning) setup
- Each code execution naturally creates a "checkpoint" where context can be compressed
- Risk: Hard olympiad problems may require maintaining full context for backtracking
- Implementation complexity is high -- requires modifying the generation loop

---

## 6. Latent Reasoning Approaches (Future/Experimental)

### 6.1 Coconut (Chain of Continuous Thought)

Paper: arxiv:2412.06769

- Replaces explicit text reasoning with continuous hidden state propagation
- The model "thinks" in embedding space rather than token space
- Enables breadth-first search over reasoning paths (vs. CoT's depth-first)
- **60%+ token reduction** but requires architectural modifications
- Currently experimental; not production-ready for competition use

### 6.2 CODI (Compressed Chain of Thought via Distillation)

- Single-stage compression of CoT into continuous representations
- Outperforms all previous implicit CoT methods
- Requires specialized training infrastructure

### 6.3 Assessment for Our Use Case

These approaches are **not recommended for AIMO3** due to:
- Require architectural changes incompatible with vLLM serving
- Not proven on olympiad-level math
- High implementation complexity with uncertain payoff

They represent the future direction but are not actionable for our April 2026 deadline.

---

## 7. Practical Recommendations for Our Pipeline

### 7.1 Immediate Actions (Dataset Curation Phase)

**A. Difficulty-aware trace selection** (Priority 1, no training needed):
- When selecting training traces, pick shorter correct solutions for easier problems
- For hard problems (pass_rate < 0.15), allow longer but high-quality traces
- Expected impact: ~30% token reduction in training data

**B. DPO pair construction** (Priority 2, needed for training):
- From our generated solutions, construct (short_correct, long_correct) DPO pairs
- Use ratio_threshold=0.7 (chosen must be <70% of rejected length)
- Enforce min_length to avoid degenerate outputs
- Expected impact: Additional 10-20% inference token reduction

### 7.2 Training Strategy

**Recommended 3-stage approach** (based on Imagination team + research):

```
Stage 1: SFT on difficulty-calibrated traces (2K-4K examples)
    - Easy problems: shortest correct trace
    - Hard problems: highest-quality trace
    - Include TIR (tool-integrated) traces with >=3 tool calls

Stage 2: DPO on length-preference pairs (5K-10K pairs)
    - Chosen: shorter correct solution
    - Rejected: longer correct solution (or incorrect)
    - Ratio threshold: 0.7
    - This stage specifically reduces output verbosity

Stage 3 (optional): GRPO with length-aware reward
    - Reward = correctness + 0.1 * cosine_length_bonus
    - Only penalize length for correct solutions
    - Dynamic difficulty targeting (AdaRFT style)
```

### 7.3 Token Budget Analysis

Current state (8 attempts, ~10K tokens/attempt):
```
50 problems x 8 attempts x 10K tokens = 4M tokens generated
Time: ~5 hours (within 9h budget)
```

With 30% token reduction (difficulty-aware SFT + DPO):
```
50 problems x 8 attempts x 7K tokens = 2.8M tokens
Time: ~3.5 hours
Freed time: 1.5 hours = room for 4 more attempts per problem (12 total)
```

With 40% token reduction (+ TokenSkip or length RL):
```
50 problems x 8 attempts x 6K tokens = 2.4M tokens
Time: ~3 hours
Freed time: 2 hours = room for 6 more attempts per problem (14 total)
```

**Critical**: More attempts only help with better selection (our feb6 experiment with 16 attempts scored 29/50 due to selection degradation). Token efficiency must be paired with improved answer selection.

### 7.4 Format Guidelines for Training Data

Based on research, the ideal TIR trace format for token efficiency:

1. **No preamble**: Skip "Let me think about this problem..." type openings
2. **Direct problem analysis**: Start with key observations about the problem structure
3. **Code-first verification**: Use Python/SymPy for computation rather than manual algebra
4. **No redundant recapping**: Each step should add new information, not repeat context
5. **Concise error handling**: If code fails, briefly state why and fix, don't elaborate
6. **Direct answer extraction**: State the answer immediately after computation confirms it

Example of verbose (bad) vs concise (good) TIR:

**Verbose** (~3000 tokens):
```
Let me carefully analyze this problem. We need to find the number of...
[repeats problem statement]
Let me think about what approach would work best here...
I'll try using Python to compute this.
```python
# First, let me define the variables
# This is a combinatorics problem
# We need to count the number of ways...
[20 lines of commented code]
```
The code gave us 42. Let me verify this is correct by trying another approach...
[another 1000 tokens of verification]
So the answer is 42.
```

**Concise** (~800 tokens):
```
Key observation: this is equivalent to counting lattice paths with constraint.
```python
from sympy import binomial
# Count paths from (0,0) to (n,k) staying below diagonal
result = binomial(2*n, n) // (n+1)  # Catalan number
print(result)
```
Output: 42

Verified: matches Catalan number formula C_n.
The answer is \boxed{42}.
```

---

## 8. Key Papers Reference

| Paper | Key Contribution | Reduction | Year | Link |
|-------|-----------------|-----------|------|------|
| Imagination AIMO2 | SFT+DPO for shorter outputs, 2nd place | ~30% | 2025 | [GitHub](https://github.com/imagination-research/aimo2) |
| DA-CoTD | Difficulty-aware trace selection | 30% | 2025 | [arxiv:2509.05226](https://arxiv.org/abs/2509.05226) |
| TokenSkip | Controllable CoT compression via fine-tuning | 40% | 2025 | [arxiv:2502.12067](https://arxiv.org/abs/2502.12067) |
| L1 (LCPO) | Length-controlled RL for reasoning models | Controllable | 2025 | [arxiv:2503.04697](https://arxiv.org/abs/2503.04697) |
| e1 | Adaptive effort control, 3x reduction | 67% | 2025 | [arxiv:2510.27042](https://arxiv.org/abs/2510.27042) |
| MCoT | Markov chain memoryless reasoning | 47% (1.9x) | 2025 | [arxiv:2410.17635](https://arxiv.org/abs/2410.17635) |
| s1 | 1K curated examples beat o1-preview | N/A (curation) | 2025 | [arxiv:2501.19393](https://arxiv.org/abs/2501.19393) |
| LIMO | 817 examples, quality > quantity | N/A (curation) | 2025 | [arxiv:2502.03387](https://arxiv.org/abs/2502.03387) |
| Coconut | Latent continuous reasoning | 60%+ | 2024 | [arxiv:2412.06769](https://arxiv.org/abs/2412.06769) |
| CODI | Single-stage CoT compression | ~50% | 2025 | [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.36/) |
| Ada-R1 | Hybrid adaptive CoT routing | Variable | 2025 | [arxiv:2504.21659](https://arxiv.org/abs/2504.21659) |

---

## 9. Impact on Our Competition Strategy

### What This Means for AIMO3

1. **Training data curation** should prioritize concise correct solutions, especially for easier problems
2. **SFT + DPO** (2-stage) is the most proven approach for our setup -- the Imagination team used exactly this to score 31/50 in AIMO2
3. **30% token reduction is realistic** with just careful trace selection + DPO, no exotic techniques needed
4. **Token savings translate to more attempts**: from 8 to 12-14 attempts per problem within the same time budget
5. **More attempts only help with better selection**: must pair token efficiency with improved answer selection strategy (not just majority vote)

### What NOT to Do

- Do NOT try latent reasoning (Coconut/CODI) -- incompatible with vLLM, too experimental
- Do NOT aggressively compress hard problems -- they need full reasoning depth
- Do NOT use length penalties without correctness gating -- the model will learn to output short wrong answers
- Do NOT expect >50% reduction while maintaining olympiad-level accuracy -- 30-40% is the realistic ceiling
