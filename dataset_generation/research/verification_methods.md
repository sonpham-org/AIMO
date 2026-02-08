# Answer Verification and Logical Correctness Checking Methods

> Author: researcher-3
> Date: 2026-02-07
> Purpose: Comprehensive guide to verifying both final answers AND reasoning correctness in math solution datasets
> Context: For curating high-quality fine-tuning data for gpt-oss-120b (AIMO3 competition)

---

## Executive Summary

Verifying math solutions requires two distinct layers:

1. **Outcome verification**: Is the final answer correct?
2. **Process verification**: Is the reasoning logically sound at every step?

Training on solutions with correct answers but flawed reasoning ("right answer, wrong reasoning") actively degrades model performance. This document catalogs all practical methods for both layers, from zero-cost tools to trained verifiers.

**Recommended approach for our pipeline**:
- **Stage 1**: Outcome verification via `math_verify` + SymPy (free, automated)
- **Stage 2**: Cross-model answer agreement (free, catches "lucky correct")
- **Stage 3**: Process verification via ThinkPRM-14B or Qwen2.5-Math-PRM-7B (free, off-the-shelf)
- **Stage 4** (optional): Self-consistency filtering (free, statistical)
- **Stage 5** (aspirational): Formal verification via MATH-VF/FANS for highest-stakes data

---

## 1. Outcome Verification (Final Answer Correctness)

### 1.1 Direct Answer Checking with math_verify

The standard tool for AIMO/math competitions. Uses SymPy under the hood.

**Library**: `math_verify` (HuggingFace, PyPI)
- Version 0.8.0 available on PyPI
- Three-step algorithm: Answer Extraction -> SymPy Conversion -> Gold Comparison
- Uses ANTLR4 grammar to parse LaTeX/text answers into SymPy expressions
- Handles equivalence checking (e.g., `x+y` = `y+x`, interval vs finite set)
- Asymmetric inequality handling prevents models from returning input without solving

**Usage**:
```python
from math_verify import verify_answer
# Extracts \boxed{} answer, parses to SymPy, compares with gold
is_correct = verify_answer(model_output, gold_answer)
```

**Limitations**:
- Only checks final answer, not reasoning
- Cannot handle all answer formats (proofs, constructions, etc.)
- Some numerical precision issues with floating-point comparisons

### 1.2 SymPy Direct Comparison

For cases where `math_verify` is insufficient:

```python
import sympy
from sympy import simplify, Eq, sympify

def check_equivalence(answer_str, gold_str):
    """Check mathematical equivalence via SymPy simplification."""
    try:
        ans = sympify(answer_str)
        gold = sympify(gold_str)
        # Try direct equality
        if ans == gold:
            return True
        # Try simplification
        if simplify(ans - gold) == 0:
            return True
        # Try numerical evaluation
        if abs(float(ans.evalf()) - float(gold.evalf())) < 1e-8:
            return True
        return False
    except:
        return False
```

### 1.3 Cross-Model Answer Agreement (AceMath Method)

**The strongest outcome filter after direct checking.**

AceMath (NVIDIA, 2024) generates solutions with two independent models and keeps only answers where both agree with ground truth. This catches "lucky correct" solutions where the model reaches the right answer through flawed reasoning.

**Method**:
1. Generate answer with Model A (e.g., gpt-oss-120b)
2. Generate answer with Model B (e.g., Qwen3-8B or GPT-4o-mini)
3. Keep only solutions where BOTH models agree with ground truth

**Why this works**: If two architecturally different models arrive at the same answer through independent reasoning, the probability of both having wrong-but-coincidentally-correct reasoning is very low.

**Result**: AceMath reduced 2.3M samples to 800K with no loss in benchmark performance. The retained samples had significantly higher reasoning quality.

**Practical for us**: Generate 8 solutions with gpt-oss-120b per problem. For each problem, also generate 2 solutions with a smaller model (e.g., Qwen3-8B). Only keep gpt-oss-120b solutions whose answer matches at least one correct Qwen3-8B answer.

### 1.4 Self-Consistency Voting

Multiple samples from the same model; answers that appear more frequently are more likely correct.

**Standard self-consistency**: Sample N solutions, take majority vote.

**Confidence-Informed Self-Consistency (CISC, 2025)**:
- Adds a self-assessment step: assign confidence score to each reasoning path
- Weighted majority vote using confidence scores
- Reduces required samples by 40-46% compared to standard self-consistency
- 10 CISC samples match 18.6 standard samples in accuracy

**Reasoning-Aware Self-Consistency (RASC, 2025)**:
- Dynamically evaluates both outputs AND rationales
- Reduces sample usage by ~70% while maintaining accuracy
- Evaluates reasoning faithfulness, not just answer frequency

**For our pipeline**: We already use self-consistency (entropy-gated consensus). CISC/RASC could improve our answer selection at inference time with fewer attempts.

---

## 2. Process Verification (Step-Level Reasoning Correctness)

This is the critical gap. A solution can reach the right answer through:
- Correct reasoning (good training signal)
- Compensating errors (two wrongs make a right -- BAD training signal)
- Lucky shortcuts (skipping steps that happen to work -- BAD training signal)
- Circular reasoning (assuming the answer, then "deriving" it -- BAD training signal)

### 2.1 Process Reward Models (PRMs) -- The Primary Tool

PRMs assign a quality/correctness score to each reasoning step, enabling detection of intermediate errors even when the final answer is correct.

#### 2.1.1 Available Off-the-Shelf PRMs

| Model | Size | Type | Training Data | Best For | HuggingFace |
|-------|------|------|--------------|----------|-------------|
| **Qwen2.5-Math-PRM-7B** | 7B | Discriminative | Consensus-filtered MC | General math | `Qwen/Qwen2.5-Math-PRM-7B` |
| **Qwen2.5-Math-PRM-72B** | 72B | Discriminative | Consensus-filtered MC | High accuracy | `Qwen/Qwen2.5-Math-PRM-72B` |
| **ThinkPRM-14B** | 14B | Generative (CoT) | 1% of PRM800K | Step verification | `launch/ThinkPRM-14B` |
| **ThinkPRM-1.5B** | 1.5B | Generative (CoT) | 1% of PRM800K | Lightweight | `launch/ThinkPRM-1.5B` |
| **ReasonFlux-PRM-7B** | 7B | Trajectory-aware | OpenThoughts-114K | Long CoT | `Gen-Verse/ReasonFlux` |
| **DeepSeek-Math-7B-RL-PRM** | 7B | Discriminative | PRM800K (LoRA) | Basic | `mukaj/deepseek-math-7b-rl-prm-v0.1` |

#### 2.1.2 Discriminative PRMs (Qwen2.5-Math-PRM)

**How they work**: Encode solution text, use classification head to produce step-level scores (0-1). Steps below threshold indicate errors.

**Usage with Qwen2.5-Math-PRM-7B**:
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("Qwen/Qwen2.5-Math-PRM-7B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-PRM-7B")

# Separate steps with \n\n, insert <extra_0> after each step
# Model returns reward 0-1 per step
# Minimum step reward = solution quality score
```

**Key finding from Qwen team**: Training PRMs with consensus-filtered Monte Carlo estimation (combining MC rollouts with LLM-as-judge evaluation, then filtering noisy labels) significantly outperforms pure MC estimation or pure LLM-as-judge alone.

**Limitation**: Discriminative PRMs struggle to generalize beyond their training distribution. A PRM trained on Qwen2.5-Math outputs may not accurately score gpt-oss-120b outputs. Use as a soft signal, not a hard filter.

#### 2.1.3 Generative PRMs (ThinkPRM) -- RECOMMENDED

**How they work**: Instead of a classification head, the PRM generates a verification chain-of-thought that explicitly checks each step, then concludes with a correctness judgment.

**Why ThinkPRM is better for us**:
- Trained on only 1% of PRM800K labels (data-efficient)
- Outperforms discriminative PRMs trained on full PRM800K
- Generalizes better across model outputs (not tied to one generator's distribution)
- Provides interpretable verification (we can read WHY a step was flagged)
- Scales with test-time compute (longer verification = more accurate)

**ThinkPRM-14B** (based on DeepSeek-R1-Distill-Qwen-14B):
- SOTA on ProcessBench for 7/8B scale
- Outperforms LLM-as-Judge approaches
- Available: `launch/ThinkPRM-14B` on HuggingFace

**Usage**: Feed solution step-by-step, model generates verification CoT, extract correctness judgment per step.

#### 2.1.4 Trajectory-Aware PRMs (ReasonFlux-PRM)

Designed specifically for long chain-of-thought reasoning with trajectory-response structure.

**Key features**:
- Incorporates both step-level AND trajectory-level supervision
- Supports offline (data filtering) and online (RL reward) use
- Trained on OpenThoughts-114K trajectory-response traces
- Better suited for our TIR traces which have complex multi-turn structure

**Available**: `Gen-Verse/ReasonFlux` (GitHub), with 7B model weights.

### 2.2 Dyve: Dynamic Process Verification (EMNLP 2025)

**Dual-system approach** inspired by Kahneman's Thinking Fast and Slow:
- **System 1 (Fast)**: Immediate token-level confirmation for straightforward steps
- **System 2 (Slow)**: Comprehensive analysis for complex steps

**Key innovation**: Adaptively decides which verification depth to use per step, saving compute on easy steps while thoroughly checking difficult ones.

**Data quality**: Uses consensus-filtered process supervision (combining Monte Carlo estimation with LLM evaluation), removing ~50% of noisy rollouts, yielding 117K high-quality supervision examples.

**Results**: SOTA F1 on ProcessBench (68.5 on GSM8K, 58.3 on MATH). Generalizes to Olympiad-level problems.

**Relevance for us**: The consensus-filtered annotation technique could be used to generate our own step-level labels for training data quality filtering.

### 2.3 TANGO: Co-Evolving Generator and Verifier

**Key insight**: Train the verifier alongside the policy model, so the verifier improves as the generator improves.

**Results**: TANGO with GRPO achieves relative gains of 50.4% on AIME 2024, 100.0% on AIME 2025, and 30.0% on AMC 2023 (averaging 24.6% improvement across all math tasks).

**How it works**:
1. Generator produces solutions
2. Verifier scores each step
3. Both are updated: generator via GRPO with verifier rewards, verifier via updated training data from generator
4. Iterate

**Relevance for us**: If we pursue RL training, TANGO's co-training approach could simultaneously improve our generator AND build a custom verifier.

### 2.4 Self-Certainty (Reward-Model-Free Verification)

**Zero-cost alternative to PRMs** for ranking candidate solutions.

Self-certainty leverages the inherent probability distribution of LLM outputs to estimate response quality without any external reward model.

**How it works**: For reasoning models (like DeepSeek-R1 distills), self-certainty measures how "sure" the model is about its answer by examining the logprob distribution at key tokens.

**Results**: Consistently outperforms greedy decoding and random sampling. Performance improves as N increases for Best-of-N selection.

**For our pipeline**: We already collect logprobs (for entropy-based selection). Self-certainty is essentially a more principled version of what we do with entropy gating. Could be used as a complementary signal.

---

## 3. Formal Verification Approaches

### 3.1 MATH-VF: Step-Wise Formal Verification (May 2025)

**Training-free framework** that formally verifies each step using external tools.

**Architecture**:
- **Formalizer**: LLM translates natural language solution into SimpleMath (a formal language closer to natural language than Lean4, easier for LLMs to produce)
- **Critic**: Integrates Computer Algebra System (CAS) + SMT solver to evaluate each statement

**Key advantage**: Not an LLM judging another LLM -- uses formal tools (CAS, SMT solvers) that provide mathematical guarantees.

**Sparsity optimization**: Most steps need only 4 premises, dramatically reducing token usage for long solutions.

**Applications**:
1. Verification: Determine if a solution is correct
2. Refinement: Provide constructive feedback on incorrect steps for regeneration

**Limitation**: Formalization step is imperfect. LLM may incorrectly translate natural language to formal language, introducing false positives/negatives.

### 3.2 FANS: Formal Answer Selection Using Lean4 (Mar 2025)

Uses Lean4 theorem prover for answer verification:
1. LLM generates multiple candidate solutions
2. Each solution is autoformalized into Lean4 code
3. Lean4 compiler verifies the proof
4. Select answers backed by verified proofs

**Advantage**: Mathematical certainty when verification succeeds.
**Limitation**: Autoformalization success rate is low (typically <50% even for correct solutions). Many correct solutions cannot be successfully formalized.

### 3.3 Practical Assessment of Formal Verification for Our Use Case

**Should we use formal verification?**

Formal verification (Lean4, MATH-VF) provides the highest confidence but has critical practical limitations:

| Factor | Assessment |
|--------|-----------|
| Accuracy when it works | Near-perfect (mathematical guarantees) |
| Coverage | Low (30-50% of solutions can be formalized) |
| Compute cost | High (LLM formalization + compiler verification) |
| Olympiad-level math | Limited (many olympiad techniques lack formal libraries) |
| TIR traces | Very difficult (multi-turn code execution hard to formalize) |
| Setup complexity | High (Lean4 environment, formalization pipeline) |

**Recommendation**: Not practical as primary verification for our dataset curation pipeline. Potentially useful as a final "gold standard" check on a small subset of the most critical training examples. PRMs are more practical for bulk verification.

---

## 4. Detecting "Right Answer, Wrong Reasoning"

This is the core challenge. Methods ranked by practicality:

### 4.1 Cross-Model Agreement (Most Practical)

If two architecturally different models solve the same problem and reach the same answer, the probability of BOTH having wrong-but-coincidentally-correct reasoning is very low.

**Implementation**:
```python
def cross_model_filter(problem, gold_answer, solutions_model_a, solutions_model_b):
    """Keep model_a solutions only if model_b independently confirms the answer."""
    correct_a = [s for s in solutions_model_a if verify_answer(s, gold_answer)]
    correct_b = [s for s in solutions_model_b if verify_answer(s, gold_answer)]

    if correct_a and correct_b:
        # Both models independently found the same answer
        # High confidence in reasoning correctness
        return correct_a  # Keep model_a solutions
    else:
        # Only one model found the answer -- may be "lucky correct"
        return []  # Discard or flag for manual review
```

### 4.2 PRM Step-Level Scoring (Recommended)

Use ThinkPRM-14B or Qwen2.5-Math-PRM-7B to score each step. Solutions where any step scores below threshold are likely to contain reasoning errors, even if the final answer is correct.

**Filtering strategy**:
```python
def prm_filter(solution_text, step_scores, threshold=0.5):
    """
    Filter solutions based on minimum step score.
    Even correct-answer solutions are rejected if any step is below threshold.
    """
    min_step_score = min(step_scores)
    if min_step_score < threshold:
        return False  # Likely has reasoning error
    return True
```

**ProcessBench findings**: On competition-level math, the best open PRMs (QwQ-32B-Preview acting as critic) achieve ~65-70% accuracy in detecting erroneous steps. This is imperfect but significantly better than no process checking.

### 4.3 Multi-Path Consistency

If the model reaches the correct answer via the SAME reasoning path in multiple samples, the reasoning is more likely correct. If it reaches the correct answer via wildly different paths each time, some paths may have compensating errors.

**Implementation**: Cluster reasoning traces (by key intermediate results or step structure), keep traces from the largest cluster.

### 4.4 Self-Verification Prompting

Ask the same model to verify its own reasoning:

```python
verification_prompt = """
Here is a math problem and a proposed solution.
Check each step of the reasoning for correctness.
Identify any logical errors, unjustified jumps, or incorrect calculations.

Problem: {problem}
Solution: {solution}

Step-by-step verification:
"""
```

**Limitation**: Models are biased toward confirming their own outputs. Self-verification catches obvious errors but misses subtle logical flaws.

### 4.5 Code Execution Verification (for TIR traces)

For Tool-Integrated Reasoning, we can verify that:
1. All code blocks execute without errors
2. Code output matches the claims in the reasoning text
3. Final answer is derived from code output, not hallucinated

```python
def tir_verification(trace):
    """Verify TIR trace integrity."""
    for code_block, claimed_output in trace.code_blocks:
        actual_output = execute_code(code_block)
        if actual_output != claimed_output:
            return False  # Output mismatch
        if code_block.has_error:
            return False  # Code execution error
    # Check final answer comes from last code output
    if trace.final_answer not in trace.last_code_output:
        return False  # Answer not derived from code
    return True
```

---

## 5. NVIDIA's GenSelect: Learned Verification at Scale

GenSelect (NVIDIA, AIMO-2 winner) trained a model to SELECT the best solution from N candidates. This is essentially a learned verifier integrated into the generator.

### 5.1 How GenSelect Works

1. Generate 64 solutions per problem
2. Create subsets of 16 solutions
3. Model evaluates all 16, reasons about which is best, selects one
4. Repeat 64 times with different subsets, majority vote over selections

### 5.2 Training Data Construction

- 566K GenSelect training samples in `nvidia/OpenMathReasoning`
- Format: multiple candidate solutions + selection reasoning + chosen answer
- Training uses full reasoning traces from DeepSeek-R1-0528-671B

### 5.3 Results

| Benchmark | Without GenSelect | With GenSelect | Improvement |
|-----------|------------------|----------------|-------------|
| AIME24 | 89.2 | 93.3 | +4.1 |
| AIME25 | 84.0 | 90.0 | +6.0 |
| HMMT | 73.8 | 96.7 | +22.9 |

### 5.4 Relevance for Our Pipeline

GenSelect addresses both verification problems simultaneously:
- It verifies answer correctness by comparing across candidates
- It implicitly verifies reasoning quality by evaluating solution coherence

For dataset curation: we could use a GenSelect-trained model to evaluate and rank candidate solutions, keeping only those it confidently selects as highest quality.

---

## 6. Automated Math-Shepherd Labels (Step-Level Annotation at Scale)

### 6.1 Math-Shepherd Method (ACL 2024)

Automatically generates step-level supervision labels without human annotation:

1. For each step in a solution, treat it as a "checkpoint"
2. From that checkpoint, generate K completions (Monte Carlo rollouts)
3. If any completion reaches the correct final answer, the step is labeled "correct"
4. If no completion reaches the correct answer, the step is labeled "incorrect"

**Results**: Mistral-7B improved from 77.9% to 84.1% on GSM8K and 28.6% to 33.0% on MATH with Math-Shepherd PPO training.

### 6.2 OmegaPRM (DeepMind, 2024)

Improved Math-Shepherd with divide-and-conquer MCTS:

- **Binary search** for the first error in a chain-of-thought (much faster than checking every step)
- Balances positive and negative examples in training data
- Collected 1.5M process supervision annotations
- Gemini Pro: 51% -> 69.4% on MATH (+36% relative)
- Outperforms human annotations from PRM800K

### 6.3 Consensus-Filtered MC (Dyve Method, 2025)

Combines Monte Carlo estimation with LLM-based evaluation:

1. Standard MC rollouts from each step
2. Also ask an LLM to evaluate each step
3. Keep only labels where MC and LLM agree (consensus filter)
4. Removes ~50% of noisy labels, yielding 117K high-quality annotations

**This is currently the best automated step-labeling technique.**

### 6.4 Practical Step-Labeling Pipeline for Our Data

```python
def label_steps(problem, solution, gold_answer, n_rollouts=8):
    """
    Generate step-level labels using Math-Shepherd approach.

    For each step:
    - Generate n_rollouts completions from that step
    - If any completion reaches correct answer, step is "correct"
    - Otherwise, step is "incorrect" (first error found)
    """
    steps = split_into_steps(solution)
    labels = []

    for i, step in enumerate(steps):
        partial_solution = "\n".join(steps[:i+1])
        completions = generate_completions(
            problem, partial_solution, n=n_rollouts
        )
        correct_completions = sum(
            1 for c in completions
            if verify_answer(c, gold_answer)
        )

        step_label = "correct" if correct_completions > 0 else "incorrect"
        step_confidence = correct_completions / n_rollouts
        labels.append({
            "step": i,
            "label": step_label,
            "confidence": step_confidence,
            "text": step
        })

        if step_label == "incorrect":
            # First error found -- remaining steps are suspect
            break

    return labels
```

**Cost estimate**: For 10K solutions with 10 steps each and 8 rollouts per step, that's 800K inference calls. With vLLM on H100, approximately 2-4 hours.

---

## 7. Practical Verification Pipeline (Recommended for Our Project)

### Stage 1: Outcome Verification (Filter ~30-50% of solutions)

```python
# Use math_verify for answer extraction and checking
from math_verify import verify_answer

correct_solutions = []
for solution in all_solutions:
    extracted = extract_boxed_answer(solution.text)
    if verify_answer(extracted, gold_answer):
        correct_solutions.append(solution)
```

**Cost**: Near-zero (string parsing + SymPy). Milliseconds per solution.

### Stage 2: TIR Integrity Check (Filter ~10-20% of correct solutions)

```python
def tir_integrity(trace):
    """For TIR traces: verify code execution integrity."""
    if trace.n_tool_calls == 0:
        return False  # Not actually TIR
    if trace.n_tool_calls > 15:
        return False  # Likely spinning
    if any(call.has_error for call in trace.tool_calls):
        return False  # Code errors
    if len(trace.text) > 80000:
        return False  # Stuck in loops
    return True
```

**Cost**: Near-zero (structural checks only).

### Stage 3: Cross-Model Agreement (Filter ~20-40% more)

```python
# Generate reference answers with a different model
# Keep only solutions where answer matches cross-model reference
cross_verified = []
for problem in problems:
    our_correct = get_correct_solutions(problem, model="gpt-oss-120b")
    ref_correct = get_correct_solutions(problem, model="qwen3-8b")

    if our_correct and ref_correct:
        # Both models found correct answer independently
        cross_verified.extend(our_correct)
```

**Cost**: Requires running a second model. If using existing dataset with multiple model traces, this is free.

### Stage 4: Process Verification with PRM (Score remaining solutions)

```python
# Use ThinkPRM-14B for step-level verification
# ThinkPRM generates verification CoT -- interpretable!
from think_prm import ThinkPRM

prm = ThinkPRM("launch/ThinkPRM-14B")
for solution in cross_verified:
    step_scores = prm.verify_steps(problem, solution)
    min_score = min(step_scores)
    avg_score = sum(step_scores) / len(step_scores)

    solution.prm_min_score = min_score
    solution.prm_avg_score = avg_score

# Keep solutions where min step score > 0.5
process_verified = [s for s in cross_verified if s.prm_min_score > 0.5]
```

**Cost**: ~1GB VRAM for 7B quantized PRM. Minutes for 10K solutions. ThinkPRM-14B needs ~8GB.

### Stage 5: Quality Ranking (Select best per problem)

Combine all signals for final ranking:

```python
def combined_score(solution):
    return (
        0.30 * solution.prm_avg_score +      # Process quality
        0.25 * solution.difficulty_score +     # Problem difficulty
        0.20 * solution.quality_keywords +     # Reasoning quality
        0.15 * solution.interaction_density +  # TIR engagement
        0.10 * solution.cross_model_confidence # Cross-model agreement
    )

# Per problem: keep top-1 (LIMO approach) or top-3 (diversity approach)
```

---

## 8. Key Findings and Recommendations

### 8.1 What Works Best (Ranked by Evidence)

| Method | Evidence Strength | Cost | Catches "Right Answer Wrong Reasoning" |
|--------|------------------|------|---------------------------------------|
| Cross-model agreement | Strong (AceMath) | Low-Medium | Yes (primary method) |
| ThinkPRM generative verification | Strong (ProcessBench SOTA) | Low | Yes (step-level) |
| math_verify outcome check | Universal | Near-zero | No (answer only) |
| Qwen2.5-Math-PRM-7B | Medium | Low | Partially (distribution mismatch) |
| Self-consistency voting | Strong | Medium | Partially (statistical) |
| Formal verification (MATH-VF) | Strong when it works | High | Yes (mathematical guarantee) |
| Self-verification prompting | Weak | Low | Partially (confirmation bias) |

### 8.2 What NOT to Do

1. **Do not rely solely on outcome verification**. Right answer + wrong reasoning = bad training data.
2. **Do not use a mismatched PRM as a hard filter**. PRM trained on Qwen outputs may incorrectly reject valid gpt-oss-120b reasoning. Use as soft signal only.
3. **Do not skip verification for "easy" problems**. Easy problems with wrong reasoning are especially harmful because the model learns shortcuts.
4. **Do not trust self-verification alone**. Models are biased toward confirming their own outputs.
5. **Do not use formal verification as the primary filter**. Coverage is too low (30-50%). Use PRMs for bulk filtering, formal verification for spot checks.

### 8.3 PRM Mismatch Warning

No PRM exists for gpt-oss-120b specifically. Using Qwen-based PRMs on gpt-oss-120b outputs introduces distribution mismatch. Mitigations:
- Use generative PRMs (ThinkPRM) which generalize better than discriminative PRMs
- Use PRM scores as soft weights, not hard filters
- Combine with cross-model agreement for robustness
- Consider training a custom PRM on our trace data (see Section 6.4 for step-labeling)

### 8.4 Estimated Verification Pipeline Costs

| Stage | Method | Time (10K solutions) | VRAM | Cost |
|-------|--------|---------------------|------|------|
| 1. Outcome | math_verify | Seconds | None | $0 |
| 2. TIR integrity | Structural checks | Seconds | None | $0 |
| 3. Cross-model | Pre-existing data | Minutes | None | $0 |
| 4. PRM scoring | ThinkPRM-14B | 30-60 min | 8GB | $0 |
| 5. Quality ranking | Scoring function | Seconds | None | $0 |
| **Total** | | **~1 hour** | **8GB** | **$0** |

---

## 9. Available Tools and Models Summary

### Libraries
| Tool | Purpose | Install |
|------|---------|---------|
| `math_verify` | Answer extraction + equivalence | `pip install math-verify` |
| `sympy` | Symbolic math comparison | `pip install sympy` |
| `trl` | PRM Trainer (train custom PRM) | `pip install trl` |

### Pre-trained Models
| Model | Size | Type | HuggingFace ID |
|-------|------|------|---------------|
| Qwen2.5-Math-PRM-7B | 7B | Discriminative PRM | `Qwen/Qwen2.5-Math-PRM-7B` |
| Qwen2.5-Math-PRM-72B | 72B | Discriminative PRM | `Qwen/Qwen2.5-Math-PRM-72B` |
| ThinkPRM-14B | 14B | Generative PRM | `launch/ThinkPRM-14B` |
| ThinkPRM-1.5B | 1.5B | Generative PRM | `launch/ThinkPRM-1.5B` |
| ReasonFlux-PRM-7B | 7B | Trajectory-aware PRM | See `Gen-Verse/ReasonFlux` |

### Benchmarks
| Benchmark | What It Tests | Size | Source |
|-----------|--------------|------|--------|
| ProcessBench | Step-error detection | 3,400 | `QwenLM/ProcessBench` |
| AceMath-RewardBench | Math reward model quality | Various | NVIDIA |

---

## 10. References

### Process Reward Models
- [ThinkPRM: Process Reward Models That Think](https://arxiv.org/abs/2504.16828) -- Generative PRMs, 1% of PRM800K labels, SOTA
- [Math-Shepherd (ACL 2024)](https://arxiv.org/abs/2312.08935) -- Automated step-level labels via MCTS
- [OmegaPRM (DeepMind 2024)](https://arxiv.org/abs/2406.06592) -- Divide-and-conquer MCTS, 1.5M annotations
- [Dyve (EMNLP 2025)](https://arxiv.org/abs/2502.11157) -- Fast/slow dynamic verification, consensus-filtered labels
- [ReasonFlux-PRM](https://arxiv.org/abs/2506.18896) -- Trajectory-aware PRMs for long CoT
- [TANGO](https://arxiv.org/abs/2505.15034) -- Co-evolving generator and verifier
- [ProcessBench (ACL 2025)](https://arxiv.org/abs/2412.06559) -- Benchmark for step-error detection
- [Qwen2.5-Math PRM Blog](https://qwenlm.github.io/blog/qwen2.5-math-prm/) -- Consensus-filtered MC annotation

### Outcome Verification
- [math_verify (PyPI)](https://libraries.io/pypi/math-verify) -- HuggingFace answer verification library
- [AceMath (NVIDIA 2024)](https://arxiv.org/abs/2412.15084) -- Cross-model verification + reward models

### Self-Consistency
- [CISC (2025)](https://arxiv.org/abs/2502.06233) -- Confidence-informed self-consistency
- [RASC (NAACL 2025)](https://aclanthology.org/2025.naacl-long.184/) -- Reasoning-aware self-consistency
- [Self-Certainty (2025)](https://arxiv.org/abs/2502.18581) -- Reward-model-free Best-of-N

### Formal Verification
- [MATH-VF (May 2025)](https://arxiv.org/abs/2505.20869) -- Step-wise formal verification with SimpleMath
- [FANS (Mar 2025)](https://arxiv.org/abs/2503.03238) -- Formal answer selection using Lean4
- [Aristotle (Harmonic)](https://arxiv.org/abs/2510.01346) -- IMO-level automated theorem proving

### GenSelect and Selection
- [OpenMathReasoning (NVIDIA)](https://arxiv.org/abs/2504.16891) -- GenSelect, won AIMO-2
- [GenSelect dataset](https://huggingface.co/datasets/nvidia/OpenMathReasoning) -- 566K training samples
