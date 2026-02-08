# Trace Generation: What to Store — Deep Research

**Date**: Feb 7, 2026
**Purpose**: Document all signals stored in trace generation notebooks and why each matters.

## Complete Per-Sample Schema

Each sample in a problem's JSON trace file contains these fields:

### Core Fields
| Field | Type | Description |
|-------|------|-------------|
| `answer` | int/null | Extracted final answer from `\boxed{}` |
| `is_correct` | bool/null | Whether answer matches ground truth |
| `finish_reason` | str | `"stop"` (natural) or `"length"` (truncated) |
| `time` | float | Wall-clock seconds for this sample |
| `n_turns` | int | Number of assistant turns |
| `turns` | list | Structured conversation with per-turn metadata |

### Logprob Signals — For PRIME, DPO, KTO, iw-SFT, Self-Certainty
| Field | Type | Description | Used By |
|-------|------|-------------|---------|
| `chosen_logprobs` | list[float] | Per-token logprob of the chosen token | **PRIME** (`log pi_theta`), DPO implicit reward, Self-Certainty proxy |
| `cumulative_logprob` | float | Sum of all chosen token logprobs | Best-of-N reranking, CISC weighted voting |
| `length_normalized_logprob` | float | `cumulative / completion_tokens` | Length-fair comparison |
| `completion_tokens` | int | Total generated tokens | Cost estimation, length filtering |
| `prompt_tokens` | int | Input tokens (P100 only) | Context utilization |
| `logprobs` | list[dict] | Top-10 token→logprob dicts per position | Entropy computation, future analysis |

### Entropy Signals — For DeepConf, PRM Step Rewards, Think Just Enough
| Field | Type | Description | Used By |
|-------|------|-------------|---------|
| `entropy` | float | Mean Shannon entropy across all tokens | Entropy-gated consensus (our method) |
| `per_token_entropy` | list[float] | Entropy at each token position | **DeepConf** group confidence, PRM step-level |
| `entropy_std` | float | Standard deviation of per-token entropy | Quality filtering |
| `entropy_min` | float | Minimum per-token entropy | Confidence peak detection |
| `entropy_max` | float | Maximum per-token entropy | Uncertainty peak detection |
| `entropy_p10` | float | 10th percentile entropy | **DeepConf** bottom-10% filtering |

### Code Execution Signals — For CodePRM, TIR Preference, GRPO
| Field | Type | Description | Used By |
|-------|------|-------------|---------|
| `code_calls` | int | Total code executions | TIR activity level |
| `code_errors` | int | Failed code executions | Quality filtering, DPO negative |
| `code_success_rate` | float | `(calls - errors) / calls` | CodePRM, TIR preference |

### Answer Stability — For TIR Preference, Quality Filtering
| Field | Type | Description | Used By |
|-------|------|-------------|---------|
| `all_turn_answers` | list[int] | Every answer extracted across turns | Answer evolution tracking |
| `answer_changed_count` | int | Times answer changed between turns | Stability signal |
| `answer_format_valid` | bool | Has proper `\boxed{}` format | GRPO format reward, DPO-VP |

### Degeneration Detection
| Field | Type | Description | Used By |
|-------|------|-------------|---------|
| `ngram_rep_4` | float | Word-level 4-gram repetition ratio (0=none, 1=all) | Filter degenerate solutions |

### Per-Turn Fields (inside `turns[]`)
| Field | Type | Description |
|-------|------|-------------|
| `role` | str | `"assistant"` or `"tool"` |
| `content` | str | Text content / code output |
| `tokens` | int | Token count for this turn (assistant only) |
| `entropy` | float | Mean entropy for this turn (assistant only) |
| `cumulative_logprob` | float | Sum of chosen logprobs for this turn (assistant only) |
| `code_success` | bool | Whether code execution succeeded (tool only) |

### Problem-Level Aggregates
| Field | Type | Description | Used By |
|-------|------|-------------|---------|
| `pass_rate` | float | `n_correct / n_samples` | iw-SFT difficulty weighting, GRPO |
| `n_correct` | int | Samples with correct answer | GRPO advantage normalization |
| `answer_distribution` | dict | `{answer: count}` | Consensus strength |
| `temperature` | float | Generation temperature | Reproducibility |
| `top_logprobs` | int | K for top-K logprobs stored | Schema documentation |

---

## What Each Training Method Needs

### PRIME (Implicit PRM) — Zero Step Labels Required
**The cheapest path to a PRM.** Needs only:
- `chosen_logprobs` → per-token `log pi_theta(y_t | y_{<t})`
- `is_correct` → binary outcome label
- Reference model identity (to compute `log pi_ref` later)

Formula: `reward_t = beta * (log pi_theta(y_t) - log pi_ref(y_t))`

**We store everything needed.** Reference model logprobs computed offline.

### DPO / KTO / iw-SFT
- `turns[].content` → full response text (chosen/rejected)
- `is_correct` → pair construction (correct=chosen, incorrect=rejected)
- `cumulative_logprob` → implicit reward for pair ranking
- `answer_format_valid` → DPO-VP 3-level scoring
- `completion_tokens` → DPO-VP length-biased selection

### GRPO (Group Relative Policy Optimization)
- `answer_distribution` → group rewards
- `is_correct` → binary accuracy reward
- `answer_format_valid` → format reward
- `pass_rate` → difficulty-aware reward scaling

### Step-DPO / Full-Step-DPO / PRM Training
- `turns[]` with per-turn `entropy` and `tokens` → step boundary detection
- `per_token_entropy` → step-level reward signals
- `is_correct` → outcome labels for Monte Carlo rollouts
- Full conversation in `turns[]` → prefix reconstruction for rollouts

### GenSelect (NVIDIA's Learned Selector)
- `turns[].content` → full solution text (GenSelect reads raw text)
- **Does NOT need logprobs** — operates purely on text comparison

### Self-Certainty (ICLR 2026)
- `chosen_logprobs` → mean chosen logprob as proxy (true Self-Certainty needs full vocabulary)
- Formula: `C = -(1/nV) * sum sum log(V * p(j|x, y<i))` — we approximate with top-K

### DeepConf (Meta FAIR, 99.9% AIME 2025)
- `per_token_entropy` or `chosen_logprobs` → group confidence computation
- `entropy_p10` → bottom-10% filtering threshold
- Groups tokens into windows, takes mean confidence per window, filters by bottom-10%

### CISC Weighted Majority Voting (ACL 2025)
- `length_normalized_logprob` → response probability confidence
- `answer` → for weighted majority vote aggregation
- Formula: `c_tilde_i = exp(c_i / T) / sum exp(c_j / T)`, then weighted vote

---

## Key Research Insights

### 1. Nobody Uses Logprobs/Entropy for Answer Selection (!)
| Team | Selection Method | Uses Logprobs? |
|------|-----------------|----------------|
| NVIDIA (1st AIMO-2) | GenSelect (LLM-as-judge) | No |
| Imagination (2nd AIMO-2) | Priority-weighted majority | No |
| NuminaMath (1st AIMO-1) | Pure majority vote (N=48) | No |
| CMU-MATH (2nd AIMO-1) | RM-weighted majority | No (separate RM) |
| **Us (40/50)** | **Entropy-gated consensus** | **Yes (novel)** |

Our approach is novel but unvalidated by any winner. All winners use either brute-force majority voting with many samples or a separately-trained selector.

### 2. NVIDIA Stores Logprobs But Doesn't Use Them
NVIDIA's NemoSkills code stores `logprobs`, `tokens`, `top_logprobs` per solution but their winning strategy (GenSelect) ignores all of it in favor of text-based comparison.

### 3. PRIME Is the Cheapest PRM Path
Only needs outcome labels (`is_correct`) + per-token policy logprobs (`chosen_logprobs`). No step annotations, no human labels, no Monte Carlo rollouts. Results: 26.7% on AIME 2024 with only 230K SFT + 150K RL queries.

### 4. DeepConf Achieves 99.9% on AIME 2025
Uses bottom-10% group confidence (from chosen token logprobs) to filter low-quality traces before majority voting. With 512 samples: 99.9% accuracy while reducing total token generation by 84.7%.

### 5. Math-Shepherd Needs Step Boundaries
For Monte Carlo step labeling, you need the conversation state at each step boundary to generate rollouts. Our `turns[]` structure with per-turn content provides this.

---

## Sources
- [PRIME: Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456)
- [Self-Certainty: Scalable Best-of-N (ICLR 2026)](https://arxiv.org/abs/2502.18581)
- [DeepConf: Deep Think with Confidence (Meta FAIR)](https://arxiv.org/abs/2508.15260)
- [CISC: Confidence Improves Self-Consistency (ACL 2025)](https://arxiv.org/abs/2502.06233)
- [OpenMathReasoning / GenSelect (NVIDIA)](https://arxiv.org/abs/2507.17797)
- [Math-Shepherd (ACL 2024)](https://arxiv.org/abs/2312.08935)
- [OmegaPRM (Google DeepMind)](https://arxiv.org/abs/2406.06592)
- [CodePRM (ACL 2025)](https://aclanthology.org/2025.findings-acl.428/)
- [Step-DPO](https://arxiv.org/abs/2406.18629)
- [Pattern-Aware TIR (ICLR 2025)](https://arxiv.org/abs/2509.23292)
- [DPO-VP: Verifiable Pairs](https://github.com/TU2021/DPO-VP)
- [GRPO / DeepSeek-Math](https://arxiv.org/abs/2402.03300)
- [Think Just Enough: Entropy as Confidence](https://arxiv.org/abs/2510.08146)
- [Rewarding Progress: PAVs (ICLR 2025)](https://arxiv.org/abs/2410.08146)
- [Process Reward Models That Think](https://arxiv.org/abs/2504.16828)
- [Lessons of Developing PRMs (Qwen Team)](https://arxiv.org/abs/2501.07301)
- [NeMo-Skills GitHub](https://github.com/NVIDIA-NeMo/Skills)
- [Imagination AIMO2 GitHub](https://github.com/imagination-research/aimo2)
