# Model Merging, Ensembling & Multi-Model Strategies for AIMO3

## Executive Summary

We have massive time headroom (15min used of 9h) and ~15-18GB free VRAM after loading gpt-oss-120b (~60GB). This opens several strategies for combining multiple fine-tuned models/adapters. The most practical approaches in priority order:

1. **vLLM Multi-LoRA serving** — Hot-swap adapters (SFT for generation, GenSelect for selection) on the same base model. Near-zero VRAM overhead per adapter. **Best bang for buck.**
2. **LoRA merging** — Merge SFT + GRPO + GenSelect adapters into one using TIES/DARE/CAT. Zero runtime overhead. **Simplest deployment.**
3. **Dual-model with Sleep Mode** — Run gpt-oss-120b for generation, wake a quantized Qwen3-8B for verification/selection. ~3-6s switch time. **Most architectural flexibility.**
4. **Self-MoA (same model, diverse samples)** — Our current approach is already close to optimal per recent research. **Validates our strategy.**

---

## 1. vLLM Multi-LoRA Serving

### How It Works
vLLM natively supports serving multiple LoRA adapters on a single base model. Adapters are loaded/unloaded dynamically without restarting the server.

### Configuration
```bash
vllm serve gpt-oss-120b \
  --enable-lora \
  --max-loras 3 \
  --max-lora-rank 64 \
  --max-cpu-loras 8 \
  --lora-modules sft-lora=/path/to/sft genselect-lora=/path/to/genselect grpo-lora=/path/to/grpo
```

### Dynamic Loading (Runtime)
```bash
# Enable runtime adapter management
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# Load adapter via API
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -d '{"lora_name": "sft-lora", "lora_path": "/path/to/adapter"}'

# Unload adapter
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -d '{"lora_name": "sft-lora"}'
```

### Per-Request Adapter Selection
```python
# Use SFT-LoRA for generation
response = client.completions.create(model="sft-lora", prompt=..., ...)

# Use GenSelect-LoRA for answer selection
response = client.completions.create(model="genselect-lora", prompt=..., ...)

# Use base model (no adapter)
response = client.completions.create(model="gpt-oss", prompt=..., ...)
```

### VRAM Overhead Per Adapter
LoRA adapter size formula: `2 * hidden_dim * rank * num_target_layers * bytes_per_param`

For gpt-oss-120b (hidden_dim ~= 4096 for active params, MoE with 5.1B active):
- Rank 16: ~50-100MB per adapter (negligible)
- Rank 32: ~100-200MB per adapter
- Rank 64: ~200-400MB per adapter

With 15-18GB free VRAM, we can easily hold 3-5 adapters simultaneously.

### Performance Impact
- **Throughput**: Up to 50% throughput drop reported with LoRA vs base model on A100 (vLLM issue #10062)
- **Latency**: Minimal per-request overhead for adapter application
- **LRU cache**: `max_cpu_loras` keeps inactive adapters in CPU RAM for fast swap
- With our 16-attempt setup running ~15min total, even a 50% throughput drop still leaves ~8.5h of headroom

### Practical AIMO3 Architecture
```
Problem → [SFT-LoRA: Generate 16 solutions with TIR]
       → [GenSelect-LoRA: Score each solution]
       → [Entropy-gated consensus on GenSelect scores]
       → Final answer
```

This is the **recommended primary approach**. It matches exactly what NVIDIA did to win AIMO2 (SFT for generation + GenSelect for selection).

### Limitations
- vLLM LoRA support requires the base model architecture to support it (check vLLM docs for gpt-oss/Qwen3-MoE compatibility)
- MoE models: LoRA is typically applied to attention layers, not expert FFNs — should work but verify
- `--enable-lora` flag may conflict with some optimization flags

---

## 2. LoRA Merging (Weight Space)

### When to Use
When you have 2+ LoRA adapters trained on the same base model and want a single merged adapter for simplest deployment (no multi-LoRA overhead, no adapter switching logic).

### Methods Ranked by Effectiveness

#### 2a. CAT (Concatenation with Learned Weights) — BEST
From "LoRA Soups" (COLING 2025 Industry):
- Concatenates LoRA matrices and learns **layer-wise merging coefficients**
- Outperforms TIES, DARE, and LoRA Hub by **43% on math word problems**
- Requires a small calibration dataset (~100-1000 examples) to learn coefficients
- Implementation: https://github.com/aksh555/LoRA-Soups

```python
# Pseudocode for CAT merging
merged_A = concat([lora1_A, lora2_A, lora3_A], dim=0)  # Stack A matrices
merged_B = concat([lora1_B, lora2_B, lora3_B], dim=1)  # Stack B matrices
# Learn per-layer alpha weights via small calibration set
```

#### 2b. TIES-Merging
Three-step method:
1. **Trim**: Remove redundant (near-zero) parameters
2. **Elect**: Resolve sign conflicts via majority vote
3. **Merge**: Average parameters with matching signs

Best for: Combining adapters that may have conflicting gradients (e.g., SFT vs GRPO)

```python
# Using PEFT
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "sft-adapter")
model.load_adapter("grpo-adapter", adapter_name="grpo")
model.add_weighted_adapter(
    adapters=["default", "grpo"],
    weights=[1.0, 0.8],
    adapter_name="merged",
    combination_type="ties",
    density=0.5  # Keep top 50% of parameters
)
```

#### 2c. DARE (Drop And Rescale)
- Randomly drops 90% of delta parameters, rescales remaining
- Reduces interference between adapters before merging
- Often used as preprocessing before TIES or linear merge

```python
model.add_weighted_adapter(
    adapters=["default", "grpo"],
    weights=[1.2, 1.0],  # >1.0 weights often work better
    adapter_name="merged",
    combination_type="dare_ties",
    density=0.1  # Keep only 10% of params
)
```

#### 2d. Linear Merge (Model Soups)
- Simple weighted average of adapter parameters
- Works well when adapters are trained from same initialization
- No hyperparameters to tune beyond weights

```python
model.add_weighted_adapter(
    adapters=["sft", "grpo", "genselect"],
    weights=[0.5, 0.3, 0.2],
    adapter_name="merged",
    combination_type="linear"
)
```

#### 2e. Task Arithmetic
- Compute task vectors (adapter - base), combine, add back to base
- Works for full model merges, less common for LoRA

### Tools
- **PEFT** (HuggingFace): Built-in `add_weighted_adapter()` supports linear, TIES, DARE, SVD
- **mergekit** (Arcee AI): Full-featured CLI tool, supports SLERP, TIES, DARE, task arithmetic, MoE conversion
  - Can run on CPU with out-of-core processing (no GPU needed for merging)
  - Supports extracting LoRA from fine-tuned models (`mergekit-extract-lora`)
  - MoE creation: `mergekit-moe` converts multiple dense models into MoE
- **LoRA Soups**: CAT method implementation at https://github.com/aksh555/LoRA-Soups

### MoE-Specific Considerations
- gpt-oss-120b is already an MoE model (117B total, 5.1B active)
- LoRA is typically applied to **attention layers only** (shared across experts)
- Merging LoRA adapters on an MoE base model should work normally since LoRA targets attention, not expert FFNs
- mergekit has an open issue (#426) about merging LoRA with MoE models — may require manual verification

### Recommendation
1. If we produce SFT + GRPO adapters for the SAME task (math generation): use **TIES or DARE** merge
2. If we produce SFT (generation) + GenSelect (selection) for DIFFERENT tasks: use **CAT** with calibration data, or keep separate and use multi-LoRA serving instead
3. Always validate merged adapter on a held-out set before deploying

---

## 3. Multi-Model Ensembling

### 3a. Can We Fit Two Models on H100?

**gpt-oss-120b**: ~60GB VRAM at gpu_memory_utilization=0.96 → ~77GB used (including KV cache)
**Remaining**: ~3GB with current settings

Options:
1. **Lower gpu_memory_utilization to 0.85**: Frees ~9GB → enough for Qwen3-8B-4bit (~5GB) but tight
2. **vLLM Sleep Mode** (recommended): Hibernate gpt-oss-120b, load Qwen3-8B, do verification, hibernate Qwen3-8B, wake gpt-oss-120b

### 3b. vLLM Sleep Mode for Model Switching
```bash
# Start gpt-oss-120b
vllm serve gpt-oss-120b --enable-sleep-mode --port 8000

# After generation phase, sleep the big model
curl -X POST http://localhost:8000/sleep?level=1  # Offload to CPU RAM

# Start Qwen3-8B for verification (separate process on now-free GPU)
vllm serve qwen3-8b-4bit --enable-sleep-mode --port 8001

# After verification, sleep small model
curl -X POST http://localhost:8001/sleep?level=2  # Discard weights

# Wake big model for next problem
curl -X POST http://localhost:8000/wake_up
```

**Wake times on H100:**
- Level 1 (CPU offload): ~3-6s for large models (keeps weights in CPU RAM, ~60-80GB RAM needed)
- Level 2 (minimal RAM): ~7-8s for 30B+ models (reloads from disk)
- Both are 18-200x faster than cold start (~60s)

**Compatibility**: Works with TP, PP, and EP (Expert Parallelism) — confirmed for MoE models.

### 3c. Does Model Diversity Help?

**Critical finding — "Rethinking Mixture-of-Agents" (Li et al., Feb 2025):**
> Self-MoA (sampling multiple outputs from ONE strong model) outperforms mixed-model MoA by **6.6% on AlpacaEval** and **3.8% average across MMLU, CRUX, MATH**

Key insight: **Intra-model diversity > inter-model diversity** for math reasoning.

This means:
- Our current approach (16 samples from gpt-oss-120b + entropy-gated consensus) is already close to optimal
- Adding a weaker model (Qwen3-8B) for generation is unlikely to help and may hurt
- A smaller model is still valuable as a **verifier/selector**, not as a generator

### 3d. When Multi-Model DOES Help
- **Different roles**: Big model generates, small model verifies/selects (NVIDIA's approach)
- **Different modalities**: CoT-specialized model + TIR-specialized model
- **Difficulty routing**: Fast model for easy problems, slow model for hard problems (see Section 5)

---

## 4. Mixture-of-Agents (MoA)

### Original MoA (Wang et al., 2024)
Architecture: Multiple LLM agents in layers. Each layer's agents see all outputs from the previous layer.
- **Proposers**: Generate initial diverse responses
- **Aggregators**: Synthesize responses into higher-quality output
- Performance: Open-source MoA beat GPT-4o on AlpacaEval 2.0 (65.1% vs 57.5%)

### Self-MoA (Li et al., 2025) — More Relevant
- Use the SAME model as both proposer and aggregator
- Sample N diverse outputs → aggregate into final answer
- Outperforms mixed-model MoA in most scenarios
- **This is essentially what our entropy-gated consensus already does**

### Practical Application for AIMO3
Our current pipeline is already a form of Self-MoA:
```
gpt-oss-120b (temp=1.0, 16 seeds) → 16 diverse solutions → entropy-gated consensus → answer
```

To improve, we should focus on **better aggregation** (GenSelect, learned scoring) rather than adding more models.

### A-HMAD (Adaptive Heterogeneous Multi-Agent Debate)
- Assigns distinct roles (logical reasoning, factual verification, strategic planning)
- 4-6% higher accuracy, 30% fewer errors vs standard methods
- Could be adapted: assign different system prompts to different attempts
  - Attempt 1-4: "Solve algebraically"
  - Attempt 5-8: "Solve with code verification"
  - Attempt 9-12: "Solve with multiple approaches"
  - Attempt 13-16: "Find edge cases and verify"
- **Risk**: Our feb5 "entropy-plus" with mixed prompts scored 33/50 vs 40/50 with uniform prompts. Prompt diversity HURT in practice.

---

## 5. Difficulty Routing

### Concept
Route easy problems to a fast/cheap model, hard problems to a slow/expensive model. Or allocate more attempts to harder problems.

### Difficulty Estimation Methods

#### 5a. First-Pass Confidence
1. Run 1 quick attempt on each problem
2. If answer is confident (low entropy): accept it, move on
3. If uncertain (high entropy): allocate more attempts

This is essentially our **adaptive resampling** approach from feb5_adaptive.

#### 5b. LLM Internal Representations
Research shows LLMs encode difficulty in hidden states (arxiv 2510.18147):
- Can probe hidden representations to predict difficulty
- Requires training a small classifier on top of embeddings
- Not practical without pre-computed difficulty labels for AIMO3 problems

#### 5c. Process Reward Models (PRMs)
- Estimate difficulty via step-level reward scores
- Calibrated PRMs produce probability ranges rather than point estimates
- Could use a PRM to decide: "This problem needs 16 attempts" vs "This needs 32"

#### 5d. Practical Implementation for AIMO3
```python
class AdaptiveRouter:
    def allocate_attempts(self, problem_entropy_from_first_pass):
        if entropy < 1.0:  # Very confident
            return 4  # Few attempts needed
        elif entropy < 3.0:  # Moderate confidence
            return 8
        elif entropy < 5.0:  # Uncertain
            return 16
        else:  # Very uncertain
            return 32  # Throw everything at it
```

**Caveat**: Our experiments show that more attempts with current selection strategy actually HURT (16 attempts → 29/50 vs 8 attempts → 40/50). So routing only helps if we also fix the selection strategy for high-N scenarios.

---

## 6. Cross-Model Answer Aggregation

### 6a. Entropy-Weighted Voting (Current)
```
score(answer) = sum(1/entropy for each attempt producing this answer)
```
Works well for single-model, but needs adaptation for multi-model.

### 6b. Cross-Model Weighted Voting
```
score(answer) = sum(model_weight * (1/entropy) for each attempt)
```
Where `model_weight` reflects model reliability on math tasks.

### 6c. GenSelect (NVIDIA, AIMO2 Winner)
- Trained model that takes N candidate solutions and selects the best one
- Input: Problem + N solution summaries (correct + incorrect)
- Output: Selected answer
- Training data: 566K GenSelect examples from OpenMathReasoning
- **Key result**: GenSelect + generation SFT achieved 93.3% on competition benchmark
- This is a **learned selector**, far superior to majority voting

### 6d. Verifier/Reward Model Scoring
- Use a separate model to score each solution
- Process Reward Models (PRMs) score each reasoning step
- Outcome Reward Models (ORMs) score final answers
- Best-of-N with reward model scoring consistently outperforms majority vote

### Recommendation for AIMO3
Priority order for improving answer selection:
1. **Train GenSelect LoRA** on gpt-oss-120b (requires OpenMathReasoning GenSelect data)
2. **Use GenSelect at inference** via multi-LoRA: generate with SFT-LoRA, select with GenSelect-LoRA
3. If GenSelect not feasible: improve entropy-gated consensus with better calibration

---

## 7. Recommended Architecture for AIMO3

### Phase 1: Immediate (No Fine-Tuning Required)
Keep current architecture but optimize selection:
```
gpt-oss-120b (base) → 16 attempts → improved entropy-gated consensus → answer
```
Focus: Fix the selection strategy that degrades at N>8.

### Phase 2: With SFT + GenSelect LoRA Adapters
```
gpt-oss-120b + multi-LoRA:
  1. SFT-LoRA generates 16-32 solutions per problem
  2. GenSelect-LoRA evaluates/scores each solution
  3. Entropy-gated consensus on GenSelect scores
```
This mirrors NVIDIA's winning approach. Multi-LoRA overhead is negligible.

### Phase 3: With Additional Small Model (Ambitious)
```
gpt-oss-120b (generation, 16 attempts)
  ↓ Sleep Mode switch (~5s)
Qwen3-8B-4bit (verification/GenSelect, fast)
  ↓ Sleep Mode switch (~5s)
gpt-oss-120b (next problem)
```
Only if Qwen3-8B is fine-tuned specifically for verification/selection.

### Phase 4: Full Pipeline (Maximum Performance)
```
For each problem:
  1. Quick difficulty estimate (1 fast attempt)
  2. Allocate attempts based on difficulty (4-32)
  3. Generate with SFT-LoRA (diverse seeds, temp=1.0)
  4. Score with GenSelect-LoRA
  5. Verify top candidate with code execution
  6. Entropy-gated consensus on verified solutions
```

---

## 8. Key Takeaways

| Strategy | Effort | Expected Impact | Risk |
|----------|--------|-----------------|------|
| Multi-LoRA (SFT + GenSelect) | Medium | **HIGH** (mirrors AIMO2 winner) | Low — vLLM native support |
| LoRA merging (TIES/CAT) | Low | Medium | Low — well-studied |
| Sleep Mode dual-model | High | Medium | Medium — timing complexity |
| Mixed-model MoA | High | **LOW** | High — Self-MoA beats mixed MoA |
| Difficulty routing | Medium | Medium | Medium — needs good estimator |
| Prompt diversity | Low | **NEGATIVE** | High — proven to hurt (33 vs 40) |

### Critical Insight
**Self-MoA research validates our current approach.** The biggest gains will come from:
1. Better selection (GenSelect LoRA), not more models
2. More attempts (16-32) with a selection strategy that scales
3. SFT LoRA to improve per-attempt solve rate

### What NOT to Do
- Do NOT mix different models for generation (Self-MoA > mixed MoA)
- Do NOT use prompt diversity across attempts (proven to hurt)
- Do NOT run two models simultaneously (VRAM too tight)
- Do NOT merge GenSelect + SFT into one adapter (they serve different roles — keep separate with multi-LoRA)

---

## Sources

### vLLM Multi-LoRA
- [vLLM LoRA Adapters Documentation](https://docs.vllm.ai/en/latest/features/lora/)
- [vLLM Multi-LoRA Performance Issue #10062](https://github.com/vllm-project/vllm/issues/10062)
- [vLLM Multi-LoRA Memory Issue #20160](https://github.com/vllm-project/vllm/issues/20160)
- [Benchmarking Multi-LoRA Adapters on vLLM (2025)](https://uksystems.org/workshop/2025/pdfs/paper24.pdf)

### LoRA Merging
- [LoRA Soups: Merging LoRAs for Practical Skill Composition (COLING 2025)](https://arxiv.org/abs/2410.13025)
- [PEFT Model Merging Guide](https://huggingface.co/docs/peft/en/developer_guides/model_merging)
- [mergekit (Arcee AI)](https://github.com/arcee-ai/mergekit)
- [From Task-Specific to Unified: Review of Model Merging](https://arxiv.org/html/2503.08998v1)

### Multi-Model Ensembling
- [Mixture-of-Agents (Wang et al., 2024)](https://arxiv.org/abs/2406.04692)
- [Rethinking Mixture-of-Agents: Self-MoA > Mixed MoA (Li et al., 2025)](https://arxiv.org/abs/2502.00674)
- [Diversity of Thought Improves Reasoning (2023)](https://arxiv.org/html/2310.07088)
- [Efficient Dynamic Ensembling for Multiple LLM Experts (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/0900.pdf)

### NVIDIA GenSelect / AIMO2
- [AIMO-2 Winning Solution (NVIDIA, 2025)](https://arxiv.org/abs/2504.16891)
- [OpenMath-Nemotron Release](https://www.marktechpost.com/2025/04/24/nvidia-ai-releases-openmath-nemotron-32b-and-14b-kaggle/)

### vLLM Sleep Mode
- [Zero-Reload Model Switching with vLLM Sleep Mode (Oct 2025)](https://blog.vllm.ai/2025/10/26/sleep-mode.html)
- [vLLM Sleep Mode Documentation](https://docs.vllm.ai/en/latest/features/sleep_mode/)

### Difficulty Routing
- [LLMs Encode How Difficult Problems Are (2025)](https://arxiv.org/html/2510.18147v1)
- [Easy2Hard-Bench (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/4e6f22305275966513990f53cec908e0-Paper-Datasets_and_Benchmarks_Track.pdf)

### ASTER
- [ASTER: Agentic Scaling with Tool-integrated Extended Reasoning (2025)](https://arxiv.org/html/2602.01204)
