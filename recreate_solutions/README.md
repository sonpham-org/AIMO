# AIMO Progress Prize 2 — Top 3 Solution Recreations

Faithful recreations of the top 3 solutions from the [AI Mathematical Olympiad Progress Prize 2](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2) competition, plus a unified "best-of-all-worlds" solver for AIMO3.

## Quick Start

All solutions use an OpenAI-compatible API backend (vLLM, lmdeploy, llama-server, etc.).

```bash
# Start your model server first, e.g.:
# vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --port 8080

# Run against AIMO3 reference problems (default)
python -m recreate_solutions.unified_solution.run --api-base http://localhost:8080/v1

# Run against AIMO2 problems (switch with one flag)
python -m recreate_solutions.unified_solution.run --competition aimo2

# Run individual recreations
python -m recreate_solutions.solution_1_nemoskills.run --competition aimo3
python -m recreate_solutions.solution_2_imagination.run --competition aimo3
python -m recreate_solutions.solution_3_aliev.run --competition aimo3
```

---

## Solution Summaries

### 1st Place: NemoSkills (NVIDIA) — 34/50

**Paper:** [arXiv:2504.16891](https://arxiv.org/abs/2504.16891)
**Code:** [NVIDIA-NeMo/Skills](https://github.com/NVIDIA-NeMo/Skills)

| Component | Detail |
|-----------|--------|
| **Base model** | Qwen2.5-14B, fine-tuned on 3.2M synthetic reasoning solutions |
| **Training data** | 540K unique math problems + 1.7M TIR solutions + 566K GenSelect samples |
| **Inference** | Tool-Integrated Reasoning (TIR): model reasons in natural language and calls Python mid-thought to verify/compute, then continues reasoning |
| **Selection** | GenSelect: a trained model evaluates all candidate solutions and picks the best one (outperforms majority voting) |
| **Quantization** | AWQ 4-bit weights |
| **Hardware** | 4x L4 GPUs |

**Key insight:** The model doesn't just "write code to solve" — it *reasons* and uses code as a verification tool within the reasoning chain. This is fundamentally different from code-only approaches.

**What made it win:** Custom training data at massive scale (3.2M solutions), TIR integration, and GenSelect replacing naive majority voting.

---

### 2nd Place: imagination-research (Tsinghua + MSR) — 31/50 private

**Code:** [imagination-research/aimo2](https://github.com/imagination-research/aimo2)

| Component | Detail |
|-----------|--------|
| **Base model** | DeepSeek-R1-Distill-Qwen-14B, fine-tuned with SFT + DPO |
| **DPO training** | 2K pairs, 4 epochs on 8×A800, specifically to reduce output length |
| **Dual prompting** | 7 CoT samples + 8 Code samples per question |
| **Early stopping** | Sample-level (stop on first `\boxed{}`) + Question-level (stop at 5/7 consensus) |
| **Speed control** | Dynamic sample count based on remaining time budget |
| **Engine** | lmdeploy TurboMind (faster than vLLM for this workload) |
| **Quantization** | AWQ 4-bit + KV cache 8-bit |

**Key insight:** Running both CoT and Code prompts doubles solution diversity from the same model. DPO specifically trained to produce shorter (but correct) outputs, saving precious inference time.

**What made it strong:** Efficiency engineering — getting more samples in the time limit through quantization, lmdeploy, early stopping, and DPO length reduction.

---

### 3rd Place: Aliev — ~30/50

**Notebook:** [3rd-place-solution-aliev](https://www.kaggle.com/code/mavicbf/3rd-place-solution-aliev)

| Component | Detail |
|-----------|--------|
| **Base model** | DeepSeek-R1-Distill-Qwen-14B-AWQ (off-the-shelf, no fine-tuning) |
| **Inference** | Standard TIR: reason + code execution + majority voting |
| **Quantization** | Pre-quantized AWQ model |
| **Selection** | Majority voting |

**Key insight:** A strong off-the-shelf reasoning model (DeepSeek-R1) with proper infrastructure can place top-3 without any custom training. The model's built-in reasoning ability does most of the heavy lifting.

**What made it competitive:** Simplicity. No training overhead, no complex pipelines — just good prompt engineering and enough samples for reliable majority voting.

---

## What I Learned

### Common Patterns Across All Winners

1. **Tool-Integrated Reasoning is essential.** All top solutions use code execution during reasoning, not as a separate step. The model reasons, calls code to verify/compute, then continues reasoning.

2. **Majority voting / self-consistency is the baseline.** Every solution samples N times and aggregates. The question is how to do it better (GenSelect, early stopping, etc.).

3. **DeepSeek-R1 family dominates.** All three solutions use DeepSeek-R1-Distill-Qwen-14B or derivatives. Its reasoning ability is best-in-class for this parameter count.

4. **AWQ quantization is free lunch.** 4-bit weight quantization has negligible accuracy loss but doubles throughput and halves memory.

5. **Time management is critical.** The 5-hour Kaggle limit means you must be strategic about how many samples per question. Dynamic budgeting matters.

### What Separates 1st from 3rd

| Factor | 3rd (Aliev) | 2nd (Imagination) | 1st (NemoSkills) |
|--------|------------|-------------------|------------------|
| Custom training | None | SFT + DPO | Full pipeline on 3.2M samples |
| Prompt strategy | Single | Dual (CoT + Code) | TIR-specific |
| Answer selection | Majority vote | Majority vote + early stop | GenSelect (learned) |
| Efficiency tricks | Basic | lmdeploy + KV8 + dynamic speed | Custom model optimized for length |

The gap between 3rd and 1st is primarily **custom training data** and **learned answer selection**.

---

## Unified Strategy for AIMO3

Based on these learnings, the `unified_solution/` combines the best ideas:

```
┌─────────────────────────────────────────────────┐
│                UNIFIED PIPELINE                  │
├─────────────────────────────────────────────────┤
│  1. Dual-Mode TIR Sampling                      │
│     ├── N/2 CoT-TIR samples (reasoning-first)   │
│     └── N/2 Code-TIR samples (computation-first) │
│                                                  │
│  2. Question-Level Early Stopping                │
│     └── Stop when K answers agree                │
│                                                  │
│  3. Answer Selection                             │
│     ├── GenSelect (if answers disagree)          │
│     └── Majority vote (fallback)                 │
│                                                  │
│  4. Verification Round                           │
│     └── Independent check of top answer          │
│                                                  │
│  5. Dynamic Time Management                      │
│     └── Adjust samples based on remaining budget │
└─────────────────────────────────────────────────┘
```

### Recommended Configuration for AIMO3

```bash
# Full competition mode
python -m recreate_solutions.unified_solution.run \
    --competition aimo3 \
    --n-cot 8 --n-code 8 \
    --early-stop 5 \
    --temperature 0.7 \
    --max-tokens 8192 \
    --max-turns 16

# Quick test mode
python -m recreate_solutions.unified_solution.run \
    --competition aimo3 \
    --n-cot 2 --n-code 2 \
    --no-genselect --no-verify
```

### For Kaggle Submission

Use `unified_solution/kaggle_notebook.py` — it's self-contained with:
- vLLM server management
- All prompts and solving logic inline
- Kaggle inference server integration
- Automatic fallback for AIMO2/AIMO3 competition formats

---

## Directory Structure

```
recreate_solutions/
├── README.md                          # This file
├── common/                            # Shared utilities
│   ├── __init__.py
│   ├── sandbox.py                     # Jupyter kernel sandbox
│   ├── answer_extraction.py           # Answer/code parsing
│   └── data_loader.py                 # Problem CSV loader (AIMO2/AIMO3)
├── solution_1_nemoskills/             # 1st place recreation
│   ├── solver.py                      # TIR + GenSelect
│   └── run.py                         # Entry point
├── solution_2_imagination/            # 2nd place recreation
│   ├── solver.py                      # Dual prompt + early stopping
│   └── run.py                         # Entry point
├── solution_3_aliev/                  # 3rd place recreation
│   ├── solver.py                      # Simple TIR + majority vote
│   └── run.py                         # Entry point
└── unified_solution/                  # Best-of-all-worlds
    ├── solver.py                      # Combined pipeline
    ├── run.py                         # Entry point
    └── kaggle_notebook.py             # Self-contained Kaggle submission
```

## Sources

- [NemoSkills 1st Place Writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills)
- [NemoSkills Paper (arXiv:2504.16891)](https://arxiv.org/abs/2504.16891)
- [NemoSkills Code](https://github.com/NVIDIA-NeMo/Skills)
- [imagination-research 2nd Place Writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/imagination-research-2nd-place-solution-team-imagi)
- [imagination-research Code](https://github.com/imagination-research/aimo2)
- [Aliev 3rd Place Writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/aliev-3rd-place-solution-report)
- [Aliev Notebook](https://www.kaggle.com/code/mavicbf/3rd-place-solution-aliev)
