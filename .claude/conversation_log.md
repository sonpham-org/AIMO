# AIMO Project - Conversation Log

## 2026-01-31 - Session 1

### Context carried from previous session:
- Discussed the nature of tool calls in AIMO competition solutions
- Key insight: The tool calls are essentially a **Python executor that is regularly triggered**
- Each solution allows approximately **16 back-and-forth rounds** between the LLM and the Python executor (Tool-Integrated Reasoning / TIR pattern)
- This is the core mechanism behind the top solutions' "code interpreter" approach

---

## 2026-01-31 - Session 2: Deep Dive on Top-3 AIMO2 Solutions

### Competition Rules (AIMO Progress Prize 2)
- **Platform:** Kaggle, offline Jupyter notebook environment
- **Hardware:** 4x NVIDIA L4 GPUs (96 GB VRAM total)
- **Time limit:** 5 hours for the entire test set
- **Problems:** 50 unreleased Olympiad-level math problems (national olympiad level: BMO, USAMO), split across public/private leaderboards
- **Answer format:** Integer answers (0-99999)
- **Submissions:** 1 per day during competition; private leaderboard evaluated only once
- **Prize pool:** $507,904 for top 5; remainder reserved for >=47/50 score (rolls over if unclaimed)
- **Open-source requirement:** Winners must publish code, methodology, data, and model parameters
- **Anti-contamination:** Problems kept hidden; no data leakage; evaluator access removed before testing

---

### Solution 1: NemoSkills (NVIDIA) — 1st Place, 34/50 private

**Sources:**
- [Kaggle Writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills)
- [arXiv Paper: 2504.16891](https://arxiv.org/abs/2504.16891)
- [GitHub: NVIDIA/NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills)
- [Dataset: nvidia/OpenMathReasoning on HuggingFace](https://huggingface.co/datasets/nvidia/OpenMathReasoning)

#### (1) Key Strategy
Three pillars:
1. **Large-scale synthetic training data** — 306K unique problems with 3.2M CoT solutions
2. **Tool-Integrated Reasoning (TIR)** — Model reasons in natural language, calls Python mid-thought to verify/compute, then continues reasoning. This is NOT "write code to solve" — the model *reasons* and uses code as a verification/computation tool within the reasoning chain
3. **Generative Solution Selection (GenSelect)** — A trained model that evaluates all candidate solutions and picks the best one, significantly outperforming naive majority voting

#### (2) Role of Training Data
Training data was THE key differentiator. Their 3.2M solution dataset enabled:
- Teaching the model the TIR pattern (reason → code → verify → continue)
- Creating GenSelect training pairs (566K samples)
- Achieving state-of-the-art on math benchmarks

#### (3) Where They Got Training Data
- **Problem sources:** Primarily **AoPS (Art of Problem Solving) forums** — scraped problem statements, refined with Qwen2.5-32B-Instruct. Also includes problems from the **MATH dataset**
- **Solution generation:** Solutions generated synthetically using **DeepSeek-R1** and **QwQ-32B** as teacher models
- **Initial count:** Started with ~540K problems, filtered down to **306K unique problems** that made it into the final OpenMathReasoning release
- **Dataset composition:** 306K problems → 3.2M CoT solutions + 1.7M TIR solutions + 566K GenSelect samples

#### (4) How They Judged Problem Quality
- Problems extracted from AoPS were preprocessed and refined using Qwen2.5-32B-Instruct
- `problem_type` field categorizes as: `has_answer_extracted`, `no_answer_extracted`, or `converted_proof` (proof problems converted to answer-type)
- Quality filtering through iterative training: generate solutions → check correctness → keep problems where model can be verified → retrain
- `pass_rate_72b_tir` metric used as quality signal
- Problems tagged by source forum difficulty level (e.g., `aops_c4_high_school_math`, `aops_c6_high_school_olympiads`)

#### (5) Training Within Competition Time Constraints
- **Training happened BEFORE submission** — there is no training during the 5-hour evaluation window
- Base model: **Qwen2.5-14B** (not DeepSeek-R1 like the others — they trained their own from scratch)
- Iterative pipeline: train model → generate solutions → filter for quality → retrain on improved data
- Final model quantized to **AWQ 4-bit** for inference — negligible accuracy loss but doubles throughput, halves memory
- The 5-hour limit only applies to **inference**, not training. Teams prepare everything offline

#### (6) What They Do at Test Time
- Load AWQ 4-bit quantized model on 4x L4 GPUs
- For each problem: generate **N candidate solutions** using TIR (each solution involves multiple LLM ↔ Python executor rounds, up to ~16 turns)
- Each TIR round: model outputs reasoning + Python code block → code is executed in sandbox → output fed back to model → model continues reasoning
- After all candidates generated: **GenSelect model evaluates all solutions** and picks the best one (instead of simple majority vote)
- Time budget managed across all 50 problems

#### (7) External Materials Prepared
- Pre-trained and fine-tuned model checkpoints (uploaded to Kaggle/HF)
- AWQ-quantized model weights
- OpenMathReasoning dataset (306K problems, 3.2M+ solutions) — released as CC-BY-4.0
- GenSelect model checkpoint
- Full NeMo-Skills pipeline (open-sourced)
- Sandbox/code execution infrastructure

---

### Solution 2: imagination-research (Tsinghua + MSR) — 2nd Place, 31/50 private

**Sources:**
- [Kaggle Writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/imagination-research-2nd-place-solution-team-imagi)
- [GitHub: imagination-research/aimo2](https://github.com/imagination-research/aimo2)
- [HuggingFace Collection](https://huggingface.co/collections/imagination-research/aimo2)

#### (1) Key Strategy
Three-part approach:
1. **Reasoning-Oriented Training** — SFT then DPO, specifically to make the model shorter but correct
2. **Efficiency Optimization** — lmdeploy TurboMind engine, W4A16 + KV8 quantization
3. **Inference-Time Strategies** — Dual prompting, self-consistency, early stopping, dynamic speed control

#### (2) Role of Training Data
Two-stage training:
- **SFT stage:** Teaches the model math reasoning from curated high-quality trajectories
- **DPO stage:** Specifically reduces output length while maintaining accuracy — this is critical for fitting more samples within the 5-hour time limit

#### (3) Where They Got Training Data
- **SFT data:** Combined **Light-R1 stage2 data** and **Limo dataset** (duplicates removed). Both contain "high-difficulty math problems' reasoning trajectories generated from DeepSeek-R1"
- **DPO data:** Constructed from model's own outputs using four filtering criteria:
  - **Correctness:** Chosen response must be correct; rejected may be either
  - **Min Length:** `len(chosen) > min_threshold` (filter out trivially short)
  - **Length ratio:** `len(chosen) < ratio_threshold * len(rejected)` (chosen must be shorter)
  - **Similarity:** Sentence transformer embeddings to ensure diversity of pairs
- Final DPO dataset: **2K pairs** (dpo-1 used in final submission)

#### (4) How They Judged Problem Quality
- Relied on curated datasets (Light-R1, Limo) that were already filtered for high-difficulty math
- DPO pairs filtered by correctness verification against known answers
- Tried GRPO but "did not observe significant improvement on accuracy" after 4 training runs

#### (5) Training Within Competition Time Constraints
- **SFT:** 8 epochs on 8×A800 GPUs, ~11 hours
- **DPO:** 4 epochs on 8×A800 GPUs, ~40 hours (using 360-LLaMA-Factory's sequence parallelism)
- All training done **before submission** — not during the 5-hour window
- DPO specifically optimized for **shorter outputs** (the entire point was to reduce inference cost)
- After fine-tuning, model less inclined to output code: ~11 out of 16 code-prompted samples still produced text-only reasoning. But when it did produce code, conditional accuracy was slightly higher (45-55% vs 42% pre-fine-tuning)

#### (6) What They Do at Test Time
- **Dual prompting:** For each problem, generate **7 CoT samples + 8 Code samples** (15 total)
  - CoT prompt: "reason step by step to put the answer in \\boxed{}"
  - Code prompt: "provide the python code...put the final answer in \\boxed{}"
- **Sample-level early stopping:** Stop generating a sample at the first `\boxed{}` answer or first successfully executable code output (avoids the "self-doubt rewriting" waste)
- **Question-level early stopping:** Stop all remaining samples when majority agrees (e.g., 5/7 answers match)
- **Streaming answer extraction:** Continuously extract answers during generation, not just at the end
- **Dynamic speed adjustment (`adjust_speed` module):**
  - Default: speed 3, 15 samples per question
  - If avg remaining time < 5 min per question: speed 1 (fastest), reduce to 10 samples
- **Inference engine:** lmdeploy TurboMind (chosen for higher throughput and shorter init time than vLLM)
- **Quantization:** AWQ 4-bit weights + 8-bit KV cache (W4KV8) — reduces time per output token by ~20% vs W4KV16, and overall latency by ~40% vs FP16
- **Answer aggregation:** Majority voting across all samples for each question

#### (7) External Materials Prepared
- Pre-fine-tuned model checkpoints uploaded to HuggingFace (deepseek-14b-sft-dpo2, deepseek-14b-sft-dpo4)
- AWQ-quantized weights
- lmdeploy inference configuration
- Custom training pipeline using 360-LLaMA-Factory
- AIME 2025 reference set and 30-problem test set for validation
- Analysis utilities (early stop visualization, answer aggregation scripts)

---

### Solution 3: Aliev — 3rd Place, ~30/50 private

**Sources:**
- [Kaggle Writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/aliev-3rd-place-solution-report)
- [Notebook: 3rd-place-solution-aliev](https://www.kaggle.com/code/mavicbf/3rd-place-solution-aliev)

#### (1) Key Strategy
**Simplicity:** Use a strong off-the-shelf reasoning model with no fine-tuning. Just run inference with TIR + majority voting.

#### (2) Role of Training Data
**None.** No custom training. This is the baseline approach — use a pre-trained model as-is.

#### (3) Where They Got Training Data
N/A — used a pre-quantized model from Kaggle Models: `shelterw/deepseek-r1/Transformers/deepseek-r1-distill-qwen-14b-awq/1`

#### (4) How They Judged Problem Quality
N/A — no training involved.

#### (5) Training Within Competition Time Constraints
No training needed. The entire approach is inference-only, which is its biggest advantage: zero setup time, zero training cost. The model is ready to use immediately.

#### (6) What They Do at Test Time
- Load **DeepSeek-R1-Distill-Qwen-14B-AWQ** (pre-quantized 4-bit)
- Standard TIR: model reasons → generates Python code → code executed → output fed back → continues reasoning
- Multiple samples per problem with **majority voting** for answer selection
- No early stopping, no GenSelect, no dual prompting — just straightforward TIR + voting

#### (7) External Materials Prepared
- Pre-quantized model weights (from Kaggle Models)
- Basic inference notebook
- That's it — the strength is in simplicity

---

### Cross-Solution Comparison

| Factor | 3rd (Aliev) | 2nd (Imagination) | 1st (NemoSkills) |
|--------|------------|-------------------|------------------|
| Base model | DeepSeek-R1-Distill-Qwen-14B | DeepSeek-R1-Distill-Qwen-14B | Qwen2.5-14B (custom trained) |
| Custom training | None | SFT + DPO (~51 hrs) | Full pipeline on 3.2M samples |
| Training data | N/A | Light-R1 + Limo + 2K DPO pairs | 306K problems, 3.2M solutions from AoPS |
| Prompt strategy | Single TIR | Dual (CoT + Code) | TIR-specific |
| Answer selection | Majority vote | Majority vote + early stop | GenSelect (learned) |
| Inference engine | vLLM | lmdeploy TurboMind | vLLM |
| Quantization | AWQ 4-bit | AWQ 4-bit + KV 8-bit | AWQ 4-bit |
| Score (private) | ~30/50 | 31/50 | 34/50 |

### Key Takeaways
1. **The gap between 1st and 3rd is custom training + learned selection**, not just prompt engineering
2. **All solutions use TIR** — code execution during reasoning is table stakes
3. **Majority voting is baseline** — improvements come from GenSelect or early stopping
4. **DeepSeek-R1-Distill-Qwen-14B is the sweet spot** — 2nd and 3rd use it directly; 1st trained their own 14B from Qwen2.5
5. **AWQ quantization is free lunch** — everyone uses it
6. **Time management is critical** — DPO for shorter outputs (2nd), dynamic speed control (2nd), GenSelect vs more samples (1st)
7. **Training happens BEFORE submission** — the 5-hour limit is inference-only. Teams prepare models, data, and infrastructure weeks/months in advance
8. **Open-source requirement** means all code, data, and weights are published after competition

---

### Implementation: Aliev-style Solver with Qwen3-30B-A3B
- Created `solve_aimo3.py` — local test script against 10 reference problems
- Created `kaggle_submission.py` — self-contained Kaggle notebook for competition submission
- Model choice: **Qwen3-30B-A3B** (MoE, 30B total / 3B active — fast, beats QwQ-32B)
  - Thinking variant (Qwen3-30B-A3B-Thinking-2507) is ideal but requires `--enable-reasoning --reasoning-parser deepseek_r1` in vLLM
  - AWQ quantized versions available from QuantTrio on HuggingFace
- Architecture: TIR + majority voting + parallel attempts + early stopping + dynamic time budget
- AIMO3 differences from AIMO2: H100 GPUs (vs L4), 9-hour limit (vs 5h), ~50 problems
- For Kaggle: model must be uploaded as a Kaggle Model or Dataset input at `/kaggle/input/...`
- Key parameters: 12 attempts, 16 max turns, temperature 0.6, 8 parallel workers
