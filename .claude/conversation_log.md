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

---

## 2026-02-01 - Session 3: Kaggle Submission Notebook & Push

### AIMO3 Competition Details (Updated)
- **Hardware:** H100 GPUs (roughly 2x compute of AIMO2's L4s)
- **Time limit:** 9 hours
- **Problems:** 110 math problems (National Olympiad to IMO level)
- **Answer format:** 5-digit integers (0-99999) — makes guessing virtually impossible
- **Subject areas:** Algebra, combinatorics, geometry, number theory
- **All problems are original** — zero data contamination risk

### Kaggle Submission Infrastructure
- AIMO3 uses `kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer`
- predict function signature: `predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame`
- Competition rerun detected via `os.getenv("KAGGLE_IS_COMPETITION_RERUN")`
- Local gateway mode: `inference_server.run_local_gateway(("path/to/test.csv",))`

### Created .ipynb Notebook
- Created `kaggle_push/kaggle_submission.ipynb` — block-by-block version of kaggle_submission.py
- 10 cells: env setup, imports, config, model path, REPL, answer extraction, model wrapper, model load, predict function, inference server
- Each cell has test outputs for easy debugging on Kaggle
- Updated `kernel-metadata.json` to use notebook format

### Kaggle Push Setup
- Auth: `~/.kaggle/kaggle.json` contains username + KGAT token
- Push command: `KAGGLE_API_TOKEN="KGAT_..." ~/.local/bin/kaggle kernels push -p kaggle_push/`
- **Successfully pushed version 1** to https://www.kaggle.com/code/sonphamorg/aimo3-qwen3-30b-tir-solver
- Kernel-metadata.json uses `"model_sources": ["qwen/qwen3-30b-a3b"]` (owner/slug format, not full path)

### Key Findings from Demo Notebook Research
- Could not scrape Kaggle notebooks (dynamic JS rendering blocks all access)
- `kaggle_evaluation` package is Kaggle-internal (not on PyPI) — only available on Kaggle notebooks
- For future reference: `kaggle kernels pull user/notebook-slug` to download notebooks (requires working CLI)

### CRITICAL BUG FIX: Predict Function Signature (from friederrr demo notebook)
User provided full content of `friederrr/aimo-3-submission-demo-notebook-2-2` which revealed:
- **Our predict function had the WRONG parameter name**: `problem` instead of `question`
- AIMO3 test CSV has columns `id` and `question` — the inference server maps these to predict() params by name
- **Correct signature**: `predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame`
- **Our old (broken) signature**: `predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame`
- Fixed in both `kaggle_push/kaggle_submission.py`, `kaggle_submission.py`, and `.ipynb`
- Pushed version 2 to Kaggle

### Demo Notebook Technical Details (friederrr, based on AIMO2 ESP winner)
- **Model**: `Qwen/Qwen3-32B-FP8` (full 32B, FP8 quantized — not the 30B-A3B MoE we chose)
- **vLLM config**: `max_num_seqs=256`, `max_model_len=32768`, `gpu_memory_utilization=0.96`
- **SamplingParams**: `temperature=1.0`, `min_p=0.01`, no seed
- **Cutoff**: 4h45m (very conservative for 9h limit)
- **Dependency installation**: Via "Utility Script" notebook pattern — pip install to `/kaggle/working` in a separate notebook, then link as input
- **Conflicts**: Must uninstall tensorflow, matplotlib, keras, scikit-learn before vLLM
- **torch version**: 2.8.0+cu128
- **Only 5 prompts** (our 10 is more), **max_rounds=1** (we do 3 with TIR feedback)
- **No TIR feedback loop** — they extract code output numbers but don't feed back to model
- **Our advantages over demo**: TIR with feedback (3 rounds), more prompts (10), early stopping on consensus
- **Test problems**: "What is 0x10?", "What is 1-1?", "Solve 4+x=4" — trivial tests, all answered 0
- **GPU**: Single H100, ~79GB VRAM, model takes 32GB, KV cache ~41GB available

---

## 2026-02-01 - Session 4: vLLM + Python 3.12 Compatibility Research

### Context
Kaggle recently upgraded to Python 3.12. Need to find compatible vLLM version for CUDA 12.4 + H100.

### Key Findings

#### vLLM Version / CUDA / PyTorch Compatibility Matrix
| vLLM Version | Default CUDA | PyTorch Version | Python 3.12 |
|---|---|---|---|
| 0.6.x | CUDA 12.1 | torch 2.4-2.5 | Yes (cp38-abi3) |
| 0.7.x | CUDA 12.1 | torch 2.5 | Yes (cp38-abi3) |
| 0.8.0-0.8.4 | **CUDA 12.4** | **torch 2.6** | Yes (cp38-abi3) |
| 0.8.5 | CUDA 12.x | torch 2.8 | Yes |
| 0.9.x | CUDA 12.8 | torch 2.8 | Yes |
| 0.11+ | CUDA 12.9 | torch 2.8-2.9 | Yes |
| 0.15.0 (latest) | CUDA 12.9 | torch 2.9 | Yes |

#### Best Match for Kaggle (CUDA 12.4 + torch 2.6 + Python 3.12)
- **vLLM 0.8.0 through 0.8.4** — built with CUDA 12.4 by default, requires torch 2.6
- This perfectly matches: torch-2.6.0-cp312 + CUDA 12.4 wheels already downloaded

#### Kaggle-Specific Issues
- GitHub Issue #27132: vLLM versions above v0.10 won't run on Kaggle
- Root cause: numpy version conflict — fix is `numpy<2.3`
- Older versions (0.8.x) should work fine on Kaggle

#### Wheel Format
- Modern vLLM uses `cp38-abi3` stable ABI wheels (no separate cp312 wheel needed)
- Compatible with Python 3.8+ including 3.12
- PyPI metadata: `Python >=3.10, <3.14` (for recent versions); older versions support `>=3.9, <3.13`

#### CUDA Forward Compatibility
- CUDA 12.x minor versions are forward-compatible
- Wheels built for CUDA 12.1 work on CUDA 12.4 systems (if driver is new enough)
- But CUDA 12.4-native builds (vLLM 0.8.x) are ideal for CUDA 12.4 environments

### Kaggle Dataset: vLLM Wheels
- **Dataset URL:** https://www.kaggle.com/datasets/sonphamorg/vllm-wheels-cp312
- **Contents:** 104 wheels (~668MB) — vLLM 0.8.0 + all deps EXCEPT torch/nvidia/triton (Kaggle already has these)
- **Key packages:** vllm-0.8.0, xformers-0.0.29.post2, xgrammar-0.1.16, numba-0.60.0, ray-2.53.0, cupy-cuda12x-13.6.0
- **Install on Kaggle:** `pip install --no-index --find-links /kaggle/input/vllm-wheels-cp312/ vllm==0.8.0`
- **Full wheels set** (3.6GB, 139 wheels including torch/nvidia) also exists locally at `kaggle_push/vllm-wheels/` as backup
- **Old dataset:** `sonphamorg/vllm-offline-install-cp312-fix` (5MB, only msgspec + xgrammar) — superseded

### Potential Issues to Watch
- vLLM 0.8.0 requires `numpy<2.0.0` — may conflict with Kaggle's pre-installed numpy 2.x
- If numpy conflict: add `pip install "numpy<2.0"` before vLLM install
- Must uninstall tensorflow, matplotlib, keras, scikit-learn before vLLM (from friederrr demo)
- Running on AMD 8060s locally — cannot test CUDA functionality, need NVIDIA machine for testing

### Skills Updated
- Added "Multi-Agent Sync via Conversation Log" skill to `.claude/skills.md`
- Added "Kaggle Dataset Upload" skill with API token instructions
- Added "vLLM Wheel Management for Kaggle" skill with version compatibility matrix

---

## 2026-02-01 - Session 5: Reorganize Submissions + 2nd-Place Kaggle Notebook

### Reorganized kaggle_submissions/ Directory
- Moved `kaggle_push/*` → `kaggle_submissions/solution_3_aliev_baseline/` (git mv, preserves history)
- Deleted root `kaggle_submission.py` (was duplicate of kaggle_push version)
- Updated `.gitignore` to use `kaggle_submissions/*/` glob patterns
- Created placeholder `kaggle_submissions/solution_1_nemoskills/README.md`

### New Directory Structure
```
kaggle_submissions/
├── solution_3_aliev_baseline/    # Existing Qwen3-30B TIR solver
│   ├── kaggle_submission.py
│   ├── kaggle_submission.ipynb
│   ├── kernel-metadata.json
│   ├── install_vllm.sh
│   └── test_vllm_install.py
├── solution_2_imagination/       # NEW — 2nd place approach
│   ├── kaggle_submission.py
│   ├── kaggle_submission.ipynb
│   ├── kernel-metadata.json
│   └── README.md
└── solution_1_nemoskills/        # Placeholder
    └── README.md
```

### 2nd-Place Solution Kaggle Notebook Created
- `kaggle_submissions/solution_2_imagination/kaggle_submission.py` — full implementation
- Adapted imagination-research approach for AIMO3 (H100, 9hrs, 110 problems)
- Key features:
  - **Dual prompting**: 7 CoT + 8 Code = 15 samples per problem
  - **TIR**: Max 3 rounds of code execution feedback per sample
  - **Sample-level early stopping**: Stop at first \boxed{}
  - **Question-level early stopping**: Stop when 5+ answers agree
  - **Dynamic speed control**: 3 speed levels based on remaining time budget
    - Speed 3: 15 samples (≥180s/q avg remaining)
    - Speed 2: 10 samples (≥90s/q)
    - Speed 1: 5 samples (desperate mode)
  - **Local RTX 4090 support**: `--api-base` or `--local-model` flags for local testing
  - **Model configuration**: Easy switch between imagination-research/deepseek-14b-sft-dpo2, custom fine-tuned, or off-the-shelf AWQ
- Kernel metadata: `sonphamorg/aimo3-imagination-deepseek14b-solver`
- vLLM config: `gpu_memory_utilization=0.95`, `enforce_eager=True`, `max_model_len=32768`
- Per-sample varied seeds (SEED + idx*13) for diversity in vLLM batch generation

### Model Candidates for Kaggle Upload
- `imagination-research/deepseek-14b-sft-dpo2` — Their best DPO checkpoint (primary)
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-AWQ` — Off-the-shelf fallback
- `Qwen/Qwen3-30B-A3B` — Already uploaded (solution 3)
