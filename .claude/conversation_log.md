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

---

## 2026-02-01 - Session 6: vLLM Python 3.12 Wheels — Resolved

### Problem
Kaggle upgraded to Python 3.12. vLLM needed to be installed offline (no internet during competition). Previous attempts with custom wheel sets failed due to dependency conflicts (numpy<2.0 requirement, torch version pinning).

### What Worked
- **Dataset:** `sonphamorg/vllm-wheels-py312-cu129` (created by other Claude agent on NVIDIA 4090)
- **vLLM version:** 0.15.0 (latest)
- **CUDA:** 12.9, torch 2.9.1+cu129
- **Install command:** `bash /kaggle/input/vllm-wheels-py312-cu129/install.sh`
- **Confirmed working:** vLLM imports OK, model loads OK on Kaggle H100

### What Didn't Work
- Our `sonphamorg/vllm-wheels-cp312` dataset (vLLM 0.8.0, CUDA 12.4) failed because:
  1. vLLM 0.8.0 requires `numpy<2.0.0` — Kaggle has numpy 2.x, no internet to downgrade
  2. Even with numpy wheel included, `--no-index` still tried to resolve torch==2.6.0 from PyPI
  3. `--no-deps` approach worked for vLLM itself but required manually listing all missing deps, and Kaggle's `%%bash` cell broke `\` line continuations (trailing whitespace issue)

### Notebook Updated
- Cell 16 now installs from `vllm-wheels-py312-cu129/wheels/`
- kernel-metadata.json includes `dataset_sources: ["sonphamorg/vllm-wheels-py312-cu129"]`
- Pushed to GitHub

### Key Lesson
For Kaggle offline wheel datasets: include ALL dependencies (including torch, numpy, CUDA libs) in the wheels directory. Don't try to be clever with a "slim" set that relies on Kaggle's pre-installed packages — version mismatches and pip's dependency resolver will cause pain. The full-fat approach (like `vllm-wheels-py312-cu129`) just works.

---

## 2026-02-02 - Session 7: Weighted Entropy vs TIR Solver Analysis

### Submissions Analyzed
1. **Weighted entropy** (`sonphamorg/41-50-aimo-3-weighted-entropy`) — Score: **38/50** (success)
2. **TIR solver** (`sonphamorg/aimo3-qwen3-30b-tir-solver`) — **Failed**

### Weighted Entropy Notebook — Key Architecture
- **Model**: `gpt-oss-120b` (MoE, 117B total / 5.1B active, MXFP4) via `danielhanchen/gpt-oss-120b`
- **Library**: `openai_harmony` for tokenization, chat templates, stop token management
- **Answer selection**: Entropy-weighted voting — each attempt's mean token entropy (Shannon, from top-5 logprobs) is computed; answers weighted by `1/entropy` (confident answers count more)
- **Code execution**: 16 persistent Jupyter kernels via `jupyter_client.KernelManager` (stateful across turns)
- **TIR**: Up to 128 turns per attempt, streaming with logprobs
- **Parallelism**: 16 threaded workers → vLLM OpenAI-compatible server
- **Config**: 8 attempts, early stop at 4 matching, temp=1.0, min_p=0.02, FP8 KV cache
- **Time budget**: Dynamic per-problem (270-900s), notebook limit 17400s
- **Dependencies**: `unsloth`, `trl`, `vllm`, `openai_harmony` from `aimo-3-utils` wheels tarball

### TIR Solver Notebook — Key Architecture
- **Model**: Qwen3-30B-A3B (MoE, 30B total / 3B active)
- **Code execution**: Subprocess-based (stateless, fresh process per code block)
- **Answer selection**: Simple majority vote (`Counter.most_common`)
- **TIR**: 3 rounds max
- **Parallelism**: Sequential vLLM Python API batch
- **Config**: 10 prompts (5 unique × 2), temp=0.6, single seed

### Why the Weighted Entropy Notebook Succeeded
1. **Model quality**: gpt-oss-120b >> Qwen3-30B-A3B on olympiad math
2. **Stateful code execution**: Persistent Jupyter kernels allow multi-step computations across turns
3. **Deep TIR**: 128 turns allows real iterative problem-solving and self-correction
4. **Entropy weighting**: Filters out low-confidence guesses in favor of answers the model was sure about
5. **Parallel workers**: 16 concurrent attempts maximize samples within time budget
6. **Streaming logprobs**: Enables entropy calculation + early answer detection mid-stream

### Why the TIR Solver Failed (TECHNICAL — not approach failure)
The TIR solver was **never successfully evaluated** on the competition problems. It failed due to **infrastructure/dependency errors** (vLLM install conflicts, wheel compatibility issues, runtime crashes) before it could solve any problems. The approach itself (TIR + majority vote with Qwen3-30B-A3B) was never tested against the actual problem set. This is distinct from the weighted entropy notebook which ran to completion and scored 38/50.

### Proposed Improvements (build on weighted entropy notebook)
- **A**: Increase to 12 attempts (from 8), early stop at 5
- **B**: Extract integers from code output as fallback answers (with entropy penalty)
- **C**: Dynamic TIR depth — go deeper on hard problems when early attempts disagree
- **D**: Add code-first prompt variants for genuine diversity (not just temperature)
- **E**: Hybrid scoring: `score = Σ(1/entropy) + 0.1 * vote_count` for tiebreaking

### Key Lesson
The winning formula is: **strong model + deep stateful TIR + confidence-weighted answer selection**. Future iterations should always build on the weighted entropy notebook as the base.

### Improvement Notebook Created: `improv1_entropy_plus`
- **Kaggle URL**: https://www.kaggle.com/code/sonphamorg/aimo3-entropy-plus-v1
- **Local path**: `kaggle_submissions/improv1_entropy_plus/`
- **Base**: Weighted entropy notebook (score 38), same model (gpt-oss-120b), same deps (aimo-3-utils)
- **Changes from base**:
  - **A. More attempts**: 8→12, early stop 4→5 (16 workers handle this without wall-time increase)
  - **B. Code output fallback**: If no `\boxed{}` found, extract last integer from successful code output; penalized with `code_fallback_entropy=8.0` so it's low-priority in scoring
  - **D. Prompt diversity**: 8 reasoning + 2 code-first + 2 case-analysis prompts (was: 1 prompt for all attempts)
  - **E. Hybrid scoring**: `score = Σ(1/entropy) + 0.1 * vote_count` — tiebreaker favors answers with more votes
- **VRAM**: No increase — same model, same vLLM server config, 12 concurrent requests well within `max_num_seqs=256`
- **Dependencies**: Same `andreasbis/aimo-3-utils` kernel source + `danielhanchen/gpt-oss-120b` model

### Iteration Strategy Plan
Full plan saved to `.claude/iteration_plan.md` — covers:
1. Faster evaluation (save-and-replay, local proxy eval, result logging)
2. Non-finetuning iteration (prompt sweeps, ablation testing, decoupled generation/selection)
3. Fine-tuning iteration (DPO from competition outputs, progressive model scaling)
**Next step**: Switch to NVIDIA machine with more VRAM, pull this repo, implement save-and-replay first.

---

## 2026-02-02 - Session 8: Trace Generation + Replay Selection System

### Implementation Complete
Built the full save-and-replay infrastructure for offline strategy iteration:

**Files Created:**
- `scripts/prompts.py` (~55 lines) — Prompt variants registry with exact text from improv1 notebook
- `scripts/generate_traces.py` (~350 lines) — TIR trace generation with logprobs, entropy, code execution tracking
- `scripts/replay_selection.py` (~480 lines) — 138 selection strategies + Bayesian optimization
- `scripts/smoke_test.py` (~120 lines) — End-to-end verification script

**Selection Strategies (138 total):**
- **Group 1 (6):** Majority vote + 4 pure entropy transforms (1/x, 1/x², exp(-x), 1/log(1+x)) + min entropy
- **Group 2 (40):** Hybrid weighting: score = Σ(transform(entropy)) + k × votes, 4 transforms × 10 k-values
- **Group 3 (44):** Normalized hybrids: (1-α)×norm_entropy + α×norm_votes, 4 transforms × 11 α-values
- **Group 4 (34):** Threshold filters (7) + Top-K (6) + Threshold×Hybrid cross-product (21)
- **Group 5 (6):** Source-aware (code_fallback weighting)
- **Group 6 (8):** Prompt-aware (meta-vote, diversity bonus, weighted meta-vote, prompt-type-only)

**Infrastructure:**
- SQLite database at `results/experiments.db` with tables: `selection_results`, `generation_runs`
- Bayesian optimization via Optuna (TPE sampler) for generation hyperparams
- All trace JSON files include: answer, answer_source, entropy, prompt_type, code executions, logprobs summary

### What Needs GPU
This machine (AMD) cannot run the pipeline — requires NVIDIA GPU with vLLM + jupyter_client. The scripts are ready; they need to be run on the NVIDIA machine.

### Update: Running on AMD Strix Halo (not NVIDIA)
Discovered this IS the AMD machine with llama.cpp Vulkan. The venv already has jupyter_client. Everything works locally.

### Overnight Run Started (2026-02-03 00:37 EST)

**Configuration:**
- Model: gpt-oss-120b Q4_K_M (actual competition model, 60GB)
- Problems: 53 (all reference problems)
- Samples per problem: 12
- Max turns: 16
- Temperature: 0.6
- Output: `output/traces/gpt_oss_120b_full/`
- Log: `output/overnight_run.log`
- PID: 324556

**Observed speeds:**
- Trivial problems: ~15s per sample
- Olympiad problems: ~6 min per sample (with Qwen3-30B-A3B)
- gpt-oss-120b expected similar or slightly slower

**Monitor:**
```bash
tail -f output/overnight_run.log
ls output/traces/gpt_oss_120b_full/problem_*.json | wc -l
```

**When done (morning):**
```bash
source .venv/bin/activate
python scripts/replay_selection.py sweep --traces-dir output/traces/gpt_oss_120b_full/
```

---

## 2026-02-03 - Session 9: Kaggle Trace Generator Notebook

### Overnight Run Status
The local trace generation job crashed multiple times due to:
1. `KeyError: 'entropy'` — sample failed to generate entropy data
2. Long gaps suggesting machine went to sleep
3. Completed only 10/53 problems (2/5 correct = 40%)

### Decision: Move to Kaggle H100
Local llama.cpp on AMD is unreliable for long runs. Created Kaggle notebook to:
- Use the actual competition infrastructure (gpt-oss-120b + vLLM + openai_harmony)
- Run on reliable H100 GPU
- Generate traces in format compatible with `replay_selection.py`

### Files Created

```
kaggle_submissions/trace_generator/
├── kaggle_trace_generator.py      # Standalone Python script
├── kaggle_trace_generator.ipynb   # Kaggle notebook (14 cells)
├── kernel-metadata.json           # Kaggle push config
├── reference_dataset/             # Dataset to upload
│   ├── reference.csv              # 53 AIMO3 problems with answers
│   └── dataset-metadata.json      # Kaggle dataset config
└── README.md                      # Setup instructions
```

### Notebook Features
- **Detailed markdown documentation** at top explaining purpose and usage
- **Same infrastructure** as competition notebook (gpt-oss-120b, vLLM, openai_harmony)
- **Compatible trace format** for `scripts/replay_selection.py`
- **12 samples per problem**: 8 reasoning + 2 code-first + 2 case-analysis
- **128 max turns** (deep TIR like competition)
- **Entropy/logprobs** collection for all attempts
- **Code execution tracking** with full history
- **Per-problem JSON files** + summary.json + config.json

### Kaggle Push Commands

```bash
# 1. Upload reference problems dataset
cd kaggle_submissions/trace_generator/reference_dataset
kaggle datasets create -p .

# 2. Push notebook
cd kaggle_submissions/trace_generator
kaggle kernels push -p .
```

### Expected Output
- 53 `problem_{id}.json` files in `/kaggle/working/traces/`
- `summary.json` with overall accuracy
- `config.json` with generation parameters
- ~9 hours runtime (fits Kaggle limit)

### Next Steps
1. Push reference dataset to Kaggle
2. Push notebook to Kaggle
3. Run with H100 GPU, 9h timeout
4. Download traces
5. Run `python scripts/replay_selection.py sweep --traces-dir ./traces/`

---

## 2026-02-03 - Session 10: Feb 3 Submission + Research

### Submission Status Summary
- **41-50-aimo-3-weighted-entropy**: Score **38** (baseline)
- **aimo3-entropy-plus-v1**: Score **33** (regression - prompt diversity hurt)

### Research Findings (arXiv 2025-2026)

Key papers on answer selection and confidence:

1. **[CISC: Confidence Improves Self-Consistency](https://arxiv.org/pdf/2502.06233)** - Weighted voting with confidence scores beats standard self-consistency, 46% cost reduction
2. **[Deep Think with Confidence](https://arxiv.org/pdf/2508.15260)** - Filter to top-η% confident traces BEFORE voting (filtering threshold η)
3. **[rStar-Math](https://arxiv.org/html/2501.04519v1)** - Keep only answers with ≥3 consistent solutions → Qwen2.5-Math-7B improved from 58.8% to 89.4% on MATH
4. **[Entropy-Guided Loop](https://arxiv.org/html/2509.00079v1)** - Captures "discarded" probability distributions; achieves 95% of reasoning-model performance at 1/3 cost
5. **[Think Just Enough](https://arxiv.org/html/2510.08146)** - Sequence-level entropy as confidence signal with closed-form thresholds

### Why Entropy-Plus Regressed (38 → 33)
Based on research:
- Prompt diversity (code-first, case prompts) likely worse than pure reasoning
- Code fallback extracted wrong integers, adding noise
- More samples (12) with 4 bad prompts < 8 samples with 8 good prompts

### Feb 3 Submission: Entropy-Gated Consensus

**Strategy**: Filter → Then Weight (research-backed)

**Implementation** (`kaggle_submissions/feb3_entropy_gated/`):
1. **8 attempts, all reasoning prompts** (revert from 12 mixed)
2. **Entropy threshold filter** - Only consider answers with entropy < 5.0
3. **Consensus requirement** - Answer must have ≥2 votes to be candidate
4. **Entropy-weighted scoring** among filtered candidates only
5. **Fallback** - If filtering too aggressive, use simple majority

**Key Code Change** (`_select_answer_gated()`):
```python
# Stage 1: Filter by entropy threshold
confident_results = [r for r in valid_results if r['Entropy'] < 5.0]

# Stage 2: Require consensus
candidates = {ans: cnt for ans, cnt in votes.items() if cnt >= 2}

# Stage 3: Entropy-weight among candidates
for r in confident_results:
    if r['Answer'] in candidates:
        scores[r['Answer']] += 1.0 / max(r['Entropy'], 0.1)

# Stage 4: Fallback to simple majority if filtering too aggressive
```

**Kaggle URL**: https://www.kaggle.com/code/sonphamorg/aimo3-entropy-gated-feb3
**Status**: RUNNING

### Trace Generator Notebooks

| Notebook | Model | Dataset | Status |
|----------|-------|---------|--------|
| `aimo3-trace-generator` | gpt-oss-120b | AIMO3 ref | ERROR (P100 assigned, needs H100) |
| `traces-t4-qwen3-4b-aime-bulk-strategy-tuning` | Qwen3-4B | AIME | ERROR |
| `traces-t4-qwen8b-aime-validation` | Qwen3-8B | AIME | Not pushed (GPU limit) |
| `traces-h100-qwen30b-aimo3-transfer` | Qwen3-30B | AIMO3 | Not pushed (GPU limit) |

**Issue**: H100 notebooks get P100 because `enable_gpu: true` doesn't specify GPU type. H100 only assigned for actual competition submissions.

### Next Steps
1. Monitor feb3 submission result
2. Fix trace generators (T4 notebooks should work for small models)
3. If feb3 scores well, tune thresholds using trace data

---

## 2026-02-03 - Session 11: Local Trace Generation Jobs + Feb3 Fixes

### Feb3 Submission Fix
- **Issue**: Notebook crashed with "Kernel died before replying to kernel_info" during Jupyter kernel initialization
- **Cause**: Parallel initialization of 16 Jupyter kernels overwhelmed the system
- **Fix**:
  - Reduced workers from 16 to 8
  - Changed kernel init from parallel to sequential with retry logic (3 attempts per kernel)
- **Status**: v5 pushed, waiting for results
- **Important**: `machine_shape: "NvidiaH100"` in API push is unreliable - must manually verify/change GPU type in Kaggle web UI

### Kaggle Skills Updated
- Added "Kaggle Authentication" skill - ALWAYS use `export $(grep KAGGLE_API_TOKEN .env | xargs)` before any kaggle CLI command
- Added "Kaggle Kernel/Notebook Push" skill with machine_shape notes
- Updated dataset upload skill to reference authentication skill

### Trace Generator Notebooks Fixed
All three trace notebooks updated to use offline vLLM wheels:
- Added `sonphamorg/vllm-wheels-py312-cu129` to dataset_sources
- Changed pip install to: `pip install --no-index --find-links /kaggle/input/vllm-wheels-py312-cu129/wheels/ vllm`
- Fixed AIME column detection (`question` vs `problem`)

| Notebook | Model | GPU | Status |
|----------|-------|-----|--------|
| traces-t4-qwen3-4b-aime-bulk-strategy-tuning | Qwen3-4B | T4 | v4 RUNNING |
| traces-t4-qwen3-8b-aime-val | Qwen3-8B | T4 | v1 pushed |
| traces-h100-qwen30b-aimo3-transfer | Qwen3-30B | H100 | v1 ERROR (GPU quota) |

### Local Trace Generation Jobs Created

**Scripts**:
- `scripts/run_traces_amd.sh` - For AMD machine (this machine, Radeon 8060S)
- `scripts/run_traces_nvidia.sh` - For NVIDIA machine (4090)

**AMD Job** (this machine):
```bash
# Run with defaults (Qwen3-8B, 8 samples)
./scripts/run_traces_amd.sh

# Or customize:
MODEL_PATH=~/models/Qwen3-4B-Q4_K_M.gguf MODEL_NAME=Qwen3-4B N_SAMPLES=12 ./scripts/run_traces_amd.sh
```

**NVIDIA Job** (pull repo on NVIDIA machine and run):
```bash
git pull origin main
./scripts/run_traces_nvidia.sh

# For larger model with vLLM:
USE_VLLM=true MODEL_PATH=/path/to/hf/model ./scripts/run_traces_nvidia.sh
```

**Available Local Models** (`~/models/`):
- Qwen3-4B-Q4_K_M.gguf (2.5GB)
- Qwen3-8B-Q4_K_M.gguf (5GB)
- Qwen3-14B-Q4_K_M.gguf (9GB)
- Qwen3-30B-A3B-Q4_K_M.gguf (18.5GB)
- Qwen3-32B-Q4_K_M.gguf (19.8GB)
- DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf (4.7GB)
- DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf (19.9GB)
- DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf (42.5GB)
- gpt-oss-120b/ (HF format, 65GB)

**Output**: Traces saved to `output/traces/{gpu}_{model}_{timestamp}/`

### Session 11 Continued - Trace Generation Running

**Fixed Issues:**
1. `--flash-attn` flag now requires explicit value (`on`, `off`, or `auto`) - updated both scripts to use `--flash-attn auto`
2. Scripts now activate venv to ensure dependencies are available
3. AMD script changed to port 8081 to avoid conflict with other servers

**Current Status:**
- Qwen3-8B llama-server running on port 8081 (PID 394788)
- Trace generation running (PID 395564) with:
  - Model: Qwen3-8B
  - 10 problems × 8 samples = 80 traces
  - Output: `output/traces/amd_Qwen3-8B_20260203_220536/`
- Code pushed to GitHub (commit 40b47e7)

**NVIDIA Machine Instructions:**
```bash
git pull origin main
./scripts/run_traces_nvidia.sh
```

---

## 2026-02-04 - Session 12: Feb4 Verified Submission + Ablation Testing Setup

### Feb3 Submission Result
- **Score: 40/50** — Our best score so far!
- Entropy-gated consensus approach is working

### Feb4 Submission: Verified Consensus (Option B + C)
**Location**: `kaggle_submissions/feb4_verified/`

**New Features** (on top of feb3):
1. **Option C - Self-consistency boost**: Track ALL `\boxed{}` answers during reasoning, boost repeated answers
2. **Option B - Code verification**: When top-2 candidates are close (ratio < 1.5), check if code output supports either

**Key Parameters (educated guesses, pending ablation)**:
```python
code_answer_boost = 1.5      # Boost when code output matches boxed answer
repeat_answer_boost = 1.3    # Boost per repeated answer during reasoning
error_penalty = 0.6          # Penalty for attempts with Python errors
verify_threshold = 1.5       # Verify if top/second score ratio < this
verify_timeout = 15          # Timeout for verification code execution
```

**Status**: Ready to push (Kaggle auth not configured on this machine)

### Ablation Testing Script Created
**Location**: `scripts/ablation_test.py`

Tests 9 configurations:
1. Simple majority vote (no filter)
2. Entropy-gated (threshold 0.5)
3. Option C only (self-consistency)
4. Option B only (code verification)
5. B + C combined
6. B + C (strict entropy 0.3)
7. B + C (relaxed entropy 0.7)
8. B + C (stronger repeat boost 1.5)
9. B + C (stronger code boost 2.0)

**Usage** (when traces are ready):
```bash
python3 scripts/ablation_test.py --traces-dir output/traces/aime_amd_20260204_073853
```

### Trace Generation Status
- **AMD**: DeepSeek-R1-0528-Qwen3-8B on AIME 2005-2022 (524 problems × 12 samples)
- **PID**: 425808 (generate_traces.py), 425365 (llama-server on port 8081)
- **Output**: `output/traces/aime_amd_20260204_073853/`
- **Progress**: Just started, no problem traces yet

### Key Insight from Ablation Pre-test
Entropy threshold 0.3 is too strict — only 2/12 attempts pass, breaking consensus. Current traces (gpt-oss-120b) show typical entropy 0.25-0.55. Use threshold 0.5 as default.

---

## 2026-02-04 - Session 13: Qwen3 /no_think Fix + AIME Trace Restart

### Critical Discovery: Qwen3 Thinking Mode Was ON
- Other machine found Qwen3 thinking mode is ultra slow — 5x slower than non-thinking
- Investigation of AMD traces confirmed: `<think>` blocks present in all outputs
- First token logprob: `<think>` at probability ~1.0
- Per-sample: ~7,576 tokens, ~370s (most tokens wasted on thinking)
- Only 7 problems completed in ~9 hours

### Fix: Added `--no-think` Flag
- `scripts/generate_traces.py`: New `--no-think` CLI flag, appends `/no_think` to user messages
- `scripts/run_traces_amd.sh`: Added `--no-think`
- `scripts/run_traces_nvidia.sh`: Added `--no-think`
- Config JSON now records `"no_think": true` for traceability

### Restarted AMD Trace Job (with /no_think)
- **Killed**: Old job (PID 436844) running aime_train_100.csv WITH thinking
- **Started**: New job on full `aime_train_2005_2022.csv` (524 problems)
- **Output**: `output/traces/aime_qwen3_8b_nothink_20260204_170730/`
- **PID**: 471724

### Speed Comparison (Qwen3-8B, 12 samples/problem)
| Metric | With thinking | Without thinking | Speedup |
|--------|--------------|-----------------|---------|
| Tokens/sample | ~7,576 | ~730 | 10x fewer |
| Time/sample | ~370s | ~29s | 13x |
| Time/problem | 74 min | 5.7 min | 13x |

### ETA
- 524 problems × 5.7 min ≈ 50 hours (Friday morning)
- First problem (2005-I-1): 12/12 correct (answer=942), unanimous vote

---

## 2026-02-04 - Session 14: Analysis of jonathanchan/aimo3-gpt-oss-120b-finetuning (LB 42)

### Key Finding: "Finetuning" = Hyperparameter Tuning, NOT Model Fine-Tuning
Despite the misleading notebook title, this notebook does **zero model fine-tuning**. No LoRA, no QLoRA, no weight updates. The "finetuning" refers entirely to **hyperparameter and prompt tuning** on top of the same base `gpt-oss-120b` model.

### Source
- Base notebook: `andreasbis/aimo-3-gpt-oss-120b-with-tools` (Version 8, LB 41)
- Jonathan's Version 17 scored LB 42 (revised prompts + hyperparams)
- Source notebook Version 11 also scored LB 42 (just reduced timeout from 300→270)

### Exact Changes from Source (andreasbis LB 41) to jonathanchan (LB 42)

#### 1. Prompt Changes
| Prompt | Source (andreasbis) | Modified (jonathanchan) |
|--------|-------------------|----------------------|
| system_prompt | "You are a world-class International Mathematical Olympiad (IMO) competitor. The final answer must be a non-negative integer between 0 and 99999. You must place the final integer answer inside \boxed{}." | "You a world-class competitor in a math competition in the style of International Mathematical Olympiad. The final answer for each question must be an integer between 0 and 99999. The final integer answer must be placed inside \boxed{}." |
| tool_prompt | "Use this tool to execute Python code. The environment is a stateful Jupyter notebook. You must use print() to output results." | "Use this tool to execute Python codes in a Jupyter notebook. Use print() to output results." |
| preference_prompt | "You have access to `math`, `numpy` and `sympy` to solve the problem." | "There is access to `math`, `numpy` and `sympy` and other modules to solve these math problems." |

#### 2. Hyperparameter Changes
| Parameter | Source | Modified | Effect |
|-----------|--------|----------|--------|
| temperature | 1.0 | 0.95 | Slightly less random |
| seed | 42 | 2570 | Different random seed |
| gpu_memory_utilization | 0.99 | 0.95 | More conservative VRAM |
| context_tokens | 81920 | 65536 | Shorter context window |
| batch_size (max_num_seqs) | 64 | 256 | Higher concurrent requests |
| base_problem_timeout | 300→270 (v20) | 270 | Less time per problem |

#### 3. vLLM Server Config Changes
| Parameter | Source | Modified |
|-----------|--------|----------|
| --max-num-batched-tokens | 2048 | REMOVED |
| --max-cudagraph-capture-size | 64 | REMOVED |
| --max-num-seqs | 64 | 256 |

#### 4. What Stayed the Same (The Core Architecture)
- Model: gpt-oss-120b (MXFP4, ~65GB)
- Answer selection: Pure entropy-weighted voting (score = 1/entropy)
- TIR: 128 max turns, persistent Jupyter kernels
- Workers: 16 parallel threads
- Attempts: 8 per problem, early stop at 4 matching
- Logprobs: top-5 for Shannon entropy calculation
- KV cache: FP8 (fp8_e4m3)
- Sampling: min_p = 0.02
- Code execution: Persistent Jupyter kernels via jupyter_client

### Implications for Our Strategy
1. The 41→42 jump came from minor prompt rewording + temperature 1.0→0.95 + seed change
2. This suggests high variance between runs — the same notebook can score 41 or 42 depending on random seed
3. The core entropy-weighted architecture is already near-optimal for this model
4. Real gains likely require model improvements (fine-tuning, better base model) not just hyperparameter tuning
5. The `batch_size=256` change (from 64) may help throughput by allowing more concurrent sequences in vLLM

---

## 2026-02-04 - Session 15: Analysis of kishanvavdara/44-50-got-lucky-but-not-for-long-eagle3 (LB 44)

### Source
- **Notebook URL**: https://www.kaggle.com/code/kishanvavdara/44-50-got-lucky-but-not-for-long-eagle3
- **Author**: kishanvavdara
- **Score**: 44/50

### Key Finding: EAGLE-3 Speculative Decoding for Speed, NOT Accuracy
The notebook is architecturally **identical** to the andreasbis base notebook (LB 38-41). The only meaningful addition is **EAGLE-3 speculative decoding** — a small draft model that generates candidate tokens ahead of the main model, which are then verified. This is purely a **speed optimization** (2-3x faster inference), not an accuracy improvement. The 44/50 score likely comes from variance/luck on the private test set.

### Title Explanation: "Got Lucky But Not For Long"
The title almost certainly means: the author scored 44/50 through favorable random variance on the specific test problems, but expects the score to regress on different test sets. With 8 attempts per problem and stochastic sampling (temp=1.0), there is significant run-to-run variance. A "lucky" run can score 44 while the same setup might score 38-42 on other runs.

### Model
- **Base model**: `gpt-oss-120b` (same as all other top AIMO3 notebooks)
  - Path: `/kaggle/input/gpt-oss-120b/transformers/default/1`
  - MoE architecture: 120B total parameters, ~5B active
  - MXFP4 quantization, FP8 KV cache
- **Eagle3 draft model**: `/kaggle/input/download-eagle3/wenliang1990/gpt-oss-120b-eagle3-aimo3`
  - ~0.2-0.9B parameter small transformer draft model
  - Takes hidden states from 3 layers of the base model
  - Generates 3 candidate tokens per step (`num_speculative_tokens: 3`)
  - `draft_tensor_parallel_size: 1`
- **No fine-tuning** of any kind — both base and draft model are off-the-shelf

### EAGLE-3 Speculative Decoding Details
- **Paper**: [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) (NeurIPS 2025)
- **Mechanism**: Draft model proposes tokens, base model verifies in parallel, accepted tokens are kept
- **Expected speedup**: ~2-3x for batch size 1, acceptance rate ~2.2-2.8 tokens per step for math tasks
- **vLLM integration**: Via `--speculative-config` JSON argument
- **Config used**:
  ```json
  {
    "method": "eagle3",
    "model": "/kaggle/input/download-eagle3/wenliang1990/gpt-oss-120b-eagle3-aimo3",
    "num_speculative_tokens": 3,
    "draft_tensor_parallel_size": 1
  }
  ```

### Inference Setup (identical to andreasbis base)
| Parameter | Value |
|-----------|-------|
| served_model_name | gpt-oss |
| kv_cache_dtype | fp8_e4m3 |
| dtype | auto |
| temperature | 1.0 |
| min_p | 0.02 |
| gpu_memory_utilization | 0.96 |
| context_tokens | 65536 |
| top_logprobs | 5 |
| batch_size (max_num_seqs) | 256 |
| attempts | 8 |
| early_stop | 4 |
| workers | 16 |
| turns | 128 |
| seed | 42 |
| tensor_parallel_size | 1 |
| enable_prefix_caching | true |
| notebook_limit | 17400s |
| high_problem_timeout | 900s |
| base_problem_timeout | 300s |

### Answer Selection Strategy
- **Same entropy-weighted voting** as andreasbis base (pure `1/entropy` weighting)
- For each attempt: compute mean Shannon entropy from top-5 logprobs
- Weight = `1 / max(entropy, 1e-9)`
- Score per answer = sum of weights from all attempts producing that answer
- Highest-scoring answer wins
- Fallback: return 0 if no valid answers

### Prompt Templates
Three prompts, all identical to andreasbis base:
1. **system_prompt**: IMO-level solver with 5-step approach (UNDERSTAND, EXPLORE, PLAN, EXECUTE, VERIFY), verification requirements, `\boxed{}` output format
2. **tool_prompt**: Instructions for Python code execution in stateful Jupyter notebook
3. **preference_prompt**: Instructions for using math/numpy/sympy, appended to each problem text

### Notable Techniques
1. **Speed tracking**: Added `Time (s)` and `Tokens/sec` columns to results DataFrame, plus cumulative stats — this is NEW compared to the base notebook, suggesting the author was monitoring the Eagle3 speedup
2. **Model weight preloading**: Reads all model files into OS page cache before starting vLLM server (this is from the base notebook)
3. **Persistent Jupyter kernels**: 16 stateful kernels pre-initialized, reused across problems
4. **Streaming answer detection**: Scans for `\boxed{}` during token streaming, not just at end
5. **Dynamic time budgeting**: Adjusts per-problem timeout based on remaining notebook time and unsolved problems
6. **Garbage collection**: `gc.disable()` during problem solving, `gc.enable()` + `gc.collect()` after

### Differences from andreasbis Base (LB 38-41)
1. **EAGLE-3 speculative decoding** — the only architectural change (speed, not accuracy)
2. **Speed tracking metrics** — `Time (s)`, `Tokens/sec`, cumulative stats, `print_final_stats()`
3. Everything else is identical (prompts, selection strategy, model, hyperparameters)

### Why 44/50?
Most likely **favorable variance**. With temp=1.0, min_p=0.02, and 8 attempts per problem, each run has high variance. Eagle3 may have helped marginally by allowing more computation within the same time budget (faster tokens = more complete reasoning chains), but the jump from ~40 to 44 is more likely random. The title "got lucky but not for long" confirms this interpretation.

---

## 2026-02-04 - Session 16: Math Fine-Tuning Datasets & Approaches Research

### Research Goal
Comprehensive survey of datasets, data quality approaches, and best practices for fine-tuning LLMs on competition-level math (AIME, olympiad style).

### Key Datasets (Ranked by Relevance)

#### Tier 1: Directly Relevant for AIMO Competition

1. **nvidia/OpenMathReasoning** — CC-BY-4.0
   - 306K unique problems (from AoPS forums), 3.2M CoT solutions, 1.7M TIR solutions, 566K GenSelect samples
   - Generated with DeepSeek-R1 and QwQ-32B; problems preprocessed with Qwen2.5-32B-Instruct
   - Foundation of AIMO-2 1st place (34/50)
   - URL: https://huggingface.co/datasets/nvidia/OpenMathReasoning

2. **AI-MO/NuminaMath-1.5** — Apache 2.0
   - 896K competition-level math problems with CoT solutions
   - 11 sources (Chinese HS, US/international olympiads, exam PDFs, forums)
   - Metadata: problem_type, question_type, validity flags, source
   - Manually verified olympiad subset (fixed parsing issues from v1)
   - URL: https://huggingface.co/datasets/AI-MO/NuminaMath-1.5

3. **AI-MO/NuminaMath-TIR** — Tool-Integrated Reasoning subset
   - 70K problems with GPT-4-generated TORA-format reasoning (code + NL interleaved)
   - Filtered for answer correctness (3 rounds)
   - URL: https://huggingface.co/datasets/AI-MO/NuminaMath-TIR

4. **AI-MO/NuminaMath-CoT** — 860K CoT problem-solution pairs
   - URL: https://huggingface.co/datasets/AI-MO/NuminaMath-CoT

5. **open-r1/OpenR1-Math-220k**
   - 220K problems from NuminaMath-1.5, 2-4 reasoning traces per problem from DeepSeek-R1
   - Verified with Math Verify + Llama-3.3-70B-Instruct judge
   - Good for rejection sampling and DPO
   - URL: https://huggingface.co/datasets/open-r1/OpenR1-Math-220k

#### Tier 2: General Math Fine-Tuning

6. **nvidia/OpenMathInstruct-2** — 14M problem-solution pairs
   - Generated with Llama3.1-405B-Instruct, based on GSM8K + MATH training sets
   - Key insight: question diversity matters more than solution count
   - URL: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2

7. **meta-math/MetaMathQA**
   - Augmented from GSM8K + MATH training sets
   - Bootstrap question augmentation approach
   - URL: https://huggingface.co/datasets/meta-math/MetaMathQA

8. **TIGER-Lab/MathInstruct**
   - 13 math rationale datasets compiled; hybrid CoT + PoT (Program of Thought)
   - Used for MAmmoTH model series
   - URL: https://huggingface.co/datasets/TIGER-Lab/MathInstruct

9. **hendrycks/competition_math** (MATH dataset)
   - AMC 10, AMC 12, AIME problems with step-by-step solutions
   - 12.5K problems across 7 subjects, 5 difficulty levels
   - URL: https://huggingface.co/datasets/hendrycks/competition_math

10. **AoPS-Instruct** — 650K+ QA pairs from AoPS forums
    - Extracted using open-source LLMs from forum discussions
    - URL: https://arxiv.org/abs/2501.14275

#### Tier 3: Competition-Specific Evaluation Sets

11. **math-ai/aime25** — AIME 2025 problems
12. **math-ai/aime24** — AIME 2024 problems
13. **math-ai/amc23** — AMC 2023 problems
14. **MathArena/aime_2025** — Uncontaminated 2025 AIME
15. **MathArena/hmmt_feb_2025** — HMMT February 2025
16. **KbsdJames/Omni-MATH** — 4,428 olympiad problems, 33+ sub-domains, 10+ difficulty levels
17. **AIME Problem Set 1983-2024** (Kaggle: hemishveeraboina/aime-problem-set-1983-2024)
18. **DART-Math datasets** — ~590K difficulty-aware rejection-tuned examples (NeurIPS 2024)

### Training Approaches That Worked

#### AIMO-1 Winner: NuminaMath (29/50)
- **Base**: DeepSeekMath-Base 7B
- **Stage 1**: SFT on ~860K NuminaMath-CoT problems (CoT format, lr=2e-5, 3 epochs)
- **Stage 2**: SFT on ~60K NuminaMath-TIR problems (code-interleaved, lr=2e-5, 4 epochs)
- **Hardware**: 8xH100, 10 hours
- **Key**: Two-stage (CoT then TIR) based on MuMath-Code paper
- **Result**: 56.3% (CoT only) -> 68.2% (CoT+TIR) on MATH

#### AIMO-2 Winner: OpenMath-Nemotron (34/50)
- **Base**: Qwen2.5-14B
- **Data**: 540K unique problems, 3.2M CoT + 1.7M TIR solutions from OpenMathReasoning
- **Three pillars**:
  1. Large-scale high-quality dataset with olympiad-level problems
  2. Novel code execution + long reasoning integration (iterative training/generation/filtering)
  3. GenSelect: train model to pick best solution from candidates
- **Model sizes**: 1.5B, 7B, 14B, 14B-Kaggle, 32B
- **Paper**: https://arxiv.org/abs/2504.16891

#### s1: Simple Test-Time Scaling (1K examples!)
- **Base**: Qwen2.5-32B-Instruct
- **Data**: Just 1,000 curated examples (from 59K candidates)
- **Sources**: NuminaMath, AIME, OmniMath, AGIEval, OlympicArena
- **Curation**: Difficulty + diversity + quality filtering
- **Training**: 26 minutes on 16 H100s
- **Budget forcing**: Append "Wait" tokens to extend reasoning at test time
- **Result**: Exceeds o1-preview on MATH and AIME24 by up to 27%
- **Paper**: https://arxiv.org/abs/2501.19393

#### DeepSeek-R1 Distillation
- **Method**: Generate 800K samples with DeepSeek-R1, fine-tune Qwen/Llama
- **Result**: R1-Distill-Qwen-32B scores 72.6% on AIME 2024, 94.3% on MATH-500
- **Key insight**: Distilling reasoning from larger models > RL on smaller models

#### Qwen2.5-Math Training
- **Pre-training**: 1T+ math tokens (English + Chinese), web + books + code + synthetic
- **CoT data**: 580K English + 500K Chinese problems
- **Synthetic generation**: MuggleMath + DotaMath query evolution
- **Online RFT**: Iteratively generate TIR paths, filter by correctness, repeat
- **Contamination prevention**: 13-gram matching to exclude benchmark-similar samples

### Data Quality Approaches

#### Correctness Verification
- **Answer matching**: Filter by final answer correctness (simplest, most common)
- **Math Verify**: Library for programmatic math answer verification (used by OpenR1-Math)
- **LLM-as-judge**: Llama-3.3-70B-Instruct for 12% of OpenR1-Math samples
- **Reward models**: PRM800K-style process reward models score intermediate steps
- **Limitation**: Correct final answer does NOT guarantee correct reasoning trace

#### Difficulty-Aware Filtering (DART-Math, NeurIPS 2024)
- Vanilla rejection sampling is biased toward easy problems
- DART-Math: Allocate MORE sampling budget to harder problems
- Two strategies: Uniform (same #correct per query) or Prop2Diff (bias toward hard)
- ~590K examples, outperforms vanilla rejection tuning
- URL: https://github.com/hkust-nlp/dart-math

#### Adaptive Curriculum Learning (AdaRFT)
- Fine-tuning on too-easy OR too-hard problems leads to poor outcomes
- Adaptive sampling maintains optimal difficulty range matching model skill
- Reduces training steps by up to 2x and improves accuracy
- On easy benchmarks: high-entropy (uncertain) samples help most
- On hard benchmarks: low-entropy (confident) samples help most

#### Importance-Weighted SFT (iw-SFT)
- Weight samples by reward model score during SFT
- Reasoning naturally emerges: AIME 66.7%, GPQA 64.1%
- Outperforms standard SFT on same data

### Data Contamination Concerns
- **AIME 2024 contamination**: Some models (esp. QWQ-Preview-32B) show 60% above expected
- **MathArena**: Live evaluation framework using only post-release problems
- **8 AIME 2025 + 1 HMMT 2025 problems** found online in similar form
- **Models score better on older (2024) vs newer (2025)** problems
- **Mitigation**: 13-gram matching (Qwen), live benchmarks, hold-out test years

### TIR-Specific Training

#### ToRA (Microsoft, ICLR 2024)
- GPT-4 generates interactive tool-use trajectories for GSM8k + MATH
- ToRA-Corpus: only 16K annotations
- Interleaves natural language reasoning with Python code blocks
- 13-19% absolute improvement over open-source baselines

#### MuMath-Code (NuminaMath basis)
- Stage 1: SFT on CoT data (learn general math reasoning)
- Stage 2: SFT on code-nested data (learn to generate + execute code)
- Multi-perspective data augmentation for query evolution

#### SimpleTIR (2025)
- RL-based multi-turn TIR training
- Filters "void turns" (no code block and no final answer) to stabilize training
- Max 5-10 turns of code execution per problem

### Practical Recommendations for AIMO3

1. **Best dataset to start fine-tuning**: nvidia/OpenMathReasoning (proven AIMO-2 winner, CC-BY-4.0)
2. **For TIR specifically**: NuminaMath-TIR (70K) + OpenMathReasoning TIR subset (1.7M)
3. **Minimum viable dataset**: ~1K-10K curated high-quality traces (s1 showed 1K is enough)
4. **Two-stage training**: Always do CoT first, then TIR (NuminaMath approach)
5. **Difficulty matching**: Focus on AIME-level difficulty, not GSM8K-level
6. **GenSelect training**: Train model to pick best answer from candidates (NVIDIA's 3rd pillar)
7. **Contamination**: Avoid training on AIME 2023-2025 problems if using them for evaluation
8. **Format**: Code-interleaved TIR traces with Python REPL execution work best for competition

---

## 2026-02-04 - Session 17: Fine-Tuning on Kaggle Hardware Research

### Research Goal
Comprehensive analysis of whether/how fine-tuning gpt-oss-120b (or alternatives) is feasible on Kaggle hardware for AIMO3.

### 1. Kaggle GPU Resources

#### Standard Free-Tier GPUs
| Accelerator | VRAM | Count | Notes |
|---|---|---|---|
| NVIDIA Tesla P100 | 16 GB HBM2 | 1 | Older architecture, FP16 support |
| NVIDIA T4 | 16 GB GDDR6 | 2 (beta) | Turing, INT8/FP16 Tensor Cores |
| Google TPU v3-8 | 128 GB HBM | 1 | Rarely available |

#### Competition-Specific Hardware (AIMO3)
| Accelerator | VRAM | Count | Notes |
|---|---|---|---|
| NVIDIA H100 | 80 GB HBM3 | 1 | For submission notebooks during competition rerun |

**Important distinctions:**
- **Standard notebooks** (non-competition): Only get T4x2 or P100 — NOT H100
- **Competition submission reruns**: Get H100 (as specified by AIMO3 rules)
- **Pre-submission fine-tuning**: AIMO3 offers up to 128 H100s for select participants via Fields Model Initiative partnership
- **Weekly GPU quota**: 30 GPU-hours/week for free accounts

### 2. Kaggle Notebook Runtime Limits
| Mode | Limit |
|---|---|
| Interactive (editing) | 12 hours (auto-saves) |
| "Save & Run All" | **9 hours** (background execution) |
| Competition rerun | **9 hours** (AIMO3 specific) |

### 3. gpt-oss-120b Architecture Recap
- **Total params**: 116.8B (marketed as 120B)
- **Active params per token**: 5.1B
- **Architecture**: MoE with 128 experts, Top-4 routing, 36 layers
- **Weight format**: BF16 except MoE projections in MXFP4
- **MoE/MLP experts**: ~19B parameters in MXFP4 format as nn.Parameter objects

### 4. Fine-Tuning Memory Requirements

#### gpt-oss-120b VRAM Requirements
| Method | VRAM Required | Fits on T4/P100? | Fits on H100? |
|---|---|---|---|
| Full fine-tune (FP16) | ~480+ GB | NO | NO |
| BF16 LoRA | ~210 GB | NO | NO |
| QLoRA 4-bit (standard) | ~80 GB | NO | BARELY (80GB H100) |
| QLoRA 4-bit (Unsloth) | **65 GB** | NO | **YES** (80GB H100) |
| Inference only (MXFP4) | ~60 GB | NO | YES |

#### Maximum Model Size on Free Kaggle GPUs (16 GB VRAM)
| Method | Max Model Size |
|---|---|
| QLoRA 4-bit | ~7B parameters |
| QLoRA 4-bit (Unsloth) | ~7-8B parameters |
| Full fine-tune | ~1-2B parameters |

**Verdict: Fine-tuning gpt-oss-120b is IMPOSSIBLE on free Kaggle GPUs (T4/P100). It requires an H100.**

### 5. Unsloth Optimizations

#### Key Benefits
- 2x faster training, 70% less VRAM vs standard HF + Flash Attention 2
- Custom handling of gpt-oss MXFP4 format (converts nn.Parameter to nn.Linear for BitsandBytes)
- Supports 10x longer context lengths during training
- Works with gpt-oss-120b on 65GB VRAM (QLoRA)

#### gpt-oss-120b Specific (Unsloth)
| Feature | Value |
|---|---|
| QLoRA VRAM | 65 GB |
| Max context (QLoRA, H100 80GB) | 81K tokens |
| Max context (BF16 LoRA, H100 80GB) | 60K tokens |
| Max context (non-Unsloth, 80GB) | 9K-15K tokens |
| Training speed | 1.5x faster than standard |

#### Unsloth LoRA Parameters (from official gpt-oss-120b notebook)
```python
model = FastModel.from_pretrained("unsloth/gpt-oss-120b-unsloth-bnb-4bit")
model = FastModel.get_peft_model(model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)
# Training: SFTTrainer, batch_size=4, lr=2e-4, adamw_8bit, max_seq_len=4096
```

#### gpt-oss-20b (the smaller variant)
- QLoRA on only 14 GB VRAM (fits on T4!)
- Free Colab/Kaggle notebooks available
- Could be a viable alternative model for fine-tuning on free hardware

### 6. Can You Load Pre-Fine-Tuned LoRA Adapters on Kaggle?

**YES — this is the recommended approach.** The workflow:

1. **Train offline** (on your own GPU, cloud, or Kaggle training notebook):
   - Fine-tune with QLoRA/LoRA
   - Save only the adapter weights (typically 50-500 MB)
2. **Upload adapter as Kaggle Dataset**:
   - Create a Kaggle dataset containing the LoRA adapter files
   - `adapter_config.json` + `adapter_model.safetensors`
3. **Load in inference notebook**:
   - Load base model (from Kaggle Models)
   - Load LoRA adapter (from Kaggle Datasets)
   - Merge or keep separate for inference
   - Example: `model.load_adapter("/kaggle/input/my-lora-adapter/")`

**This is exactly what AIMO2 winners did:**
- NVIDIA (1st): Pre-trained model uploaded to Kaggle/HF
- Imagination (2nd): Pre-fine-tuned checkpoints on HuggingFace

### 7. Realistic Training Estimates on H100

#### gpt-oss-120b QLoRA on single H100 (80GB)
| Dataset Size | Est. Time | Fits in 9h Kaggle? | Notes |
|---|---|---|---|
| 100 examples | ~10-20 min | YES | Minimal fine-tune |
| 1,000 examples | ~2-3 hours | YES | s1-style curated set |
| 10,000 examples | ~20-30 hours | NO | Need cloud or pre-compute |
| 50,000 examples | ~100+ hours | NO | Multi-day job |

**Note**: These are rough estimates. Actual time depends on sequence length, batch size, gradient accumulation, and number of epochs.

#### Training throughput expectations (QLoRA, H100, gpt-oss-120b)
- ~20-50 tokens/s training throughput (much slower than inference)
- With max_seq_len=4096, batch_size=4: ~1-2 examples/second
- 1 epoch over 1K examples: ~10-20 minutes
- 3 epochs over 1K examples: ~30-60 minutes

### 8. Practical Strategies for AIMO3 Fine-Tuning

#### Option A: Fine-tune gpt-oss-120b on H100 (competition hardware)
- **Where**: Kaggle competition notebook (H100), or cloud H100 rental (~$3/hr)
- **Method**: Unsloth QLoRA, 65GB VRAM
- **Dataset**: 1K-10K curated math examples (s1 showed 1K suffices)
- **Time**: 1-3 hours for 1K examples
- **Upload**: Save LoRA adapter as Kaggle Dataset (~100-500MB)
- **Inference**: Load base model + adapter in competition notebook
- **Risk**: Competition rerun may not allow enough time for both training + inference

#### Option B: Fine-tune gpt-oss-20b on free Kaggle T4 (14GB VRAM)
- **Where**: Free Kaggle notebook with T4 GPU
- **Method**: Unsloth QLoRA, 14GB VRAM
- **Dataset**: 1K curated examples
- **Time**: ~30 min for 1K examples
- **Trade-off**: 20B is much weaker than 120B for olympiad math
- **Use case**: Experimental, unlikely to beat 120B inference-only

#### Option C: Fine-tune offline, deploy on Kaggle (RECOMMENDED)
- **Training**: Own hardware or cloud rental (H100, A100, etc.)
- **Method**: Unsloth QLoRA for gpt-oss-120b (65GB) or any other model
- **Upload**: LoRA adapter weights as Kaggle Dataset
- **Inference**: Load in competition notebook on H100
- **Advantage**: No time constraint for training; full 9 hours for inference
- **This is what all AIMO2 winners did**

#### Option D: Use AIMO3's 128 H100 pre-competition compute
- AIMO3 offers up to 128 H100s for fine-tuning through Fields Model Initiative partnership
- Requires applying/qualifying as select participant
- Full multi-GPU training possible (SFT, DPO, GRPO)

### 9. Key Constraints Summary

| Constraint | Value | Impact |
|---|---|---|
| T4/P100 VRAM | 16 GB | Can only fine-tune up to ~7B models |
| H100 VRAM | 80 GB | Can fine-tune gpt-oss-120b with QLoRA (65GB) |
| Kaggle runtime | 9 hours | Limits in-notebook training to ~1K-5K examples |
| Weekly GPU quota | 30 hours | Limits experimentation on free tier |
| LoRA adapter size | 50-500 MB | Easily uploaded as Kaggle Dataset |
| Base model loading | ~2-5 min | Minor overhead for adapter merging |

### 10. Verdict

**Fine-tuning gpt-oss-120b on Kaggle free-tier hardware is NOT feasible** (16GB VRAM vs 65GB required). However:

1. **Pre-compute approach works**: Train offline (own GPU/cloud), upload LoRA adapter as dataset
2. **H100 during competition**: If the rerun gives H100, you could theoretically fine-tune + infer within 9 hours (but risky)
3. **Smaller models on free tier**: gpt-oss-20b fits on T4 (14GB with Unsloth QLoRA)
4. **Best strategy**: Train offline, upload adapter, use full 9 hours for inference only (proven by all AIMO2 winners)

## 2026-02-04 - Session 17: Eagle3 Research + Fine-Tuning Plan

### Research Findings

**jonathanchan 42/50 notebook**: NOT actually fine-tuning. Just hyperparameter tweaks (temp 1.0→0.95, seed change). Score improvement likely run-to-run variance.

**kishanvavdara 44/50 Eagle3 notebook**: NOT fine-tuning either. Uses Eagle3 speculative decoding (small draft model proposes tokens, base verifies). Title "Got Lucky But Not For Long" admits score is variance. Architecture identical to andreasbis baseline.

**No public AIMO3 fine-tuning notebooks exist** with reported scores.

### Eagle3 Integration

Created two Eagle3 notebooks:
- `feb5_eagle3/` — feb3 (40/50) + Eagle3 speculative decoding + timing metrics
- `feb5_eagle3_verified/` — feb4 + Eagle3 + verified consensus + timing metrics

Eagle3 draft models available (6 total):
- `wenliang1990/gpt-oss-120b-eagle3-aimo3` — 588MB, AIMO3-specific, recommended
- `nvidia/gpt-oss-120b-Eagle3-v2` — 1.85GB, vLLM-recommended
- Must upload to Kaggle as private dataset (not publicly available)

**Known risks:**
- vLLM logprobs instability with speculative decoding (affects entropy consensus)
- vLLM issue #27626: accuracy degradation reported (MMLU 0.73→0.28)
- Speedup at high concurrency likely 1.2-1.5x (not 2-3x batch=1 numbers)

### Fine-Tuning Plan (saved to fine_tuning.md)
- Best dataset: nvidia/OpenMathReasoning (won AIMO2)
- Method: QLoRA with Unsloth on gpt-oss-120b (65GB VRAM on H100)
- 1K-5K curated examples sufficient (s1 paper)
- Train offline → upload LoRA adapter → combine with entropy consensus

### Scaling Strategy
Created SCALING_STRATEGY.md with parameter matrix based on measured speedup:
- 1.3x → 10 attempts, 10 workers
- 1.5x → 12 attempts, 12 workers, early_stop=5
- 2.0x → 16 attempts, 16 workers, 160 turns, early_stop=6

### Next Steps
1. Upload Eagle3 model to Kaggle as dataset
2. Submit feb5_eagle3 notebook and measure tokens/sec
3. Based on speedup, pick scaling parameters
4. Create scaled variant for follow-up submission


## Feb 7, 2026 - Trace Analysis & Fine-Tuning Research

### Strategy Simulation Results (524 AIME problems, Qwen3-8B)
- **Key insight**: Entropy < 0.3 + consensus ≥ 4 → 95.2% accuracy
- AIME difficulty progression: 1-5 (33.5%), 6-10 (20.7%), 11-15 (10.3%)
- Majority vote: 15.3%, best strategy: 15.6% (+2 problems)
- Saved analysis to `output/traces/analysis_insights.md`

### Fine-Tuning Research
- **Tinker API**: $0.40/M tokens for Qwen3-8B, $150 free credits
- **GRPO**: Go-to algorithm for math reasoning (DeepSeek-R1)
- **OpenMathReasoning**: NVIDIA's dataset that won AIMO-2
- Saved resources to `data/fine_tuning_resources.md`

### Notebooks Pushed
- `traces-h100-test`: https://www.kaggle.com/code/sonphamorg/traces-h100-test
- `aimo3-curation-test-feb6`: https://www.kaggle.com/code/sonphamorg/aimo3-curation-test-feb6

### Fixed Issues
- Kaggle CLI "Notebook not found" = slug already exists (use new slug)
- Added KAGGLE_API_TOKEN to ~/.bashrc (before interactive check)
