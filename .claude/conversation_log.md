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

