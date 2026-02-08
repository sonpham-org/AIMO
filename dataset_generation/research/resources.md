# Resources for Dataset Generation

> Last updated: 2026-02-07
> Author: resource-finder agent
> Purpose: Tools, libraries, notebooks, papers, and compute resources for AIMO3 dataset generation and fine-tuning

---

## 1. Tools & Libraries for Data Processing

### 1.1 Format Conversion

| Tool | Purpose | Location / URL |
|------|---------|----------------|
| **openai-harmony** (PyPI) | Official Harmony protocol SDK. Rendering, parsing, format conversion for gpt-oss. **Required** -- gpt-oss will not work without Harmony format. | `pip install openai-harmony` / [GitHub](https://github.com/openai/harmony) |
| **convert_harmony_format.py** | Convert AIMO3 TIR dataset from Harmony to ChatML, Anthropic, Llama 3.1 tool-use formats. Includes `harmony_to_chatml()`, `harmony_to_anthropic()`, `harmony_to_llama_tool()`. | `data/downloads/aimo3_tir/convert_harmony_format.py` (local) |
| **NeMo-Skills converters** | NVIDIA's data conversion scripts (used for OpenMathReasoning). Handles CoT, TIR, GenSelect formats. | [GitHub](https://github.com/NVIDIA-NeMo/Skills) |

### 1.2 Answer Verification

| Tool | Purpose | URL |
|------|---------|-----|
| **math-verify** (v0.8.0) | HuggingFace's math answer verification library. Parse -> SymPy -> compare. Fixed Open LLM Leaderboard (+4.66 pts on MATH). Standard for the community. | `pip install math-verify` / [Blog](https://huggingface.co/blog/math_verify_leaderboard) |
| **SymPy** | Symbolic mathematics. Core dependency for answer checking, expression normalization, equivalence testing. | `pip install sympy` |
| **MATH-VF** | Formal verification: Formalizer + Critic using Z3 + SymPy for step-level verification. More rigorous than answer-only checking. | [arxiv 2505.20869](https://arxiv.org/abs/2505.20869) |

### 1.3 Dataset Curation & Deduplication

| Tool | Purpose | URL |
|------|---------|-----|
| **sentence-transformers** | Embedding-based semantic deduplication. Use `all-MiniLM-L6-v2` for fast embeddings, cosine similarity > 0.95 for near-duplicate detection. | `pip install sentence-transformers` / [GitHub](https://github.com/huggingface/sentence-transformers) |
| **NeMo Curator** | NVIDIA's data curation framework. Includes semantic deduplication, fuzzy dedup, PII removal. GPU-accelerated at scale. | [Developer Page](https://developer.nvidia.com/nemo-curator) |
| **NeMo Data Designer** | NVIDIA's synthetic data generation library. From-scratch or seed-based generation. | [GitHub](https://github.com/NVIDIA-NeMo/DataDesigner) |
| **datasketch** | MinHash-based deduplication. Fast approximate dedup for large datasets. | `pip install datasketch` |
| **HDBSCAN** | Density-based clustering for grouping similar problems. Better than k-means for uneven clusters. | `pip install hdbscan` |

### 1.4 Fine-Tuning Frameworks

| Tool | Purpose | Key Specs | URL |
|------|---------|-----------|-----|
| **Unsloth** | QLoRA fine-tuning for gpt-oss-120b. 2x faster, 70% less VRAM. **Only framework supporting gpt-oss QLoRA**. VRAM: 65GB for 120B model. Exports to vLLM, GGUF, HF. | [Docs](https://docs.unsloth.ai/models/gpt-oss-how-to-run-and-fine-tune) / [GitHub](https://github.com/unslothai/unsloth) |
| **TRL** (HuggingFace) | SFTTrainer, GRPO, DPO. Full Unsloth integration. Standard high-level training API. | `pip install trl` / [GitHub](https://github.com/huggingface/trl) |
| **Tinker** (Thinking Machines Lab) | API-based fine-tuning. Per-token pricing (~$0.40/M tokens). Supports gpt-oss-120b, LoRA, GRPO, DPO. $150 free credits. | [Docs](https://tinker-docs.thinkingmachines.ai/) / [Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) (local: `fine_tuning/tinker-cookbook/`) |

### 1.5 Inference / Serving

| Tool | Purpose | URL |
|------|---------|-----|
| **vLLM** | Production LLM serving. Harmony format support for gpt-oss. PagedAttention, continuous batching. Used in all AIMO submissions. | [Docs](https://docs.vllm.ai/) / [GPT-OSS recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html) |
| **SGLang** | Alternative to vLLM. Used by OpenR1 project for generating DeepSeek-R1 solutions. | [GitHub](https://github.com/sgl-project/sglang) |

### 1.6 Our Local Scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_traces.py` | TIR trace generation with logprobs using vLLM + Jupyter kernels |
| `scripts/replay_selection.py` | 138 offline answer-selection strategies for sweep |
| `scripts/curate_dataset.py` | Dataset curation utilities |
| `scripts/extract_problems.py` | Problem extraction from various formats |
| `scripts/split_aime.py` | Split AIME problems for train/test |
| `scripts/ablation_test.py` | Ablation testing framework |

---

## 2. Kaggle Datasets (Math Problems & Solutions)

### 2.1 AIMO3-Specific (Highest Priority)

| Dataset | Kaggle Slug | Size | Key Feature |
|---------|-------------|------|-------------|
| **AIMO3 TIR** | `jeannkouagou/aimo3-tool-integrated-reasoning` | 141K traces | gpt-oss-120b generated, Harmony format |
| **AIMO3 Hard** | `wenliangtlh/aimo3-high-difficulty-tool-calling-dataset` | 7.3K problems / ~70K traces | Has pass_rate metadata |
| **AIMO External** | `alejopaullier/aimo-external-dataset` | 4.5MB | Most popular (78 votes), curated |
| **AIMO3 Prize Dataset** | `abdoulrachidlengane/ai-mathematical-olympiad-progress-prize3` | Varies | Competition data |

### 2.2 General Math

| Dataset | Kaggle Slug | Size | Notes |
|---------|-------------|------|-------|
| **NuminaMath-TIR** | `jorgeplazas/numinamath-tir` | ~70K | GPT-4 generated TIR (TORA format) |
| **OpenR1-Math-220k** | `alejopaullier/openr1-math-220k` | 220K | DeepSeek-R1 traces, DPO potential |
| **AIME 1983-2024** | `hemishveeraboina/aime-problem-set-1983-2024` | ~1.2K | Evaluation set |

### 2.3 Download Commands

```bash
# Set auth
export $(grep KAGGLE_API_TOKEN /home/son/GitHub/AIMO/.env)

# AIMO3 datasets (already downloaded)
kaggle datasets download jeannkouagou/aimo3-tool-integrated-reasoning -p data/downloads/aimo3_tir/
kaggle datasets download wenliangtlh/aimo3-high-difficulty-tool-calling-dataset -p data/downloads/aimo3_hard/

# Additional Kaggle datasets
kaggle datasets download jorgeplazas/numinamath-tir -p data/downloads/numinamath_tir/
kaggle datasets download alejopaullier/openr1-math-220k -p data/downloads/openr1_math/
```

---

## 3. HuggingFace Datasets

### 3.1 Top Priority for Our Pipeline

| Dataset | HF ID | Size | Why Priority |
|---------|-------|------|--------------|
| **OpenMathReasoning** | `nvidia/OpenMathReasoning` | 306K problems / 5.5M solutions | Won AIMO2, has TIR + GenSelect + pass_rate |
| **LIMO-v2** | `GAIR/LIMO-v2` | ~800 | Updated curation, gold standard |
| **Light-R1-SFTData** | `qihoo360/Light-R1-SFTData` | 76K + 3K | Pre-curated stage1 + hard stage2 |

### 3.2 Large-Scale

| Dataset | HF ID | Size | Notes |
|---------|-------|------|-------|
| **OpenThoughts3-1.2M** | `open-thoughts/OpenThoughts3-1.2M` | 1.2M | 850K math, QwQ-32B generated |
| **OpenR1-Math-220k** | `open-r1/OpenR1-Math-220k` | 220K | DeepSeek-R1 traces |
| **OpenMathInstruct-2** | `nvidia/OpenMathInstruct-2` | 14M | Llama-3.1-405B generated |
| **AM-DeepSeek-R1-1.4M** | `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` | 1.4M | Long reasoning traces |
| **DART-Math-Hard** | `hkust-nlp/dart-math-hard` | ~590K | Difficulty-aware rejection tuning |

### 3.3 Curated Small

| Dataset | HF ID | Size | Result |
|---------|-------|------|--------|
| **LIMO** | `GAIR/LIMO` | 817 | 57.1% AIME, 94.8% MATH |
| **s1K** | `simplescaling/s1K` | 1,000 | Beat o1-preview |

### 3.4 RL & Reward Models

| Dataset | HF ID | Size | Purpose |
|---------|-------|------|---------|
| **Big-Math-RL-Verified** | `SynthLabsAI/Big-Math-RL-Verified` | 251K | RL training, has llama8b_solve_rate |
| **PRM800K** | `openai/prm800k` (GitHub) | 800K step labels | Process Reward Model training |
| **AceMath-RM-Training** | `nvidia/AceMath-RM-Training-Data` | Large | Outcome reward model training |

### 3.5 Download Commands

```bash
# Top priority
huggingface-cli download nvidia/OpenMathReasoning --local-dir data/downloads/OpenMathReasoning/
huggingface-cli download GAIR/LIMO-v2 --local-dir data/downloads/LIMO-v2/
huggingface-cli download qihoo360/Light-R1-SFTData --local-dir data/downloads/Light-R1-SFTData/

# Large-scale (download when needed)
huggingface-cli download open-thoughts/OpenThoughts3-1.2M --local-dir data/downloads/OpenThoughts3/
huggingface-cli download open-r1/OpenR1-Math-220k --local-dir data/downloads/OpenR1-Math/
```

---

## 4. Key Notebooks & Competition Solutions

### 4.1 AIMO3 Public Notebooks (Kaggle)

| Notebook | Author | Key Approach | URL |
|----------|--------|-------------|-----|
| **AIMO 3 Submission Demo** | ryanholbrook | Official demo, submission format | [Link](https://www.kaggle.com/code/ryanholbrook/aimo-3-submission-demo) |
| **AIMO 3 GPT-OSS-120B (with tools)** | andreasbis | TIR with Harmony protocol | [Link](https://www.kaggle.com/code/andreasbis/aimo-3-gpt-oss-120b-with-tools) |
| **AIMO 3 GPT-OSS-120B ~3hrs** | seshurajup | Optimized runtime | [Link](https://www.kaggle.com/code/seshurajup/aimo-3-gpt-oss-120b-3hours-wow-h100) |
| **AIMO 3 GPT-OSS-120B + Agentic** | seshurajup | Agentic solver approach | [Link](https://www.kaggle.com/code/seshurajup/aimo-3-gpt-oss-120b-agentic-solver) |
| **AIMO 3 Baseline GPT-OSS-120B** | takuji | Baseline implementation | [Link](https://www.kaggle.com/code/takuji/aimo-3-baseline-gpt-oss-120b) |

### 4.2 AIMO2 Winning Solutions (Reference)

| Solution | Team | Score | Key Resource |
|----------|------|-------|-------------|
| **1st Place: NemoSkills** | NVIDIA | 34/50 | [Kaggle Writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills) / [Paper](https://arxiv.org/abs/2504.16891) |
| **AIMO1 Winner: Numina** | Project Numina | 29/50 | [GitHub](https://github.com/project-numina/aimo-progress-prize) / [Blog](https://huggingface.co/blog/winning-aimo-progress-prize) |

### 4.3 Our Submissions (Local)

| Submission | Score | Directory |
|------------|-------|-----------|
| **feb3 (best)** | 40/50 | `submissions/feb3_entropy_gated/` |
| feb4 verified | 32/50 | `submissions/feb4_verified/` |
| feb5 more attempts | 37/50 | `submissions/feb5_more_attempts/` |
| feb6 16 attempts | 29/50 | `submissions/feb6_16_attempts/` |

---

## 5. Papers on Math Dataset Generation

### 5.1 Dataset Creation & Curation

| Paper | Key Contribution | Year | Link |
|-------|-----------------|------|------|
| **OpenMathReasoning** | 3-stage industrial pipeline (540K problems -> 5.5M solutions), iterative generate-filter-retrain | 2025 | [arxiv](https://arxiv.org/abs/2504.16891) |
| **LIMO** | 817 curated samples beat 100K random; 4-dimension quality scoring | 2025 | [arxiv](https://arxiv.org/abs/2502.03387) |
| **s1** | 59K -> 1K via difficulty+diversity+quality; dual-model gating | 2025 | [arxiv](https://arxiv.org/abs/2501.19393) |
| **DART-Math** | Difficulty-aware rejection tuning (Prop2Diff); NeurIPS 2024 | 2024 | [arxiv](https://arxiv.org/abs/2407.13690) |
| **ASTER** | 4K interaction-dense TIR cold-start; interaction collapse prevention | 2026 | [arxiv](https://arxiv.org/html/2602.01204) |
| **Big-Math** | 250K+ RL-verified problems, reformulated multiple-choice | 2025 | [arxiv](https://arxiv.org/abs/2502.17387) |

### 5.2 Training Methods

| Paper | Key Contribution | Year | Link |
|-------|-----------------|------|------|
| **iw-SFT** | Importance-weighted SFT = +10% AIME with zero additional data | 2025 | [arxiv](https://arxiv.org/abs/2507.12856) |
| **RSR** | Rank-Surprisal Ratio for trajectory selection (0.86 Spearman) | 2026 | [arxiv](https://arxiv.org/abs/2601.14249) |
| **Light-R1** | Curriculum SFT + DPO + RL pipeline from scratch | 2025 | [arxiv](https://arxiv.org/abs/2503.10460) |
| **GRPO-LEAD** | Difficulty-aware advantage reweighting for RL | 2025 | [arxiv](https://arxiv.org/abs/2504.09696) |
| **AdaRFT** | Dynamic difficulty curriculum for RL | 2025 | [arxiv](https://arxiv.org/abs/2504.05520) |
| **Front-Loading** | SFT quality > quantity (+15% vs -5% from doubling data) | 2025 | [arxiv](https://arxiv.org/abs/2510.03264) |

### 5.3 Verification & Selection

| Paper | Key Contribution | Year | Link |
|-------|-----------------|------|------|
| **AceMath** | Cross-model verification + reward models | 2024 | [arxiv](https://arxiv.org/abs/2412.15084) |
| **PRM800K (Let's Verify)** | 800K step-level labels, process > outcome supervision | 2023 | [arxiv](https://arxiv.org/abs/2305.20050) |
| **Math-Shepherd** | Automated step-level PRM labels via MCTS | 2024 | [arxiv](https://arxiv.org/abs/2312.08935) |
| **OmegaPRM** | Divide-and-conquer MCTS process supervision | 2024 | [arxiv](https://arxiv.org/abs/2406.06592) |
| **Skill-Aware Selection** | Hierarchical skill tree, weak-skill oversampling | 2025 | [arxiv](https://arxiv.org/abs/2601.10109) |
| **Guided by Trajectories** | Repairing & rewarding TIR trajectories | 2026 | [arxiv](https://arxiv.org/abs/2601.23032) |

---

## 6. Compute Resources

### 6.1 Google API Credits (~$93 Available)

**Pricing (Gemini Developer API, Feb 2026):**

| Model | Input $/M tokens | Output $/M tokens | Best For |
|-------|------------------|-------------------|----------|
| **Gemini 2.0 Flash** | $0.10 | $0.40 | Cheapest. Verification, classification, scoring. |
| **Gemini 2.5 Flash-Lite** | $0.10 | $0.40 | Same price as 2.0 Flash, newer architecture. |
| **Gemini 2.5 Flash** | $0.30 | $2.50 | Higher quality reasoning, harder tasks. |
| **Batch API** | 50% off all models | 50% off | Non-urgent bulk processing (asynchronous). |

**Budget Analysis -- What $93 Buys:**

| Task | Model | Estimated Cost | Volume |
|------|-------|---------------|--------|
| **Topic classification** (short prompts, short outputs) | 2.0 Flash | ~$0.50 per 10K problems | ~1.8M problems |
| **Answer verification** (medium prompts, short outputs) | 2.0 Flash | ~$1.50 per 10K solutions | ~620K solutions |
| **Quality scoring** (read full solution, score) | 2.0 Flash | ~$5 per 10K solutions | ~186K solutions |
| **Solution generation** (long outputs) | 2.5 Flash | ~$25 per 1K solutions | ~3.7K solutions |
| **Cross-model verification** (generate + compare) | 2.5 Flash | ~$30 per 1K problems | ~3.1K problems |

**Recommended allocation of $93:**

| Use Case | Budget | What You Get |
|----------|--------|-------------|
| Topic classification of 141K AIMO3 TIR problems | ~$7 | MSC domain labels for diversity selection (s1 approach) |
| Quality scoring of 70K AIMO3 Hard traces | ~$35 | LLM-based quality scores (better than keyword heuristics) |
| Answer verification of top 20K candidates | ~$3 | Cross-check extracted answers against ground truth |
| Difficulty estimation via Gemini solve attempts | ~$30 | Generate 4 attempts per problem on ~2K problems |
| Reserve for iteration | ~$18 | Buffer for retries, prompt tuning, additional batches |

**Key optimizations:**
- Use **Batch API** (50% off) for all non-urgent processing
- Use **context caching** (10% of input price) when processing many solutions for the same problem
- Gemini 2.0 Flash free tier: 1,000 requests/day (useful for small-scale testing)

### 6.2 Kaggle H100 (9h Sessions)

- 1x H100 (80GB) per session
- 9 hour time limit
- Offline (no internet) for competition submissions
- **Can also be used for dataset generation** (separate from competition runs)

**Kaggle H100 time budget for dataset generation:**

| Task | Time Estimate | Output |
|------|--------------|--------|
| Generate 8 TIR traces per problem (1K problems) | ~3-4 hours | 8K traces with logprobs (like our existing trace generation) |
| RSR scoring of 10K candidate trajectories | ~30 min | Forward pass through gpt-oss-120b for trajectory selection |
| Unsloth QLoRA fine-tuning (2K examples, 1 epoch) | ~2-3 hours | LoRA adapter for gpt-oss-120b |
| Validation inference (50 test problems) | ~1-2 hours | Pre-submission accuracy check |

**Recommended strategy:** Use separate Kaggle notebooks for dataset generation vs. competition submissions. Dataset generation notebooks can upload results to a Kaggle dataset for later use.

**Dual-use approach per 9h session:**
- Option A: 4h trace generation + 3h fine-tuning + 2h validation
- Option B: Full 8h trace generation (maximize data)
- Option C: Full 9h fine-tuning + validation (after data is ready)

### 6.3 Free Compute Resources

| Resource | GPU | Limits | Best For |
|----------|-----|--------|----------|
| **Google Colab (free)** | T4 (16GB) | 15-30 GPU hrs/week, 12h sessions | Data processing, small model inference, embeddings |
| **HuggingFace ZeroGPU** | H200 (~70GB VRAM) | Free for all users, Gradio-only | Quick inference tests, embedding generation |
| **HuggingFace ZeroGPU (PRO, $9/mo)** | H200 (~70GB VRAM) | 8x quota | More sustained inference |
| **Kaggle free GPU** | T4/P100 (16GB) | 30 hrs/week | Data processing, small model tasks |

**Practical uses of free compute:**
- **Colab/Kaggle T4**: Run sentence-transformers for semantic deduplication of 141K problems (~2-3 hours)
- **Colab/Kaggle T4**: Run Qwen2.5-Math-7B-PRM800K for step-level quality scoring (~4-6 hours for 10K traces)
- **HF ZeroGPU**: Quick tests of embedding models, small inference tasks
- **Colab T4**: Topic classification with small LLM (Qwen3-4B) as free alternative to Gemini API

### 6.4 Fields Model Initiative (AIMO3 Partner)

- **Up to 128 H100 GPUs** available for fine-tuning
- Partnership with LLMC (National Institute of Informatics, Tokyo) + Benchmarks+Baselines (Vienna)
- Access via AIMO3 competition participation
- Apply through Kaggle competition page or AIMO Prize website
- **This is the best option for serious fine-tuning** if we can get access

### 6.5 Tinker API

- Per-token pricing (~$0.40/M tokens for Qwen3-8B)
- $150 free credits for new users
- Supports gpt-oss-120b, LoRA, GRPO, DPO
- API-based (no GPU management)
- Local cookbook: `fine_tuning/tinker-cookbook/`

### 6.6 Local Development

- AMD Radeon 8060S (not NVIDIA -- limited for training)
- llama.cpp with Vulkan backend for inference testing
- Models in `~/models/`
- Port 8081 for local inference server

---

## 7. Code & Repository Resources

### 7.1 Official Repositories

| Repository | Purpose | URL |
|------------|---------|-----|
| **openai/harmony** | Harmony format SDK | [GitHub](https://github.com/openai/harmony) |
| **openai/gpt-oss** | gpt-oss model documentation | [GitHub](https://github.com/openai/gpt-oss) |
| **NVIDIA-NeMo/Skills** | NeMo-Skills pipeline (data gen, training, eval) | [GitHub](https://github.com/NVIDIA-NeMo/Skills) |
| **unslothai/unsloth** | QLoRA fine-tuning framework | [GitHub](https://github.com/unslothai/unsloth) |
| **unslothai/notebooks** | Unsloth training notebooks including gpt-oss-120b | [GitHub](https://github.com/unslothai/notebooks) |
| **huggingface/trl** | TRL training library (SFT, GRPO, DPO) | [GitHub](https://github.com/huggingface/trl) |
| **huggingface/open-r1** | Open reproduction of DeepSeek-R1 | [GitHub](https://github.com/huggingface/open-r1) |
| **project-numina/aimo-progress-prize** | AIMO1 winning solution | [GitHub](https://github.com/project-numina/aimo-progress-prize) |
| **GAIR-NLP/LIMO** | LIMO dataset + training code | [GitHub](https://github.com/GAIR-NLP/LIMO) |
| **simplescaling/s1** | s1K dataset + training code | [GitHub](https://github.com/simplescaling/s1) |
| **Qihoo360/Light-R1** | Light-R1 training pipeline | [GitHub](https://github.com/Qihoo360/Light-R1) |
| **SynthLabsAI/big-math** | Big-Math dataset creation filters + MC reformulation | [GitHub](https://github.com/SynthLabsAI/big-math) |
| **openai/prm800k** | PRM800K step-level labels | [GitHub](https://github.com/openai/prm800k) |

### 7.2 Key Tutorials & Cookbooks

| Resource | Description | URL |
|----------|-------------|-----|
| **Unsloth gpt-oss Fine-Tuning Tutorial** | Step-by-step QLoRA training for gpt-oss-120b | [Docs](https://docs.unsloth.ai/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss) |
| **Unsloth gpt-oss Long Context** | Training with extended context (up to 128K) | [Blog](https://unsloth.ai/blog/gpt-oss-context) |
| **OpenAI Harmony Cookbook** | Harmony format reference and examples | [Cookbook](https://cookbook.openai.com/articles/openai-harmony) |
| **vLLM GPT-OSS Recipe** | Running gpt-oss with vLLM | [Docs](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html) |
| **TRL + Unsloth Integration** | Using TRL SFTTrainer with Unsloth acceleration | [Docs](https://huggingface.co/docs/trl/en/unsloth_integration) |

---

## 8. Competition-Specific Resources

### 8.1 AIMO3 Extra Prizes

| Prize | Description |
|-------|-------------|
| **Math Corpus Prize** | Novel datasets that help the wider community |
| **Longest Leader Prize** | Team staying longest on top of public leaderboard |
| **Write-up Prizes** | Best technical explanation of approach |

### 8.2 Key Deadlines

- **Entry deadline:** April 8, 2026
- **Final submission deadline:** April 15, 2026

### 8.3 Submission Requirements

- Must produce `submission.parquet`
- Uses `kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer`
- `andreasbis/aimo-3-utils` kernel source dependency (includes vLLM, openai_harmony, etc.)
- H100 GPU with NvidiaH100 machine shape

---

## 9. Resource Gaps & Recommendations

### What We Have
- Comprehensive dataset catalog (30+ datasets across 7 tiers)
- Local copies of AIMO3 TIR, AIMO3 Hard, LIMO, s1K, Big-Math-RL-Verified
- Harmony format conversion tools
- Tinker cookbook for API-based training
- Unsloth documentation for local QLoRA
- **~$93 Google API credits** (Gemini Flash)
- **Kaggle H100** (9h sessions, shareable between dataset gen and competition)
- **Free compute**: Colab T4 (15-30 hrs/week), HF ZeroGPU (H200), Kaggle T4 (30 hrs/week)

### What We Need
1. **OpenMathReasoning download** -- highest-priority dataset, won AIMO2. Need the TIR subset at minimum (~50GB total).
2. **Light-R1-SFTData download** -- pre-curated 3K hard examples could be immediately useful.
3. **Fields Model Initiative access** -- 128 H100 GPUs for proper fine-tuning. Need to apply through competition.
4. **math-verify integration** -- should be added to our pipeline for answer verification during dataset curation.
5. **Embedding model for dedup** -- need to set up sentence-transformers for semantic deduplication.
6. **RSR implementation** -- the trajectory selection metric code is available at https://github.com/UmeanNever/RankSurprisalRatio but needs integration.

### Recommended Resource Allocation Plan

**Phase 1: Data Curation (cost: ~$45 Google API + free compute)**
1. Use Colab T4 to run sentence-transformers dedup on 141K AIMO3 TIR problems (~3h free)
2. Use Gemini 2.0 Flash Batch API to classify all problems by MSC topic (~$7)
3. Use Gemini 2.0 Flash to quality-score 70K AIMO3 Hard traces (~$35)
4. Use Colab T4 to run PRM scoring with Qwen2.5-Math-7B-PRM800K (~6h free)
5. Result: 2K-4K curated, classified, scored training examples

**Phase 2: Trace Generation (cost: 1 Kaggle H100 session)**
1. Use Kaggle H100 (9h) to generate traces for problems where existing data is sparse
2. Focus on hard problems (pass_rate 1/8-3/8 from AIMO3 Hard) needing more samples
3. Generate 8 attempts per problem with logprobs for ~500-1K new problems
4. Result: Additional high-quality traces for underrepresented topics/difficulties

**Phase 3: Fine-Tuning (cost: 1 Kaggle H100 session or Tinker $150 credits)**
1. Option A: Kaggle H100 with Unsloth QLoRA (~3h training + 2h validation)
2. Option B: Tinker API for GRPO RL ($50-150 from free credits)
3. Result: LoRA adapter for gpt-oss-120b

**Phase 4: Validation & Submission (cost: ~$18 Google API + 1 Kaggle session)**
1. Use remaining Google API for Gemini-based cross-verification of fine-tuned model outputs
2. Use Kaggle H100 for competition submission with fine-tuned model
3. Result: New submission with improved accuracy

**Total estimated cost: ~$45-63 Google API + $0 free compute + Kaggle sessions (free)**
**Remaining Google API budget: ~$30-48 for iteration**
