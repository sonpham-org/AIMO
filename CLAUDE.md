# AIMO Project — Claude Code Instructions

## At Session Start
- Read `.claude/conversation_log.md` to recall prior discussions and decisions
- Read `.claude/skills.md` for project-specific behaviors

## During Session
- After significant research findings, decisions, or insights: append to `.claude/conversation_log.md`
- Keep entries specific (model names, scores, URLs, parameters)

## Creating New Submission Notebooks

### Quick Start: Copy feb3 and modify CFG only
The base notebook is `submissions/feb3_entropy_gated/kaggle_submission.ipynb` (scored **40/50**).

To create a new variant:
1. Copy `feb3_entropy_gated/` to new directory inside `submissions/`
2. Update `kernel-metadata.json` with new id/title
3. **Only modify the CFG class** — the rest of the code is stable

### Iterable Parameters (CFG class, cell-8)

```python
class CFG:
    # === ITERABLE PARAMETERS (safe to change) ===

    # Attempts & Early Stopping
    attempts = 8          # Number of parallel solution attempts (try: 12, 16, 24)
    workers = 8           # Parallel threads (usually = attempts)
    early_stop = 4        # Stop if N answers agree (try: attempts // 2)

    # Answer Selection
    entropy_threshold = 5.0   # Only trust answers with entropy < this (try: 4.0, 4.5, 6.0)
    min_consensus = 2         # Minimum votes to be candidate (try: 2, 3)

    # Generation
    temperature = 1.0     # Sampling temperature (try: 0.8, 0.9)
    min_p = 0.02          # Min-p sampling (try: 0.01, 0.05)
    seed = 42             # Random seed (try: different values for variance)

    # === FIXED PARAMETERS (don't change unless you know why) ===

    # Model
    served_model_name = 'gpt-oss'
    model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'
    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'

    # Timing (tuned for 50 problems in ~5 hours)
    high_problem_timeout = 900      # Max time per problem
    base_problem_timeout = 270      # Min time per problem
    notebook_limit = 17400          # ~4h 50m self-limit (Kaggle allows 9h)
    server_timeout = 180
    session_timeout = 960
    jupyter_timeout = 6
    sandbox_timeout = 3

    # vLLM settings
    context_tokens = 65536
    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 5
    batch_size = 256               # vLLM max_num_seqs
    gpu_memory_utilization = 0.96
    stream_interval = 200
    turns = 128

    # Prompts (proven to work, don't change)
    system_prompt = '...'
    tool_prompt = '...'
    preference_prompt = '...'
```

### What Others Use (for reference)
| Team | Samples/Problem | Strategy |
|------|-----------------|----------|
| AIMO-2 Winner (Numina) | **48** samples × 4 generations | Simple majority vote |
| CMU-MATH (2nd AIMO-1) | **64** samples | Weighted majority (reward model) |
| Our feb3 | **8** attempts | Entropy-gated consensus |

### Notebook Structure (cells)
- **cell-0**: Markdown header (update description)
- **cell-1 to cell-7**: Setup (don't touch)
- **cell-8**: **CFG class** ← MODIFY THIS
- **cell-9 to cell-13**: Classes (don't touch)
- **cell-14**: AIMO3Solver (don't touch unless changing selection algorithm)
- **cell-15 to cell-17**: Execution (don't touch)

### kernel-metadata.json Template
```json
{
  "id": "sonphamorg/YOUR-SLUG-HERE",
  "title": "YOUR-SLUG-HERE",
  "code_file": "kaggle_submission.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "keywords": ["gpu"],
  "dataset_sources": [],
  "kernel_sources": ["andreasbis/aimo-3-utils"],
  "competition_sources": ["ai-mathematical-olympiad-progress-prize-3"],
  "model_sources": ["danielhanchen/gpt-oss-120b"],
  "machine_shape": "NvidiaH100"
}
```

## Kaggle Submission Notebooks — CRITICAL

Every notebook MUST produce `submission.parquet` **REGARDLESS of mode**. Required last cell:
```python
inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',))
```

## Kaggle CLI Authentication

**Preferred method** (works in non-interactive shells):
```bash
export $(grep KAGGLE_API_TOKEN /home/son/GitHub/RNA3D/.env) && kaggle kernels push
```

Credentials also in `~/.bashrc` (before interactive check) for interactive use.

**Common errors:**
- "Notebook not found" → kernel slug already exists, use a new slug
- "401 Unauthorized" → regenerate token at kaggle.com/settings → API

## Submission History

| Date | Directory | Strategy | Score | Key Changes |
|------|-----------|----------|-------|-------------|
| Feb 3 | `feb3_entropy_gated/` | Entropy-gated consensus | **40/50** | attempts=8, entropy<5.0, consensus>=2 |
| Feb 4 | `feb4_verified/` | Verified consensus | **32/50** | Added code_boost, repeat_boost — HURT |
| Feb 5 | `feb5_eagle3/` | Eagle3 speculative | FAILED | Eagle3 broken with gpt-oss-120b |
| Feb 5 | `feb5_more_attempts/` | More attempts | PENDING | attempts=12, early_stop=5 |
| Feb 5 | `feb5_adaptive/` | Adaptive resampling | PENDING | Phase 1: 6 attempts, Phase 2: extra attempts on uncertain problems |

### Key Lessons
- **Simple is better**: feb3 (40/50) > feb4 (32/50)
- **Don't add complexity**: code_boost, repeat_boost, error_penalty all hurt
- **Eagle3 is broken**: Accuracy drops 73%→28% even when it loads ([vLLM #27626](https://github.com/vllm-project/vllm/issues/27626))
- **More samples might help**: Winners use 48-64 samples vs our 8

## Eagle3 — DON'T USE
Eagle3 + gpt-oss-120b has critical bugs:
- [vLLM #26328](https://github.com/vllm-project/vllm/issues/26328): `NotImplementedError: Mxfp4 linear layer`
- [vLLM #27626](https://github.com/vllm-project/vllm/issues/27626): Accuracy drops dramatically even when working

## Project Context
AIMO3 competition on Kaggle. 50 math problems, 9 hour limit, H100 GPU. Best open model is gpt-oss-120b.
