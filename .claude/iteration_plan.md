# AIMO3 Iteration Plan

> Created: 2026-02-02, Session 7
> Updated: 2026-02-02, Session 8 — trace generation + replay system implemented
> Updated: 2026-02-04, Session 12 — switched to Qwen3-8B for faster trace generation
> Context: Score **40/50** with entropy-gated consensus (feb3). Iterating on verification methods.

## Current State

### What We Have
- **Best submission**: `sonphamorg/aimo3-entropy-gated-feb3` — score **40/50** ⭐
  - Model: `gpt-oss-120b` (MoE, 117B/5.1B active, MXFP4)
  - Key: entropy-gated consensus (filter by entropy < 5.0, require ≥2 votes)
- **Feb4 submission**: `sonphamorg/aimo3-verified-consensus-feb4` — PENDING
  - Adds Option B (code verification) + Option C (self-consistency boost)
- **Trace + Replay System**: `scripts/generate_traces.py` + `scripts/replay_selection.py` — READY
- **Ablation testing**: `scripts/ablation_test.py` — tests 9 selection strategies

### Local Test Data
- `data/aimo3/reference.csv` — 53 AIMO3 reference problems with ground truth
- `data/aime/aime_train_100.csv` — 100 AIME problems (random subset for overnight runs)
- `data/aime/aime_train_2005_2022.csv` — 524 AIME problems (full set)

---

## OVERNIGHT RUN (2026-02-04)

### Currently Running
- **Model**: Qwen3-8B (fast, ~50 tok/s, generates shorter responses than DeepSeek-R1)
- **Dataset**: 100 AIME problems × 12 samples = 1,200 traces
- **Output**: `output/traces/aime_qwen3_8b_20260204_081902/`
- **Server**: llama-server on port 8081 (PID varies)
- **Generator**: `generate_traces.py` (PID varies)

### Why Qwen3-8B instead of gpt-oss-120b or DeepSeek-R1?
- DeepSeek-R1-0528-Qwen3-8B generates **13k+ tokens per response** (very slow)
- gpt-oss-120b is slower (~20 tok/s vs ~50 tok/s for Qwen3-8B)
- **Goal**: More traces for statistical significance on selection strategies
- Selection strategies should transfer across models

### Monitor Progress
```bash
# Check process
ps aux | grep generate_traces

# Count completed traces
ls output/traces/aime_qwen3_8b_*/problem_*.json 2>/dev/null | wc -l

# View log
tail -f logs/trace_qwen3_8b.log
```

### When Done (morning)
```bash
# Run ablation testing on new traces
python3 scripts/ablation_test.py --traces-dir output/traces/aime_qwen3_8b_*/

# Run full selection sweep (138 strategies)
python3 scripts/replay_selection.py sweep --traces-dir output/traces/aime_qwen3_8b_*/
```

---

## AMD Strix Halo Setup (THIS MACHINE)

### Hardware
- CPU: AMD Ryzen AI MAX+ 395
- GPU: Radeon 8060S (RDNA 3.5), 96 GB unified VRAM
- Backend: **llama.cpp Vulkan** (llama-server with OpenAI-compatible API)
- **gpt-oss-120b Q4_K_M fits locally** (58.5 GiB model + 37 GiB KV cache)

### Model Speeds (llama.cpp Vulkan, Q4_K_M)
| Model | tok/s | Size | Notes |
|---|---|---|---|
| Qwen3-30B-A3B | 89 | 17.3 GiB | Fast proxy for broad sweeps |
| **gpt-oss-120b** | 55 | 58.5 GiB | **Actual competition model** |
| Qwen3-4B | 75 | 2.3 GiB | Quick smoke tests |

### Python Environment
```bash
# Use the existing venv (has jupyter_client, pandas, requests, etc.)
source /home/son/GitHub/AIMO/.venv/bin/activate
# OR use directly:
/home/son/GitHub/AIMO/.venv/bin/python scripts/generate_traces.py ...
```

---

## RUN NOW: Quick Iteration Loop

### Step 1: Start llama-server (if not running)
```bash
# For quick smoke test (Qwen3-4B, ~75 tok/s)
~/llama.cpp/build/bin/llama-server \
  -m ~/models/Qwen3-4B/Q4_K_M.gguf \
  -ngl 99 -c 16384 --port 8080

# For real evaluation (gpt-oss-120b, ~55 tok/s)
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gpt-oss-120b/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  -ngl 99 -c 32768 --port 8080
```

### Step 2: Smoke Test (~5 min)
```bash
/home/son/GitHub/AIMO/.venv/bin/python scripts/smoke_test.py \
  --api-base http://localhost:8080/v1
```

### Step 3: Generate Traces (~40 min with gpt-oss-120b, 12 samples × 53 problems)
```bash
/home/son/GitHub/AIMO/.venv/bin/python scripts/generate_traces.py \
  --api-base http://localhost:8080/v1 \
  --model gpt-oss-120b \
  --n-samples 12 \
  --max-turns 16
```

**For faster iteration (subset of problems):**
```bash
# Just 5 problems for quick validation
/home/son/GitHub/AIMO/.venv/bin/python scripts/generate_traces.py \
  --api-base http://localhost:8080/v1 \
  --problems "0e644e,1a2b3c,..." \
  --n-samples 12
```

### Step 4: Selection Sweep (<5 seconds, tests 138 strategies)
```bash
/home/son/GitHub/AIMO/.venv/bin/python scripts/replay_selection.py sweep \
  --traces-dir output/traces/<timestamp>/
```

Output shows ranked strategies:
```
=================================================================
 #   Strategy                                  Accuracy   Correct
=================================================================
 1   hybrid_inv_k0.25                          0.528      28/53
 2   threshold_2.0_hybrid_inv_k0.1             0.509      27/53
 3   majority_vote                             0.491      26/53
...
```

### Step 5: Deploy Winner
Take the best strategy from the sweep and update `kaggle_submissions/improv1_entropy_plus/kaggle_submission.ipynb` to use those exact parameters.

---

## Time Estimates

| Task | Model | Problems | Samples | Time |
|---|---|---|---|---|
| Smoke test | Qwen3-4B | 2 | 3 | ~3 min |
| Quick validation | gpt-oss-120b | 5 | 5 | ~15 min |
| Full baseline | gpt-oss-120b | 53 | 5 | ~2.5 hrs |
| Full + scaled | gpt-oss-120b | 53 | 12 | ~6 hrs |
| Selection sweep | CPU only | N/A | N/A | <5 sec |

---

## What the Scripts Do

### `scripts/generate_traces.py`
- Runs TIR (Tool-Integrated Reasoning) with code execution
- Saves per-problem JSON with: all attempts, answers, entropy, code executions, logprobs
- Works with any OpenAI-compatible API (llama-server, vLLM, etc.)

### `scripts/replay_selection.py`
- Loads saved traces, tests 138 selection strategies
- Records everything to SQLite (`results/experiments.db`)
- **sweep mode**: Grid search over all strategies (instant)
- **optimize mode**: Bayesian optimization of generation params via Optuna (hours)

### 138 Selection Strategies
| Group | Count | Description |
|---|---|---|
| Pure | 6 | Majority vote, min entropy, 4 entropy transforms |
| Hybrid | 40 | entropy + k×votes (4 transforms × 10 k-values) |
| Normalized | 44 | (1-α)×entropy + α×votes (11 α-values) |
| Threshold | 7 | Filter low-entropy, then vote |
| Top-K | 6 | Keep K best, then vote |
| Thresh+Hybrid | 21 | Filter then hybrid scoring |
| Source-aware | 6 | Penalize code-fallback answers |
| Prompt-aware | 8 | Meta-vote, diversity bonus, etc. |

---

## After Finding Best Strategy

1. **Update improv1 notebook** with winning strategy parameters
2. **Push to Kaggle** — `kaggle kernels push -p kaggle_submissions/improv1_entropy_plus/`
3. **If strategy gains > 2%**: Consider Bayesian optimization for generation params

### Bayesian Optimization (if needed, ~20 hrs)
```bash
/home/son/GitHub/AIMO/.venv/bin/python scripts/replay_selection.py optimize \
  --api-base http://localhost:8080/v1 \
  --model gpt-oss-120b \
  --n-trials 30 \
  --problems "0e644e,1a2b3c,..."  # subset for speed
```

---

## Key Principles

1. **Run everything locally on AMD** — you have the actual competition model
2. **Separate generation from selection** — one trace run enables hundreds of strategy tests
3. **Test on gpt-oss-120b directly** — results transfer to Kaggle
4. **Use Qwen3-30B-A3B only for broad sweeps** — when speed > precision
5. **Every trace file is reusable data** — don't regenerate unless changing generation params
