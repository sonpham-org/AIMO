# AIMO3 Iteration Plan

> Created: 2026-02-02, Session 7
> Context: Score 38/50 with weighted entropy notebook. Need faster iteration loops.

## Current State

### What We Have
- **Best submission**: `sonphamorg/41-50-aimo-3-weighted-entropy` — score **38/50**
  - Model: `gpt-oss-120b` (MoE, 117B/5.1B active, MXFP4)
  - Key: entropy-weighted answer selection, 16 persistent Jupyter kernels, 128-turn TIR, streaming logprobs
  - Deps: `andreasbis/aimo-3-utils` (wheels tarball with openai_harmony, vllm, etc.) + `danielhanchen/gpt-oss-120b`
- **Improvement v1**: `sonphamorg/aimo3-entropy-plus-v1` — pushed, awaiting results
  - Local path: `kaggle_submissions/improv1_entropy_plus/`
  - Changes: 12 attempts (was 8), early stop at 5, code output fallback, prompt diversity (8 reasoning + 2 code-first + 2 case-analysis), hybrid scoring with vote bonus
- **TIR solver** (`sonphamorg/aimo3-qwen3-30b-tir-solver`): Failed due to **technical/infra errors** (vLLM install conflicts), never evaluated on actual problems. Approach not invalidated.

### Local Test Data (50 problems with ground truth)
- `data/aimo3/reference.csv` — 10 AIMO3 reference problems (olympiad level, 0-99999 answers)
- `data/aimo2/reference.csv` — 10 AIMO2 reference problems
- `data/aimo2/aime_2025_30.csv` — 30 AIME 2025 problems
- `data/aimo3/test.csv` — 3 trivial stub problems (smoke test only)

### Local Eval Scripts
- `solve_aimo3.py` — runs against reference.csv, outputs JSON traces + accuracy
- `recreate_solutions/*/run.py` — per-solution eval scripts with `--competition` flag
- `recreate_solutions/common/data_loader.py` — shared CSV loader

### Hardware
- **Kaggle**: H100 80GB, 9hr limit, 1 submit/day, no internet during eval
- **Local (switching to)**: NVIDIA machine with more VRAM (need to confirm specs)
- **gpt-oss-120b** needs ~60GB VRAM (MXFP4) + KV cache — requires H100 or equivalent

---

## Priority 1: Save-and-Replay (Highest ROI)

### What
Modify the competition notebook to save raw model outputs (full text + logprobs) as JSON to `/kaggle/working/`. After a Kaggle run, download these outputs. Then test different selection strategies locally — **no GPU needed, runs in seconds**.

### Why
One Kaggle run (~hours) produces data for hundreds of offline experiments. Currently, model outputs are consumed in-flight and discarded. This is the single biggest leverage point.

### How to Implement
Add to `AIMO3Solver._process_attempt()` — after streaming completes, save:
```python
{
    "attempt_idx": idx,
    "problem": problem,
    "system_prompt": sys_prompt,
    "answer": ans,
    "source": source,  # "boxed" or "code_fallback"
    "entropy": entropy,
    "logprobs": logprobs,  # raw top-5 logprobs per token
    "response_tokens": total_toks,
    "python_calls": py_calls,
    "python_errors": py_errs,
    "last_code_output": last_code_output,
    "seed": seed,
}
```

Add to `AIMO3Solver.solve_problem()` — after all attempts complete, write:
```python
import json
output_dir = "/kaggle/working/traces"
os.makedirs(output_dir, exist_ok=True)
with open(f"{output_dir}/problem_{problem_hash}.json", "w") as f:
    json.dump({"problem": problem, "results": results, "final_answer": final}, f)
```

Then create a local replay script `scripts/replay_selection.py` that:
1. Reads all `problem_*.json` files from a downloaded traces directory
2. Applies different selection strategies (pure majority, pure entropy, hybrid with various `vote_bonus` values, etc.)
3. Compares against ground truth from reference.csv
4. Outputs accuracy table per strategy

### Effort
- Notebook modification: ~30 lines
- Replay script: ~100 lines
- Payoff: unlimited offline experiments per Kaggle run

---

## Priority 2: Local Proxy Evaluation

### What
Run a smaller model (14B or 30B) locally on the NVIDIA machine against the 50 reference problems. Use this as a fast iteration loop for testing infrastructure changes, prompt strategies, and selection algorithms.

### Why
Absolute scores will be lower than gpt-oss-120b, but **relative improvements** from strategy changes should transfer. Testing locally takes minutes per problem instead of hours on Kaggle.

### How to Implement
The existing `solve_aimo3.py` already supports this. Steps:
1. Download a model to the NVIDIA machine (e.g., `DeepSeek-R1-Distill-Qwen-14B-AWQ` or `Qwen3-30B-A3B`)
2. Start vLLM server locally: `python -m vllm.entrypoints.openai.api_server --model <path> --dtype auto`
3. Run: `python solve_aimo3.py --api-base http://localhost:8000/v1 --problems 1-10`
4. Compare accuracy across strategy variants

### Model Candidates for Local Testing
| Model | VRAM | Active Params | Notes |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-14B-AWQ | ~8GB | 14B | Good baseline, AWQ quantized |
| Qwen3-30B-A3B | ~18GB | 3B | MoE, very fast, but weak on hard math |
| Qwen3-30B-A3B-AWQ | ~10GB | 3B | Even smaller footprint |
| DeepSeek-R1-Distill-Qwen-32B-AWQ | ~18GB | 32B | Strongest option if VRAM allows |

### Effort
- Setup: 30 min (download model, start server)
- Per-run: ~5-20 min for 10 problems (depends on model size)

---

## Priority 3: Result Logging in Competition Notebook

### What
Add structured logging to the competition notebook that writes per-problem results to `/kaggle/working/results.json`. This makes every Kaggle run more informative even without save-and-replay.

### How to Implement
Add after `solve_problem()` returns:
```python
# In the predict() function
all_results.append({
    "id": pid, "answer": result,
    "time_elapsed": time.time() - start,
    "problems_remaining": solver.problems_remaining,
})
# Write incrementally so partial results survive timeouts
with open("/kaggle/working/results.json", "w") as f:
    json.dump(all_results, f)
```

### Effort
- ~10 lines of code
- Download results from Kaggle notebook output after each run

---

## Priority 4: Prompt Sweep

### What
Test 20+ prompt variants on the local NVIDIA machine with a 14B model against 50 reference problems. Take the top 3-5 winning prompts and deploy in the gpt-oss-120b notebook.

### Prompt Variants to Test
1. Original reasoning prompt (baseline)
2. Code-first: "Write Python code to solve this computationally..."
3. Case-analysis: "Break the problem into cases..."
4. Verify-first: "Before solving, identify what type of problem this is..."
5. Short-answer: "Solve concisely. Skip obvious steps..."
6. Multi-method: "Try at least two different approaches..."
7. Number-theory specific: "Consider modular arithmetic, divisibility..."
8. Geometry specific: "Use coordinate geometry or trigonometric identities..."
9. Combinatorics specific: "Think about counting, bijections, generating functions..."
10. Step-verify: "After each step, verify it's correct before proceeding..."

### How to Implement
Create `scripts/prompt_sweep.py`:
1. Define all prompt variants
2. For each variant, run N problems with that prompt
3. Track accuracy per prompt
4. Output ranked table

### Effort
- Script: ~150 lines
- Per-sweep: ~1-2 hours on local GPU for 50 problems × 10 prompts

---

## Priority 5: Ablation Testing

### What
For each improvement in `improv1_entropy_plus`, create a variant that disables just that one change. Measures which improvements actually help.

### Variants to Test
| Variant | Description |
|---|---|
| baseline | Original weighted entropy (8 attempts, single prompt, pure entropy) |
| +attempts_only | 12 attempts, but no other changes |
| +prompts_only | Prompt diversity, but 8 attempts + pure entropy |
| +hybrid_only | Hybrid scoring, but 8 attempts + single prompt |
| +fallback_only | Code fallback, but 8 attempts + single prompt |
| improv1 (all) | All changes combined |

### How to Implement
Parameterize CFG so each variant can be toggled. Run each against reference problems locally.

---

## Priority 6: Fine-Tuning (Medium-Term)

### A. DPO from Competition Outputs
After implementing save-and-replay (Priority 1), each Kaggle run produces (problem, correct_attempt, incorrect_attempt) triples. These are free DPO training pairs.

Pipeline:
1. Run competition → download traces
2. Filter: attempts where answer matches ground truth = "chosen"
3. Filter: attempts where answer is wrong = "rejected"
4. Fine-tune 14B model with QLoRA DPO on 4090
5. Evaluate on held-out problems
6. If improved, quantize and deploy

### B. Progressive Model Scaling
1. Validate training recipe on 7B (fast, fits on 4090)
2. Confirm improvement → scale to 14B
3. Confirm → scale to 32B (may need multi-GPU or Kaggle training notebook)
4. Each step narrows hyperparameters

### C. Use Kaggle Free GPU for Training
Kaggle gives 30 hrs/week free GPU (separate from competition). Upload training data as dataset, run training notebook, save checkpoints as new dataset, reference in competition notebook.

---

## Implementation Order

```
Week 1 (immediate):
  [1] Save-and-replay modification to improv1 notebook
  [2] Set up local proxy eval on NVIDIA machine
  [3] Add result logging to competition notebook

Week 2:
  [4] Download first competition traces, build replay script
  [5] Prompt sweep on local GPU
  [6] Ablation testing

Week 3+:
  [7] DPO training from competition outputs
  [8] Progressive model scaling
```

---

## Key Principles

1. **Never regress from the weighted entropy architecture.** It works. Build on it.
2. **Separate generation from selection.** Most gains come from better selection strategies, which are free to test offline.
3. **Use smaller models for strategy validation, large models for competition.** Relative improvements transfer.
4. **Every Kaggle run should produce reusable data.** Don't waste the 1/day submission.
5. **The TIR solver failed technically, not conceptually.** Its approach (subprocess code execution, simpler infra) may still be worth revisiting once infra issues are resolved, but the weighted entropy architecture is strictly better for now.
