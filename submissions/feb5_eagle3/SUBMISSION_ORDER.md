# Eagle3 Submission Plan — Step-by-Step

**H100 is scarce. Each notebook runs ~5 hours. Follow this order exactly.**

---

## Step 0: Pre-requisite (no GPU needed)
Upload the Eagle3 draft model to Kaggle as a private dataset.

```bash
# Install tools if needed
pip install huggingface_hub kaggle

# Download Eagle3 model (~588MB)
huggingface-cli download wenliang1990/gpt-oss-120b-eagle3-aimo3 \
  --local-dir /tmp/eagle3-gpt-oss-120b-aimo3

# Initialize Kaggle dataset metadata
cd /tmp/eagle3-gpt-oss-120b-aimo3
kaggle datasets init -p .
# IMPORTANT: Edit dataset-metadata.json and set:
#   "title": "eagle3-gpt-oss-120b-aimo3"
#   "id": "sonphamorg/eagle3-gpt-oss-120b-aimo3"

# Upload
kaggle datasets create -p .
```

**Verify**: Go to https://www.kaggle.com/datasets/sonphamorg/eagle3-gpt-oss-120b-aimo3
and confirm it exists. Note the exact path it mounts to (should be
`/kaggle/input/eagle3-gpt-oss-120b-aimo3`).

If the mount path differs, update `CFG.eagle3_model_path` in the notebook before submitting.

---

## Step 1: DIAGNOSTIC RUN — feb5_eagle3 (1 H100 slot)
**Purpose**: Does Eagle3 work? What's the speedup? Are logprobs valid?
**Notebook**: `kaggle_submissions/feb5_eagle3/`
**Based on**: feb3 (scored 40/50) — our best known baseline

```bash
cd /home/son/GitHub/AIMO/kaggle_submissions/feb5_eagle3
kaggle kernels push -p .
```

### What to look for in the output:
1. **Did the server start?** Check "Server is ready" message. If it crashed, Eagle3 is incompatible with this vLLM version.
2. **Tokens/sec per attempt** — compare to feb3's implicit rate (~baseline)
3. **Entropy values** — are they reasonable (1.0-5.0 range) or broken (all inf/0)?
4. **Score** — should be ~38-42 if Eagle3 doesn't hurt accuracy
5. **Total time per problem** — used to calculate actual speedup factor

### Decision after Step 1:

| Outcome | Next Step |
|---------|-----------|
| Server crashes / Eagle3 incompatible | STOP. Revert to feb3 baseline. Skip all Eagle3 work. |
| Works but logprobs broken (entropy all inf) | Go to Step 2A (majority vote fallback) |
| Works, logprobs OK, speedup < 1.2x | Go to Step 2B (submit feb5_eagle3_verified as-is) |
| Works, logprobs OK, speedup 1.2-1.5x | Go to Step 2C (scaled 10 attempts) |
| Works, logprobs OK, speedup > 1.5x | Go to Step 2D (scaled 12-16 attempts) |

---

## Step 2A: FALLBACK — Eagle3 + Majority Vote (if logprobs broken)
**Purpose**: Use Eagle3 speed without relying on broken entropy
**Action**: I will create a variant that uses simple majority vote instead of entropy-gated consensus
**Skip if**: Logprobs are working fine

---

## Step 2B: VERIFIED VARIANT — feb5_eagle3_verified (if speedup minimal)
**Purpose**: Test if feb4's self-consistency + code verification helps with Eagle3
**Notebook**: `kaggle_submissions/feb5_eagle3_verified/`
**Skip if**: Speedup is significant enough to justify scaling attempts instead

```bash
cd /home/son/GitHub/AIMO/kaggle_submissions/feb5_eagle3_verified
kaggle kernels push -p .
```

---

## Step 2C: SCALED 10 ATTEMPTS (if speedup 1.2-1.5x)
**Purpose**: Use the extra speed for more consensus votes
**Action**: I will create `feb5_eagle3_scaled/` with:
- attempts = 10, workers = 10, early_stop = 4

---

## Step 2D: SCALED 12-16 ATTEMPTS (if speedup > 1.5x)
**Purpose**: Maximize consensus quality with the speed budget
**Action**: I will create `feb5_eagle3_scaled/` with:
- attempts = 12-16, workers = 12-16, early_stop = 5-6

---

## Step 3: BEST COMBO (1 H100 slot)
**Purpose**: Submit the best-performing variant from Step 2
**Based on**: Whichever Step 2 variant scores highest

---

## Summary: Maximum 3 H100 submissions needed

| Order | Notebook | Purpose | H100 Slots |
|-------|----------|---------|------------|
| Step 0 | (upload dataset) | Pre-req | 0 |
| Step 1 | feb5_eagle3 | Diagnostic: speed + compatibility | 1 |
| Step 2 | (depends on Step 1) | Optimized variant | 1 |
| Step 3 | (best combo) | Final submission | 1 |

**Total: 3 H100 runs maximum** (could be just 1-2 if Step 1 reveals issues)
