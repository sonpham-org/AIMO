# Feb 5 — Eagle3 + Entropy-Gated Consensus

## Score: PENDING (based on feb3 which scored 40/50)

## What's New
- **Eagle3 speculative decoding** added to feb3 (40/50) baseline
- Small draft model (~588MB) proposes 3 tokens ahead, base model verifies in parallel
- Expected ~1.3-2x inference speedup with zero accuracy change
- Added timing metrics (tokens/sec, time per problem) to measure speedup

## Setup Required: Upload Eagle3 Model to Kaggle

The Eagle3 draft model is NOT available on Kaggle by default. You need to upload it:

### Option A: wenliang1990 AIMO3-specific model (recommended, smaller)
```bash
# Download from HuggingFace (~588MB)
pip install huggingface_hub
huggingface-cli download wenliang1990/gpt-oss-120b-eagle3-aimo3 --local-dir eagle3-gpt-oss-120b-aimo3

# Upload to Kaggle as dataset
kaggle datasets init -p eagle3-gpt-oss-120b-aimo3
# Edit dataset-metadata.json: set slug to "sonphamorg/eagle3-gpt-oss-120b-aimo3"
kaggle datasets create -p eagle3-gpt-oss-120b-aimo3
```

### Option B: NVIDIA v2 model (vLLM-recommended, larger)
```bash
# Download from HuggingFace (~1.85GB)
huggingface-cli download nvidia/gpt-oss-120b-Eagle3-v2 --local-dir eagle3-gpt-oss-120b-v2

# Upload to Kaggle
kaggle datasets init -p eagle3-gpt-oss-120b-v2
kaggle datasets create -p eagle3-gpt-oss-120b-v2
```

If using Option B, update `eagle3_model_path` in the notebook CFG class.

## Known Risks
1. **Logprobs + speculative decoding**: vLLM may have unstable logprobs with Eagle3, which could affect entropy-gated consensus
2. **vLLM compatibility**: Needs vLLM >= 0.11.1 for Eagle3 support (check aimo-3-utils wheel version)
3. **VRAM**: Eagle3 adds ~600MB VRAM overhead, should fit with gpu_memory_utilization=0.96

## Parameter Scaling (after measuring speedup)
Once we know the actual speedup factor on H100, we can increase:
- `attempts`: 8 → 10-12 (more votes for consensus)
- `turns`: 128 → 160 (deeper reasoning chains)
- `workers`: 8 → 10-12 (more parallel kernels)
- `early_stop`: 4 → 5 (higher consensus bar)
