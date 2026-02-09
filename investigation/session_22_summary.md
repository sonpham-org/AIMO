# Session 22 Summary: Multi-Model Strategy Research
> Date: Feb 8, 2026

## Objective
Can we run multiple diverse models on H100 80GB for AIMO3?

## Key Findings

### 1. Self-MoA > Mixed-Model MoA (Critical Insight)
- Recent research (Li et al., Feb 2025): Sampling multiple outputs from ONE strong model outperforms mixing different models
- Self-MoA beats mixed-model MoA by **6.6% on AlpacaEval** and **3.8% on MATH**
- **Validates our current approach**: 16 samples from gpt-oss-120b is already near-optimal
- Adding a weaker model for generation would likely HURT

### 2. Multi-LoRA is the Proven Winner (RECOMMENDED)
- NVIDIA won AIMO2 with: SFT LoRA (generation) + GenSelect LoRA (selection) on same base model
- VRAM overhead per adapter: ~100-400 MB (negligible)
- Can hold 3-5 adapters in our 15-18GB free VRAM
- Performance penalty: Up to 50% throughput drop (but we have 8.75h headroom)

**Recommended Architecture**:
```
gpt-oss-120b + multi-LoRA:
  1. SFT-LoRA generates 16-32 solutions (better solve rate)
  2. GenSelect-LoRA scores each solution (learned selection)
  3. Entropy-gated consensus (fallback)
```

### 3. vLLM Sleep Mode Enables Sequential Multi-Model
- Can swap models in **3-6s** using Sleep Mode (Level 1 CPU offload)
- Requires ~60GB CPU RAM (Kaggle H100 has 80GB)
- Per-problem overhead: ~6-10s for wake → run → sleep → wake
- 50 problems: ~5-8 min total swap time (acceptable)
- Use case: Different model architectures for different phases (generation vs verification)

### 4. Running 2 Models Simultaneously is NOT Recommended
- Current: gpt-oss-120b uses ~77GB of 80GB
- Remaining: ~3GB (not enough for another model)
- Option: Lower gpu_memory_utilization from 0.96 to 0.85 → frees ~9GB → could fit Qwen3-8B-4bit (~5GB)
- **Risk**: Reduced KV cache → slower inference, two servers competing for resources
- **Verdict**: Technically possible but operationally fragile

### 5. Strong Math Models (2025-2026)
| Model | Size | Strengths | Our Use |
|-------|------|-----------|---------|
| gpt-oss-120b | 117B (5.1B active MoE) | Best open model | Current (40/50) |
| Qwen2.5-Math-72B | 72B | Math-specialized | Research target |
| Qwen3-8B | 8B | Fast, good reasoning | Local trace gen |
| DeepSeek-R1-Distill-Qwen-32B | 32B | Reasoning focus | AIMO2 2nd place used 14B variant |

### 6. Our 40/50 Score Beats All AIMO2 Winners
- NVIDIA (1st AIMO2): 34/50 with Qwen2.5-14B + full FT + GenSelect
- Imagination (2nd AIMO2): 31/50 with DeepSeek-R1-Distill-Qwen-14B + SFT+DPO
- Aliev (3rd AIMO2): 30/50 with DeepSeek-R1-Distill-14B + zero FT
- **Our base gpt-oss-120b**: 40/50 with entropy-gated consensus, no FT

**Implication**: Model swapping for diversity is NOT the bottleneck. Focus on:
1. Better selection (GenSelect LoRA)
2. Better generation (SFT LoRA)
3. More attempts (scale to 32-48 like AIMO2 winners)

## Recommended Implementation Roadmap

**Week 1: SFT LoRA Training**
- Curate 1K-5K TIR traces (pipeline ready)
- Train SFT LoRA on rented H100 (~2-5h)
- Expected gain: +2-3 points (better per-attempt solve rate)

**Week 2: GenSelect LoRA Training**
- Download OpenMathReasoning GenSelect (566K examples)
- Train GenSelect LoRA on rented H100 (~4-8h)
- Expected gain: +3-5 points (learned selection)

**Week 3: Multi-LoRA Integration**
- Modify feb3 notebook for multi-LoRA serving
- Test on Kaggle with SFT + GenSelect
- Expected combined gain: +5-8 points → 45-48/50

**Week 4: Scale to 32-48 Attempts**
- Fix selection strategy for N>8 (current breaks at 16)
- Scale attempts to match AIMO2 winners (48 samples)
- Expected gain: +2-4 points from better coverage

## Key Decisions
- **Multi-LoRA is the path forward** (not multi-model)
- **Sleep Mode is feasible backup** (if multi-LoRA doesn't work)
- **DO NOT mix models for generation** (Self-MoA proven better)
- **DO train GenSelect LoRA** (NVIDIA's secret weapon)
- **Target 32-48 attempts** (after fixing selection)

## Output Files
- **`investigation/multi_model_research.md`** - Comprehensive 11-section research document (9,000+ words)

## Next Steps
1. Start dataset curation for SFT training (Week 1)
2. Rent H100 and train SFT LoRA
3. Test single-LoRA inference on Kaggle
4. Proceed to GenSelect training (Week 2)
5. Integrate multi-LoRA (Week 3)
