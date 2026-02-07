# AIMO3 Fine-Tuning Plan

## Status: Saved for later — focusing on Eagle3 inference first

---

## What Worked in AIMO2

### 1st Place: NVIDIA NemoSkills (34/50 private)
- **Base model:** Qwen2.5-14B-base (NOT instruct)
- **Full fine-tuning** (not LoRA)
- **Two-stage training:**
  - Stage 1 (CoT SFT): 8 epochs on 2.2M CoT solutions from DeepSeek-R1
  - Stage 2 (TIR SFT): 400 steps on 15K TIR samples
- **Checkpoint merging:** Linear interpolation of CoT and TIR checkpoints
- **Checkpoint averaging:** 4 equally-spaced checkpoints averaged
- **Dataset:** nvidia/OpenMathReasoning (540K problems, 5.5M solutions)
- **Hardware:** 8x H100 GPUs

### 2nd Place: imagination-research (31/50 private)
- **Base model:** DeepSeek-R1-Distill-Qwen-14B
- **Stage 1 (SFT):** Light-R1 + LIMO data, 8 epochs, 11h on 8x A800
- **Stage 2 (DPO):** 2K DPO pairs from OpenR1-Math-220k, 2-4 epochs, 40h on 8x A800
- **Key insight:** DPO reduced output length while maintaining quality

### s1 Paper (beat o1-preview on MATH/AIME)
- **Base model:** Qwen2.5-32B-Instruct
- **Just 1,000 curated examples** from 59K candidates
- **26 minutes training** on 16 H100s
- Budget forcing at test time: append "Wait" tokens for longer reasoning

---

## Recommended Datasets

| Dataset | Size | Best For | License |
|---------|------|----------|---------|
| nvidia/OpenMathReasoning | 540K problems, 5.5M solutions | Primary training (won AIMO2) | CC-BY-4.0 |
| AI-MO/NuminaMath-TIR | 70K TIR traces | TIR format training | Apache 2.0 |
| AI-MO/NuminaMath-1.5 | 896K problems | Broad math coverage | Apache 2.0 |
| open-r1/OpenR1-Math-220k | 220K multi-trace | DPO pairs | Apache 2.0 |
| hendrycks/competition_math | 12.5K AMC/AIME | Olympiad-level | MIT |

---

## Fine-Tuning gpt-oss-120b on Kaggle

### Hardware: H100 (80GB)
- QLoRA 4-bit with Unsloth: **65GB VRAM** (fits!)
- Config: `r=16, lora_alpha=32, targets=q/k/v/o/gate/up/down_proj`
- Training: `lr=2e-4, adamw_8bit, batch_size=4`

### Time Budget (within 9h Kaggle runtime)
| Examples | Est. Time | Feasible? |
|----------|-----------|-----------|
| 100 | ~10-20 min | Yes |
| 1,000 (3 epochs) | ~30-60 min | Yes |
| 5,000 (1 epoch) | ~3-5 hours | Yes, tight |
| 10,000+ | ~20+ hours | No — train offline |

### Recommended Approach
1. **Train offline** on rented H100 or own hardware
2. **Upload LoRA adapter** as Kaggle Dataset (~100-500MB)
3. **Load at inference** — full 9 hours for inference
4. **Combine with entropy-weighted consensus** — fine-tuning and selection are independent

---

## Data Quality Strategies

### Filtering Metrics
1. **Correctness verification** — final answer matches ground truth
2. **Difficulty matching** — focus on AIME/olympiad level, not GSM8K
3. **DART-Math difficulty-aware sampling** — more budget to harder problems
4. **Process Reward Models (PRM)** — score intermediate steps, not just final answer
5. **Importance-weighted SFT** — weight samples by reward model score
6. **Adaptive curriculum (AdaRFT)** — dynamically adjust difficulty to model skill

### Synthetic Data Generation
- Use gpt-oss-120b itself to generate TIR traces on curated problems
- Filter for correctness with Math Verify library
- This gives traces in exactly the format our inference pipeline uses
- Quality > Quantity: 1K curated > 100K random (s1 paper)

### Contamination Prevention
- Exclude AIME 2023-2025 from training
- 13-gram matching with text normalization
- Longest common subsequence ratio > 0.6

---

## GenSelect Training (NVIDIA's secret weapon)
- Train the model to evaluate multiple candidate solutions and pick the best
- Third pillar of NVIDIA's approach alongside CoT and TIR
- 566K GenSelect samples in OpenMathReasoning
- Essentially bakes answer selection INTO the model

---

## TODO When Starting Fine-Tuning
1. Download nvidia/OpenMathReasoning TIR subset
2. Filter to olympiad-level difficulty (AIME-hard and above)
3. Verify correctness with Math Verify
4. Curate 1K-5K highest quality examples
5. Set up Unsloth QLoRA training script
6. Train offline → save LoRA adapter
7. Upload as Kaggle Dataset
8. Test combined fine-tuned model + entropy consensus
