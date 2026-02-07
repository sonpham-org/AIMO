# Fine-Tuning Resources for Math LLMs

Research compiled: Feb 7, 2026

---

## 1. Tinker API (Thinking Machines Lab)

**Overview:** Released by Mira Murati's Thinking Machines Lab in October 2025, GA in December 2025.

### Key Capabilities
- Write single-processor Python scripts that auto-scale to multi-GPU
- Supports LoRA fine-tuning from supervised learning to RL
- Models: Llama 70B, Qwen 235B, gpt-oss-120b, and other open-source models
- Live sampling tool to test prompts during training

### Pricing
- **Per-token pricing** (not hourly GPU): ~$0.40 per million tokens for Qwen3-8B
- New users get **$150 in credits** when cleared from waitlist
- Currently in private beta

### Resources
- [Tinker Documentation](https://tinker-docs.thinkingmachines.ai/)
- [Tinker Cookbook (GitHub)](https://github.com/thinking-machines-lab/tinker-cookbook)
- [DataCamp Tutorial](https://www.datacamp.com/tutorial/tinker-tutorial)

---

## 2. Unsloth for Efficient LoRA Training

**Overview:** 2x faster training with 70% less VRAM. Most widely used open-source framework.

### Supported Models
- Qwen3 (all sizes, 128K context)
- DeepSeek-R1 and distilled variants
- Llama, Gemma, any transformers-compatible model

### Key Features
- QLoRA (4-bit) and LoRA (16-bit) training
- GRPO support for R1-style reasoning models
- Dynamic 2.0 format with long context

### Recommended Hyperparameters
```python
# Starting point for math fine-tuning
learning_rate = 2e-4
epochs = 1-3  # More risks overfitting
lora_r = 16-64  # Rank, higher = more capacity
lora_alpha = 32  # Scaling factor
```

### Resources
- [Unsloth Documentation](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)
- [GitHub](https://github.com/unslothai/unsloth)
- [Train Your Own R1 (GRPO)](https://unsloth.ai/blog/r1-reasoning)

---

## 3. Training Methods for Math Reasoning

### GRPO (Group Relative Policy Optimization)
The algorithm behind DeepSeek-R1's success.
- Variant of PPO without critic model (saves memory)
- Best for training reasoning capabilities
- [Illustrated breakdown](https://epichka.com/blog/2025/grpo/)

### DPO (Direct Preference Optimization)
- ~50% less compute than PPO-RLHF
- Single-stage training, more stable
- Good for alignment after SFT

### Spectrum (Layer Selection)
- Uses SNR analysis to identify most informative layers
- Trains only top ~30% of layers
- Higher accuracy than QLoRA on math with similar resources

---

## 4. Key Datasets for Math Fine-Tuning

### NVIDIA OpenMathReasoning
- 306K unique problems from AoPS forums
- Solutions from DeepSeek-R1 and Qwen2.5-72B-Math
- Splits: `cot`, `tir`, `genselect`
- **Won AIMO-2 competition**
- [HuggingFace Dataset](https://huggingface.co/datasets/nvidia/OpenMathReasoning)

### AIMO3 Tool-Integrated Reasoning (Kaggle)
- 141K samples with Python tool execution
- GPT-OSS-120b generated
- `kaggle datasets download jeannkouagou/aimo3-tool-integrated-reasoning`

### AIMO3 High-Difficulty (Thinking Machines Lab)
- 70K trajectories from 7,293 hard problems
- Eagle3 trained on this data
- `kaggle datasets download wenliangtlh/aimo3-high-difficulty-tool-calling-dataset`

---

## 5. Best Practices

### Data Selection
- **Quality > Quantity**: ~1,000 high-quality examples often sufficient
- Use skill-based data selection with hierarchical skill trees
- Filter by pass rate: harder problems (1-3 correct out of 8) are more valuable

### Curriculum Learning
- **Self-Evolving Curriculum (SEC)**: Learn curriculum policy with RL
- **Prompt Curriculum Learning**: Select ~50% success rate prompts for efficiency
- Order by complexity: start easy, progress to hard

### Training Tips
1. Start with QLoRA for memory efficiency
2. Use 2e-4 learning rate as baseline
3. Limit to 1-3 epochs to avoid overfitting
4. Include tool-use examples for TIR capability
5. Verify with external validator (best-of-N sampling)

---

## 6. Recommended Pipeline for AIMO

### Phase 1: Smoke Test (~$1-5)
```bash
# Use Tinker with small dataset
tinker train --model qwen3-8b --data small_sample.jsonl --epochs 1
```

### Phase 2: Full TIR Training (~$50-100)
- Dataset: `aimo3-tool-integrated-reasoning` (141K samples)
- Model: Qwen3-8B or Qwen3-32B
- Method: SFT with Harmony format conversion

### Phase 3: RL Fine-Tuning (~$100-200)
- Method: GRPO on hard problems (pass rate 1-3/8)
- Reward: Correct answer = +1, wrong = -1
- Use curriculum learning for efficiency

### Phase 4: Deploy
- Quantize with MXFP4 or AWQ
- Use Eagle3 for speculative decoding (36% speedup)

---

## 7. Recent Papers (2025-2026)

| Paper | Key Contribution |
|-------|------------------|
| [Self-Evolving Curriculum](https://arxiv.org/abs/2505.14970) | Auto curriculum for RL |
| [GRPO-LEAD](https://arxiv.org/html/2504.09696v2) | Difficulty-aware RL |
| [Training-Free GRPO](https://arxiv.org/abs/2510.08191) | No-train enhancement |
| [AIMO-2 Solution](https://arxiv.org/pdf/2504.16891) | Competition winner |

---

## Key Insights

1. **GRPO is the go-to RL algorithm** for math reasoning (DeepSeek-R1 recipe)
2. **OpenMathReasoning dataset** won AIMO-2 - directly applicable
3. **Tinker per-token pricing** makes distributed training affordable
4. **Unsloth enables local GRPO** on consumer hardware
5. **Curriculum learning** (50% difficulty) maximizes sample efficiency
6. **Inference-time scaling** is 2026 trend - invest compute at generation
