# Fine-Tuning Datasets for AIMO3

Research compiled: Feb 6, 2026

## Top Recommended Datasets for LoRA Fine-Tuning

### 1. AIMO3 Tool-Integrated Reasoning Dataset ⭐ BEST FOR TIR
**Kaggle:** `jeannkouagou/aimo3-tool-integrated-reasoning`
- **Size:** 141,277 samples (902MB)
- **Model:** GPT-OSS-120b generated
- **Format:** Harmony protocol (with conversion scripts)
- **License:** Apache-2.0
- **Key Features:**
  - Real Python execution traces (not synthetic)
  - Solution hint methodology ensures correctness
  - Average completion: 21,825 characters
  - Includes `convert_harmony_format.py` for ChatML/Anthropic/Llama conversion
- **Best for:** TIR fine-tuning with tool use patterns

### 2. AIMO3 High-Difficulty Tool-Calling Dataset ⭐ FROM THINKING MACHINES LAB
**Kaggle:** `wenliangtlh/aimo3-high-difficulty-tool-calling-dataset`
- **Size:** ~70,000 trajectories from 7,293 problems
- **Model:** GPT-OSS-120b (8 samples per problem)
- **Format:** Harmony JSONL (direct training ready)
- **License:** Apache-2.0
- **Key Features:**
  - High difficulty focus (pass rate ≤7/8)
  - This is what Eagle3 was trained on!
  - Reported IMO improvement: 50% → 60%
  - Includes `convert_harmony_to_messages.py`
- **Best for:** Eagle3 draft model training, high-difficulty math

### 3. OpenR1-Math-220k
**Kaggle:** `alejopaullier/openr1-math-220k`
- **Size:** 220,000 samples (1.2GB)
- **Votes:** 15
- **Format:** Standard chat format
- **Best for:** General math reasoning, large-scale SFT

### 4. NuminaMath-TIR
**Kaggle:** `jorgeplazas/numinamath-tir`
- **Size:** ~70,000 samples (196MB)
- **License:** Apache-2.0
- **Key Features:**
  - Tool-integrated reasoning from NuminaMath
  - Easier difficulty than AIMO3 datasets
- **Best for:** Warm-up/pre-training before AIMO3-specific data

### 5. AIMO External Dataset
**Kaggle:** `alejopaullier/aimo-external-dataset`
- **Size:** 4.5MB (smaller, curated)
- **Votes:** 78 (most popular)
- **Best for:** Quick experiments, baseline

---

## Dataset Comparison

| Dataset | Samples | Difficulty | Tool Use | Model Source | Format |
|---------|---------|------------|----------|--------------|--------|
| aimo3-tool-integrated-reasoning | 141,277 | Olympiad | ✅ Full | GPT-OSS-120b | Harmony CSV |
| aimo3-high-difficulty-tool-calling | ~70,000 | High (AIME+) | ✅ Full | GPT-OSS-120b | Harmony JSONL |
| openr1-math-220k | 220,000 | Mixed | ❌ | Various | Chat |
| numinamath-tir | ~70,000 | Medium | ✅ | NuminaMath | Standard |

---

## Recommended Fine-Tuning Strategy

### Phase 1: Quick Validation ($1-5)
- Use AIMO External Dataset (small, curated)
- Validate pipeline works end-to-end

### Phase 2: TIR Training (~$50-100)
- Primary: `aimo3-tool-integrated-reasoning` (141k samples)
- Or: `aimo3-high-difficulty-tool-calling` (70k samples)
- Target: Qwen3-8B or Qwen3-32B

### Phase 3: Full Model (~$200-500)
- Fine-tune gpt-oss-120b with high-difficulty data
- Use curriculum: start with easier, progress to hard

---

## Download Commands

```bash
# Top TIR dataset (141k samples)
kaggle datasets download jeannkouagou/aimo3-tool-integrated-reasoning

# High-difficulty (Eagle3 training data)
kaggle datasets download wenliangtlh/aimo3-high-difficulty-tool-calling-dataset

# OpenR1 for general math
kaggle datasets download alejopaullier/openr1-math-220k

# NuminaMath-TIR (medium difficulty)
kaggle datasets download jorgeplazas/numinamath-tir
```

---

## Key Insight from Dataset #2 (Thinking Machines Lab)

Their Eagle3 model training showed:
- **IMO accuracy: 50% → 60%** after training on this dataset
- **Inference speedup: 36-42%** with speculative decoding

This validates that fine-tuning on high-quality TIR traces improves both accuracy AND efficiency.

---

## Format Conversion

Both top datasets include conversion scripts:

```python
# For jeannkouagou dataset
from convert_harmony_format import convert_dataset_to_format
converted_df = convert_dataset_to_format('data.csv', output_format='chatml')

# For wenliangtlh dataset
# Use convert_harmony_to_messages.py
```

---

## Notes

- All datasets Apache-2.0 licensed
- Harmony format is gpt-oss-120b native format
- For non-Harmony models (Qwen, Llama), use conversion scripts
- Solution hints in training data ensure correctness (hints excluded from final traces)
