# MathCorpus Prize: Submission Format & Process

> Last updated: 2026-02-07
> Author: resource-finder agent
> Purpose: How to submit a dataset for the AIMO3 MathCorpus Prize

---

## 1. What We Know

### 1.1 MathCorpus Prize Overview

- **Part of**: AIMO3 Extra Prizes (EPs), total pool $110,000 shared across 4 prizes
- **Purpose**: "Publishing novel datasets that will help the wider community"
- **4 Extra Prize categories**: Longest Leader, Write-up, MathCorpus, Hardest Problem
- **Requirement**: All entrants must make code and datasets publicly available

### 1.2 What's NOT Publicly Documented

The detailed MathCorpus Prize rules (specific prize amount, judging rubric, exact submission mechanism, format requirements) are on the Kaggle competition rules page, which is JavaScript-rendered and not accessible via web scraping. The AIMO Prize website and CompeteHub summaries only provide high-level descriptions.

**IMPORTANT**: The team lead says the deadline is Feb 9, 2026 -- this could not be independently verified via web search. The main competition entry deadline is April 8, 2026. The Feb 9 date may be an interim deadline specific to the MathCorpus Prize or an Early Dataset Sharing deadline. **The user should verify this directly on the Kaggle competition page.**

---

## 2. Submission Mechanism: Kaggle Dataset (Most Likely)

Based on how the existing AIMO3 community datasets were published, and the requirement that datasets must be "publicly available," the submission mechanism is almost certainly a **Kaggle Dataset**.

### 2.1 How Existing AIMO3 Datasets Were Published

| Dataset | Format | Files |
|---------|--------|-------|
| **AIMO3 TIR** (`jeannkouagou/aimo3-tool-integrated-reasoning`) | CSV | `data.csv`, `solver.py`, `convert_harmony_format.py`, `convert_traces_to_csv.py`, `DATASET_DESCRIPTION.md` |
| **AIMO3 Hard** (`wenliangtlh/aimo3-high-difficulty-tool-calling-dataset`) | JSONL | `AIMO3-High-Difficulty-Tool-Calling-Dataset.jsonl`, `benchmark.py`, `convert_harmony_to_messages.py`, `extract_gpt_oss_120b_response.py` |
| **AIMO External** (`alejopaullier/aimo-external-dataset`) | Various | Smaller, curated |

**Pattern**: Both major AIMO3 datasets include:
1. Main data file (CSV or JSONL)
2. Conversion/utility scripts
3. Description document
4. Benchmark/evaluation scripts

### 2.2 How to Create a Kaggle Dataset

```bash
# 1. Initialize metadata
mkdir my-dataset && cd my-dataset
kaggle datasets init

# 2. Edit dataset-metadata.json (see template below)

# 3. Add data files + README

# 4. Create the dataset
export $(grep KAGGLE_API_TOKEN /home/son/GitHub/AIMO/.env)
kaggle datasets create -p ./my-dataset

# 5. Make it public (required for MathCorpus Prize)
# Set "isPrivate": false in metadata, or update via web UI
```

### 2.3 Dataset Metadata Template

```json
{
  "title": "AIMO3 Curated Math Reasoning Dataset",
  "id": "sonphamorg/aimo3-curated-math-reasoning",
  "subtitle": "Difficulty-calibrated, quality-scored TIR traces for math olympiad fine-tuning",
  "description": "A curated dataset of high-quality Tool-Integrated Reasoning traces for fine-tuning math reasoning models, built from AIMO3 competition data with systematic quality filtering.",
  "licenses": [{"name": "apache-2.0"}],
  "keywords": ["mathematics", "reasoning", "fine-tuning", "olympiad", "tool-integrated-reasoning"],
  "resources": [
    {
      "path": "curated_traces.jsonl",
      "description": "Main dataset: curated TIR traces with quality metadata"
    },
    {
      "path": "selection_metadata.csv",
      "description": "Per-problem selection metadata (difficulty, topic, quality scores)"
    },
    {
      "path": "README.md",
      "description": "Dataset documentation and methodology"
    }
  ]
}
```

**Validation rules:**
- `title`: 6-50 characters
- `id`: `username/slug`, slug 3-50 characters
- `licenses`: Exactly one entry. Use `apache-2.0` (matches source datasets)
- `subtitle`: 20-80 characters (optional but recommended)

---

## 3. Recommended File Format

### 3.1 Primary Data: JSONL (Preferred)

JSONL is preferred over CSV because:
- Handles nested fields cleanly (tool calls, multi-turn conversations)
- No escaping issues with LaTeX in math problems
- Can include structured metadata per record
- Matches AIMO3 Hard format

**Schema per record:**

```json
{
  "problem_id": "unique_identifier",
  "problem": "Full problem text in LaTeX",
  "solution": "Full TIR solution trace",
  "answer": 42,
  "source": "aimo3_hard",
  "pass_rate": 0.25,
  "topic": "number_theory",
  "difficulty_tier": "hard",
  "quality_score": 0.82,
  "n_tool_calls": 6,
  "solution_length": 15234,
  "has_verification": true,
  "selection_method": "entropy_quality_diversity"
}
```

### 3.2 Supplementary: Selection Metadata (CSV)

A separate CSV showing the full selection pipeline statistics:

```csv
problem_id,source,pass_rate,topic,quality_score,n_tool_calls,solution_length,selected,selection_reason
prob_001,aimo3_hard,0.25,number_theory,0.82,6,15234,true,high_quality_hard
prob_002,aimo3_hard,0.875,algebra,0.45,2,3201,false,too_easy
```

### 3.3 File Size Considerations

- Kaggle API limit: ~2GB per file
- Web upload limit: 500MB
- Our estimated dataset size: 2K-4K JSONL records at ~20K chars each = ~40-80MB (well within limits)
- Use Kaggle API for upload (not web UI) for reliability

---

## 4. Required Documentation (README.md)

Based on the pattern from existing AIMO3 datasets and general Kaggle best practices:

### 4.1 Sections to Include

```markdown
# [Dataset Title]

## Summary
- What: Curated dataset of N high-quality TIR traces for math olympiad fine-tuning
- Why: Quality > quantity (LIMO: 817 examples > 100K random)
- How: 6-stage curation pipeline with difficulty calibration and diversity selection

## Dataset Statistics
- Total examples: N
- Problems from: [sources]
- Difficulty distribution: [table]
- Topic distribution: [table]
- Average solution length: X tokens
- Average tool calls: Y

## Curation Methodology
1. Hard filtering (length, tool calls, answer format)
2. Correctness verification
3. Difficulty calibration (pass_rate filtering)
4. Quality scoring (LIMO + ASTER methodology)
5. Topic classification and diversity selection
6. Decontamination (AIME 2023-2025, MATH500)

## Format
- Main file: `curated_traces.jsonl` — JSONL with fields [list]
- Metadata: `selection_metadata.csv` — per-problem curation statistics

## Usage
```python
import json
with open('curated_traces.jsonl') as f:
    data = [json.loads(line) for line in f]
```

## License
Apache 2.0

## Citation
[bibtex]

## Acknowledgments
Built from [source datasets] using methodology from [papers]
```

### 4.2 What Makes a Winning Dataset Card

Looking at the AIMO3 TIR dataset description as a gold standard:
1. **Novelty section** -- what's new about this dataset vs existing ones
2. **Performance claims** -- how training on this data improves scores
3. **Technical details** -- generation process, statistics, format
4. **Reproducibility** -- include scripts to reproduce the curation
5. **Conversion utilities** -- scripts to convert to different formats

---

## 5. Which Category Best Fits Our Approach

### 5.1 The 4 Extra Prize Categories

| Category | What It Rewards | Fits Us? |
|----------|----------------|----------|
| **Longest Leader Prize** | Staying on top of leaderboard longest | No (competition model) |
| **Write-up Prizes** | Best technical explanation | Maybe (separate from dataset) |
| **MathCorpus Prize** | Novel datasets for the community | **YES -- primary target** |
| **Hardest Problem Prize** | Solving the hardest single problem | No (competition model) |

### 5.2 What Makes Our Dataset "Novel"

Existing AIMO3 datasets on Kaggle are **raw dumps** -- all traces without curation. Our value proposition:

1. **Difficulty-calibrated**: Filtered by pass_rate metadata (AIMO3 Hard has this, but it's not filtered in the raw data)
2. **Quality-scored**: Each trace scored using published methodology (LIMO + ASTER)
3. **Topic-balanced**: Diversity selection across math domains
4. **Decontaminated**: Against evaluation benchmarks
5. **Ready-to-train**: Clean format with metadata, not requiring preprocessing
6. **Selection transparency**: Full metadata showing WHY each trace was selected
7. **Includes DPO pairs**: Best vs. worst solution per problem for preference training
8. **Compact**: 2K-4K curated traces that outperform raw 141K (if validated)

### 5.3 Positioning

Frame it as: **"The LIMO of TIR"** -- proving that a small, carefully curated TIR dataset outperforms massive raw collections, following the methodology of LIMO (817 >> 100K) and s1K (1K beat o1-preview).

---

## 6. Submission Checklist (2-Day Sprint)

### Day 1 (Feb 7-8): Build the dataset

- [ ] Run curation pipeline on AIMO3 Hard (7.3K problems, ~70K traces)
- [ ] Apply hard filters (length, tool calls, answer format, no repetition)
- [ ] Filter by pass_rate (keep 1/8 to 3/8 for SFT sweet spot)
- [ ] Score quality (LIMO heuristics + interaction density)
- [ ] Classify topics (keyword-based or Gemini Batch API)
- [ ] Decontaminate (9-gram against AIME 2023-2025)
- [ ] Select final 2K-4K examples with diversity weighting
- [ ] Generate DPO pairs (best vs worst solution per problem)
- [ ] Export as JSONL + metadata CSV

### Day 2 (Feb 8-9): Package and submit

- [ ] Write comprehensive README.md / dataset card
- [ ] Compute and include statistics (distributions, plots)
- [ ] Include curation scripts for reproducibility
- [ ] Include format conversion utilities
- [ ] Create dataset-metadata.json
- [ ] Upload to Kaggle: `kaggle datasets create -p ./dataset_dir`
- [ ] Make public
- [ ] Verify dataset loads correctly
- [ ] Submit/link for MathCorpus Prize (check exact mechanism on Kaggle)

### Pre-submission verification
- [ ] All files load without errors
- [ ] README is clear and complete
- [ ] License is Apache 2.0 (matching sources)
- [ ] No test set contamination
- [ ] Dataset is publicly accessible

---

## 7. Upload Commands

```bash
# Set credentials
export $(grep KAGGLE_API_TOKEN /home/son/GitHub/AIMO/.env)

# Initialize dataset directory
mkdir -p /home/son/GitHub/AIMO/dataset_generation/kaggle_dataset
cd /home/son/GitHub/AIMO/dataset_generation/kaggle_dataset
kaggle datasets init

# After adding files and editing metadata:
kaggle datasets create -p /home/son/GitHub/AIMO/dataset_generation/kaggle_dataset

# To update an existing dataset:
kaggle datasets version -p /home/son/GitHub/AIMO/dataset_generation/kaggle_dataset -m "Update description"
```
