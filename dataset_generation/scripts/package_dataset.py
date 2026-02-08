#!/usr/bin/env python3
"""
Package Dataset for MathCorpus Prize / Kaggle Submission

Takes curated JSONL from quality_filter.py and packages it into a
submission-ready format with:
  - Properly formatted data files (JSONL, Parquet, CSV)
  - Dataset card / metadata (README.md, dataset_info.json)
  - Statistics summary
  - Train/val split
  - Kaggle dataset metadata (dataset-metadata.json)

Usage:
  # Package curated SFT dataset
  python dataset_generation/scripts/package_dataset.py \
      --input dataset_generation/output/curated_sft_3000.jsonl \
      --name "aimo3-curated-sft" \
      --output-dir dataset_generation/output/packaged_sft

  # Package with DPO pairs included
  python dataset_generation/scripts/package_dataset.py \
      --input dataset_generation/output/curated_sft_3000.jsonl \
      --dpo-input dataset_generation/output/dpo_pairs.jsonl \
      --name "aimo3-curated-complete" \
      --output-dir dataset_generation/output/packaged_complete

  # Package for Kaggle upload
  python dataset_generation/scripts/package_dataset.py \
      --input dataset_generation/output/curated_sft_3000.jsonl \
      --name "aimo3-curated-sft" \
      --output-dir dataset_generation/output/packaged_sft \
      --kaggle
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path):
    """Load JSONL file into list of dicts."""
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------

def split_data(samples, val_ratio=0.05, seed=42):
    """Split samples into train/val with stratification by topic."""
    rng = random.Random(seed)

    by_topic = defaultdict(list)
    for s in samples:
        topic = s.get("topic", "other")
        by_topic[topic].append(s)

    train = []
    val = []

    for topic, topic_samples in by_topic.items():
        rng.shuffle(topic_samples)
        n_val = max(1, int(len(topic_samples) * val_ratio))
        val.extend(topic_samples[:n_val])
        train.extend(topic_samples[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    return train, val


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(samples):
    """Compute dataset statistics."""
    n = len(samples)
    if n == 0:
        return {"count": 0}

    sol_lengths = [len(s.get("solution_text", "")) for s in samples]
    prob_lengths = [len(s.get("problem_text", "")) for s in samples]
    tool_calls = [s.get("n_tool_calls", 0) for s in samples]
    pass_rates = [s["pass_rate"] for s in samples if s.get("pass_rate") is not None]
    scores = [s.get("combined_score", 0) for s in samples]

    topics = Counter(s.get("topic", "other") for s in samples)
    sources = Counter(s.get("source", "unknown") for s in samples)

    sorted_lengths = sorted(sol_lengths)
    sorted_scores = sorted(scores)

    return {
        "count": n,
        "sources": dict(sources.most_common()),
        "topics": dict(topics.most_common()),
        "solution_length": {
            "min": min(sol_lengths),
            "max": max(sol_lengths),
            "mean": round(sum(sol_lengths) / n),
            "median": sorted_lengths[n // 2],
            "p95": sorted_lengths[int(n * 0.95)],
        },
        "problem_length": {
            "mean": round(sum(prob_lengths) / n),
        },
        "tool_calls": {
            "mean": round(sum(tool_calls) / n, 1),
            "with_tools": sum(1 for t in tool_calls if t > 0),
            "without_tools": sum(1 for t in tool_calls if t == 0),
        },
        "quality_score": {
            "min": round(min(scores), 4) if scores else 0,
            "max": round(max(scores), 4) if scores else 0,
            "mean": round(sum(scores) / n, 4) if scores else 0,
            "median": round(sorted_scores[n // 2], 4) if scores else 0,
        },
        "pass_rate": {
            "available": len(pass_rates),
            "mean": round(sum(pass_rates) / len(pass_rates), 3) if pass_rates else None,
        },
    }


# ---------------------------------------------------------------------------
# Dataset card generation
# ---------------------------------------------------------------------------

def generate_dataset_card(name, stats, train_stats, val_stats, has_dpo=False):
    """Generate a README.md dataset card."""
    card = f"""# {name}

## Overview

Curated mathematics training dataset for the AIMO Progress Prize 3 competition.
Generated by a 6-stage quality filtering pipeline from multiple source datasets.

**Created:** {datetime.now().strftime('%Y-%m-%d')}
**License:** Apache-2.0

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total samples | {stats['count']:,} |
| Training samples | {train_stats['count']:,} |
| Validation samples | {val_stats['count']:,} |
| Avg solution length | {stats['solution_length']['mean']:,} chars |
| Median solution length | {stats['solution_length']['median']:,} chars |
| Avg tool calls | {stats['tool_calls']['mean']:.1f} |
| With tool use | {stats['tool_calls']['with_tools']:,} ({100*stats['tool_calls']['with_tools']/stats['count']:.1f}%) |
| Mean quality score | {stats['quality_score']['mean']:.4f} |

## Source Distribution

| Source | Count | Percentage |
|--------|-------|------------|
"""
    for source, count in stats["sources"].items():
        pct = 100 * count / stats["count"]
        card += f"| {source} | {count:,} | {pct:.1f}% |\n"

    card += """
## Topic Distribution

| Topic | Count | Percentage |
|-------|-------|------------|
"""
    for topic, count in stats["topics"].items():
        pct = 100 * count / stats["count"]
        card += f"| {topic} | {count:,} | {pct:.1f}% |\n"

    card += f"""
## Quality Pipeline

This dataset was produced by a 6-stage filtering pipeline:

1. **Hard Filters**: Remove too-short (<500 chars), too-long (>100K chars), repetitive solutions, missing answer markers
2. **Decontamination**: 9-gram overlap check against AIME 2023-2025 evaluation problems
3. **Difficulty Calibration**: Keep pass_rate 0.03-0.30 (SFT sweet spot, per LIMO/DART-Math research)
4. **Quality Scoring**: LIMO-style scoring (verification keywords, exploration, logical connectives) + ASTER interaction density
5. **Diversity Selection**: Topic-balanced sampling with minimum per-topic representation
6. **Importance Weighting**: Prop2Diff-style weights for iw-SFT training

## Files

- `train.jsonl` - Training split ({train_stats['count']:,} samples)
- `val.jsonl` - Validation split ({val_stats['count']:,} samples)
- `all.jsonl` - Complete dataset ({stats['count']:,} samples)
- `stats.json` - Detailed statistics
"""

    if has_dpo:
        card += "- `dpo_pairs.jsonl` - DPO preference pairs for RLHF/DPO training\n"

    card += """
## Fields

Each sample contains:
- `uid`: Unique identifier
- `source`: Source dataset (aimo3_hard, aimo3_tir, limo, s1k)
- `problem_text`: Math problem statement
- `solution_text`: Complete solution with reasoning and tool calls
- `answer`: Ground truth answer (where available)
- `pass_rate`: Difficulty estimate (fraction of correct solutions)
- `n_tool_calls`: Number of Python code execution blocks
- `topic`: Classified math topic (number_theory, algebra, geometry, combinatorics, other)
- `combined_score`: Overall quality score (0-1)
- `difficulty_score`, `quality_score`, `interaction_score`, `structure_score`: Component scores

## Citation

If you use this dataset, please cite:
```
@dataset{aimo3_curated,
  title={AIMO3 Curated Mathematics Training Dataset},
  year={2026},
  url={https://kaggle.com/datasets/sonphamorg/""" + name.replace(" ", "-").lower() + """}
}
```
"""
    return card


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

def package_dataset(args):
    """Package curated dataset into submission-ready format."""
    print(f"Loading input from {args.input}...")
    samples = load_jsonl(args.input)
    print(f"  Loaded {len(samples):,} samples")

    if not samples:
        print("Error: no samples loaded")
        sys.exit(1)

    # Split
    print(f"  Splitting (val_ratio={args.val_ratio})...")
    train, val = split_data(samples, val_ratio=args.val_ratio)
    print(f"  Train: {len(train):,}  Val: {len(val):,}")

    # Stats
    all_stats = compute_stats(samples)
    train_stats = compute_stats(train)
    val_stats = compute_stats(val)

    # Create output directory
    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Write data files
    for name, data in [("all.jsonl", samples), ("train.jsonl", train), ("val.jsonl", val)]:
        path = out_dir / name
        with open(path, "w") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Wrote {path} ({len(data):,} samples, {size_mb:.1f} MB)")

    # Write Parquet if pandas available
    try:
        import pandas as pd
        for name, data in [("train.parquet", train), ("val.parquet", val)]:
            df = pd.DataFrame(data)
            path = out_dir / name
            df.to_parquet(path, index=False)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  Wrote {path} ({size_mb:.1f} MB)")
    except ImportError:
        print("  Warning: pandas not available, skipping Parquet output")

    # DPO pairs
    has_dpo = False
    if args.dpo_input and os.path.exists(args.dpo_input):
        dpo_data = load_jsonl(args.dpo_input)
        dpo_path = out_dir / "dpo_pairs.jsonl"
        with open(dpo_path, "w") as f:
            for d in dpo_data:
                f.write(json.dumps(d) + "\n")
        size_mb = os.path.getsize(dpo_path) / (1024 * 1024)
        print(f"  Wrote {dpo_path} ({len(dpo_data):,} pairs, {size_mb:.1f} MB)")
        has_dpo = True

    # Stats JSON
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "name": args.name,
            "created": datetime.now().isoformat(),
            "all": all_stats,
            "train": train_stats,
            "val": val_stats,
        }, f, indent=2)
    print(f"  Wrote {stats_path}")

    # Dataset card
    card = generate_dataset_card(args.name, all_stats, train_stats, val_stats, has_dpo)
    card_path = out_dir / "README.md"
    with open(card_path, "w") as f:
        f.write(card)
    print(f"  Wrote {card_path}")

    # Kaggle metadata
    if args.kaggle:
        slug = args.name.lower().replace(" ", "-").replace("_", "-")
        kaggle_meta = {
            "title": args.name,
            "id": f"sonphamorg/{slug}",
            "licenses": [{"name": "Apache 2.0"}],
        }
        meta_path = out_dir / "dataset-metadata.json"
        with open(meta_path, "w") as f:
            json.dump(kaggle_meta, f, indent=2)
        print(f"  Wrote {meta_path}")
        print(f"\n  To upload to Kaggle:")
        print(f"    export $(grep KAGGLE_API_TOKEN /home/son/GitHub/AIMO/.env) && "
              f"kaggle datasets create -p {out_dir}")

    # Summary
    print(f"\n{'='*60}")
    print(f"PACKAGED: {args.name}")
    print(f"{'='*60}")
    print(f"  Total: {all_stats['count']:,} samples")
    print(f"  Train: {train_stats['count']:,}  Val: {val_stats['count']:,}")
    print(f"  Sources: {all_stats['sources']}")
    print(f"  Topics: {all_stats['topics']}")
    print(f"  Quality: mean={all_stats['quality_score']['mean']:.4f}")
    print(f"  Output: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Package Dataset for Submission")
    parser.add_argument("--input", type=str, required=True,
                        help="Input curated JSONL file")
    parser.add_argument("--dpo-input", type=str, default=None,
                        help="DPO pairs JSONL to include")
    parser.add_argument("--name", type=str, default="aimo3-curated",
                        help="Dataset name")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Validation split ratio (default: 0.05)")
    parser.add_argument("--kaggle", action="store_true",
                        help="Generate Kaggle dataset metadata")

    args = parser.parse_args()
    package_dataset(args)


if __name__ == "__main__":
    main()
