#!/usr/bin/env python3
"""
Token Length Analysis for AIMO3 Fine-Tuning Datasets

Analyzes token counts across datasets to ensure training samples fit within
model context windows. Critical for gpt-oss-120b (65536 token context).

Supports two tokenization modes:
  1. tiktoken (fast, approximate) - uses cl100k_base as proxy
  2. transformers (accurate) - uses actual model tokenizer if available

Usage:
  # Quick analysis with tiktoken (no GPU needed)
  python dataset_generation/scripts/token_length_analysis.py \
      --datasets aimo3_tir aimo3_hard limo s1k \
      --tokenizer tiktoken \
      --context-limit 65536

  # Detailed analysis with percentile breakdown
  python dataset_generation/scripts/token_length_analysis.py \
      --datasets aimo3_tir \
      --tokenizer tiktoken \
      --context-limit 65536 \
      --percentiles 50 75 90 95 99 \
      --output data/token_analysis.json

  # Analyze a single JSONL file
  python dataset_generation/scripts/token_length_analysis.py \
      --input-file data/curated_3000.jsonl \
      --tokenizer tiktoken \
      --context-limit 65536
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Tokenizer backends
# ---------------------------------------------------------------------------

def get_tiktoken_encoder():
    """Load tiktoken cl100k_base as a fast proxy tokenizer."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        print("Error: tiktoken not installed. Run: pip install tiktoken")
        sys.exit(1)


def count_tokens_tiktoken(text, encoder):
    """Count tokens using tiktoken."""
    return len(encoder.encode(text, disallowed_special=()))


def get_transformers_tokenizer(model_name):
    """Load a HuggingFace tokenizer."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except ImportError:
        print("Error: transformers not installed. Run: pip install transformers")
        sys.exit(1)


def count_tokens_transformers(text, tokenizer):
    """Count tokens using a HuggingFace tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Dataset loaders (return iterator of (sample_id, text) tuples)
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/downloads")


def iter_aimo3_hard(max_rows=0):
    """Iterate AIMO3 Hard dataset, yielding (id, full_text) pairs."""
    path = DATA_DIR / "aimo3_hard" / "AIMO3-High-Difficulty-Tool-Calling-Dataset.jsonl"
    if not path.exists():
        print(f"  Warning: {path} not found, skipping")
        return
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            d = json.loads(line)
            meta = d.get("metadata_infos", {})
            problem = meta.get("problem", "")
            text = d.get("text", "")
            full = problem + "\n" + text
            yield f"hard_{i}", full


def iter_aimo3_tir(max_rows=0):
    """Iterate AIMO3 TIR dataset, yielding (id, full_text) pairs."""
    path = DATA_DIR / "aimo3_tir" / "data.csv"
    if not path.exists():
        print(f"  Warning: {path} not found, skipping")
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            prompt = row.get("prompt", "")
            comp = row.get("completion", "")
            full = prompt + "\n" + comp
            yield f"tir_{i}", full


def iter_limo(max_rows=0):
    """Iterate LIMO dataset."""
    path = DATA_DIR / "LIMO" / "limo.jsonl"
    if not path.exists():
        print(f"  Warning: {path} not found, skipping")
        return
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            d = json.loads(line)
            question = d.get("question", "")
            solution = d.get("solution", "")
            full = question + "\n" + solution
            yield f"limo_{i}", full


def iter_s1k(max_rows=0):
    """Iterate s1K dataset."""
    path = DATA_DIR / "s1K" / "data" / "train-00000-of-00001.parquet"
    if not path.exists():
        print(f"  Warning: {path} not found, skipping")
        return
    try:
        import pandas as pd
    except ImportError:
        print("  Warning: pandas not available, skipping s1K")
        return
    df = pd.read_parquet(path)
    for i, row in df.iterrows():
        if max_rows and i >= max_rows:
            break
        question = str(row.get("question", ""))
        cot = str(row.get("cot", "") or "")
        solution = str(row.get("solution", "") or "")
        text = cot if len(cot) > len(solution) else solution
        full = question + "\n" + text
        yield f"s1k_{i}", full


def iter_jsonl_file(path, max_rows=0):
    """Iterate a JSONL file (curated output), yielding (id, full_text) pairs."""
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            d = json.loads(line)
            problem = d.get("problem_text", "")
            solution = d.get("solution_text", "")
            full = problem + "\n" + solution
            yield d.get("uid", f"row_{i}"), full


DATASET_ITERATORS = {
    "aimo3_hard": iter_aimo3_hard,
    "aimo3_tir": iter_aimo3_tir,
    "limo": iter_limo,
    "s1k": iter_s1k,
}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_percentiles(values, percentiles):
    """Compute percentile values from a sorted list."""
    if not values:
        return {p: 0 for p in percentiles}
    values_sorted = sorted(values)
    n = len(values_sorted)
    result = {}
    for p in percentiles:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        result[p] = values_sorted[idx]
    return result


def analyze_dataset(name, iterator, count_fn, encoder_or_tokenizer, context_limit,
                    max_rows=0, percentiles=None):
    """Analyze token lengths for a dataset. Returns a dict with statistics."""
    if percentiles is None:
        percentiles = [50, 75, 90, 95, 99]

    token_counts = []
    over_limit = 0
    total_chars = 0

    print(f"  Analyzing {name}...")

    for sample_id, text in iterator(max_rows=max_rows):
        n_tokens = count_fn(text, encoder_or_tokenizer)
        token_counts.append(n_tokens)
        total_chars += len(text)
        if n_tokens > context_limit:
            over_limit += 1

    if not token_counts:
        return {"name": name, "count": 0, "error": "no samples found"}

    n = len(token_counts)
    pctiles = compute_percentiles(token_counts, percentiles)

    stats = {
        "name": name,
        "count": n,
        "context_limit": context_limit,
        "over_limit": over_limit,
        "over_limit_pct": round(100 * over_limit / n, 2),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "mean_tokens": round(sum(token_counts) / n, 1),
        "median_tokens": pctiles.get(50, 0),
        "percentiles": {str(p): v for p, v in pctiles.items()},
        "total_chars": total_chars,
        "avg_chars": round(total_chars / n, 1),
        "chars_per_token": round(total_chars / sum(token_counts), 2) if sum(token_counts) > 0 else 0,
    }

    # Token count histogram buckets
    buckets = [0, 1000, 2000, 4000, 8000, 16000, 32000, 65536, 131072, float("inf")]
    bucket_labels = []
    bucket_counts = []
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        label = f"{lo}-{hi}" if hi != float("inf") else f"{lo}+"
        count = sum(1 for t in token_counts if lo <= t < hi)
        bucket_labels.append(label)
        bucket_counts.append(count)
    stats["histogram"] = dict(zip(bucket_labels, bucket_counts))

    return stats


def print_report(results, context_limit):
    """Print a formatted report of token analysis results."""
    print(f"\n{'='*80}")
    print(f"TOKEN LENGTH ANALYSIS REPORT (context limit: {context_limit:,} tokens)")
    print(f"{'='*80}\n")

    for r in results:
        if r.get("error"):
            print(f"--- {r['name']}: {r['error']} ---\n")
            continue

        print(f"--- {r['name']} ({r['count']:,} samples) ---")
        print(f"  Tokens: min={r['min_tokens']:,}  mean={r['mean_tokens']:,.0f}  "
              f"median={r['median_tokens']:,}  max={r['max_tokens']:,}")
        print(f"  Chars:  avg={r['avg_chars']:,.0f}  chars/token={r['chars_per_token']:.2f}")
        print(f"  Over context limit ({context_limit:,}): "
              f"{r['over_limit']:,}/{r['count']:,} ({r['over_limit_pct']:.1f}%)")

        pctiles = r.get("percentiles", {})
        if pctiles:
            pct_str = "  Percentiles: " + "  ".join(
                f"p{p}={v:,}" for p, v in sorted(pctiles.items(), key=lambda x: int(x[0]))
            )
            print(pct_str)

        hist = r.get("histogram", {})
        if hist:
            print("  Distribution:")
            for bucket, count in hist.items():
                pct = 100 * count / r["count"] if r["count"] > 0 else 0
                bar = "#" * int(pct / 2)
                print(f"    {bucket:>12s}: {count:>6,} ({pct:5.1f}%) {bar}")
        print()

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<20s} {'Samples':>10s} {'Mean Tok':>10s} {'p95 Tok':>10s} "
          f"{'Over Limit':>12s} {'% Over':>8s}")
    print("-" * 74)
    for r in results:
        if r.get("error"):
            continue
        p95 = r.get("percentiles", {}).get("95", "N/A")
        print(f"{r['name']:<20s} {r['count']:>10,} {r['mean_tokens']:>10,.0f} "
              f"{p95:>10,} {r['over_limit']:>12,} {r['over_limit_pct']:>7.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Token Length Analysis for AIMO3 Datasets")
    parser.add_argument("--datasets", nargs="*", default=None,
                        choices=list(DATASET_ITERATORS.keys()),
                        help="Datasets to analyze (default: all available)")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Analyze a single JSONL file instead of named datasets")
    parser.add_argument("--tokenizer", type=str, default="tiktoken",
                        choices=["tiktoken"],
                        help="Tokenizer backend (default: tiktoken)")
    parser.add_argument("--hf-model", type=str, default=None,
                        help="HuggingFace model name for transformers tokenizer")
    parser.add_argument("--context-limit", type=int, default=65536,
                        help="Context window limit in tokens (default: 65536)")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Max rows per dataset (0=all, useful for quick profiling)")
    parser.add_argument("--percentiles", nargs="*", type=int, default=[50, 75, 90, 95, 99],
                        help="Percentiles to compute (default: 50 75 90 95 99)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    # Set up tokenizer
    if args.hf_model:
        tokenizer = get_transformers_tokenizer(args.hf_model)
        count_fn = count_tokens_transformers
        print(f"Using HuggingFace tokenizer: {args.hf_model}")
    else:
        tokenizer = get_tiktoken_encoder()
        count_fn = count_tokens_tiktoken
        print("Using tiktoken (cl100k_base) as proxy tokenizer")

    print(f"Context limit: {args.context_limit:,} tokens")
    if args.max_rows:
        print(f"Max rows per dataset: {args.max_rows:,}")
    print()

    results = []

    if args.input_file:
        name = Path(args.input_file).stem
        iterator = lambda max_rows=0: iter_jsonl_file(args.input_file, max_rows=max_rows)
        stats = analyze_dataset(
            name, iterator, count_fn, tokenizer,
            args.context_limit, args.max_rows, args.percentiles,
        )
        results.append(stats)
    else:
        datasets = args.datasets or list(DATASET_ITERATORS.keys())
        for ds_name in datasets:
            iterator = DATASET_ITERATORS[ds_name]
            stats = analyze_dataset(
                ds_name, iterator, count_fn, tokenizer,
                args.context_limit, args.max_rows, args.percentiles,
            )
            results.append(stats)

    print_report(results, args.context_limit)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
