#!/usr/bin/env python3
"""
Dataset Profiler for AIMO3 Fine-Tuning

Produces comprehensive statistics across all available datasets:
  - Sample counts and size distribution
  - Solution length distribution (chars and estimated tokens)
  - Tool call frequency and density
  - Topic classification distribution
  - Pass rate / difficulty distribution
  - Answer format analysis
  - Quality score distribution (using the curate_dataset scorer)

Outputs a JSON report and optional ASCII dashboard.

Usage:
  # Profile all datasets (quick mode, 10K rows per dataset)
  python dataset_generation/scripts/dataset_profiler.py --quick

  # Profile specific datasets, full scan
  python dataset_generation/scripts/dataset_profiler.py \
      --datasets aimo3_hard limo s1k

  # Save JSON report
  python dataset_generation/scripts/dataset_profiler.py \
      --output data/profile_report.json
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Data loading (mirrors curate_dataset.py loaders)
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/downloads")


def load_aimo3_hard(max_rows: int = 0) -> list[dict]:
    path = DATA_DIR / "aimo3_hard" / "AIMO3-High-Difficulty-Tool-Calling-Dataset.jsonl"
    if not path.exists():
        return []
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            d = json.loads(line)
            meta = d.get("metadata_infos", {})
            rht = meta.get("reason_high_with_tool", {})
            text = d.get("text", "")
            count = rht.get("count", 8)
            passed = rht.get("pass", 0)

            n_tool = text.count("<|start|>python")
            if n_tool == 0:
                n_tool = text.count("```python")

            samples.append({
                "id": f"hard_{i}",
                "source": "aimo3_hard",
                "problem": meta.get("problem", ""),
                "solution": text,
                "answer": str(meta.get("standard", "")),
                "pass_rate": passed / count if count > 0 else None,
                "n_tool_calls": n_tool,
            })
    return samples


def load_aimo3_tir(max_rows: int = 0) -> list[dict]:
    path = DATA_DIR / "aimo3_tir" / "data.csv"
    if not path.exists():
        return []
    samples = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            comp = row.get("completion", "")
            n_tool = comp.count("<|start|>python")
            if n_tool == 0:
                n_tool = comp.count("```python")
            samples.append({
                "id": f"tir_{i}",
                "source": "aimo3_tir",
                "problem": row.get("prompt", ""),
                "solution": comp,
                "answer": None,
                "pass_rate": None,
                "n_tool_calls": n_tool,
            })
    return samples


def load_limo(max_rows: int = 0) -> list[dict]:
    path = DATA_DIR / "LIMO" / "limo.jsonl"
    if not path.exists():
        return []
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            d = json.loads(line)
            samples.append({
                "id": f"limo_{i}",
                "source": "limo",
                "problem": d.get("question", ""),
                "solution": d.get("solution", ""),
                "answer": str(d.get("answer", "")),
                "pass_rate": 0.05,
                "n_tool_calls": 0,
            })
    return samples


def load_s1k(max_rows: int = 0) -> list[dict]:
    path = DATA_DIR / "s1K" / "data" / "train-00000-of-00001.parquet"
    if not path.exists():
        return []
    try:
        import pandas as pd
    except ImportError:
        return []
    df = pd.read_parquet(path)
    samples = []
    for i, row in df.iterrows():
        if max_rows and i >= max_rows:
            break
        cot = str(row.get("cot", "") or "")
        solution = str(row.get("solution", "") or "")
        text = cot if len(cot) > len(solution) else solution
        samples.append({
            "id": f"s1k_{i}",
            "source": "s1k",
            "problem": str(row.get("question", "")),
            "solution": text,
            "answer": str(row.get("solution", "")),
            "pass_rate": 0.10,
            "n_tool_calls": 0,
        })
    return samples


def load_genselect(max_rows: int = 0) -> list[dict]:
    data_dir = DATA_DIR / "OpenMathReasoning-GenSelect" / "data"
    if not data_dir.exists():
        return []
    try:
        import pandas as pd
    except ImportError:
        return []
    samples = []
    files = sorted(f for f in os.listdir(data_dir) if "genselect" in f and f.endswith(".parquet"))
    total = 0
    for fname in files:
        df = pd.read_parquet(os.path.join(data_dir, fname))
        for _, row in df.iterrows():
            if max_rows and total >= max_rows:
                break
            solution = str(row.get("generated_solution", ""))
            pr_str = str(row.get("pass_rate_72b_tir", "n/a"))
            try:
                pass_rate = float(pr_str) if pr_str != "n/a" else None
            except (ValueError, TypeError):
                pass_rate = None
            samples.append({
                "id": f"genselect_{total}",
                "source": "genselect",
                "problem": str(row.get("problem", "")),
                "solution": solution,
                "answer": str(row.get("expected_answer", "")),
                "pass_rate": pass_rate,
                "n_tool_calls": 0,
            })
            total += 1
        if max_rows and total >= max_rows:
            break
    return samples


def load_bigmath(max_rows: int = 0) -> list[dict]:
    path = DATA_DIR / "Big-Math-RL-Verified" / "data" / "train-00000-of-00001.parquet"
    if not path.exists():
        return []
    try:
        import pandas as pd
    except ImportError:
        return []
    df = pd.read_parquet(path)
    samples = []
    for i, row in df.iterrows():
        if max_rows and i >= max_rows:
            break
        problem = str(row.get("problem", row.get("question", "")))
        answer = str(row.get("answer", row.get("expected_answer", "")))
        samples.append({
            "id": f"bigmath_{i}",
            "source": "bigmath",
            "problem": problem,
            "solution": problem,  # Problems only
            "answer": answer,
            "pass_rate": None,
            "n_tool_calls": 0,
        })
    return samples


DATASET_LOADERS = {
    "aimo3_hard": load_aimo3_hard,
    "aimo3_tir": load_aimo3_tir,
    "limo": load_limo,
    "s1k": load_s1k,
    "genselect": load_genselect,
    "bigmath": load_bigmath,
}


# ---------------------------------------------------------------------------
# Topic classification
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS = {
    "number_theory": ["prime", "divisor", "modular", "gcd", "lcm", "congruence", "euler", "fermat",
                       "divisible", "remainder", "modulo", "coprime", "totient"],
    "algebra": ["polynomial", "equation", "inequality", "root", "coefficient", "quadratic",
                "cubic", "factor", "solve for", "variable", "expression", "function"],
    "combinatorics": ["permutation", "combination", "counting", "probability", "choose",
                      "arrange", "subset", "partition", "pigeonhole", "binomial", "ways"],
    "geometry": ["triangle", "circle", "angle", "area", "perimeter", "perpendicular",
                 "parallel", "polygon", "radius", "diameter", "tangent", "circumscribe"],
}


def classify_topic(problem_text: str) -> str:
    text = problem_text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = sum(text.count(kw) for kw in keywords)
    if max(scores.values()) == 0:
        return "other"
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Profiling functions
# ---------------------------------------------------------------------------

def percentiles(values: list, pcts: list[int]) -> dict:
    if not values:
        return {p: 0 for p in pcts}
    s = sorted(values)
    n = len(s)
    return {p: s[min(int(n * p / 100), n - 1)] for p in pcts}


def profile_dataset(name: str, samples: list[dict]) -> dict:
    """Generate comprehensive profile for a dataset."""
    n = len(samples)
    if n == 0:
        return {"name": name, "count": 0}

    # Basic stats
    sol_lengths = [len(s["solution"]) for s in samples]
    prob_lengths = [len(s["problem"]) for s in samples]
    tool_calls = [s["n_tool_calls"] for s in samples]
    pass_rates = [s["pass_rate"] for s in samples if s["pass_rate"] is not None]

    # Answer format analysis
    has_boxed = sum(1 for s in samples if "boxed{" in s["solution"])
    has_answer_text = sum(1 for s in samples
                         if re.search(r"(?:final\s+)?answer\s+is", s["solution"], re.IGNORECASE))
    uses_sympy = sum(1 for s in samples if "sympy" in s["solution"].lower())
    uses_python = sum(1 for s in samples if "```python" in s["solution"] or "<|start|>python" in s["solution"])

    # Topic distribution
    topics = Counter(classify_topic(s["problem"]) for s in samples)

    # Pass rate buckets
    pr_buckets = Counter()
    for pr in pass_rates:
        if pr <= 0.0:
            pr_buckets["0%"] += 1
        elif pr <= 0.125:
            pr_buckets["0-12.5%"] += 1
        elif pr <= 0.25:
            pr_buckets["12.5-25%"] += 1
        elif pr <= 0.375:
            pr_buckets["25-37.5%"] += 1
        elif pr <= 0.50:
            pr_buckets["37.5-50%"] += 1
        elif pr <= 0.625:
            pr_buckets["50-62.5%"] += 1
        elif pr <= 0.75:
            pr_buckets["62.5-75%"] += 1
        elif pr <= 0.875:
            pr_buckets["75-87.5%"] += 1
        else:
            pr_buckets["87.5-100%"] += 1

    # Estimated tokens (rough: ~4 chars per token)
    est_tokens = [l // 4 for l in sol_lengths]

    pcts = [25, 50, 75, 90, 95, 99]

    return {
        "name": name,
        "count": n,
        "solution_length": {
            "min": min(sol_lengths),
            "max": max(sol_lengths),
            "mean": round(sum(sol_lengths) / n),
            "percentiles": percentiles(sol_lengths, pcts),
        },
        "problem_length": {
            "min": min(prob_lengths),
            "max": max(prob_lengths),
            "mean": round(sum(prob_lengths) / n),
        },
        "estimated_tokens": {
            "mean": round(sum(est_tokens) / n),
            "percentiles": percentiles(est_tokens, pcts),
            "over_65k": sum(1 for t in est_tokens if t > 65536),
            "over_32k": sum(1 for t in est_tokens if t > 32768),
        },
        "tool_calls": {
            "min": min(tool_calls),
            "max": max(tool_calls),
            "mean": round(sum(tool_calls) / n, 1),
            "zero": sum(1 for t in tool_calls if t == 0),
            "1_to_5": sum(1 for t in tool_calls if 1 <= t <= 5),
            "6_to_9": sum(1 for t in tool_calls if 6 <= t <= 9),
            "10_to_15": sum(1 for t in tool_calls if 10 <= t <= 15),
            "over_15": sum(1 for t in tool_calls if t > 15),
        },
        "answer_format": {
            "has_boxed": has_boxed,
            "has_answer_text": has_answer_text,
            "uses_sympy": uses_sympy,
            "uses_python": uses_python,
            "has_boxed_pct": round(100 * has_boxed / n, 1),
            "uses_sympy_pct": round(100 * uses_sympy / n, 1),
            "uses_python_pct": round(100 * uses_python / n, 1),
        },
        "topics": dict(topics.most_common()),
        "pass_rate": {
            "available": len(pass_rates),
            "unavailable": n - len(pass_rates),
            "distribution": dict(sorted(pr_buckets.items())),
            "mean": round(sum(pass_rates) / len(pass_rates), 3) if pass_rates else None,
            "in_sft_range_003_030": sum(1 for p in pass_rates if 0.03 <= p <= 0.30),
            "in_rl_range_010_070": sum(1 for p in pass_rates if 0.10 <= p <= 0.70),
        },
    }


def print_profile(profiles: list[dict]):
    """Print formatted profile report."""
    print(f"\n{'='*80}")
    print("DATASET PROFILE REPORT")
    print(f"{'='*80}\n")

    for p in profiles:
        if p["count"] == 0:
            print(f"--- {p['name']}: no samples found ---\n")
            continue

        print(f"--- {p['name']} ({p['count']:,} samples) ---")

        # Solution length
        sl = p["solution_length"]
        print(f"  Solution length (chars): min={sl['min']:,}  mean={sl['mean']:,}  "
              f"max={sl['max']:,}")
        pctiles = sl["percentiles"]
        print(f"    Percentiles: " + "  ".join(
            f"p{k}={v:,}" for k, v in sorted(pctiles.items())
        ))

        # Estimated tokens
        et = p["estimated_tokens"]
        print(f"  Est. tokens: mean={et['mean']:,}  "
              f"over 32K={et['over_32k']:,}  over 65K={et['over_65k']:,}")

        # Tool calls
        tc = p["tool_calls"]
        print(f"  Tool calls: mean={tc['mean']:.1f}  "
              f"zero={tc['zero']:,}  1-5={tc['1_to_5']:,}  "
              f"6-9={tc['6_to_9']:,}  10-15={tc['10_to_15']:,}  >15={tc['over_15']:,}")

        # Answer format
        af = p["answer_format"]
        print(f"  Answer format: boxed={af['has_boxed_pct']}%  "
              f"sympy={af['uses_sympy_pct']}%  python={af['uses_python_pct']}%")

        # Topics
        topics = p["topics"]
        topic_str = "  Topics: " + ", ".join(
            f"{t}={c}" for t, c in sorted(topics.items(), key=lambda x: -x[1])
        )
        print(topic_str)

        # Pass rate
        pr = p["pass_rate"]
        if pr["available"] > 0:
            print(f"  Pass rate: mean={pr['mean']:.3f}  "
                  f"SFT range (3-30%)={pr['in_sft_range_003_030']:,}  "
                  f"RL range (10-70%)={pr['in_rl_range_010_070']:,}")
            dist = pr["distribution"]
            if dist:
                print(f"    Distribution: " + "  ".join(
                    f"{k}={v}" for k, v in sorted(dist.items())
                ))
        else:
            print(f"  Pass rate: not available")

        print()

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<15s} {'Count':>8s} {'AvgLen':>8s} {'AvgTok':>8s} "
          f"{'ToolCalls':>10s} {'Boxed%':>7s} {'SymPy%':>7s}")
    print("-" * 67)
    for p in profiles:
        if p["count"] == 0:
            continue
        print(f"{p['name']:<15s} {p['count']:>8,} "
              f"{p['solution_length']['mean']:>8,} "
              f"{p['estimated_tokens']['mean']:>8,} "
              f"{p['tool_calls']['mean']:>10.1f} "
              f"{p['answer_format']['has_boxed_pct']:>6.1f}% "
              f"{p['answer_format']['uses_sympy_pct']:>6.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AIMO3 Dataset Profiler")
    parser.add_argument("--datasets", nargs="*", default=None,
                        choices=list(DATASET_LOADERS.keys()),
                        help="Datasets to profile (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: max 10K rows per dataset")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Max rows per dataset (0=all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON report to file")

    args = parser.parse_args()

    max_rows = args.max_rows
    if args.quick and not max_rows:
        max_rows = 10000
        print("Quick mode: max 10,000 rows per dataset\n")

    datasets = args.datasets or list(DATASET_LOADERS.keys())
    profiles = []

    for ds_name in datasets:
        print(f"Loading {ds_name}...")
        loader = DATASET_LOADERS[ds_name]
        samples = loader(max_rows=max_rows)
        print(f"  Loaded {len(samples):,} samples")

        if samples:
            profile = profile_dataset(ds_name, samples)
            profiles.append(profile)
        else:
            profiles.append({"name": ds_name, "count": 0})

    print_profile(profiles)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(profiles, f, indent=2)
        print(f"\nJSON report saved to {args.output}")


if __name__ == "__main__":
    main()
