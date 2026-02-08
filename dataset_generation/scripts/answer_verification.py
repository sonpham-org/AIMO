#!/usr/bin/env python3
"""
Answer Verification for AIMO3 Fine-Tuning Datasets

Extracts and verifies answers from solution traces against ground truth.
Supports multiple answer formats:
  - \\boxed{...} (standard LaTeX)
  - "The answer is ..." / "Final answer: ..."
  - Harmony format extraction

Verification methods:
  1. Exact match (after normalization)
  2. Numeric equivalence (handles 1.0 == 1, fractions, etc.)

Usage:
  # Verify AIMO3 Hard dataset (has ground truth)
  python dataset_generation/scripts/answer_verification.py \
      --dataset aimo3_hard \
      --max-rows 1000

  # Verify a curated JSONL file
  python dataset_generation/scripts/answer_verification.py \
      --input-file data/curated_3000.jsonl

  # Output verified samples to new file
  python dataset_generation/scripts/answer_verification.py \
      --dataset aimo3_hard \
      --output data/verified_hard.jsonl
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed_answer(text):
    """Extract answer from \\boxed{...}, handling nested braces."""
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    last_match = matches[-1]
    start = last_match.end()

    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth == 0:
        return text[start:i - 1].strip()
    return None


def extract_answer_from_text(text):
    """Extract answer using multiple heuristics."""
    answer = extract_boxed_answer(text)
    if answer:
        return answer

    patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(\S+)',
        r'(?:the\s+)?(?:final\s+)?answer\s*[:=]\s*(\S+)',
        r'(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?answer\s+is\s*[:\s]*(\S+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip('.')

    return None


def normalize_answer(answer):
    """Normalize an answer string for comparison."""
    if answer is None:
        return ""
    s = str(answer).strip()
    s = s.replace('\\,', '')
    s = s.replace('\\;', '')
    s = s.replace('\\!', '')
    s = s.replace('\\text{', '').rstrip('}')
    s = s.replace('\\mathrm{', '').rstrip('}')
    s = s.replace('$', '')
    s = s.replace(' ', '')
    s = s.lower()
    return s


def answers_match(extracted, ground_truth):
    """Check if two answers match (exact or numeric equivalence)."""
    if extracted is None or ground_truth is None:
        return False

    norm_ext = normalize_answer(extracted)
    norm_gt = normalize_answer(ground_truth)

    if norm_ext == norm_gt:
        return True

    try:
        val_ext = float(norm_ext)
        val_gt = float(norm_gt)
        if math.isclose(val_ext, val_gt, rel_tol=1e-9, abs_tol=1e-9):
            return True
        if val_ext == int(val_ext) and val_gt == int(val_gt):
            return int(val_ext) == int(val_gt)
    except (ValueError, OverflowError):
        pass

    frac_pattern = r'^(\d+)/(\d+)$'
    for a, b in [(norm_ext, norm_gt), (norm_gt, norm_ext)]:
        m = re.match(frac_pattern, a)
        if m:
            try:
                frac_val = int(m.group(1)) / int(m.group(2))
                other_val = float(b)
                if math.isclose(frac_val, other_val, rel_tol=1e-9):
                    return True
            except (ValueError, ZeroDivisionError):
                pass

    return False


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/downloads")


def iter_aimo3_hard(max_rows=0):
    """Load AIMO3 Hard with ground truth."""
    path = DATA_DIR / "aimo3_hard" / "AIMO3-High-Difficulty-Tool-Calling-Dataset.jsonl"
    if not path.exists():
        print(f"  Warning: {path} not found")
        return
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            d = json.loads(line)
            meta = d.get("metadata_infos", {})
            rht = meta.get("reason_high_with_tool", {})
            yield {
                "id": f"hard_{i}",
                "problem": meta.get("problem", ""),
                "solution": d.get("text", ""),
                "ground_truth": str(meta.get("standard", "")),
                "pass_rate": rht.get("pass", 0) / max(rht.get("count", 8), 1),
            }


def iter_aimo3_tir(max_rows=0):
    """Load AIMO3 TIR (no ground truth in CSV)."""
    path = DATA_DIR / "aimo3_tir" / "data.csv"
    if not path.exists():
        print(f"  Warning: {path} not found")
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            yield {
                "id": f"tir_{i}",
                "problem": row.get("prompt", ""),
                "solution": row.get("completion", ""),
                "ground_truth": None,
                "pass_rate": None,
            }


def iter_limo(max_rows=0):
    """Load LIMO with ground truth."""
    path = DATA_DIR / "LIMO" / "limo.jsonl"
    if not path.exists():
        print(f"  Warning: {path} not found")
        return
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            d = json.loads(line)
            yield {
                "id": f"limo_{i}",
                "problem": d.get("question", ""),
                "solution": d.get("solution", ""),
                "ground_truth": str(d.get("answer", "")),
                "pass_rate": 0.05,
            }


def iter_jsonl_file(path, max_rows=0):
    """Load a curated JSONL file."""
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            d = json.loads(line)
            yield {
                "id": d.get("uid", f"row_{i}"),
                "problem": d.get("problem_text", ""),
                "solution": d.get("solution_text", ""),
                "ground_truth": d.get("answer"),
                "pass_rate": d.get("pass_rate"),
            }


DATASET_LOADERS = {
    "aimo3_hard": iter_aimo3_hard,
    "aimo3_tir": iter_aimo3_tir,
    "limo": iter_limo,
}


# ---------------------------------------------------------------------------
# Verification pipeline
# ---------------------------------------------------------------------------

def verify_dataset(name, iterator, max_rows=0, output_path=None):
    """Run answer verification on a dataset. Returns statistics dict."""
    stats = {
        "name": name,
        "total": 0,
        "has_ground_truth": 0,
        "answer_extracted": 0,
        "answer_verified": 0,
        "answer_wrong": 0,
        "no_answer_found": 0,
        "no_ground_truth": 0,
    }

    verified_samples = []
    extraction_methods = Counter()

    print(f"  Verifying {name}...")

    for sample in iterator(max_rows=max_rows):
        stats["total"] += 1
        solution = sample["solution"]
        gt = sample["ground_truth"]

        extracted = extract_answer_from_text(solution)

        if extracted is None:
            stats["no_answer_found"] += 1
            extraction_methods["none"] += 1
        else:
            stats["answer_extracted"] += 1
            if extract_boxed_answer(solution):
                extraction_methods["boxed"] += 1
            else:
                extraction_methods["text_pattern"] += 1

        if gt is None or gt == "" or gt == "None":
            stats["no_ground_truth"] += 1
        else:
            stats["has_ground_truth"] += 1
            if extracted is not None:
                if answers_match(extracted, gt):
                    stats["answer_verified"] += 1
                    if output_path:
                        verified_samples.append({
                            "uid": sample["id"],
                            "problem_text": sample["problem"],
                            "solution_text": solution,
                            "extracted_answer": extracted,
                            "ground_truth": gt,
                            "pass_rate": sample.get("pass_rate"),
                            "verified": True,
                        })
                else:
                    stats["answer_wrong"] += 1

    stats["extraction_methods"] = dict(extraction_methods)

    if stats["has_ground_truth"] > 0:
        verifiable = stats["has_ground_truth"]
        stats["verification_rate"] = round(100 * stats["answer_verified"] / verifiable, 2)
        stats["wrong_rate"] = round(100 * stats["answer_wrong"] / verifiable, 2)
    if stats["total"] > 0:
        stats["extraction_rate"] = round(100 * stats["answer_extracted"] / stats["total"], 2)

    if output_path and verified_samples:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for s in verified_samples:
                f.write(json.dumps(s) + "\n")
        stats["output_path"] = output_path
        stats["output_count"] = len(verified_samples)
        print(f"  Wrote {len(verified_samples)} verified samples to {output_path}")

    return stats


def print_report(results):
    """Print verification report."""
    print(f"\n{'='*80}")
    print("ANSWER VERIFICATION REPORT")
    print(f"{'='*80}\n")

    for r in results:
        print(f"--- {r['name']} ({r['total']:,} samples) ---")
        print(f"  Answer extracted:    {r['answer_extracted']:>6,} / {r['total']:,} "
              f"({r.get('extraction_rate', 0):.1f}%)")
        print(f"  No answer found:     {r['no_answer_found']:>6,}")

        if r["has_ground_truth"] > 0:
            print(f"  Has ground truth:    {r['has_ground_truth']:>6,}")
            print(f"  Verified correct:    {r['answer_verified']:>6,} / {r['has_ground_truth']:,} "
                  f"({r.get('verification_rate', 0):.1f}%)")
            print(f"  Wrong answer:        {r['answer_wrong']:>6,} / {r['has_ground_truth']:,} "
                  f"({r.get('wrong_rate', 0):.1f}%)")
        else:
            print(f"  No ground truth available for verification")

        methods = r.get("extraction_methods", {})
        if methods:
            print(f"  Extraction methods:  " + ", ".join(
                f"{k}={v}" for k, v in sorted(methods.items(), key=lambda x: -x[1])
            ))

        if r.get("output_path"):
            print(f"  Output: {r['output_count']} samples -> {r['output_path']}")
        print()

    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<20s} {'Total':>8s} {'Extracted':>10s} "
          f"{'Verified':>10s} {'Wrong':>8s} {'V.Rate':>8s}")
    print("-" * 68)
    for r in results:
        vrate = f"{r.get('verification_rate', 0):.1f}%" if r["has_ground_truth"] > 0 else "N/A"
        print(f"{r['name']:<20s} {r['total']:>8,} {r['answer_extracted']:>10,} "
              f"{r['answer_verified']:>10,} {r['answer_wrong']:>8,} {vrate:>8s}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Answer Verification for AIMO3 Datasets")
    parser.add_argument("--datasets", nargs="*", default=None,
                        choices=list(DATASET_LOADERS.keys()),
                        help="Datasets to verify")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Verify a single JSONL file")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Max rows per dataset (0=all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output verified samples to JSONL")

    args = parser.parse_args()

    results = []

    if args.input_file:
        name = Path(args.input_file).stem
        iterator = lambda max_rows=0: iter_jsonl_file(args.input_file, max_rows=max_rows)
        stats = verify_dataset(name, iterator, args.max_rows, args.output)
        results.append(stats)
    else:
        datasets = args.datasets or list(DATASET_LOADERS.keys())
        for ds_name in datasets:
            out = None
            if args.output and len(datasets) == 1:
                out = args.output
            elif args.output:
                base, ext = os.path.splitext(args.output)
                out = f"{base}_{ds_name}{ext}"
            stats = verify_dataset(ds_name, DATASET_LOADERS[ds_name], args.max_rows, out)
            results.append(stats)

    print_report(results)


if __name__ == "__main__":
    main()
