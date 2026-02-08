#!/usr/bin/env python3
"""
Build DPO (Direct Preference Optimization) Pairs from AIMO3 Datasets

Creates chosen/rejected solution pairs for preference training:
  - Chosen: correct solutions with high quality scores
  - Rejected: incorrect solutions OR correct but low quality

Sources:
  - AIMO3 Hard: 8 solutions per problem with pass_rate metadata
    -> Group by problem, pick best correct as chosen, worst/incorrect as rejected
  - Curated JSONL: already scored, pair high vs low quality

Output format (JSONL, one pair per line):
  {"prompt": "...", "chosen": "...", "rejected": "...", "metadata": {...}}

Usage:
  # Build DPO pairs from AIMO3 Hard (has multiple solutions per problem)
  python dataset_generation/scripts/build_dpo_pairs.py \
      --source aimo3_hard \
      --output dataset_generation/output/dpo_pairs.jsonl \
      --max-problems 5000

  # Build from curated JSONL (pairs high-score vs low-score)
  python dataset_generation/scripts/build_dpo_pairs.py \
      --source curated \
      --input dataset_generation/output/curated_sft_3000.jsonl \
      --output dataset_generation/output/dpo_curated.jsonl

  # Build with Tinker-compatible format
  python dataset_generation/scripts/build_dpo_pairs.py \
      --source aimo3_hard \
      --format tinker \
      --output dataset_generation/output/dpo_tinker.jsonl
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Answer extraction (shared with answer_verification.py)
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


def normalize_answer(answer):
    if answer is None:
        return ""
    s = str(answer).strip()
    s = s.replace('\\,', '').replace('\\;', '').replace('\\!', '')
    s = s.replace('\\text{', '').rstrip('}')
    s = s.replace('\\mathrm{', '').rstrip('}')
    s = s.replace('$', '').replace(' ', '').lower()
    return s


def answers_match(extracted, ground_truth):
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
    except (ValueError, OverflowError):
        pass
    return False


# ---------------------------------------------------------------------------
# Quality scoring (lightweight version)
# ---------------------------------------------------------------------------

def quick_quality_score(solution_text, n_tool_calls=0):
    """Quick quality score for ranking solutions."""
    text = solution_text
    length = len(text)
    score = 0.0

    # Length component
    score += 0.25 * min(length / 30000, 1.0)
    if length > 80000:
        score -= 0.1

    # Verification keywords
    verify_words = ["check", "verify", "confirm", "let's test", "validate"]
    verify_count = sum(text.lower().count(w) for w in verify_words)
    score += 0.20 * min(verify_count / max(length / 5000, 1), 1.0)

    # Logical connectives
    logic_words = ["therefore", "since", "hence", "thus", "because"]
    logic_count = sum(text.lower().count(w) for w in logic_words)
    score += 0.15 * min(logic_count / max(length / 3000, 1), 1.0)

    # Structure
    if "boxed{" in text:
        score += 0.15
    if "sympy" in text.lower():
        score += 0.10
    if re.search(r"step\s*\d", text, re.IGNORECASE):
        score += 0.05

    # Tool usage bonus
    if 3 <= n_tool_calls <= 12:
        score += 0.10

    # Repetition penalty (check for repeated 60-char substrings without backtracking regex)
    has_repetition = False
    if length > 200:
        # Sample a few positions to check for repeated blocks
        step = max(length // 20, 60)
        for i in range(0, length - 120, step):
            chunk = text[i:i + 60]
            if text.find(chunk, i + 60) != -1:
                has_repetition = True
                break
    if has_repetition:
        score -= 0.20

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# AIMO3 Hard: group by problem, build pairs
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/downloads")

SYSTEM_PROMPT = (
    "You are a helpful math assistant that solves problems step by step, "
    "using Python code execution when helpful for computation and verification. "
    "Always provide your final answer in \\boxed{} format."
)


def build_pairs_aimo3_hard(max_problems=0, min_quality_gap=0.05):
    """Build DPO pairs from AIMO3 Hard (multiple solutions per problem).

    Groups solutions by problem_id, picks best correct as chosen,
    worst incorrect (or lowest quality correct) as rejected.
    """
    path = DATA_DIR / "aimo3_hard" / "AIMO3-High-Difficulty-Tool-Calling-Dataset.jsonl"
    if not path.exists():
        print(f"Error: {path} not found")
        return []

    # Load and group by problem
    print("  Loading AIMO3 Hard...")
    problems = defaultdict(list)
    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            meta = d.get("metadata_infos", {})
            problem = meta.get("problem", "")
            answer = str(meta.get("standard", ""))
            text = d.get("text", "")

            n_tool = text.count("<|start|>python")
            if n_tool == 0:
                n_tool = text.count("```python")

            extracted = extract_boxed_answer(text)
            is_correct = answers_match(extracted, answer) if extracted else False
            quality = quick_quality_score(text, n_tool)

            problems[problem].append({
                "solution": text,
                "answer": answer,
                "extracted": extracted,
                "correct": is_correct,
                "quality": quality,
                "n_tool_calls": n_tool,
                "idx": i,
            })

    print(f"  Found {len(problems):,} unique problems with {sum(len(v) for v in problems.values()):,} total solutions")

    # Build pairs
    pairs = []
    skipped_no_correct = 0
    skipped_no_rejected = 0

    for problem_text, solutions in problems.items():
        if max_problems and len(pairs) >= max_problems:
            break

        correct = [s for s in solutions if s["correct"]]
        incorrect = [s for s in solutions if not s["correct"]]

        if not correct:
            skipped_no_correct += 1
            continue

        # Sort correct by quality descending
        correct.sort(key=lambda s: s["quality"], reverse=True)
        best_correct = correct[0]

        # Rejected: prefer incorrect solution, fallback to lowest quality correct
        rejected = None
        if incorrect:
            # Pick the incorrect solution with highest quality (most plausible wrong answer)
            incorrect.sort(key=lambda s: s["quality"], reverse=True)
            rejected = incorrect[0]
        elif len(correct) > 1:
            # No incorrect solutions available, use lowest quality correct
            worst_correct = correct[-1]
            if best_correct["quality"] - worst_correct["quality"] >= min_quality_gap:
                rejected = worst_correct
            else:
                skipped_no_rejected += 1
                continue
        else:
            skipped_no_rejected += 1
            continue

        pairs.append({
            "prompt": problem_text,
            "chosen": best_correct["solution"],
            "rejected": rejected["solution"],
            "metadata": {
                "ground_truth": best_correct["answer"],
                "chosen_correct": best_correct["correct"],
                "chosen_quality": round(best_correct["quality"], 4),
                "rejected_correct": rejected["correct"],
                "rejected_quality": round(rejected["quality"], 4),
                "n_solutions": len(solutions),
                "n_correct": len(correct),
                "n_incorrect": len(incorrect),
            }
        })

    print(f"  Built {len(pairs):,} DPO pairs")
    print(f"  Skipped: no correct={skipped_no_correct:,}, no rejected={skipped_no_rejected:,}")

    # Stats
    if pairs:
        chosen_correct = sum(1 for p in pairs if p["metadata"]["chosen_correct"])
        rejected_correct = sum(1 for p in pairs if p["metadata"]["rejected_correct"])
        avg_quality_gap = sum(
            p["metadata"]["chosen_quality"] - p["metadata"]["rejected_quality"]
            for p in pairs
        ) / len(pairs)
        print(f"  Chosen correct: {chosen_correct}/{len(pairs)} "
              f"({100 * chosen_correct / len(pairs):.1f}%)")
        print(f"  Rejected correct: {rejected_correct}/{len(pairs)} "
              f"({100 * rejected_correct / len(pairs):.1f}%)")
        print(f"  Avg quality gap: {avg_quality_gap:.4f}")

    return pairs


# ---------------------------------------------------------------------------
# Curated JSONL: pair high vs low quality per topic
# ---------------------------------------------------------------------------

def build_pairs_curated(input_path, min_quality_gap=0.05):
    """Build DPO pairs from curated JSONL by pairing high vs low quality."""
    samples = []
    with open(input_path) as f:
        for line in f:
            d = json.loads(line)
            samples.append(d)

    print(f"  Loaded {len(samples):,} curated samples")

    # Group by topic
    by_topic = defaultdict(list)
    for s in samples:
        topic = s.get("topic", "other")
        by_topic[topic].append(s)

    pairs = []
    for topic, topic_samples in by_topic.items():
        # Sort by combined_score
        topic_samples.sort(key=lambda s: s.get("combined_score", 0), reverse=True)

        # Pair top half with bottom half
        n = len(topic_samples)
        top_half = topic_samples[:n // 2]
        bottom_half = topic_samples[n // 2:]

        for chosen, rejected in zip(top_half, bottom_half):
            gap = chosen.get("combined_score", 0) - rejected.get("combined_score", 0)
            if gap < min_quality_gap:
                continue

            pairs.append({
                "prompt": chosen.get("problem_text", ""),
                "chosen": chosen.get("solution_text", ""),
                "rejected": rejected.get("solution_text", ""),
                "metadata": {
                    "chosen_score": chosen.get("combined_score"),
                    "rejected_score": rejected.get("combined_score"),
                    "quality_gap": round(gap, 4),
                    "topic": topic,
                }
            })

    print(f"  Built {len(pairs):,} DPO pairs from curated data")
    return pairs


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def format_standard(pair):
    """Standard DPO format."""
    return pair


def format_tinker(pair):
    """Tinker-compatible DPO format."""
    return {
        "chosen": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["chosen"]},
        ],
        "rejected": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["rejected"]},
        ],
        "metadata": pair.get("metadata", {}),
    }


def format_sharegpt_dpo(pair):
    """ShareGPT-style DPO format (used by LLaMA-Factory)."""
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": pair["prompt"]},
        ],
        "chosen": {"from": "gpt", "value": pair["chosen"]},
        "rejected": {"from": "gpt", "value": pair["rejected"]},
    }


FORMAT_MAP = {
    "standard": format_standard,
    "tinker": format_tinker,
    "sharegpt": format_sharegpt_dpo,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build DPO Pairs for AIMO3")
    parser.add_argument("--source", type=str, required=True,
                        choices=["aimo3_hard", "curated"],
                        help="Source: aimo3_hard (grouped by problem) or curated (by quality)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSONL (required for --source curated)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path")
    parser.add_argument("--format", type=str, default="standard",
                        choices=list(FORMAT_MAP.keys()),
                        help="Output format (default: standard)")
    parser.add_argument("--max-problems", type=int, default=0,
                        help="Max number of pairs (0=all)")
    parser.add_argument("--min-quality-gap", type=float, default=0.05,
                        help="Minimum quality gap between chosen/rejected")

    args = parser.parse_args()

    if args.source == "aimo3_hard":
        pairs = build_pairs_aimo3_hard(
            max_problems=args.max_problems,
            min_quality_gap=args.min_quality_gap,
        )
    elif args.source == "curated":
        if not args.input:
            print("Error: --input required for --source curated")
            sys.exit(1)
        pairs = build_pairs_curated(args.input, args.min_quality_gap)

    if not pairs:
        print("No pairs generated!")
        sys.exit(1)

    # Format and write
    formatter = FORMAT_MAP[args.format]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w") as f:
        for pair in pairs:
            record = formatter(pair)
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(pairs):,} DPO pairs to {args.output}")
    print(f"Format: {args.format}")

    # File size
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
