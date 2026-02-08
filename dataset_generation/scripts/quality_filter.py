#!/usr/bin/env python3
"""
Quality Filtering Pipeline for AIMO3 Fine-Tuning Datasets

Implements the full 6-stage quality filtering pipeline based on research:
  Stage 0: Load datasets into unified format
  Stage 1: Hard filters (length, tool calls, answer format, repetition)
  Stage 2: Decontamination (9-gram overlap vs eval sets)
  Stage 3: Difficulty calibration (pass rate filtering)
  Stage 4: Quality scoring (LIMO-style + interaction density)
  Stage 5: Diversity selection (topic-balanced sampling)

This is a streamlined version of scripts/curate_dataset.py with additional
features from the research (ASTER interaction density, iw-SFT weights).

Usage:
  # Run full pipeline, output 3000 samples
  python dataset_generation/scripts/quality_filter.py \
      --target 3000 \
      --output dataset_generation/output/curated_3000.jsonl

  # Run with custom difficulty range for RL
  python dataset_generation/scripts/quality_filter.py \
      --target 4000 \
      --min-pass-rate 0.10 --max-pass-rate 0.70 \
      --output dataset_generation/output/curated_rl_4000.jsonl

  # SFT mode (strict difficulty, high quality)
  python dataset_generation/scripts/quality_filter.py \
      --mode sft --target 2000

  # RL mode (broader difficulty, more samples)
  python dataset_generation/scripts/quality_filter.py \
      --mode rl --target 5000

  # Quick test (first 1000 rows per dataset)
  python dataset_generation/scripts/quality_filter.py \
      --target 500 --max-rows 1000

  # Profile only (show stats at each stage, no output)
  python dataset_generation/scripts/quality_filter.py --profile
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Sample model
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    uid: str
    source: str
    problem: str
    solution: str
    answer: Optional[str] = None
    pass_rate: Optional[float] = None
    n_tool_calls: int = 0
    topic: Optional[str] = None
    quality_score: float = 0.0
    difficulty_score: float = 0.0
    interaction_score: float = 0.0
    structure_score: float = 0.0
    combined_score: float = 0.0
    iw_weight: float = 1.0
    filter_stage: str = "loaded"
    filter_reason: str = ""


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/downloads")


def load_all(datasets: list[str], max_rows: int = 0) -> list[Sample]:
    """Load requested datasets into unified Sample format."""
    all_samples = []

    if "aimo3_hard" in datasets:
        path = DATA_DIR / "aimo3_hard" / "AIMO3-High-Difficulty-Tool-Calling-Dataset.jsonl"
        if path.exists():
            print(f"  Loading aimo3_hard from {path}...")
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
                    all_samples.append(Sample(
                        uid=f"hard_{i}",
                        source="aimo3_hard",
                        problem=meta.get("problem", ""),
                        solution=text,
                        answer=str(meta.get("standard", "")),
                        pass_rate=passed / count if count > 0 else None,
                        n_tool_calls=n_tool,
                    ))
            print(f"    Loaded {sum(1 for s in all_samples if s.source == 'aimo3_hard'):,}")

    if "aimo3_tir" in datasets:
        path = DATA_DIR / "aimo3_tir" / "data.csv"
        if path.exists():
            print(f"  Loading aimo3_tir from {path}...")
            count = 0
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if max_rows and i >= max_rows:
                        break
                    comp = row.get("completion", "")
                    n_tool = comp.count("<|start|>python")
                    if n_tool == 0:
                        n_tool = comp.count("```python")
                    all_samples.append(Sample(
                        uid=f"tir_{i}",
                        source="aimo3_tir",
                        problem=row.get("prompt", ""),
                        solution=comp,
                        n_tool_calls=n_tool,
                    ))
                    count += 1
            print(f"    Loaded {count:,}")

    if "limo" in datasets:
        path = DATA_DIR / "LIMO" / "limo.jsonl"
        if path.exists():
            print(f"  Loading limo from {path}...")
            count = 0
            with open(path) as f:
                for i, line in enumerate(f):
                    if max_rows and i >= max_rows:
                        break
                    d = json.loads(line)
                    all_samples.append(Sample(
                        uid=f"limo_{i}",
                        source="limo",
                        problem=d.get("question", ""),
                        solution=d.get("solution", ""),
                        answer=str(d.get("answer", "")),
                        pass_rate=0.05,
                    ))
                    count += 1
            print(f"    Loaded {count:,}")

    if "s1k" in datasets:
        path = DATA_DIR / "s1K" / "data" / "train-00000-of-00001.parquet"
        if path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(path)
                print(f"  Loading s1k from {path}...")
                count = 0
                for i, row in df.iterrows():
                    if max_rows and i >= max_rows:
                        break
                    cot = str(row.get("cot", "") or "")
                    sol = str(row.get("solution", "") or "")
                    text = cot if len(cot) > len(sol) else sol
                    all_samples.append(Sample(
                        uid=f"s1k_{i}",
                        source="s1k",
                        problem=str(row.get("question", "")),
                        solution=text,
                        answer=str(row.get("solution", "")),
                        pass_rate=0.10,
                    ))
                    count += 1
                print(f"    Loaded {count:,}")
            except ImportError:
                print("  Warning: pandas not available, skipping s1k")

    return all_samples


# ---------------------------------------------------------------------------
# Stage 1: Hard filters
# ---------------------------------------------------------------------------

def hard_filter(s: Sample) -> tuple[bool, str]:
    """Binary filters. Returns (passed, reason)."""
    text = s.solution

    # Length checks
    if len(text) < 500:
        return False, "too_short"
    if len(text) > 100_000:
        return False, "too_long"

    # Answer marker
    if "boxed{" not in text and "boxed" not in text:
        if "final answer" not in text.lower() and "the answer is" not in text.lower():
            return False, "no_answer_marker"

    # Repetition (80+ char repeated substring, no backtracking regex)
    if len(text) > 300:
        step = max(len(text) // 20, 80)
        for i in range(0, len(text) - 160, step):
            chunk = text[i:i + 80]
            if text.find(chunk, i + 80) != -1:
                return False, "repetitive"

    # TIR-specific: for aimo3_hard, require tool usage (it's a tool-calling dataset)
    # aimo3_tir is mostly CoT without tools, so don't require tool calls
    if s.source == "aimo3_hard":
        if s.n_tool_calls < 1:
            return False, "no_tool_calls"
        if s.n_tool_calls > 20:
            return False, "too_many_tool_calls"

    return True, ""


# ---------------------------------------------------------------------------
# Stage 2: Decontamination
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\\[a-z]+\{", "", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_ngram_set(texts: list[str], n: int = 9) -> set:
    """Build word-level n-gram set from evaluation texts.

    Uses word-level n-grams (not character-level) to avoid false positives
    from common math phrases like 'determine the number of'.
    """
    ngrams = set()
    for text in texts:
        norm = normalize_text(text)
        words = norm.split()
        for i in range(len(words) - n + 1):
            ngrams.add(" ".join(words[i:i + n]))
    return ngrams


def load_eval_problems() -> list[str]:
    """Load evaluation problem texts for decontamination."""
    eval_texts = []
    for fpath in [
        "data/aime_test_2023_2024.csv",
        "data/aime_train_2005_2022.csv",
    ]:
        if os.path.exists(fpath):
            with open(fpath, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    p = row.get("problem", row.get("question", ""))
                    if p:
                        eval_texts.append(p)

    imo_path = "data/downloads/aimo3_hard/imo-benchmark.csv"
    if os.path.exists(imo_path):
        with open(imo_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("problem", row.get("question", ""))
                if p:
                    eval_texts.append(p)

    return eval_texts


def is_contaminated(problem: str, eval_ngrams: set, n: int = 9, threshold: float = 0.30) -> bool:
    """Check if problem is contaminated using word-level n-gram overlap.

    Threshold of 0.30 (30% word n-gram overlap) catches actual duplicates
    while avoiding false positives from common math vocabulary.
    """
    norm = normalize_text(problem)
    words = norm.split()
    if len(words) < n:
        return False
    sample_ngrams = set(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    if not sample_ngrams:
        return False
    overlap = len(sample_ngrams & eval_ngrams) / len(sample_ngrams)
    return overlap >= threshold


# ---------------------------------------------------------------------------
# Stage 3: Difficulty calibration
# ---------------------------------------------------------------------------

def difficulty_filter(s: Sample, min_pr: float, max_pr: float) -> tuple[bool, str]:
    if s.pass_rate is None:
        return True, "no_pass_rate"
    if s.pass_rate < min_pr:
        return False, f"too_hard_{s.pass_rate:.3f}"
    if s.pass_rate > max_pr:
        return False, f"too_easy_{s.pass_rate:.3f}"
    return True, ""


# ---------------------------------------------------------------------------
# Stage 4: Quality scoring
# ---------------------------------------------------------------------------

def score_quality(s: Sample) -> dict[str, float]:
    """Score using LIMO-style quality + ASTER interaction density."""
    text = s.solution
    length = len(text)
    scores = {}

    # Difficulty (25%)
    pr = s.pass_rate
    if pr is None:
        scores["difficulty"] = 0.5
    elif pr <= 0.0:
        scores["difficulty"] = 0.0
    elif pr <= 0.03:
        scores["difficulty"] = 0.3
    elif pr <= 0.15:
        scores["difficulty"] = 1.0
    elif pr <= 0.30:
        scores["difficulty"] = 0.7
    elif pr <= 0.50:
        scores["difficulty"] = 0.4
    else:
        scores["difficulty"] = 0.1

    # Solution quality (35%)
    length_score = min(length / 30000, 1.0)
    if length > 80000:
        length_score = 0.0

    verify_words = ["check", "verify", "confirm", "let's test", "validate", "double-check"]
    verify_count = sum(text.lower().count(w) for w in verify_words)
    verify_score = min(verify_count / max(length / 5000, 1), 1.0)

    explore_words = ["perhaps", "might", "alternatively", "another approach", "let me try", "consider"]
    explore_count = sum(text.lower().count(w) for w in explore_words)
    explore_score = min(explore_count / max(length / 5000, 1), 1.0)

    logic_words = ["therefore", "since", "hence", "thus", "because", "it follows", "we conclude"]
    logic_count = sum(text.lower().count(w) for w in logic_words)
    logic_score = min(logic_count / max(length / 3000, 1), 1.0)

    scores["quality"] = (
        0.23 * length_score
        + 0.26 * verify_score
        + 0.26 * explore_score
        + 0.25 * logic_score
    )

    # Interaction density (20%) - ASTER finding
    tc = s.n_tool_calls
    if tc == 0:
        scores["interaction"] = 0.2  # CoT only
    elif tc <= 2:
        scores["interaction"] = 0.3
    elif tc <= 5:
        scores["interaction"] = 0.6
    elif tc <= 9:
        scores["interaction"] = 0.9
    elif tc <= 15:
        scores["interaction"] = 1.0  # ASTER sweet spot
    else:
        scores["interaction"] = 0.4

    # Structure (20%)
    has_steps = bool(re.search(r"step\s*\d|^\d+[\.\)]", text, re.MULTILINE | re.IGNORECASE))
    has_boxed = "boxed{" in text
    uses_sympy = "sympy" in text.lower() or "from sympy" in text
    # Check for repeated 60-char substrings (avoiding catastrophic backtracking regex)
    has_repetition = False
    if length > 200:
        step = max(length // 20, 60)
        for i in range(0, length - 120, step):
            chunk = text[i:i + 60]
            if text.find(chunk, i + 60) != -1:
                has_repetition = True
                break

    scores["structure"] = (
        0.20 * float(has_steps)
        + 0.30 * float(has_boxed)
        + 0.30 * float(uses_sympy)
        + 0.20 * float(not has_repetition)
    )

    scores["combined"] = (
        0.25 * scores["difficulty"]
        + 0.35 * scores["quality"]
        + 0.20 * scores["interaction"]
        + 0.20 * scores["structure"]
    )

    return scores


# ---------------------------------------------------------------------------
# Stage 5: Diversity selection
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


def classify_topic(problem: str) -> str:
    text = problem.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = sum(text.count(kw) for kw in keywords)
    if max(scores.values()) == 0:
        return "other"
    return max(scores, key=scores.get)


def diversity_select(samples: list[Sample], target: int) -> list[Sample]:
    """Topic-balanced, quality-weighted selection."""
    for s in samples:
        if s.topic is None:
            s.topic = classify_topic(s.problem)

    by_topic = defaultdict(list)
    for s in samples:
        by_topic[s.topic].append(s)

    for topic in by_topic:
        by_topic[topic].sort(key=lambda s: s.combined_score, reverse=True)

    topics = sorted(by_topic.keys())
    selected = []

    # Ensure minimum per topic
    min_per_topic = max(target // (len(topics) * 2), 10)
    for topic in topics:
        avail = by_topic[topic]
        take = min(min_per_topic, len(avail))
        selected.extend(avail[:take])

    # Fill remaining by global score
    remaining = target - len(selected)
    if remaining > 0:
        selected_uids = {s.uid for s in selected}
        candidates = [(s.combined_score, s) for s in samples if s.uid not in selected_uids]
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected.extend([s for _, s in candidates[:remaining]])

    return selected[:target]


# ---------------------------------------------------------------------------
# Importance weighting
# ---------------------------------------------------------------------------

def compute_iw_weights(samples: list[Sample]) -> list[Sample]:
    """Prop2Diff-style importance weights."""
    for s in samples:
        if s.pass_rate is not None and s.pass_rate > 0:
            raw = 1.0 - s.pass_rate
        else:
            raw = 0.5
        raw *= (0.5 + 0.5 * s.combined_score)
        s.iw_weight = raw

    mean_w = sum(s.iw_weight for s in samples) / len(samples) if samples else 1.0
    for s in samples:
        s.iw_weight = s.iw_weight / mean_w if mean_w > 0 else 1.0

    return samples


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

MODE_PRESETS = {
    "sft": {"min_pass_rate": 0.03, "max_pass_rate": 0.30, "target": 3000},
    "rl": {"min_pass_rate": 0.10, "max_pass_rate": 0.70, "target": 5000},
}


def run_pipeline(args):
    """Execute the full filtering pipeline."""
    # Apply mode preset if specified
    if args.mode:
        preset = MODE_PRESETS[args.mode]
        if not args.min_pass_rate:
            args.min_pass_rate = preset["min_pass_rate"]
        if not args.max_pass_rate:
            args.max_pass_rate = preset["max_pass_rate"]
        if not args.target:
            args.target = preset["target"]

    # Defaults
    min_pr = args.min_pass_rate or 0.03
    max_pr = args.max_pass_rate or 0.50
    target = args.target or 3000

    datasets = args.datasets or ["aimo3_hard", "aimo3_tir", "limo", "s1k"]

    print(f"Pipeline: target={target}, difficulty={min_pr:.2f}-{max_pr:.2f}")
    print(f"Datasets: {', '.join(datasets)}")
    if args.iw_sft:
        print("iw-SFT weights: enabled")
    print()

    # Stage 0: Load
    samples = load_all(datasets, max_rows=args.max_rows)
    total = len(samples)
    print(f"\nSTAGE 0 - LOADED: {total:,} samples\n")

    if args.profile:
        _print_stage_stats("Loaded", samples)

    # Stage 1: Hard filters
    passed = []
    reasons = Counter()
    for s in samples:
        ok, reason = hard_filter(s)
        if ok:
            s.filter_stage = "hard_filter"
            passed.append(s)
        else:
            reasons[reason] += 1

    print(f"STAGE 1 - HARD FILTER: {len(passed):,}/{total:,} "
          f"({100 * len(passed) / total:.1f}%)")
    for r, c in reasons.most_common():
        print(f"  Rejected: {r} = {c:,}")
    samples = passed
    print()

    if args.profile:
        _print_stage_stats("After hard filter", samples)

    # Stage 2: Decontamination
    eval_texts = load_eval_problems()
    if eval_texts:
        eval_ngrams = build_ngram_set(eval_texts, n=9)
        passed = []
        removed = 0
        for s in samples:
            if is_contaminated(s.problem, eval_ngrams, n=9):
                removed += 1
            else:
                s.filter_stage = "decontaminated"
                passed.append(s)
        print(f"STAGE 2 - DECONTAMINATION: {len(passed):,}/{len(samples):,} "
              f"({removed:,} removed)")
        samples = passed
    else:
        print("STAGE 2 - DECONTAMINATION: skipped (no eval sets found)")
    print()

    # Stage 3: Difficulty
    passed = []
    too_hard = too_easy = 0
    for s in samples:
        ok, reason = difficulty_filter(s, min_pr, max_pr)
        if ok:
            s.filter_stage = "difficulty"
            passed.append(s)
        elif "too_hard" in reason:
            too_hard += 1
        elif "too_easy" in reason:
            too_easy += 1

    no_pr = sum(1 for s in passed if s.pass_rate is None)
    print(f"STAGE 3 - DIFFICULTY: {len(passed):,}/{len(samples):,}")
    print(f"  Too hard (<{min_pr}): {too_hard:,}  Too easy (>{max_pr}): {too_easy:,}  "
          f"No pass_rate (kept): {no_pr:,}")
    samples = passed
    print()

    # Stage 4: Quality scoring
    for s in samples:
        scores = score_quality(s)
        s.difficulty_score = scores["difficulty"]
        s.quality_score = scores["quality"]
        s.interaction_score = scores["interaction"]
        s.structure_score = scores["structure"]
        s.combined_score = scores["combined"]
        s.filter_stage = "scored"

    samples.sort(key=lambda s: s.combined_score, reverse=True)
    all_scores = [s.combined_score for s in samples]
    print(f"STAGE 4 - QUALITY SCORING: {len(samples):,} scored")
    if all_scores:
        print(f"  Score range: {min(all_scores):.3f} - {max(all_scores):.3f}")
        print(f"  Mean: {sum(all_scores) / len(all_scores):.3f}  "
              f"Median: {all_scores[len(all_scores) // 2]:.3f}")

    by_ds = defaultdict(list)
    for s in samples:
        by_ds[s.source].append(s.combined_score)
    for ds, ds_scores in sorted(by_ds.items()):
        print(f"  {ds}: mean={sum(ds_scores) / len(ds_scores):.3f}, n={len(ds_scores):,}")
    print()

    if args.profile:
        _print_stage_stats("After scoring", samples)

    # Classify topics (needed for diversity selection and output)
    for s in samples:
        if s.topic is None:
            s.topic = classify_topic(s.problem)

    # Stage 5: Diversity selection
    if len(samples) > target:
        selected = diversity_select(samples, target)
    else:
        selected = samples
        print(f"STAGE 5: Only {len(samples):,} available (target {target:,}), keeping all")

    print(f"STAGE 5 - DIVERSITY SELECTION: {len(selected):,}")
    topics = Counter(s.topic for s in selected)
    for t, c in topics.most_common():
        print(f"  {t}: {c:,} ({100 * c / len(selected):.1f}%)")
    print()

    # Optional: iw-SFT weights
    if args.iw_sft:
        selected = compute_iw_weights(selected)
        weights = [s.iw_weight for s in selected]
        print(f"iw-SFT WEIGHTS: min={min(weights):.3f} mean={sum(weights) / len(weights):.3f} "
              f"max={max(weights):.3f}")
        print()

    # Save output
    output_path = args.output or f"dataset_generation/output/curated_{len(selected)}.jsonl"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        for s in selected:
            record = {
                "uid": s.uid,
                "source": s.source,
                "problem_text": s.problem,
                "solution_text": s.solution,
                "answer": s.answer,
                "pass_rate": s.pass_rate,
                "n_tool_calls": s.n_tool_calls,
                "topic": s.topic,
                "combined_score": round(s.combined_score, 4),
                "difficulty_score": round(s.difficulty_score, 4),
                "quality_score": round(s.quality_score, 4),
                "interaction_score": round(s.interaction_score, 4),
                "structure_score": round(s.structure_score, 4),
            }
            if args.iw_sft:
                record["iw_weight"] = round(s.iw_weight, 4)
            f.write(json.dumps(record) + "\n")

    print(f"{'=' * 60}")
    print(f"SAVED: {len(selected):,} samples to {output_path}")
    print(f"{'=' * 60}")

    # Final summary
    by_ds = Counter(s.source for s in selected)
    by_topic = Counter(s.topic for s in selected)
    print(f"\nBy dataset: {dict(by_ds)}")
    print(f"By topic: {dict(by_topic)}")

    return selected


def _print_stage_stats(label: str, samples: list[Sample]):
    """Print brief stats at each pipeline stage."""
    if not samples:
        return
    by_ds = Counter(s.source for s in samples)
    lengths = sorted(len(s.solution) for s in samples)
    n = len(lengths)
    print(f"  [{label}] n={n:,}  "
          f"len: p50={lengths[n // 2]:,}  p95={lengths[int(n * 0.95)]:,}")
    print(f"    Sources: " + ", ".join(f"{k}={v:,}" for k, v in by_ds.most_common()))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AIMO3 Quality Filtering Pipeline")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Datasets to include (default: aimo3_hard aimo3_tir limo s1k)")
    parser.add_argument("--mode", type=str, choices=["sft", "rl"], default=None,
                        help="Preset mode (sets difficulty range and target)")
    parser.add_argument("--target", type=int, default=None,
                        help="Target number of samples")
    parser.add_argument("--min-pass-rate", type=float, default=None,
                        help="Minimum pass rate for difficulty filter")
    parser.add_argument("--max-pass-rate", type=float, default=None,
                        help="Maximum pass rate for difficulty filter")
    parser.add_argument("--iw-sft", action="store_true",
                        help="Compute importance weights for iw-SFT")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Max rows per dataset (0=all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path")
    parser.add_argument("--profile", action="store_true",
                        help="Print detailed stats at each stage")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
