#!/usr/bin/env python3
"""
Dataset Curation Script for AIMO3 Fine-Tuning

6-stage pipeline to select high-quality training samples:
  1. Load & normalize all datasets into unified format
  2. Hard filters (length, tool calls, boxed answer, no repetition)
  3. Decontamination (9-gram vs eval sets)
  4. Difficulty calibration (pass_rate filtering)
  5. Quality scoring (LIMO-style + interaction density)
  6. Diversity selection (topic-balanced, target size)

Usage:
  # Profile all datasets (no filtering, just stats)
  python scripts/curate_dataset.py profile

  # Run full pipeline, select 3000 samples
  python scripts/curate_dataset.py select --target 3000

  # Run full pipeline with custom difficulty range
  python scripts/curate_dataset.py select --target 4000 --min-pass-rate 0.03 --max-pass-rate 0.30

  # Run with importance weighting output (for iw-SFT)
  python scripts/curate_dataset.py select --target 3000 --iw-sft

  # Only run stages 1-2 (load + hard filter) to see retention
  python scripts/curate_dataset.py select --target 3000 --stop-after hard_filter
"""

import argparse
import csv
import json
import hashlib
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """Unified sample format across all datasets."""
    uid: str                          # unique id
    source_dataset: str               # aimo3_tir | aimo3_hard | limo | s1k
    problem_text: str                 # the math problem
    solution_text: str                # full solution / completion
    answer: Optional[str] = None      # ground truth answer
    pass_rate: Optional[float] = None # fraction correct (from metadata or estimated)
    n_tool_calls: int = 0             # number of code execution blocks
    topic: Optional[str] = None       # algebra, number_theory, combinatorics, geometry, etc.
    problem_id: Optional[str] = None  # original problem id

    # Computed during scoring
    quality_score: float = 0.0
    difficulty_score: float = 0.0
    interaction_score: float = 0.0
    structure_score: float = 0.0
    combined_score: float = 0.0
    iw_weight: float = 1.0           # importance weight for iw-SFT

    # Filter tracking
    filter_stage: str = "loaded"      # which stage this sample passed/failed
    filter_reason: str = ""

# ---------------------------------------------------------------------------
# Stage 0: Loaders
# ---------------------------------------------------------------------------

def load_aimo3_hard(path: str) -> list[Sample]:
    """Load AIMO3 High-Difficulty Tool-Calling Dataset (JSONL, Harmony format)."""
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            meta = d.get("metadata_infos", {})
            rht = meta.get("reason_high_with_tool", {})

            problem = meta.get("problem", "")
            answer = str(meta.get("standard", ""))
            text = d.get("text", "")

            # pass_rate from metadata
            count = rht.get("count", 8)
            passed = rht.get("pass", 0)
            pass_rate = passed / count if count > 0 else None

            # Count tool calls (Harmony format: <|start|>python segments)
            n_tool = text.count("<|start|>python")
            if n_tool == 0:
                # Fallback: count ```python blocks
                n_tool = text.count("```python")

            samples.append(Sample(
                uid=f"hard_{i}",
                source_dataset="aimo3_hard",
                problem_text=problem,
                solution_text=text,
                answer=answer,
                pass_rate=pass_rate,
                n_tool_calls=n_tool,
                problem_id=str(i),
            ))
    return samples


def load_aimo3_tir(path: str, max_rows: int = 0) -> list[Sample]:
    """Load AIMO3 TIR dataset (CSV, Harmony format)."""
    samples = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            comp = row.get("completion", "")

            # Count tool calls (Harmony format: <|start|>python segments)
            n_tool = comp.count("<|start|>python")
            if n_tool == 0:
                n_tool = comp.count("```python")

            samples.append(Sample(
                uid=f"tir_{i}",
                source_dataset="aimo3_tir",
                problem_text=row.get("prompt", ""),
                solution_text=comp,
                answer=None,  # TIR dataset doesn't have ground truth in CSV
                pass_rate=None,  # No pass_rate metadata
                n_tool_calls=n_tool,
                problem_id=row.get("problem_id", str(i)),
            ))
    return samples


def load_limo(path: str) -> list[Sample]:
    """Load LIMO dataset (JSONL)."""
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            solution = d.get("solution", "")
            samples.append(Sample(
                uid=f"limo_{i}",
                source_dataset="limo",
                problem_text=d.get("question", ""),
                solution_text=solution,
                answer=str(d.get("answer", "")),
                pass_rate=0.05,  # LIMO is curated for ~3-9% pass rate
                n_tool_calls=0,  # LIMO is CoT, not TIR
                problem_id=str(i),
            ))
    return samples


def load_s1k(path: str) -> list[Sample]:
    """Load s1K dataset (Parquet)."""
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not available, skipping s1K")
        return []

    df = pd.read_parquet(path)
    samples = []
    for i, row in df.iterrows():
        cot = row.get("cot", "") or ""
        solution = row.get("solution", "") or ""
        text = cot if len(cot) > len(solution) else solution

        samples.append(Sample(
            uid=f"s1k_{i}",
            source_dataset="s1k",
            problem_text=str(row.get("question", "")),
            solution_text=text,
            answer=str(row.get("solution", "")),
            pass_rate=0.10,  # s1 curated for hard problems
            n_tool_calls=0,  # s1K is CoT
            problem_id=str(i),
        ))
    return samples


def load_genselect(data_dir: str, max_rows: int = 0) -> list[Sample]:
    """Load OpenMathReasoning GenSelect dataset (multiple Parquet shards)."""
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not available, skipping GenSelect")
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
            problem = str(row.get("problem", ""))

            # Parse pass_rate from pass_rate_72b_tir
            pr_str = str(row.get("pass_rate_72b_tir", "n/a"))
            try:
                pass_rate = float(pr_str) if pr_str != "n/a" else None
            except (ValueError, TypeError):
                pass_rate = None

            samples.append(Sample(
                uid=f"genselect_{total}",
                source_dataset="genselect",
                problem_text=problem,
                solution_text=solution,
                answer=str(row.get("expected_answer", "")),
                pass_rate=pass_rate,
                n_tool_calls=0,  # GenSelect is answer selection, not TIR
                problem_id=str(total),
            ))
            total += 1
        if max_rows and total >= max_rows:
            break
    return samples


def load_bigmath(path: str, max_rows: int = 0) -> list[Sample]:
    """Load Big-Math-RL-Verified dataset (Parquet)."""
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not available, skipping Big-Math")
        return []

    df = pd.read_parquet(path)
    samples = []
    for i, row in df.iterrows():
        if max_rows and i >= max_rows:
            break
        problem = str(row.get("problem", row.get("question", "")))
        answer = str(row.get("answer", row.get("expected_answer", "")))

        samples.append(Sample(
            uid=f"bigmath_{i}",
            source_dataset="bigmath",
            problem_text=problem,
            solution_text=problem,  # Big-Math is problems only (for RL)
            answer=answer,
            pass_rate=None,
            n_tool_calls=0,
            problem_id=str(i),
        ))
    return samples


def _has_repetition(text: str, block_size: int = 80) -> bool:
    """Fast repetition check using hashing (O(n) instead of O(n^2) regex)."""
    if len(text) < block_size * 2:
        return False
    # Check if any block_size substring appears again within the next block_size*3 chars
    seen = {}
    step = block_size // 2
    for i in range(0, len(text) - block_size, step):
        chunk = text[i:i + block_size]
        h = hash(chunk)
        if h in seen:
            prev_i = seen[h]
            # Verify (hash collision check) and ensure it's a real repeat
            if text[prev_i:prev_i + block_size] == chunk and i - prev_i >= block_size:
                return True
        seen[h] = i
    return False


# ---------------------------------------------------------------------------
# Stage 1: Hard Filters
# ---------------------------------------------------------------------------

def hard_filter(sample: Sample) -> tuple[bool, str]:
    """Binary filters — must pass ALL."""
    text = sample.solution_text

    # Big-Math is problems-only (for RL), skip solution filters
    if sample.source_dataset == "bigmath":
        if len(sample.problem_text) < 20:
            return False, "too_short"
        return True, ""

    if len(text) < 500:
        return False, "too_short"
    if len(text) > 100_000:
        return False, "too_long"

    # For aimo3_hard, require at least 1 tool call (TIR traces)
    # aimo3_tir has mixed format — some pure CoT, some with tool calls
    if sample.source_dataset == "aimo3_hard":
        if sample.n_tool_calls < 1:
            return False, "no_tool_calls"
        if sample.n_tool_calls > 20:
            return False, "too_many_tool_calls"

    # Check for boxed answer in solution
    if "boxed{" not in text and "boxed" not in text:
        # Some formats use different answer markers
        if "final answer" not in text.lower() and "the answer is" not in text.lower():
            return False, "no_answer_marker"

    # Repetition check — use sliding window instead of catastrophic regex
    if _has_repetition(text, block_size=80):
        return False, "repetitive"

    return True, ""


# ---------------------------------------------------------------------------
# Stage 2: Decontamination
# ---------------------------------------------------------------------------

def build_ngram_set(texts: list[str], n: int = 9) -> set:
    """Build n-gram set from evaluation texts."""
    ngrams = set()
    for text in texts:
        normalized = normalize_text(text)
        for i in range(len(normalized) - n + 1):
            ngrams.add(normalized[i:i + n])
    return ngrams


def normalize_text(text: str) -> str:
    """Normalize text for n-gram matching."""
    text = text.lower()
    text = re.sub(r"\\[a-z]+\{", "", text)  # strip LaTeX commands
    text = re.sub(r"[^a-z0-9 ]", "", text)   # keep only alphanum + space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def check_contamination(sample: Sample, eval_ngrams: set, n: int = 9) -> tuple[bool, float]:
    """Check if sample overlaps with evaluation set. Returns (is_clean, overlap_ratio)."""
    normalized = normalize_text(sample.problem_text)
    if len(normalized) < n:
        return True, 0.0

    sample_ngrams = set()
    for i in range(len(normalized) - n + 1):
        sample_ngrams.add(normalized[i:i + n])

    if not sample_ngrams:
        return True, 0.0

    overlap = len(sample_ngrams & eval_ngrams) / len(sample_ngrams)
    return overlap < 0.10, overlap


def load_eval_problems() -> list[str]:
    """Load evaluation problem texts for decontamination."""
    eval_texts = []

    # AIME 2023-2024 (from our CSV files)
    aime_files = [
        "data/aime_test_2023_2024.csv",
        "data/aime_train_2005_2022.csv",  # exclude later years
    ]
    for fpath in aime_files:
        if os.path.exists(fpath):
            with open(fpath, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    problem = row.get("problem", row.get("question", ""))
                    if problem:
                        eval_texts.append(problem)

    # IMO benchmark from aimo3_hard
    imo_path = "data/downloads/aimo3_hard/imo-benchmark.csv"
    if os.path.exists(imo_path):
        with open(imo_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                problem = row.get("problem", row.get("question", ""))
                if problem:
                    eval_texts.append(problem)

    print(f"  Loaded {len(eval_texts)} evaluation problems for decontamination")
    return eval_texts


# ---------------------------------------------------------------------------
# Stage 3: Difficulty Calibration
# ---------------------------------------------------------------------------

def difficulty_filter(sample: Sample, min_pass_rate: float, max_pass_rate: float) -> tuple[bool, str]:
    """Filter by difficulty (pass_rate range)."""
    if sample.pass_rate is None:
        # No pass_rate metadata — keep but flag
        return True, "no_pass_rate"

    if sample.pass_rate < min_pass_rate:
        return False, f"too_hard_{sample.pass_rate:.3f}"
    if sample.pass_rate > max_pass_rate:
        return False, f"too_easy_{sample.pass_rate:.3f}"

    return True, ""


# ---------------------------------------------------------------------------
# Stage 4: Quality Scoring
# ---------------------------------------------------------------------------

def score_quality(sample: Sample) -> dict[str, float]:
    """Score a sample using LIMO-style quality metrics + interaction density."""
    text = sample.solution_text
    length = len(text)
    scores = {}

    # --- Difficulty component (25%) ---
    pr = sample.pass_rate
    if pr is None:
        scores["difficulty"] = 0.5  # Unknown = neutral
    elif pr <= 0.0:
        scores["difficulty"] = 0.0
    elif pr <= 0.03:
        scores["difficulty"] = 0.3
    elif pr <= 0.15:
        scores["difficulty"] = 1.0  # LIMO sweet spot
    elif pr <= 0.30:
        scores["difficulty"] = 0.7  # s1 range
    elif pr <= 0.50:
        scores["difficulty"] = 0.4
    else:
        scores["difficulty"] = 0.1

    # --- Solution quality (35%) ---
    # Length sub-score
    length_score = min(length / 30000, 1.0)
    if length > 80000:
        length_score = 0.0

    # Verification keywords
    verify_words = ["check", "verify", "confirm", "let's test", "validate", "double-check"]
    verify_count = sum(text.lower().count(w) for w in verify_words)
    verify_score = min(verify_count / max(length / 5000, 1), 1.0)

    # Exploration keywords
    explore_words = ["perhaps", "might", "alternatively", "another approach", "let me try", "consider"]
    explore_count = sum(text.lower().count(w) for w in explore_words)
    explore_score = min(explore_count / max(length / 5000, 1), 1.0)

    # Logical connectives
    logic_words = ["therefore", "since", "hence", "thus", "because", "it follows", "we conclude"]
    logic_count = sum(text.lower().count(w) for w in logic_words)
    logic_score = min(logic_count / max(length / 3000, 1), 1.0)

    scores["quality"] = (
        0.23 * length_score
        + 0.26 * verify_score
        + 0.26 * explore_score
        + 0.25 * logic_score
    )

    # --- Interaction density (20%) ---
    tc = sample.n_tool_calls
    if tc == 0:
        scores["interaction"] = 0.2  # CoT-only (LIMO, s1K)
    elif tc <= 2:
        scores["interaction"] = 0.3
    elif tc <= 5:
        scores["interaction"] = 0.6
    elif tc <= 9:
        scores["interaction"] = 0.9
    elif tc <= 15:
        scores["interaction"] = 1.0  # ASTER sweet spot
    else:
        scores["interaction"] = 0.4  # Spinning

    # --- Structure (20%) ---
    has_steps = bool(re.search(r"step\s*\d|^\d+[\.\)]", text, re.MULTILINE | re.IGNORECASE))
    has_boxed = "boxed{" in text
    uses_sympy = "sympy" in text.lower() or "from sympy" in text
    has_repetition = _has_repetition(text, block_size=60)

    scores["structure"] = (
        0.20 * float(has_steps)
        + 0.30 * float(has_boxed)
        + 0.30 * float(uses_sympy)
        + 0.20 * float(not has_repetition)
    )

    # --- Combined ---
    scores["combined"] = (
        0.25 * scores["difficulty"]
        + 0.35 * scores["quality"]
        + 0.20 * scores["interaction"]
        + 0.20 * scores["structure"]
    )

    return scores


# ---------------------------------------------------------------------------
# Stage 5: Diversity Selection
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS = {
    # Number Theory (split into sub-areas)
    "nt_primes": ["prime", "primality", "sieve", "mersenne", "twin prime", "prime factorization"],
    "nt_divisibility": ["divisor", "divisible", "gcd", "lcm", "coprime", "totient", "euler phi",
                        "greatest common", "least common"],
    "nt_modular": ["modular", "congruence", "modulo", "remainder", "residue", "fermat's little",
                   "chinese remainder", "quadratic residue"],
    "nt_diophantine": ["diophantine", "integer solution", "pell", "pythagorean triple"],

    # Algebra (split into sub-areas)
    "alg_polynomial": ["polynomial", "root", "coefficient", "quadratic", "cubic", "degree",
                       "vieta", "factor theorem", "rational root"],
    "alg_inequality": ["inequality", "am-gm", "cauchy-schwarz", "jensen", "schur",
                       "rearrangement", "power mean", "weighted mean"],
    "alg_equations": ["equation", "solve for", "system of equations", "linear system",
                      "functional equation"],
    "alg_sequences": ["sequence", "series", "recurrence", "arithmetic progression",
                      "geometric progression", "fibonacci", "telescoping", "generating function"],

    # Combinatorics (split into sub-areas)
    "comb_counting": ["permutation", "combination", "counting", "choose", "arrange",
                      "derangement", "stars and bars", "inclusion-exclusion", "overcounting"],
    "comb_probability": ["probability", "expected value", "random", "dice", "coin",
                         "independent", "conditional", "bayes"],
    "comb_graph": ["graph", "vertex", "edge", "tree", "path", "cycle", "coloring",
                   "chromatic", "hamiltonian", "eulerian", "bipartite"],
    "comb_pigeonhole": ["pigeonhole", "partition", "ramsey", "subset", "extremal"],

    # Geometry (split into sub-areas)
    "geo_triangle": ["triangle", "altitude", "median", "incircle", "circumcircle",
                     "orthocenter", "centroid", "angle bisector", "cevian"],
    "geo_circle": ["circle", "tangent", "chord", "arc", "radius", "diameter",
                   "power of a point", "radical axis", "cyclic quadrilateral", "inscribed angle"],
    "geo_polygon": ["polygon", "quadrilateral", "parallelogram", "trapezoid", "rhombus",
                    "regular polygon", "convex", "diagonal"],
    "geo_coordinate": ["coordinate", "distance formula", "slope", "midpoint", "locus",
                       "transformation", "rotation", "reflection", "vector"],
    "geo_3d": ["sphere", "cube", "tetrahedron", "volume", "surface area", "cross section",
               "solid", "cone", "cylinder", "prism"],

    # Analysis / Calculus
    "analysis": ["limit", "derivative", "integral", "continuous", "differentiable",
                 "convergent", "divergent", "infinite series", "taylor", "calculus"],
}


def classify_topic(problem_text: str) -> str:
    """Simple keyword-based topic classification."""
    text = problem_text.lower()
    topic_scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        topic_scores[topic] = sum(text.count(kw) for kw in keywords)

    if max(topic_scores.values()) == 0:
        return "other"
    return max(topic_scores, key=topic_scores.get)


def diversity_select(samples: list[Sample], target: int,
                     diversity_mode: str = "topic") -> list[Sample]:
    """Select target samples with diversity balancing.

    diversity_mode:
      - "topic": balance by fine-grained topic (19 categories)
      - "source": balance by source dataset
      - "difficulty": balance by difficulty bands
      - "multi": combined topic + source + difficulty
    """
    # Classify topics
    for s in samples:
        if s.topic is None:
            s.topic = classify_topic(s.problem_text)

    if diversity_mode == "topic":
        return _select_balanced(samples, target, key=lambda s: s.topic)

    elif diversity_mode == "source":
        return _select_balanced(samples, target, key=lambda s: s.source_dataset)

    elif diversity_mode == "difficulty":
        def diff_band(s):
            if s.pass_rate is None:
                return "unknown"
            elif s.pass_rate <= 0.10:
                return "very_hard"
            elif s.pass_rate <= 0.25:
                return "hard"
            elif s.pass_rate <= 0.40:
                return "medium"
            else:
                return "easy"
        return _select_balanced(samples, target, key=diff_band)

    elif diversity_mode == "multi":
        # Multi-axis: allocate budget across source datasets proportionally,
        # then within each source, balance by topic
        by_source = defaultdict(list)
        for s in samples:
            by_source[s.source_dataset].append(s)

        # Allocate budget: proportional to sqrt(n) to prevent dominant datasets
        source_weights = {src: math.sqrt(len(samps)) for src, samps in by_source.items()}
        total_weight = sum(source_weights.values())

        selected = []
        for src, samps in by_source.items():
            src_target = max(int(target * source_weights[src] / total_weight), 5)
            src_selected = _select_balanced(samps, src_target, key=lambda s: s.topic)
            selected.extend(src_selected)

        # Trim to target (take highest scores if over)
        if len(selected) > target:
            selected.sort(key=lambda s: s.combined_score, reverse=True)
            selected = selected[:target]
        # Fill if under
        elif len(selected) < target:
            selected_uids = {s.uid for s in selected}
            remaining = [s for s in samples if s.uid not in selected_uids]
            remaining.sort(key=lambda s: s.combined_score, reverse=True)
            selected.extend(remaining[:target - len(selected)])

        return selected[:target]

    else:
        return _select_balanced(samples, target, key=lambda s: s.topic)


def _select_balanced(samples: list[Sample], target: int, key) -> list[Sample]:
    """Generic balanced selection by any grouping key."""
    by_group = defaultdict(list)
    for s in samples:
        by_group[key(s)].append(s)

    # Sort each group by combined score (descending)
    for group in by_group:
        by_group[group].sort(key=lambda s: s.combined_score, reverse=True)

    # Balanced round-robin
    groups = sorted(by_group.keys())
    selected = []

    # First pass: ensure minimum per group
    min_per_group = max(target // (len(groups) * 2), 5)
    for group in groups:
        available = by_group[group]
        take = min(min_per_group, len(available))
        selected.extend(available[:take])

    # Second pass: fill remaining by global score
    remaining = target - len(selected)
    if remaining > 0:
        selected_uids = {s.uid for s in selected}
        candidates = [(s.combined_score, s) for s in samples if s.uid not in selected_uids]
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected.extend([s for _, s in candidates[:remaining]])

    return selected[:target]


# ---------------------------------------------------------------------------
# Stage 6: Importance Weighting (for iw-SFT)
# ---------------------------------------------------------------------------

def compute_importance_weights(samples: list[Sample]) -> list[Sample]:
    """Compute importance weights based on difficulty (Prop2Diff style)."""
    # Weight proportional to difficulty
    for s in samples:
        if s.pass_rate is not None and s.pass_rate > 0:
            # Prop2Diff: weight = (1 - pass_rate) = difficulty
            raw_weight = 1.0 - s.pass_rate
        else:
            raw_weight = 0.5  # default for unknown difficulty

        # Also factor in quality score
        raw_weight *= (0.5 + 0.5 * s.combined_score)
        s.iw_weight = raw_weight

    # Normalize weights to mean=1.0
    mean_w = sum(s.iw_weight for s in samples) / len(samples) if samples else 1.0
    for s in samples:
        s.iw_weight = s.iw_weight / mean_w if mean_w > 0 else 1.0

    return samples


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/downloads")


def load_all_datasets(max_tir_rows: int = 0) -> list[Sample]:
    """Load all available datasets."""
    all_samples = []

    # AIMO3 Hard
    hard_path = DATA_DIR / "aimo3_hard" / "AIMO3-High-Difficulty-Tool-Calling-Dataset.jsonl"
    if hard_path.exists():
        print(f"Loading AIMO3 Hard from {hard_path}...")
        samples = load_aimo3_hard(str(hard_path))
        print(f"  Loaded {len(samples)} samples")
        all_samples.extend(samples)

    # AIMO3 TIR
    tir_path = DATA_DIR / "aimo3_tir" / "data.csv"
    if tir_path.exists():
        print(f"Loading AIMO3 TIR from {tir_path}...")
        samples = load_aimo3_tir(str(tir_path), max_rows=max_tir_rows)
        print(f"  Loaded {len(samples)} samples")
        all_samples.extend(samples)

    # LIMO
    limo_path = DATA_DIR / "LIMO" / "limo.jsonl"
    if limo_path.exists():
        print(f"Loading LIMO from {limo_path}...")
        samples = load_limo(str(limo_path))
        print(f"  Loaded {len(samples)} samples")
        all_samples.extend(samples)

    # s1K
    s1k_path = DATA_DIR / "s1K" / "data" / "train-00000-of-00001.parquet"
    if s1k_path.exists():
        print(f"Loading s1K from {s1k_path}...")
        samples = load_s1k(str(s1k_path))
        print(f"  Loaded {len(samples)} samples")
        all_samples.extend(samples)

    # OpenMathReasoning GenSelect
    genselect_dir = DATA_DIR / "OpenMathReasoning-GenSelect" / "data"
    if genselect_dir.exists():
        print(f"Loading GenSelect from {genselect_dir}...")
        samples = load_genselect(str(genselect_dir), max_rows=max_tir_rows)
        print(f"  Loaded {len(samples)} samples")
        all_samples.extend(samples)

    # Big-Math-RL-Verified
    bigmath_path = DATA_DIR / "Big-Math-RL-Verified" / "data" / "train-00000-of-00001.parquet"
    if bigmath_path.exists():
        print(f"Loading Big-Math-RL-Verified from {bigmath_path}...")
        samples = load_bigmath(str(bigmath_path))
        print(f"  Loaded {len(samples)} samples")
        all_samples.extend(samples)

    return all_samples


def run_profile(args):
    """Profile all datasets without filtering."""
    samples = load_all_datasets(max_tir_rows=args.max_tir_rows)
    print(f"\n{'='*60}")
    print(f"TOTAL LOADED: {len(samples)} samples")
    print(f"{'='*60}\n")

    # Per-dataset stats
    by_dataset = defaultdict(list)
    for s in samples:
        by_dataset[s.source_dataset].append(s)

    for ds, ds_samples in sorted(by_dataset.items()):
        lengths = [len(s.solution_text) for s in ds_samples]
        tool_calls = [s.n_tool_calls for s in ds_samples]
        pass_rates = [s.pass_rate for s in ds_samples if s.pass_rate is not None]

        print(f"--- {ds} ({len(ds_samples)} samples) ---")
        print(f"  Solution length: min={min(lengths)}, median={sorted(lengths)[len(lengths)//2]}, max={max(lengths)}")
        print(f"  Tool calls: min={min(tool_calls)}, median={sorted(tool_calls)[len(tool_calls)//2]}, max={max(tool_calls)}")
        if pass_rates:
            print(f"  Pass rate: min={min(pass_rates):.3f}, median={sorted(pass_rates)[len(pass_rates)//2]:.3f}, max={max(pass_rates):.3f}")
            # Distribution
            buckets = Counter()
            for pr in pass_rates:
                if pr <= 0.125:
                    buckets["0-12.5%"] += 1
                elif pr <= 0.25:
                    buckets["12.5-25%"] += 1
                elif pr <= 0.375:
                    buckets["25-37.5%"] += 1
                elif pr <= 0.50:
                    buckets["37.5-50%"] += 1
                elif pr <= 0.625:
                    buckets["50-62.5%"] += 1
                elif pr <= 0.75:
                    buckets["62.5-75%"] += 1
                else:
                    buckets["75-100%"] += 1
            print(f"  Pass rate distribution: {dict(sorted(buckets.items()))}")
        else:
            print(f"  Pass rate: not available")

        # Hard filter preview
        pass_count = sum(1 for s in ds_samples if hard_filter(s)[0])
        print(f"  Would pass hard filter: {pass_count}/{len(ds_samples)} ({100*pass_count/len(ds_samples):.1f}%)")
        print()

    # Quick topic classification
    print("--- Topic Distribution (keyword-based) ---")
    topics = Counter(classify_topic(s.problem_text) for s in samples)
    for topic, count in topics.most_common():
        print(f"  {topic}: {count} ({100*count/len(samples):.1f}%)")


def run_select(args):
    """Run full curation pipeline."""
    target = args.target
    stop_after = args.stop_after

    print(f"Target: {target} samples")
    print(f"Difficulty range: pass_rate {args.min_pass_rate:.2f} - {args.max_pass_rate:.2f}")
    print()

    # Stage 0: Load
    samples = load_all_datasets(max_tir_rows=args.max_tir_rows)
    total_loaded = len(samples)
    print(f"\n{'='*60}")
    print(f"STAGE 0 - LOADED: {total_loaded} samples")
    print(f"{'='*60}\n")

    if stop_after == "load":
        return samples

    # Stage 1: Hard filters
    passed = []
    rejected_reasons = Counter()
    for s in samples:
        ok, reason = hard_filter(s)
        if ok:
            s.filter_stage = "hard_filter"
            passed.append(s)
        else:
            rejected_reasons[reason] += 1

    print(f"STAGE 1 - HARD FILTER: {len(passed)}/{total_loaded} passed ({100*len(passed)/total_loaded:.1f}%)")
    for reason, count in rejected_reasons.most_common():
        print(f"  Rejected: {reason} = {count}")
    print()
    samples = passed

    if stop_after == "hard_filter":
        return samples

    # Stage 2: Decontamination
    eval_texts = load_eval_problems()
    if eval_texts:
        eval_ngrams = build_ngram_set(eval_texts, n=9)
        passed = []
        contaminated = 0
        for s in samples:
            is_clean, overlap = check_contamination(s, eval_ngrams, n=9)
            if is_clean:
                s.filter_stage = "decontaminated"
                passed.append(s)
            else:
                contaminated += 1

        print(f"STAGE 2 - DECONTAMINATION: {len(passed)}/{len(samples)} passed ({contaminated} removed)")
        samples = passed
    else:
        print("STAGE 2 - DECONTAMINATION: skipped (no eval problems found)")
    print()

    if stop_after == "decontaminate":
        return samples

    # Stage 3: Difficulty calibration
    passed = []
    no_passrate = 0
    too_hard = 0
    too_easy = 0
    for s in samples:
        ok, reason = difficulty_filter(s, args.min_pass_rate, args.max_pass_rate)
        if ok:
            s.filter_stage = "difficulty"
            passed.append(s)
        elif "too_hard" in reason:
            too_hard += 1
        elif "too_easy" in reason:
            too_easy += 1

    # Count samples without pass_rate
    no_passrate = sum(1 for s in passed if s.pass_rate is None)
    print(f"STAGE 3 - DIFFICULTY: {len(passed)}/{len(samples)} passed")
    print(f"  Too hard (pass_rate < {args.min_pass_rate}): {too_hard}")
    print(f"  Too easy (pass_rate > {args.max_pass_rate}): {too_easy}")
    print(f"  No pass_rate (kept): {no_passrate}")
    print()
    samples = passed

    if stop_after == "difficulty":
        return samples

    # Stage 4: Quality scoring
    for s in samples:
        scores = score_quality(s)
        s.difficulty_score = scores["difficulty"]
        s.quality_score = scores["quality"]
        s.interaction_score = scores["interaction"]
        s.structure_score = scores["structure"]
        s.combined_score = scores["combined"]
        s.filter_stage = "scored"

    # Sort by combined score
    samples.sort(key=lambda s: s.combined_score, reverse=True)

    # Stats
    scores = [s.combined_score for s in samples]
    print(f"STAGE 4 - QUALITY SCORING: {len(samples)} scored")
    print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
    print(f"  Median: {scores[len(scores)//2]:.3f}")
    print(f"  Mean: {sum(scores)/len(scores):.3f}")

    # Show score breakdown by dataset
    by_ds = defaultdict(list)
    for s in samples:
        by_ds[s.source_dataset].append(s.combined_score)
    for ds, ds_scores in sorted(by_ds.items()):
        print(f"  {ds}: mean={sum(ds_scores)/len(ds_scores):.3f}, n={len(ds_scores)}")
    print()

    if stop_after == "scoring":
        return samples

    # Stage 5: Diversity selection
    diversity_mode = getattr(args, "diversity", "topic")
    if len(samples) > target:
        selected = diversity_select(samples, target, diversity_mode=diversity_mode)
    else:
        selected = samples
        print(f"STAGE 5 - DIVERSITY: Only {len(samples)} available, keeping all (target was {target})")

    print(f"STAGE 5 - DIVERSITY SELECTION ({diversity_mode}): {len(selected)} selected from {len(samples)}")

    # Topic distribution
    topics = Counter(s.topic for s in selected)
    for topic, count in topics.most_common():
        print(f"  {topic}: {count} ({100*count/len(selected):.1f}%)")

    # Source distribution
    sources = Counter(s.source_dataset for s in selected)
    print(f"  --- By source ---")
    for src, count in sources.most_common():
        print(f"  {src}: {count} ({100*count/len(selected):.1f}%)")

    # Difficulty distribution
    diff_bands = Counter()
    for s in selected:
        if s.pass_rate is None:
            diff_bands["unknown"] += 1
        elif s.pass_rate <= 0.10:
            diff_bands["very_hard"] += 1
        elif s.pass_rate <= 0.25:
            diff_bands["hard"] += 1
        elif s.pass_rate <= 0.40:
            diff_bands["medium"] += 1
        else:
            diff_bands["easy"] += 1
    print(f"  --- By difficulty ---")
    for band, count in sorted(diff_bands.items()):
        print(f"  {band}: {count} ({100*count/len(selected):.1f}%)")
    print()

    # Stage 6: Importance weighting (optional)
    if args.iw_sft:
        selected = compute_importance_weights(selected)
        weights = [s.iw_weight for s in selected]
        print(f"STAGE 6 - IMPORTANCE WEIGHTING: applied")
        print(f"  Weight range: {min(weights):.3f} - {max(weights):.3f}")
        print(f"  Mean: {sum(weights)/len(weights):.3f}")
        print()

    # Save output
    output_path = args.output or f"data/curated_{len(selected)}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for s in selected:
            record = {
                "uid": s.uid,
                "source_dataset": s.source_dataset,
                "problem_text": s.problem_text,
                "solution_text": s.solution_text,
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

    print(f"{'='*60}")
    print(f"SAVED: {len(selected)} samples to {output_path}")
    print(f"{'='*60}")

    # Summary stats
    by_ds = Counter(s.source_dataset for s in selected)
    print(f"\nBy dataset: {dict(by_ds)}")
    by_topic = Counter(s.topic for s in selected)
    print(f"By topic: {dict(by_topic)}")
    if any(s.pass_rate is not None for s in selected):
        prs = [s.pass_rate for s in selected if s.pass_rate is not None]
        print(f"Pass rate: mean={sum(prs)/len(prs):.3f}, range={min(prs):.3f}-{max(prs):.3f}")

    return selected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AIMO3 Dataset Curation Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # Profile
    p_profile = subparsers.add_parser("profile", help="Profile all datasets")
    p_profile.add_argument("--max-tir-rows", type=int, default=10000,
                           help="Max rows to load from AIMO3 TIR (0=all, default=10000 for speed)")

    # Select
    p_select = subparsers.add_parser("select", help="Run curation pipeline")
    p_select.add_argument("--target", type=int, default=3000, help="Target number of samples")
    p_select.add_argument("--min-pass-rate", type=float, default=0.03, help="Min pass rate (default 0.03)")
    p_select.add_argument("--max-pass-rate", type=float, default=0.50, help="Max pass rate (default 0.50)")
    p_select.add_argument("--iw-sft", action="store_true", help="Compute importance weights for iw-SFT")
    p_select.add_argument("--output", type=str, default=None, help="Output path (default: data/curated_N.jsonl)")
    p_select.add_argument("--max-tir-rows", type=int, default=0,
                           help="Max rows from AIMO3 TIR (0=all)")
    p_select.add_argument("--diversity", type=str, default="topic",
                           choices=["topic", "source", "difficulty", "multi"],
                           help="Diversity mode: topic, source, difficulty, or multi (default: topic)")
    p_select.add_argument("--stop-after", type=str, default=None,
                           choices=["load", "hard_filter", "decontaminate", "difficulty", "scoring"],
                           help="Stop after this stage (for debugging)")

    args = parser.parse_args()

    if args.command == "profile":
        run_profile(args)
    elif args.command == "select":
        run_select(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
