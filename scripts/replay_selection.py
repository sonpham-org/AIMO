"""Offline replay of selection strategies on saved traces.

Two modes:
  1. sweep  — Grid search over 800+ selection strategy configs on saved traces (seconds)
  2. optimize — Bayesian optimization of generation hyperparams via Optuna (hours)

Usage:
    # Sweep selection strategies on existing traces
    python scripts/replay_selection.py sweep \
        --traces-dir output/traces/20260202_143000/

    # Bayesian optimization of generation params
    python scripts/replay_selection.py optimize \
        --api-base http://localhost:8080/v1 \
        --model Qwen3-4B \
        --n-trials 30
"""

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

DEFAULT_DB_PATH = os.path.join(_REPO_ROOT, "results", "experiments.db")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Attempt:
    answer: Optional[int]
    answer_source: Optional[str]  # "boxed", "code_output", "boxed_lastditch"
    entropy: float
    prompt_type: str
    n_python_calls: int
    n_python_errors: int
    turns_used: int
    total_response_tokens: int


@dataclass
class ProblemTraces:
    problem_id: str
    ground_truth: Optional[int]
    attempts: List[Attempt]


def load_traces(traces_dir: str) -> List[ProblemTraces]:
    """Load all problem trace files from a traces directory."""
    problems = []
    for fname in sorted(os.listdir(traces_dir)):
        if not fname.startswith("problem_") or not fname.endswith(".json"):
            continue
        with open(os.path.join(traces_dir, fname)) as f:
            data = json.load(f)

        attempts = []
        for a in data.get("attempts", []):
            ent = a.get("entropy", float("inf"))
            if isinstance(ent, str):
                ent = float("inf") if ent == "Infinity" else float(ent)
            attempts.append(Attempt(
                answer=a.get("answer"),
                answer_source=a.get("answer_source"),
                entropy=ent,
                prompt_type=a.get("prompt_type", "reasoning"),
                n_python_calls=a.get("n_python_calls", 0),
                n_python_errors=a.get("n_python_errors", 0),
                turns_used=a.get("turns_used", 0),
                total_response_tokens=a.get("total_response_tokens", 0),
            ))

        gt = data.get("ground_truth")
        problems.append(ProblemTraces(
            problem_id=data["problem_id"],
            ground_truth=int(gt) if gt is not None else None,
            attempts=attempts,
        ))
    return problems


# ---------------------------------------------------------------------------
# Entropy transforms
# ---------------------------------------------------------------------------

def inv(x: float) -> float:
    return 1.0 / max(x, 1e-9)

def inv_sq(x: float) -> float:
    return 1.0 / max(x * x, 1e-18)

def exp_neg(x: float) -> float:
    return math.exp(-x)

def inv_log(x: float) -> float:
    return 1.0 / math.log(1.0 + max(x, 0.0))

TRANSFORMS = {
    "inv": inv,
    "inv_sq": inv_sq,
    "exp_neg": exp_neg,
    "inv_log": inv_log,
}


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

# Type alias: a strategy takes a list of Attempts and returns an answer (or None)
Strategy = Callable[[List[Attempt]], Optional[int]]


def _valid(attempts: List[Attempt]) -> List[Attempt]:
    return [a for a in attempts if a.answer is not None]


def majority_vote(attempts: List[Attempt]) -> Optional[int]:
    valid = _valid(attempts)
    if not valid:
        return None
    counter = Counter(a.answer for a in valid)
    return counter.most_common(1)[0][0]


def min_entropy(attempts: List[Attempt]) -> Optional[int]:
    valid = _valid(attempts)
    if not valid:
        return None
    best = min(valid, key=lambda a: a.entropy)
    return best.answer


def pure_entropy(attempts: List[Attempt], transform: Callable = inv) -> Optional[int]:
    valid = _valid(attempts)
    if not valid:
        return None
    scores: Dict[int, float] = defaultdict(float)
    for a in valid:
        scores[a.answer] += transform(a.entropy)
    return max(scores, key=scores.get)


def hybrid_weighted(
    attempts: List[Attempt],
    transform: Callable = inv,
    k: float = 0.1,
) -> Optional[int]:
    valid = _valid(attempts)
    if not valid:
        return None
    ent_scores: Dict[int, float] = defaultdict(float)
    vote_counts: Dict[int, int] = defaultdict(int)
    for a in valid:
        ent_scores[a.answer] += transform(a.entropy)
        vote_counts[a.answer] += 1
    scores = {
        ans: ent_scores[ans] + k * vote_counts[ans]
        for ans in ent_scores
    }
    return max(scores, key=scores.get)


def normalized_hybrid(
    attempts: List[Attempt],
    transform: Callable = inv,
    alpha: float = 0.5,
) -> Optional[int]:
    """Blend normalized entropy scores and normalized vote counts.

    score = (1 - alpha) * norm_entropy_score + alpha * norm_votes
    """
    valid = _valid(attempts)
    if not valid:
        return None

    ent_scores: Dict[int, float] = defaultdict(float)
    vote_counts: Dict[int, int] = defaultdict(int)
    for a in valid:
        ent_scores[a.answer] += transform(a.entropy)
        vote_counts[a.answer] += 1

    # Normalize to [0, 1]
    ent_vals = list(ent_scores.values())
    ent_max = max(ent_vals) if ent_vals else 1.0
    vote_max = max(vote_counts.values()) if vote_counts else 1

    scores = {}
    for ans in ent_scores:
        ne = ent_scores[ans] / ent_max if ent_max > 0 else 0
        nv = vote_counts[ans] / vote_max if vote_max > 0 else 0
        scores[ans] = (1 - alpha) * ne + alpha * nv
    return max(scores, key=scores.get)


def threshold_vote(
    attempts: List[Attempt],
    threshold: float = 2.0,
) -> Optional[int]:
    """Keep attempts with entropy < threshold, then majority vote. Fallback to all."""
    valid = _valid(attempts)
    if not valid:
        return None
    filtered = [a for a in valid if a.entropy < threshold]
    if not filtered:
        filtered = valid  # fallback
    counter = Counter(a.answer for a in filtered)
    return counter.most_common(1)[0][0]


def topk_vote(
    attempts: List[Attempt],
    k: int = 3,
) -> Optional[int]:
    """Keep K lowest-entropy attempts, then majority vote."""
    valid = _valid(attempts)
    if not valid:
        return None
    sorted_a = sorted(valid, key=lambda a: a.entropy)
    top = sorted_a[:k] if len(sorted_a) >= k else sorted_a
    counter = Counter(a.answer for a in top)
    return counter.most_common(1)[0][0]


def threshold_hybrid(
    attempts: List[Attempt],
    threshold: float = 2.0,
    transform: Callable = inv,
    k: float = 0.1,
) -> Optional[int]:
    """Filter by entropy threshold, then apply hybrid scoring."""
    valid = _valid(attempts)
    if not valid:
        return None
    filtered = [a for a in valid if a.entropy < threshold]
    if not filtered:
        filtered = valid
    return hybrid_weighted(filtered, transform=transform, k=k)


def source_aware(
    attempts: List[Attempt],
    transform: Callable = inv,
    k: float = 0.1,
    code_fallback_weight: float = 0.5,
) -> Optional[int]:
    """Hybrid scoring with reduced weight for code_output-sourced answers."""
    valid = _valid(attempts)
    if not valid:
        return None
    ent_scores: Dict[int, float] = defaultdict(float)
    vote_counts: Dict[int, int] = defaultdict(int)
    for a in valid:
        weight = code_fallback_weight if a.answer_source == "code_output" else 1.0
        ent_scores[a.answer] += weight * transform(a.entropy)
        vote_counts[a.answer] += 1
    scores = {
        ans: ent_scores[ans] + k * vote_counts[ans]
        for ans in ent_scores
    }
    return max(scores, key=scores.get)


def meta_vote(attempts: List[Attempt]) -> Optional[int]:
    """Majority within each prompt type, then majority across types."""
    valid = _valid(attempts)
    if not valid:
        return None
    by_type: Dict[str, List[Attempt]] = defaultdict(list)
    for a in valid:
        by_type[a.prompt_type].append(a)

    type_answers = []
    for t_attempts in by_type.values():
        counter = Counter(a.answer for a in t_attempts)
        type_answers.append(counter.most_common(1)[0][0])

    if not type_answers:
        return None
    return Counter(type_answers).most_common(1)[0][0]


def diversity_bonus(
    attempts: List[Attempt],
    transform: Callable = inv,
    k: float = 0.1,
    bonus: float = 1.0,
) -> Optional[int]:
    """Hybrid scoring + bonus when answer spans multiple prompt types."""
    valid = _valid(attempts)
    if not valid:
        return None

    ent_scores: Dict[int, float] = defaultdict(float)
    vote_counts: Dict[int, int] = defaultdict(int)
    prompt_types_per_answer: Dict[int, set] = defaultdict(set)

    for a in valid:
        ent_scores[a.answer] += transform(a.entropy)
        vote_counts[a.answer] += 1
        prompt_types_per_answer[a.answer].add(a.prompt_type)

    scores = {}
    for ans in ent_scores:
        n_types = len(prompt_types_per_answer[ans])
        div_bonus = bonus * (n_types - 1) if n_types > 1 else 0
        scores[ans] = ent_scores[ans] + k * vote_counts[ans] + div_bonus
    return max(scores, key=scores.get)


def weighted_meta_vote(attempts: List[Attempt], transform: Callable = inv) -> Optional[int]:
    """Per-prompt-type vote weighted by that type's mean entropy."""
    valid = _valid(attempts)
    if not valid:
        return None

    by_type: Dict[str, List[Attempt]] = defaultdict(list)
    for a in valid:
        by_type[a.prompt_type].append(a)

    answer_scores: Dict[int, float] = defaultdict(float)
    for ptype, t_attempts in by_type.items():
        mean_ent = sum(a.entropy for a in t_attempts) / len(t_attempts)
        weight = transform(mean_ent)
        counter = Counter(a.answer for a in t_attempts)
        winner = counter.most_common(1)[0][0]
        answer_scores[winner] += weight

    return max(answer_scores, key=answer_scores.get) if answer_scores else None


def prompt_type_only(
    attempts: List[Attempt],
    prompt_type: str = "reasoning",
) -> Optional[int]:
    """Only use attempts from a specific prompt type, majority vote."""
    valid = [a for a in _valid(attempts) if a.prompt_type == prompt_type]
    if not valid:
        return majority_vote(attempts)  # fallback
    counter = Counter(a.answer for a in valid)
    return counter.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

K_VALUES = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
ALPHA_VALUES = [round(a * 0.1, 1) for a in range(11)]  # 0.0 to 1.0
THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
TOPK_VALUES = [2, 3, 4, 5, 6, 8]
CODE_FALLBACK_WEIGHTS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
DIVERSITY_BONUSES = [0.5, 1.0, 2.0, 5.0]


def build_all_strategies() -> List[Tuple[str, Strategy, Dict[str, Any]]]:
    """Build all strategy configs. Returns list of (name, fn, params)."""
    strategies = []

    # Group 1: Pure methods (6)
    strategies.append(("majority_vote", majority_vote, {}))
    for tname, tfn in TRANSFORMS.items():
        strategies.append((
            f"pure_entropy_{tname}",
            partial(pure_entropy, transform=tfn),
            {"transform": tname},
        ))
    strategies.append(("min_entropy", min_entropy, {}))

    # Group 2: Hybrid weighting (40)
    for tname, tfn in TRANSFORMS.items():
        for k in K_VALUES:
            strategies.append((
                f"hybrid_{tname}_k{k}",
                partial(hybrid_weighted, transform=tfn, k=k),
                {"transform": tname, "k": k},
            ))

    # Group 3: Normalized hybrids (44)
    for tname, tfn in TRANSFORMS.items():
        for alpha in ALPHA_VALUES:
            strategies.append((
                f"norm_{tname}_a{alpha}",
                partial(normalized_hybrid, transform=tfn, alpha=alpha),
                {"transform": tname, "alpha": alpha},
            ))

    # Group 4a: Threshold vote (7)
    for t in THRESHOLDS:
        strategies.append((
            f"threshold_{t}",
            partial(threshold_vote, threshold=t),
            {"threshold": t},
        ))

    # Group 4b: Top-K vote (6)
    for topk in TOPK_VALUES:
        strategies.append((
            f"topk_{topk}",
            partial(topk_vote, k=topk),
            {"k": topk},
        ))

    # Group 4c: Threshold + hybrid cross-product
    # Top-3 hybrid params from Group 2 (we'll use inv with k=0.1, 0.25, 0.5 as reasonable defaults)
    top_hybrid_params = [
        ("inv", inv, 0.1),
        ("inv", inv, 0.25),
        ("inv", inv, 0.5),
    ]
    for t in THRESHOLDS:
        for tname, tfn, k in top_hybrid_params:
            strategies.append((
                f"thresh_{t}_{tname}_k{k}",
                partial(threshold_hybrid, threshold=t, transform=tfn, k=k),
                {"threshold": t, "transform": tname, "k": k},
            ))

    # Group 5: Source-aware (6)
    for cfw in CODE_FALLBACK_WEIGHTS:
        strategies.append((
            f"source_cfw{cfw}",
            partial(source_aware, transform=inv, k=0.1, code_fallback_weight=cfw),
            {"code_fallback_weight": cfw, "transform": "inv", "k": 0.1},
        ))

    # Group 6: Prompt-aware (8)
    strategies.append(("meta_vote", meta_vote, {}))

    for b in DIVERSITY_BONUSES:
        strategies.append((
            f"diversity_bonus_{b}",
            partial(diversity_bonus, transform=inv, k=0.1, bonus=b),
            {"bonus": b, "transform": "inv", "k": 0.1},
        ))

    strategies.append((
        "weighted_meta_vote",
        partial(weighted_meta_vote, transform=inv),
        {"transform": "inv"},
    ))

    strategies.append((
        "prompt_only_reasoning",
        partial(prompt_type_only, prompt_type="reasoning"),
        {"prompt_type": "reasoning"},
    ))
    strategies.append((
        "prompt_only_code_first",
        partial(prompt_type_only, prompt_type="code_first"),
        {"prompt_type": "code_first"},
    ))

    return strategies


# ---------------------------------------------------------------------------
# SQLite results database
# ---------------------------------------------------------------------------

def init_db(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS selection_results (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            traces_dir TEXT,
            strategy_name TEXT,
            strategy_params TEXT,
            correct INTEGER,
            total INTEGER,
            accuracy REAL,
            per_problem TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generation_runs (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            model TEXT,
            temperature REAL,
            min_p REAL,
            n_samples INTEGER,
            max_turns INTEGER,
            prompt_mix TEXT,
            traces_dir TEXT,
            majority_vote_accuracy REAL,
            best_selection_accuracy REAL,
            best_selection_strategy TEXT,
            wall_time_s REAL
        )
    """)
    conn.commit()
    return conn


def record_selection_result(
    conn: sqlite3.Connection,
    traces_dir: str,
    strategy_name: str,
    strategy_params: Dict,
    correct: int,
    total: int,
    per_problem: Dict,
):
    conn.execute(
        """INSERT INTO selection_results
           (timestamp, traces_dir, strategy_name, strategy_params,
            correct, total, accuracy, per_problem)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            time.strftime("%Y-%m-%d %H:%M:%S"),
            traces_dir,
            strategy_name,
            json.dumps(strategy_params),
            correct,
            total,
            correct / total if total else 0.0,
            json.dumps(per_problem),
        ),
    )


def record_generation_run(
    conn: sqlite3.Connection,
    model: str,
    temperature: float,
    min_p: float,
    n_samples: int,
    max_turns: int,
    prompt_mix: Dict,
    traces_dir: str,
    mv_accuracy: float,
    best_accuracy: float,
    best_strategy: str,
    wall_time: float,
):
    conn.execute(
        """INSERT INTO generation_runs
           (timestamp, model, temperature, min_p, n_samples, max_turns,
            prompt_mix, traces_dir, majority_vote_accuracy,
            best_selection_accuracy, best_selection_strategy, wall_time_s)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            time.strftime("%Y-%m-%d %H:%M:%S"),
            model,
            temperature,
            min_p,
            n_samples,
            max_turns,
            json.dumps(prompt_mix),
            traces_dir,
            mv_accuracy,
            best_accuracy,
            best_strategy,
            wall_time,
        ),
    )


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------

def run_selection_sweep(
    traces_dir: str,
    db_path: str = DEFAULT_DB_PATH,
    quiet: bool = False,
) -> Tuple[float, str]:
    """Run all selection strategies on saved traces.

    Returns (best_accuracy, best_strategy_name).
    """
    problems = load_traces(traces_dir)
    if not problems:
        print(f"No trace files found in {traces_dir}")
        return 0.0, ""

    # Filter to problems with ground truth
    problems = [p for p in problems if p.ground_truth is not None]
    if not problems:
        print("No problems with ground truth found.")
        return 0.0, ""

    total = len(problems)
    strategies = build_all_strategies()

    if not quiet:
        print(f"Loaded {total} problems with ground truth")
        print(f"Running {len(strategies)} strategies...")

    conn = init_db(db_path)
    results = []
    t0 = time.time()

    for sname, sfn, sparams in strategies:
        correct = 0
        per_problem = {}

        for p in problems:
            predicted = sfn(p.attempts)
            if predicted is None:
                predicted = 0
            is_correct = predicted == p.ground_truth
            if is_correct:
                correct += 1
            per_problem[p.problem_id] = {
                "predicted": predicted,
                "expected": p.ground_truth,
                "correct": is_correct,
            }

        accuracy = correct / total
        results.append((sname, accuracy, correct, total, sparams, per_problem))
        record_selection_result(conn, traces_dir, sname, sparams, correct, total, per_problem)

    conn.commit()
    conn.close()
    dt = time.time() - t0

    # Sort by accuracy descending
    results.sort(key=lambda r: (-r[1], r[0]))

    if not quiet:
        print(f"\nCompleted in {dt:.2f}s")
        print(f"\n{'='*65}")
        print(f"{'#':>3}  {'Strategy':<40} {'Accuracy':>8}  {'Correct':>10}")
        print(f"{'='*65}")
        for i, (name, acc, cor, tot, _, _) in enumerate(results[:20]):
            print(f"{i+1:>3}  {name:<40} {acc:>8.3f}  {cor:>3}/{tot}")
        print(f"{'='*65}")

        # Also show majority vote result specifically
        mv = next((r for r in results if r[0] == "majority_vote"), None)
        if mv:
            mv_rank = results.index(mv) + 1
            print(f"\nMajority vote: {mv[1]:.3f} ({mv[2]}/{mv[3]}) — rank #{mv_rank}")

        best = results[0]
        print(f"Best: {best[0]} — {best[1]:.3f} ({best[2]}/{best[3]})")
        print(f"\nResults saved to {db_path}")

    return results[0][1], results[0][0] if results else (0.0, "")


# ---------------------------------------------------------------------------
# Optimize mode (Bayesian optimization with Optuna)
# ---------------------------------------------------------------------------

def run_optimization(args):
    """Bayesian optimization of generation hyperparams via Optuna."""
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Run: pip install optuna")
        sys.exit(1)

    from scripts.generate_traces import generate_all

    db_path = args.db_path or DEFAULT_DB_PATH
    conn = init_db(db_path)

    def objective(trial):
        temperature = trial.suggest_float("temperature", 0.3, 1.5)
        min_p = trial.suggest_float("min_p", 0.005, 0.15, log=True)
        n_samples = trial.suggest_int("n_samples", 5, 20)
        max_turns = trial.suggest_int("max_turns", 1, 16)
        prompt_mix_str = trial.suggest_categorical(
            "prompt_mix", ["8:2:2", "6:3:3", "4:4:4", "10:1:1"]
        )

        # Parse prompt mix string to full format
        parts = [int(x) for x in prompt_mix_str.split(":")]
        prompt_mix_cli = f"reasoning:{parts[0]},code_first:{parts[1]},case_analysis:{parts[2]}"

        # Build args namespace for generate_all
        class GenArgs:
            pass

        ga = GenArgs()
        ga.api_base = args.api_base
        ga.model = args.model
        ga.n_samples = n_samples
        ga.max_turns = max_turns
        ga.temperature = temperature
        ga.max_tokens = args.max_tokens
        ga.min_p = min_p
        ga.seed = 42
        ga.logprobs = args.logprobs
        ga.code_timeout = 30.0
        ga.competition = args.competition
        ga.csv_path = args.csv_path
        ga.problems = args.problems
        ga.prompt_mix = prompt_mix_cli
        ga.output_dir = None  # auto-generate

        # Run generation
        t0 = time.time()
        traces_dir = generate_all(ga)
        wall_time = time.time() - t0

        if traces_dir is None:
            return 0.0

        # Run selection sweep
        best_acc, best_strat = run_selection_sweep(traces_dir, db_path=db_path, quiet=True)

        # Also get majority vote accuracy for comparison
        problems = load_traces(traces_dir)
        problems = [p for p in problems if p.ground_truth is not None]
        mv_correct = sum(
            1 for p in problems
            if majority_vote(p.attempts) == p.ground_truth
        )
        mv_acc = mv_correct / len(problems) if problems else 0.0

        # Record
        prompt_mix_dict = {"reasoning": parts[0], "code_first": parts[1], "case_analysis": parts[2]}
        record_generation_run(
            conn, args.model, temperature, min_p, n_samples, max_turns,
            prompt_mix_dict, traces_dir, mv_acc, best_acc, best_strat, wall_time,
        )
        conn.commit()

        print(f"\n  Trial {trial.number}: temp={temperature:.2f}, min_p={min_p:.4f}, "
              f"n={n_samples}, turns={max_turns}, mix={prompt_mix_str}")
        print(f"  MV={mv_acc:.3f}, Best={best_acc:.3f} ({best_strat}), "
              f"time={wall_time:.0f}s")

        return best_acc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=args.n_trials)

    print(f"\n{'='*60}")
    print("Optimization complete!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.3f}")
    print(f"Best params: {study.best_params}")
    print(f"{'='*60}")

    conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Replay selection strategies on saved traces or optimize generation params"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Sweep subcommand
    sweep_p = subparsers.add_parser("sweep", help="Grid search over selection strategies")
    sweep_p.add_argument("--traces-dir", required=True,
                         help="Directory containing trace JSON files")
    sweep_p.add_argument("--db-path", default=DEFAULT_DB_PATH,
                         help="SQLite database path for results")

    # Optimize subcommand
    opt_p = subparsers.add_parser("optimize",
                                  help="Bayesian optimization of generation hyperparams")
    opt_p.add_argument("--api-base", default="http://localhost:8080/v1")
    opt_p.add_argument("--model", default="Qwen3-4B")
    opt_p.add_argument("--n-trials", type=int, default=30)
    opt_p.add_argument("--max-tokens", type=int, default=16384)
    opt_p.add_argument("--logprobs", type=int, default=5)
    opt_p.add_argument("--competition", default="aimo3")
    opt_p.add_argument("--csv-path", default=None)
    opt_p.add_argument("--problems", default=None,
                        help="Comma-separated problem IDs (subset for faster iteration)")
    opt_p.add_argument("--db-path", default=DEFAULT_DB_PATH)

    args = parser.parse_args()

    if args.command == "sweep":
        run_selection_sweep(args.traces_dir, db_path=args.db_path)
    elif args.command == "optimize":
        run_optimization(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
