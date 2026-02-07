#!/usr/bin/env python3
"""Run the Unified solution (best-of-all-worlds for AIMO3).

Usage:
    # Default: AIMO3 reference problems
    python -m recreate_solutions.unified_solution.run

    # AIMO2 problems
    python -m recreate_solutions.unified_solution.run --competition aimo2

    # Quick test with fewer samples
    python -m recreate_solutions.unified_solution.run --n-cot 2 --n-code 2 --no-genselect --no-verify

    # Full competition mode
    python -m recreate_solutions.unified_solution.run --n-cot 8 --n-code 8 --early-stop 5
"""

import argparse
import json
import os
import time
from datetime import datetime

from recreate_solutions.common.data_loader import load_problems, get_known_answers
from recreate_solutions.unified_solution.solver import UnifiedSolver


def main():
    p = argparse.ArgumentParser(description="Unified AIMO Solver (Best-of-All-Worlds)")
    p.add_argument("--api-base", default="http://localhost:8080/v1")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    p.add_argument("--competition", choices=["aimo2", "aimo3"], default="aimo3")
    p.add_argument("--input-csv", default=None)
    p.add_argument("--output-dir", default="output/unified")
    p.add_argument("--n-cot", type=int, default=8)
    p.add_argument("--n-code", type=int, default=8)
    p.add_argument("--max-turns", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--early-stop", type=int, default=5)
    p.add_argument("--no-genselect", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--time-budget", type=float, default=17400.0)
    args = p.parse_args()

    df = load_problems(csv_path=args.input_csv, competition=args.competition)
    known = get_known_answers(df)
    print(f"Loaded {len(df)} problems from {args.competition}")

    run_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    solver = UnifiedSolver(
        api_base=args.api_base,
        model=args.model,
        n_cot_samples=args.n_cot,
        n_code_samples=args.n_code,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        early_stop_consensus=args.early_stop,
        use_genselect=not args.no_genselect,
        use_verification=not args.no_verify,
        total_time_budget=args.time_budget,
        n_problems=len(df),
    )

    results = []
    correct = 0
    total_t0 = time.time()

    try:
        for idx, row in df.iterrows():
            prob_id = str(row["id"])
            problem = row["problem"]
            print(f"\n{'=' * 70}")
            print(f"Problem {idx + 1}/{len(df)} (ID: {prob_id})")
            print(f"{'=' * 70}")
            print(f"{problem[:300]}...\n")

            t0 = time.time()
            result = solver.solve(problem)
            elapsed = time.time() - t0
            answer = result["answer"]

            expected = known.get(prob_id)
            tag = ""
            if expected is not None:
                verdict = "CORRECT" if answer == expected else "WRONG"
                tag = f" [{verdict}, expected={expected}]"
                if answer == expected:
                    correct += 1

            print(f"  => Answer: {answer}{tag} ({elapsed:.1f}s)")
            results.append({"id": prob_id, "answer": answer})

            # Save trace
            trace = {
                "id": prob_id,
                "answer": answer,
                "expected": expected,
                "correct": answer == expected if expected is not None else None,
                "method": result.get("method"),
                "early_stopped": result.get("early_stopped"),
                "votes": result.get("votes"),
                "time_s": round(elapsed, 1),
            }
            with open(os.path.join(run_dir, f"{prob_id}.json"), "w") as f:
                json.dump(trace, f, indent=2)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        total = time.time() - total_t0
        print(f"\n{'=' * 70}")
        print(f"RESULTS: {correct}/{len(results)} correct, {total:.1f}s total")
        print(f"{'=' * 70}")
        if results:
            import pandas as pd
            sub_path = os.path.join(run_dir, "submission.csv")
            pd.DataFrame(results).to_csv(sub_path, index=False)
            print(f"Saved to {sub_path}")

            summary = {
                "model": args.model,
                "competition": args.competition,
                "n_cot": args.n_cot,
                "n_code": args.n_code,
                "correct": correct if known else None,
                "total": len(results),
                "accuracy": correct / len(results) if known and results else None,
                "total_time_s": round(total, 1),
            }
            with open(os.path.join(run_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
        solver.shutdown()


if __name__ == "__main__":
    main()
