#!/usr/bin/env python3
"""Run the NemoSkills (1st place) solution recreation.

Usage:
    # Against AIMO3 reference problems (default)
    python -m recreate_solutions.solution_1_nemoskills.run

    # Against AIMO2 problems
    python -m recreate_solutions.solution_1_nemoskills.run --competition aimo2

    # Custom CSV
    python -m recreate_solutions.solution_1_nemoskills.run --input-csv path/to/problems.csv

    # With GenSelect disabled (majority voting only)
    python -m recreate_solutions.solution_1_nemoskills.run --no-genselect
"""

import argparse
import json
import os
import time
from datetime import datetime

from recreate_solutions.common.data_loader import load_problems, get_known_answers
from recreate_solutions.solution_1_nemoskills.solver import NemoSkillsSolver


def main():
    p = argparse.ArgumentParser(description="NemoSkills 1st Place Recreation")
    p.add_argument("--api-base", default="http://localhost:8080/v1")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    p.add_argument("--competition", choices=["aimo2", "aimo3"], default="aimo3")
    p.add_argument("--input-csv", default=None)
    p.add_argument("--output-dir", default="output/nemoskills")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--max-turns", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--no-genselect", action="store_true")
    args = p.parse_args()

    df = load_problems(csv_path=args.input_csv, competition=args.competition)
    known = get_known_answers(df)
    print(f"Loaded {len(df)} problems from {args.competition}")

    run_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    solver = NemoSkillsSolver(
        api_base=args.api_base,
        model=args.model,
        n_samples=args.n_samples,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_genselect=not args.no_genselect,
    )

    results = []
    correct = 0
    total_t0 = time.time()

    try:
        for idx, row in df.iterrows():
            prob_id = str(row["id"])
            problem = row["problem"]
            print(f"\n{'=' * 60}")
            print(f"Problem {idx + 1}/{len(df)} (ID: {prob_id})")
            print(f"{'=' * 60}")
            print(f"{problem[:200]}...\n")

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
            trace_path = os.path.join(run_dir, f"{prob_id}.json")
            with open(trace_path, "w") as f:
                json.dump({"id": prob_id, "answer": answer, "expected": expected,
                           "time_s": round(elapsed, 1)}, f, indent=2)
    finally:
        total = time.time() - total_t0
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {correct}/{len(results)} correct, {total:.1f}s total")
        if results:
            import pandas as pd
            pd.DataFrame(results).to_csv(os.path.join(run_dir, "submission.csv"), index=False)
            print(f"Saved to {run_dir}/submission.csv")
        solver.shutdown()


if __name__ == "__main__":
    main()
