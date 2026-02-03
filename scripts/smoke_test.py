#!/usr/bin/env python3
"""Smoke test for trace generation + replay system.

Run this on a GPU machine with vLLM server running to verify everything works
before starting long optimization runs.

Usage:
    # Start your model server first
    vllm serve Qwen/Qwen3-4B --port 8080

    # Run smoke test (2 problems, 3 samples each — should take ~5 min)
    python scripts/smoke_test.py --api-base http://localhost:8080/v1
"""

import argparse
import os
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Smoke test trace generation + replay")
    parser.add_argument("--api-base", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="Qwen3-4B")
    args = parser.parse_args()

    print("=" * 60)
    print("SMOKE TEST: Trace Generation + Replay Selection")
    print("=" * 60)

    # Step 1: Test imports
    print("\n[1/4] Testing imports...")
    try:
        from scripts.prompts import PROMPTS, get_prompt_sequence
        from scripts.replay_selection import build_all_strategies, run_selection_sweep
        from scripts.generate_traces import generate_all
        print("  OK: All imports successful")
    except ImportError as e:
        print(f"  FAIL: {e}")
        print("  Install missing deps: pip install jupyter_client pandas requests")
        sys.exit(1)

    # Step 2: Test API connectivity
    print("\n[2/4] Testing API connectivity...")
    try:
        import requests
        r = requests.get(f"{args.api_base}/models", timeout=10)
        r.raise_for_status()
        models = r.json().get("data", [])
        print(f"  OK: API reachable, {len(models)} model(s) available")
        if models:
            print(f"      Models: {[m.get('id') for m in models[:3]]}")
    except Exception as e:
        print(f"  FAIL: Cannot reach API at {args.api_base}")
        print(f"        Error: {e}")
        print("  Start your model server first:")
        print(f"    vllm serve {args.model} --port 8080")
        sys.exit(1)

    # Step 3: Generate traces on 2 trivial problems
    print("\n[3/4] Generating traces (2 problems, 3 samples each)...")

    # Use test.csv which has trivial problems
    csv_path = os.path.join(_REPO_ROOT, "data", "aimo3", "test.csv")
    if not os.path.exists(csv_path):
        # Fallback to reference.csv, limit to 2 problems
        csv_path = os.path.join(_REPO_ROOT, "data", "aimo3", "reference.csv")
        problems_arg = None  # will use --problems to limit
    else:
        problems_arg = None

    with tempfile.TemporaryDirectory() as tmpdir:
        class GenArgs:
            pass

        ga = GenArgs()
        ga.api_base = args.api_base
        ga.model = args.model
        ga.n_samples = 3
        ga.max_turns = 4  # Short for smoke test
        ga.temperature = 0.6
        ga.max_tokens = 4096
        ga.min_p = None
        ga.seed = 42
        ga.logprobs = 5
        ga.code_timeout = 15.0
        ga.competition = "aimo3"
        ga.csv_path = csv_path
        ga.problems = None
        ga.prompt_mix = "reasoning:2,code_first:1"
        ga.output_dir = os.path.join(tmpdir, "traces")

        # Limit to 2 problems for speed
        import pandas as pd
        df = pd.read_csv(csv_path)
        if len(df) > 2:
            ga.problems = ",".join(df["id"].astype(str).head(2).tolist())

        try:
            traces_dir = generate_all(ga)
            if traces_dir and os.path.exists(traces_dir):
                n_files = len([f for f in os.listdir(traces_dir) if f.startswith("problem_")])
                print(f"  OK: Generated {n_files} trace file(s)")
            else:
                print("  FAIL: No traces generated")
                sys.exit(1)
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Step 4: Run selection sweep
        print("\n[4/4] Running selection sweep...")
        try:
            db_path = os.path.join(tmpdir, "test.db")
            best_acc, best_strat = run_selection_sweep(traces_dir, db_path=db_path, quiet=True)

            # Count strategies
            strategies = build_all_strategies()
            print(f"  OK: Tested {len(strategies)} strategies")
            print(f"      Best: {best_strat} ({best_acc:.1%})")
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    print("\nYou can now run the full pipeline:")
    print(f"""
# Generate traces on all reference problems (12 samples each)
python scripts/generate_traces.py \\
    --api-base {args.api_base} \\
    --model {args.model} \\
    --n-samples 12 \\
    --max-turns 16

# Sweep selection strategies on generated traces
python scripts/replay_selection.py sweep \\
    --traces-dir output/traces/<timestamp>/

# Or run Bayesian optimization (10+ hours)
python scripts/replay_selection.py optimize \\
    --api-base {args.api_base} \\
    --model {args.model} \\
    --n-trials 30
""")


if __name__ == "__main__":
    main()
