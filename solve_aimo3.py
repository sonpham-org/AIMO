"""
AIMO3 Solver — Aliev-style (TIR + Majority Voting)
====================================================
Simple but effective: use a strong reasoning model off-the-shelf with
Tool-Integrated Reasoning and majority voting. No fine-tuning needed.

Tested with: Qwen3-30B-A3B, DeepSeek-R1-Distill-Qwen-14B, etc.

Usage:
    # Start your model server first, e.g.:
    # vllm serve Qwen/Qwen3-30B-A3B --enable-reasoning --reasoning-parser deepseek_r1

    # Run against AIMO3 reference problems (10 problems with known answers)
    python solve_aimo3.py --api-base http://localhost:8080/v1 --model Qwen/Qwen3-30B-A3B

    # Fewer samples for quick testing
    python solve_aimo3.py --n-samples 2 --max-turns 8
"""

import os
import re
import sys
import json
import time
import queue
import argparse
import threading
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd
from jupyter_client import KernelManager

# ---------------------------------------------------------------------------
# Sandbox: Isolated Jupyter kernel for code execution
# ---------------------------------------------------------------------------

class Sandbox:
    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _alloc_ports(cls, n: int = 5) -> List[int]:
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + n))
            cls._next_port += n
            return ports

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout
        ports = self._alloc_ports()
        env = os.environ.copy()
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["MPLBACKEND"] = "Agg"

        self._km = KernelManager()
        self._km.shell_port, self._km.iopub_port = ports[0], ports[1]
        self._km.stdin_port, self._km.hb_port, self._km.control_port = ports[2], ports[3], ports[4]
        self._km.start_kernel(env=env, extra_arguments=["--Application.log_level=CRITICAL"])

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._timeout)

        self._run_setup()

    def _run_setup(self):
        self.execute(
            "import math, numpy, sympy, mpmath, itertools, collections, functools\n"
            "from sympy import *\n"
            "mpmath.mp.dps = 64\n"
        )

    def execute(self, code: str, timeout: float = None) -> str:
        t = timeout or self._timeout
        msg_id = self._client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)
        stdout, stderr = [], []
        start = time.time()

        while True:
            if time.time() - start > t:
                self._km.interrupt_kernel()
                return f"[ERROR] Timed out after {t}s"
            try:
                msg = self._client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            mt = msg.get("msg_type")
            c = msg.get("content", {})
            if mt == "stream":
                (stdout if c.get("name") == "stdout" else stderr).append(c.get("text", ""))
            elif mt == "error":
                tb = c.get("traceback", [])
                stderr.append("\n".join(re.sub(r"\x1b\[[0-9;]*m", "", f) for f in tb))
            elif mt in ("execute_result", "display_data"):
                text = c.get("data", {}).get("text/plain", "")
                if text:
                    stdout.append(text if text.endswith("\n") else text + "\n")
            elif mt == "status" and c.get("execution_state") == "idle":
                break

        out = "".join(stdout).rstrip()
        err = "".join(stderr).rstrip()
        if err:
            return f"{out}\n{err}" if out else err
        return out if out else "[No output — use print()]"

    def reset(self):
        self.execute("%reset -f")
        self._run_setup()

    def close(self):
        try:
            self._client.stop_channels()
        except Exception:
            pass
        try:
            self._km.shutdown_kernel(now=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> Optional[int]:
    """Extract integer answer from \\boxed{} or 'final answer is N'."""
    for pattern in [
        r"\\boxed\s*\{\s*([0-9,\s]+)\s*\}",
        r"[Ff]inal\s+[Aa]nswer\s*(?:is|=)\s*\$?\\?boxed\{?\s*([0-9,]+)",
        r"[Ff]inal\s+[Aa]nswer\s*(?:is|=)\s*\$?\s*([0-9,]+)",
    ]:
        matches = re.findall(pattern, text)
        if matches:
            try:
                val = int(matches[-1].replace(",", "").replace(" ", ""))
                if 0 <= val <= 99999:
                    return val
            except ValueError:
                continue
    return None


def extract_code_blocks(text: str) -> List[str]:
    """Extract ```python ... ``` code blocks."""
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ensure_print(code: str) -> str:
    """Wrap last expression in print() if it isn't already a statement."""
    lines = code.strip().split("\n")
    if not lines:
        return code
    last = lines[-1].strip()
    skip = ("print", "import", "from ", "#", "plt.", "if ", "for ", "while ",
            "def ", "class ", "return", "raise", "assert", "with ", "try:", "except")
    if any(last.startswith(s) for s in skip) or last.endswith(":") or "=" in last.split("#")[0]:
        return code
    lines[-1] = f"print({last})"
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a world-class mathematical olympiad competitor. Solve the problem step by step.

Rules:
- Think carefully. Break the problem into parts.
- You may write Python code in ```python ... ``` blocks. The environment has \
sympy, numpy, mpmath, itertools, collections pre-imported. Always use print() to show results.
- After computing, reason about the result before giving the final answer.
- Your final answer MUST be a non-negative integer between 0 and 99999.
- Place ONLY your final integer answer inside \\boxed{}.
- Do NOT write anything after \\boxed{}."""


class AIMO3Solver:
    def __init__(
        self,
        api_base: str = "http://localhost:8080/v1",
        model: str = "Qwen/Qwen3-30B-A3B",
        n_samples: int = 8,
        max_turns: int = 16,
        temperature: float = 0.6,
        max_tokens: int = 16384,
        code_timeout: float = 30.0,
    ):
        import requests
        self._requests = requests

        self.api_base = api_base.rstrip("/")
        self.model = model
        self.n_samples = n_samples
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.code_timeout = code_timeout

        self.sandbox = Sandbox(timeout=code_timeout)
        print(f"[Solver] model={model}, n_samples={n_samples}, max_turns={max_turns}")

    def _chat(self, messages: List[Dict], seed: int = 42) -> str:
        """Call the OpenAI-compatible chat completions endpoint."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": seed,
        }
        r = self._requests.post(
            f"{self.api_base}/chat/completions", json=payload, timeout=600
        )
        r.raise_for_status()
        choice = r.json()["choices"][0]["message"]
        content = choice.get("content") or ""
        reasoning = choice.get("reasoning_content") or ""
        if reasoning:
            return f"<think>{reasoning}</think>\n{content}"
        return content

    def _solve_once(self, problem: str, seed: int) -> Dict[str, Any]:
        """Single TIR attempt: reason → code → verify → answer."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        full_text = ""

        for turn in range(self.max_turns):
            try:
                reply = self._chat(messages, seed=seed + turn)
            except Exception as e:
                print(f"      API error on turn {turn}: {e}")
                break

            full_text += reply + "\n"
            clean = strip_think_tags(reply)
            messages.append({"role": "assistant", "content": reply})

            # Check for answer
            answer = extract_answer(clean) or extract_answer(reply)
            if answer is not None:
                return {"answer": answer, "turns": turn + 1, "text": full_text}

            # Check for code to execute
            code_blocks = extract_code_blocks(clean) or extract_code_blocks(reply)
            if code_blocks:
                outputs = []
                for code in code_blocks:
                    out = self.sandbox.execute(ensure_print(code))
                    outputs.append(out)
                exec_text = "\n---\n".join(outputs)
                messages.append({
                    "role": "user",
                    "content": (
                        f"Code output:\n```\n{exec_text}\n```\n"
                        "Continue solving. Put your final answer in \\boxed{}."
                    ),
                })
                self.sandbox.reset()
            elif turn < self.max_turns - 1:
                messages.append({
                    "role": "user",
                    "content": "Continue solving. Put your final answer in \\boxed{}.",
                })

        # Last-ditch: scan the full text for an answer
        return {
            "answer": extract_answer(full_text),
            "turns": self.max_turns,
            "text": full_text,
        }

    def solve(self, problem: str) -> Dict[str, Any]:
        """Solve with N samples + majority voting."""
        answers: List[int] = []
        candidates: List[Dict] = []

        for i in range(self.n_samples):
            t0 = time.time()
            result = self._solve_once(problem, seed=42 + i * 37)
            dt = time.time() - t0
            ans = result["answer"]
            print(f"    Sample {i+1}/{self.n_samples}: answer={ans}, "
                  f"turns={result['turns']}, time={dt:.1f}s")
            candidates.append(result)
            if ans is not None:
                answers.append(ans)

            # Early stop: if we already have strong consensus
            if len(answers) >= 3:
                top_count = Counter(answers).most_common(1)[0][1]
                if top_count >= max(3, self.n_samples // 2):
                    print(f"    Early stop: {top_count} answers agree")
                    break

        if not answers:
            return {"answer": 0, "candidates": candidates, "votes": {}}

        counter = Counter(answers)
        answer = counter.most_common(1)[0][0]
        return {"answer": answer, "candidates": candidates, "votes": dict(counter)}

    def shutdown(self):
        self.sandbox.close()


# ---------------------------------------------------------------------------
# Main: Test against reference problems
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AIMO3 Solver — test against reference problems")
    parser.add_argument("--api-base", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--problems", type=str, default=None,
                        help="Comma-separated problem IDs to solve (default: all)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    # Load reference problems
    csv_path = os.path.join(os.path.dirname(__file__), "data", "aimo3", "reference.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} reference problems from {csv_path}\n")

    # Filter to specific problems if requested
    if args.problems:
        ids = [p.strip() for p in args.problems.split(",")]
        df = df[df["id"].isin(ids)]
        print(f"Filtered to {len(df)} problems: {ids}\n")

    # Output directory
    out_dir = args.output_dir or os.path.join(
        "logs", f"aimo3_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Initialize solver
    solver = AIMO3Solver(
        api_base=args.api_base,
        model=args.model,
        n_samples=args.n_samples,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Solve each problem
    results = []
    correct = 0
    total_start = time.time()

    for idx, row in df.iterrows():
        pid = row["id"]
        problem = row["problem"]
        expected = int(row["answer"]) if "answer" in row else None

        print(f"\n{'='*60}")
        print(f"Problem {idx+1}/{len(df)} [{pid}]")
        print(f"  Expected: {expected}")
        print(f"  {problem[:120]}...")
        print(f"{'='*60}")

        t0 = time.time()
        result = solver.solve(problem)
        dt = time.time() - t0

        predicted = result["answer"]
        is_correct = predicted == expected if expected is not None else None
        if is_correct:
            correct += 1
        tag = "CORRECT" if is_correct else ("WRONG" if is_correct is False else "?")

        print(f"\n  >> [{tag}] Predicted={predicted}, Expected={expected}, Time={dt:.1f}s")
        print(f"  >> Votes: {result.get('votes', {})}")

        entry = {
            "id": pid,
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
            "votes": result.get("votes", {}),
            "time": round(dt, 1),
            "n_candidates": len(result.get("candidates", [])),
        }
        results.append(entry)

        # Save per-problem trace
        trace_path = os.path.join(out_dir, f"{pid}.json")
        with open(trace_path, "w") as f:
            trace = {**entry}
            # Save candidate texts (truncated)
            trace["candidates"] = [
                {"answer": c.get("answer"), "turns": c.get("turns")}
                for c in result.get("candidates", [])
            ]
            json.dump(trace, f, indent=2)

    total_time = time.time() - total_start

    # Summary
    answered = sum(1 for r in results if r["expected"] is not None)
    print(f"\n{'='*60}")
    print(f"RESULTS: {correct}/{answered} correct ({correct/answered*100:.0f}%)" if answered else "No answers to check")
    print(f"Total time: {total_time:.0f}s ({total_time/len(df):.0f}s per problem)")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    # Save summary
    summary = {
        "model": args.model,
        "n_samples": args.n_samples,
        "max_turns": args.max_turns,
        "correct": correct,
        "total": answered,
        "accuracy": round(correct / answered, 3) if answered else 0,
        "total_time": round(total_time, 1),
        "results": results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save submission CSV
    sub = pd.DataFrame([{"id": r["id"], "answer": r["predicted"]} for r in results])
    sub.to_csv(os.path.join(out_dir, "submission.csv"), index=False)

    solver.shutdown()


if __name__ == "__main__":
    main()
