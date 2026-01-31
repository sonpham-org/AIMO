"""
Kaggle Notebook Template: Unified AIMO3 Solver
================================================
This is a self-contained Kaggle submission notebook that combines techniques
from all three top AIMO2 solutions into one pipeline.

Hardware target: 4x L4 GPUs (AIMO2) or 1x MI300X (AIMO3)
Model: DeepSeek-R1-Distill-Qwen-14B (or any whitelisted model)

To use on Kaggle:
  1. Upload this as a notebook
  2. Add the model as a Kaggle dataset/model input
  3. Adjust MODEL_PATH and CONFIG below
  4. Submit
"""

import os
import sys
import re
import gc
import time
import queue
import subprocess
import threading
import contextlib
from collections import Counter
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# ── Configuration ──
# Change these to match your setup / competition year
CONFIG = {
    # Model
    "model_path": "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-14b-awq/1",
    "served_model_name": "deepseek-r1",

    # Inference engine
    "engine": "vllm",  # "vllm" or "lmdeploy"
    "gpu_memory_utilization": 0.95,
    "max_model_len": 16384,
    "dtype": "auto",
    "kv_cache_dtype": "auto",
    "tensor_parallel_size": 1,  # Set to 4 for multi-GPU

    # Sampling
    "n_cot_samples": 8,
    "n_code_samples": 8,
    "max_turns": 12,
    "temperature": 0.7,
    "max_tokens": 8192,

    # Strategy
    "early_stop_consensus": 5,
    "use_genselect": True,
    "use_verification": True,

    # Timeouts
    "code_timeout": 30.0,
    "server_startup_timeout": 300,
    "total_time_budget": 17400.0,  # ~4h50m
    "port": 8000,
}


# ── Prompts ──

COT_TIR_PROMPT = """\
You are a world-class IMO competitor. Solve the problem step by step with rigorous reasoning.

RULES:
1. Think carefully about each step. Show all mathematical reasoning.
2. When you need to compute, verify, or enumerate, write Python code in ```python ... ``` blocks.
3. The code runs in Jupyter with sympy, numpy, mpmath, itertools pre-imported.
4. After seeing code output, CONTINUE reasoning with the result.
5. Always use print() to display results.
6. VERIFY your answer with a final computation before boxing it.
7. Final answer must be an integer 0-99999 inside \\boxed{}."""

CODE_TIR_PROMPT = """\
You are a computational mathematician. Solve the problem primarily through Python code.

RULES:
1. Break the problem into computational steps.
2. Write Python code in ```python ... ``` blocks to solve each step.
3. The code runs in Jupyter with sympy, numpy, mpmath, itertools pre-imported.
4. Use print() to display intermediate and final results.
5. After each code execution, briefly reason about the result before the next step.
6. For the final answer, write a verification script.
7. Final answer must be an integer 0-99999 inside \\boxed{}."""

GENSELECT_PROMPT = """\
You are a mathematical judge evaluating {n} candidate solutions.

PROBLEM:
{problem}

{solutions}

Evaluate each solution for correctness. Output:
BEST: <solution_number>
ANSWER: <integer>"""


# ── Sandbox ──

from jupyter_client import KernelManager


class Sandbox:
    _port_lock = threading.Lock()
    _next_port = 50000
    PREIMPORT = (
        "import math, numpy, sympy, mpmath, itertools, collections, functools\n"
        "from sympy import *\n"
        "mpmath.mp.dps = 64"
    )

    @classmethod
    def _alloc_ports(cls, n=5):
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + n))
            cls._next_port += n
            return ports

    def __init__(self, timeout=30.0):
        self._timeout = timeout
        ports = self._alloc_ports()
        self._km = KernelManager()
        self._km.shell_port, self._km.iopub_port = ports[0], ports[1]
        self._km.stdin_port, self._km.hb_port, self._km.control_port = ports[2], ports[3], ports[4]
        env = os.environ.copy()
        env.update({"PYDEVD_DISABLE_FILE_VALIDATION": "1", "JUPYTER_PLATFORM_DIRS": "1",
                     "PYTHONWARNINGS": "ignore", "MPLBACKEND": "Agg"})
        self._km.start_kernel(env=env, extra_arguments=["--Application.log_level=CRITICAL"])
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._timeout)
        self.execute(self.PREIMPORT)

    def execute(self, code, timeout=None):
        timeout = timeout or self._timeout
        msg_id = self._client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)
        stdout, stderr = [], []
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout:
                self._km.interrupt_kernel()
                return f"[ERROR] Timed out after {timeout}s"
            try:
                msg = self._client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mt, ct = msg.get("msg_type"), msg.get("content", {})
            if mt == "stream":
                (stdout if ct.get("name") == "stdout" else stderr).append(ct.get("text", ""))
            elif mt == "error":
                stderr.append("\n".join(re.sub(r"\x1b\[[0-9;]*m", "", f) for f in ct.get("traceback", [])))
            elif mt in ("execute_result", "display_data"):
                t = ct.get("data", {}).get("text/plain")
                if t:
                    stdout.append(t if t.endswith("\n") else f"{t}\n")
            elif mt == "status" and ct.get("execution_state") == "idle":
                break
        out, err = "".join(stdout).rstrip(), "".join(stderr).rstrip()
        if err:
            return f"{out}\n{err}" if out else err
        return out or "[No output]"

    def reset(self):
        self.execute(f"%reset -f\n{self.PREIMPORT}")

    def close(self):
        with contextlib.suppress(Exception):
            self._client.stop_channels()
        with contextlib.suppress(Exception):
            self._km.shutdown_kernel(now=True)


# ── Helpers ──

def extract_answer(text):
    for pat in [r"\\boxed\s*\{\s*([0-9,]+)\s*\}", r"final\s+answer\s+is\s*[:\s]*([0-9,]+)"]:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            try:
                val = int(matches[-1].replace(",", ""))
                if 0 <= val <= 99999:
                    return val
            except ValueError:
                pass
    return None


def extract_code_blocks(text):
    return re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)


def strip_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ensure_print(code):
    lines = code.strip().split("\n")
    if not lines:
        return code
    last = lines[-1].strip()
    if "print" in last or "import" in last or not last or last.startswith("#") or "=" in last:
        return code
    lines[-1] = f"print({last})"
    return "\n".join(lines)


# ── vLLM Server Management ──

def start_vllm_server(cfg):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg["model_path"],
        "--served-model-name", cfg["served_model_name"],
        "--tensor-parallel-size", str(cfg["tensor_parallel_size"]),
        "--max-num-seqs", "64",
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--host", "0.0.0.0",
        "--port", str(cfg["port"]),
        "--dtype", cfg["dtype"],
        "--kv-cache-dtype", cfg["kv_cache_dtype"],
        "--max-model-len", str(cfg["max_model_len"]),
        "--disable-log-stats",
        "--enable-prefix-caching",
    ]
    os.makedirs("logs", exist_ok=True)
    log = open("logs/vllm_server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, log


def wait_for_server(base_url, timeout=300):
    import requests
    print("Waiting for inference server...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{base_url}/models", timeout=5)
            if r.status_code == 200:
                print(f"Server ready ({time.time() - t0:.1f}s)")
                return True
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("Server failed to start")


# ── Solver ──

class KaggleSolver:
    def __init__(self, cfg):
        import requests
        self._requests = requests
        self.cfg = cfg
        self.base_url = f"http://0.0.0.0:{cfg['port']}/v1"
        self.sandbox = Sandbox(timeout=cfg["code_timeout"])
        self._global_start = time.time()
        self._problems_remaining = 50

    def _chat(self, messages, seed=42, temperature=None):
        payload = {
            "model": self.cfg["served_model_name"],
            "messages": messages,
            "temperature": temperature or self.cfg["temperature"],
            "max_tokens": self.cfg["max_tokens"],
            "seed": seed,
        }
        r = self._requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=600)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        return f"<think>{reasoning}</think>\n{content}" if reasoning else content

    def _tir_attempt(self, problem, system_prompt, seed):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": problem}]
        full_response = ""
        code_calls = 0

        for turn in range(self.cfg["max_turns"]):
            reply = self._chat(messages, seed=seed)
            reply_clean = strip_think_tags(reply)
            full_response += reply + "\n"
            messages.append({"role": "assistant", "content": reply})

            answer = extract_answer(reply_clean) or extract_answer(reply)
            if answer is not None:
                return {"answer": answer, "response": full_response, "code_calls": code_calls}

            code_blocks = extract_code_blocks(reply_clean) or extract_code_blocks(reply)
            if code_blocks:
                outputs = [self.sandbox.execute(ensure_print(c)) for c in code_blocks]
                code_calls += len(code_blocks)
                messages.append({"role": "user", "content": f"Output:\n```\n{'---'.join(outputs)}\n```\nContinue. Answer in \\boxed{{}}."})
                self.sandbox.reset()
            elif turn < self.cfg["max_turns"] - 1:
                messages.append({"role": "user", "content": "Continue or answer in \\boxed{}."})

        return {"answer": extract_answer(full_response), "response": full_response, "code_calls": code_calls}

    def _genselect(self, problem, candidates):
        valid = [c for c in candidates if c["answer"] is not None]
        if len(valid) <= 1:
            return valid[0]["answer"] if valid else None
        solutions = ""
        for i, c in enumerate(valid[:8]):
            solutions += f"\n--- SOLUTION {i+1} (Answer: {c['answer']}) ---\n{c['response'][:2000]}\n"
        prompt = GENSELECT_PROMPT.format(n=min(len(valid), 8), problem=problem, solutions=solutions)
        try:
            reply = strip_think_tags(self._chat([{"role": "user", "content": prompt}], seed=0, temperature=0.1))
            m = re.search(r"ANSWER:\s*(\d+)", reply)
            if m:
                val = int(m.group(1))
                if 0 <= val <= 99999:
                    return val
        except Exception:
            pass
        return None

    def solve(self, problem):
        # Dynamic budget
        elapsed = time.time() - self._global_start
        remaining = max(0, self.cfg["total_time_budget"] - elapsed)
        avg = remaining / max(1, self._problems_remaining)
        if avg < 60:
            n_cot, n_code = 3, 3
        elif avg < 120:
            n_cot, n_code = 4, 4
        else:
            n_cot, n_code = self.cfg["n_cot_samples"], self.cfg["n_code_samples"]

        schedule = []
        for i in range(max(n_cot, n_code)):
            if i < n_cot:
                schedule.append(("cot", i))
            if i < n_code:
                schedule.append(("code", i))

        answers, candidates = [], []
        for idx, (mode, _) in enumerate(schedule):
            prompt = COT_TIR_PROMPT if mode == "cot" else CODE_TIR_PROMPT
            result = self._tir_attempt(problem, prompt, seed=42 + idx * 17)
            result["mode"] = mode
            candidates.append(result)
            if result["answer"] is not None:
                answers.append(result["answer"])
                top = Counter(answers).most_common(1)[0]
                if top[1] >= self.cfg["early_stop_consensus"]:
                    break

        if not answers:
            self._problems_remaining = max(0, self._problems_remaining - 1)
            return 0

        counter = Counter(answers)

        # GenSelect
        if self.cfg["use_genselect"] and len(set(answers)) > 1:
            gs = self._genselect(problem, candidates)
            answer = gs if gs is not None else counter.most_common(1)[0][0]
        else:
            answer = counter.most_common(1)[0][0]

        self._problems_remaining = max(0, self._problems_remaining - 1)
        return answer

    def shutdown(self):
        self.sandbox.close()


# ── Main ──

def main():
    # Start server
    proc, log = start_vllm_server(CONFIG)
    base_url = f"http://0.0.0.0:{CONFIG['port']}/v1"

    try:
        wait_for_server(base_url, timeout=CONFIG["server_startup_timeout"])
        solver = KaggleSolver(CONFIG)

        # Check if running in Kaggle competition mode
        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            import polars as pl
            try:
                import kaggle_evaluation.aimo_3_inference_server as aimo_server
            except ImportError:
                # Fallback for AIMO2
                import kaggle_evaluation.aimo_2_inference_server as aimo_server

            def predict(id_df, question_df, answer_df=None):
                prob_id = id_df.item(0)
                question = question_df.item(0)
                gc.disable()
                answer = solver.solve(question)
                gc.enable()
                gc.collect()
                return pl.DataFrame({"id": prob_id, "answer": answer})

            server = aimo_server.AIMO3InferenceServer(predict)
            server.serve()
        else:
            # Local evaluation mode
            test_csv = "test.csv"
            if not os.path.exists(test_csv):
                test_csv = "data/ai-mathematical-olympiad-progress-prize-3/reference.csv"
            df = pd.read_csv(test_csv)
            results = []

            for _, row in df.iterrows():
                prob_id = str(row.get("id", ""))
                problem = row.get("problem") or row.get("question", "")
                print(f"\nSolving {prob_id}...")
                answer = solver.solve(problem)
                print(f"  Answer: {answer}")
                results.append({"id": prob_id, "answer": answer})

            pd.DataFrame(results).to_csv("submission.csv", index=False)
            print("Saved submission.csv")

        solver.shutdown()
    finally:
        proc.terminate()
        proc.wait()
        log.close()


if __name__ == "__main__":
    main()
