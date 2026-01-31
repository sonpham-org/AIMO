#!/usr/bin/env python3
"""
Demo: Solve AIMO3 Reference Problems with a local LLM + code execution sandbox.

Supports two backends:
  - vllm:   Direct vLLM LLM class (requires ROCm/CUDA GPU + vllm installed)
  - openai: Any OpenAI-compatible API (e.g. llama-server, vLLM server, Ollama)

Pipeline:
  1. Extract problems from the AIMO3 PDF (or use existing reference.csv)
  2. Load model via chosen backend
  3. For each problem, multi-turn solve with Jupyter code execution
  4. Write results in sample_submission.csv format

Usage:
    # vLLM backend (default)
    .venv/bin/python demo.py
    .venv/bin/python demo.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # OpenAI-compatible backend (e.g. llama-server)
    .venv/bin/python demo.py --backend openai --api-base http://localhost:8080/v1 \
        --model Qwen3-30B-A3B --attempts 1 --max-turns 16
"""

import os
import sys
import re
import time
import json
import queue
import threading
import contextlib
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import Counter

import pandas as pd
from jupyter_client import KernelManager


# ==============================================================================
# Step 1: PDF to CSV Extraction
# ==============================================================================

def extract_problems_from_pdf(pdf_path: str, output_csv: str) -> str:
    """Extract problems from the AIMO3 PDF, or fall back to reference.csv."""
    ref_csv = os.path.join(os.path.dirname(pdf_path), "reference.csv")

    if os.path.exists(ref_csv):
        print(f"[Step 1] Found reference.csv at {ref_csv} — using it directly.")
        return ref_csv

    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError(
            "pdfplumber not installed and no reference.csv found. "
            "Install with: uv pip install pdfplumber"
        )

    print(f"[Step 1] Extracting problems from {pdf_path} ...")
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    # Heuristic parser for "Problem N" blocks
    pattern = re.compile(
        r'Problem\s+(\d+)\s*[.\n]\s*Problem:\s*(.*?)(?=\nAnswer:|\nSolution:)',
        re.DOTALL,
    )
    problems = []
    for m in pattern.finditer(full_text):
        num = int(m.group(1))
        text = m.group(2).strip()
        if len(text) > 30:
            problems.append({"id": f"prob_{num:02d}", "problem": text})

    if not problems:
        if os.path.exists(ref_csv):
            print("  PDF parsing found no problems; falling back to reference.csv.")
            return ref_csv
        raise ValueError("Could not parse problems from PDF and no fallback CSV exists")

    df = pd.DataFrame(problems)
    df.to_csv(output_csv, index=False)
    print(f"  Extracted {len(problems)} problems -> {output_csv}")
    return output_csv


# ==============================================================================
# Step 2: Jupyter Sandbox (simplified from kaggle_example.py)
# ==============================================================================

class Sandbox:
    """Isolated Jupyter kernel for safe Python code execution."""

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

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        env = os.environ.copy()
        env.update({
            "PYDEVD_DISABLE_FILE_VALIDATION": "1",
            "JUPYTER_PLATFORM_DIRS": "1",
            "PYTHONWARNINGS": "ignore",
            "MPLBACKEND": "Agg",
        })
        self._km.start_kernel(
            env=env, extra_arguments=["--Application.log_level=CRITICAL"]
        )

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._timeout)

        # Pre-import common math libraries
        self.execute(
            "import math, numpy, sympy, mpmath, itertools, collections\n"
            "mpmath.mp.dps = 64"
        )

    def execute(self, code: str, timeout: float = None) -> str:
        timeout = timeout or self._timeout
        msg_id = self._client.execute(
            code, store_history=True, allow_stdin=False, stop_on_error=False
        )
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
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

            mt = msg.get("msg_type")
            ct = msg.get("content", {})

            if mt == "stream":
                target = stdout_parts if ct.get("name") == "stdout" else stderr_parts
                target.append(ct.get("text", ""))
            elif mt == "error":
                tb = ct.get("traceback", [])
                clean = [re.sub(r"\x1b\[[0-9;]*m", "", f) for f in tb]
                stderr_parts.append("\n".join(clean))
            elif mt in ("execute_result", "display_data"):
                text = ct.get("data", {}).get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif mt == "status" and ct.get("execution_state") == "idle":
                break

        out = "".join(stdout_parts).rstrip()
        err = "".join(stderr_parts).rstrip()
        if err:
            return f"{out}\n{err}" if out else err
        return out or "[No output — use print() to see results.]"

    def reset(self):
        self.execute(
            "%reset -f\n"
            "import math, numpy, sympy, mpmath, itertools, collections\n"
            "mpmath.mp.dps = 64"
        )

    def close(self):
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()
        with contextlib.suppress(Exception):
            self._km.shutdown_kernel(now=True)
        with contextlib.suppress(Exception):
            self._km.cleanup_resources()

    def __del__(self):
        self.close()


# ==============================================================================
# Step 3: Answer Extraction Helpers
# ==============================================================================

SYSTEM_PROMPT = (
    "You are a world-class International Mathematical Olympiad (IMO) competitor.\n"
    "Solve the given problem step by step. Be extremely careful — double-check "
    "every calculation.\n\n"
    "TOOLS: When you need to compute something, write Python code inside a "
    "```python\\n...\\n``` block. The code runs in a Jupyter environment with "
    "sympy, numpy, mpmath pre-imported. Always use print() to display results.\n\n"
    "ANSWER: The final answer must be a non-negative integer between 0 and 99999. "
    "When you are confident, write your final answer inside \\boxed{} and stop. "
    "Place ONLY the integer inside \\boxed{} — nothing else after it."
)


def extract_code_blocks(text: str) -> List[str]:
    """Extract ```python ... ``` code blocks from model output."""
    return re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)


def extract_answer(text: str) -> Optional[int]:
    """Extract the final \\boxed{} answer from model output."""
    patterns = [
        r"\\boxed\s*\{\s*([0-9,]+)\s*\}",
        r"final\s+answer\s+is\s*[:\s]*([0-9,]+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            try:
                val = int(matches[-1].replace(",", ""))
                if 0 <= val <= 99999:
                    return val
            except ValueError:
                pass
    return None


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from DeepSeek-R1 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ==============================================================================
# Step 4: Solvers
# ==============================================================================

# ---------- vLLM Solver (direct, no server) ----------

class Solver:
    """Direct vLLM inference with Jupyter code execution."""

    def __init__(
        self,
        model: str,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "float16",
        quantization: str = None,
        attempts: int = 3,
        max_turns: int = 16,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        code_timeout: float = 30.0,
    ):
        from vllm import LLM, SamplingParams

        # ── AMD Strix Halo (gfx1151) environment ──
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")
        os.environ.setdefault("VLLM_TARGET_DEVICE", "rocm")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("AMD_LOG_LEVEL", "0")

        self.model_name = model
        self.attempts = attempts
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.code_timeout = code_timeout
        self._SamplingParams = SamplingParams

        print(f"[Step 2] Loading model: {model}")
        print(f"  dtype={dtype}, gpu_memory_utilization={gpu_memory_utilization}")
        print(f"  max_model_len={max_model_len}")

        t0 = time.perf_counter()
        self.llm = LLM(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True,
        )
        load_time = time.perf_counter() - t0
        print(f"  Model loaded in {load_time:.1f}s")

        self.tokenizer = self.llm.get_tokenizer()

        print("  Warmup run...")
        warmup_params = SamplingParams(temperature=0, max_tokens=8)
        self.llm.generate(["Hi"], warmup_params)
        print("  Warmup done.\n")

        print(f"[Step 3] Creating Jupyter sandbox...")
        self.sandbox = Sandbox(timeout=code_timeout)
        print(f"  Sandbox ready.\n")

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    parts.append(f"<|system|>\n{content}")
                elif role == "user":
                    parts.append(f"<|user|>\n{content}")
                elif role == "assistant":
                    parts.append(f"<|assistant|>\n{content}")
            parts.append("<|assistant|>\n")
            return "\n".join(parts)

    def _solve_once(self, problem: str, attempt_idx: int) -> Dict[str, Any]:
        """Single attempt. Returns dict with 'answer', 'messages', 'turns'."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]

        full_response = ""
        sampling = self._SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.95,
            seed=42 + attempt_idx,
        )
        turns_used = 0

        for turn in range(self.max_turns):
            turns_used = turn + 1
            prompt = self._format_prompt(messages)
            outputs = self.llm.generate([prompt], sampling)
            reply = outputs[0].outputs[0].text

            reply_clean = strip_think_tags(reply)
            full_response += reply + "\n"
            messages.append({"role": "assistant", "content": reply})

            answer = extract_answer(reply_clean) or extract_answer(reply)
            if answer is not None:
                return {"answer": answer, "messages": messages, "turns": turns_used}

            code_blocks = extract_code_blocks(reply_clean) or extract_code_blocks(reply)
            if not code_blocks:
                if turn < self.max_turns - 1:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Continue solving. Write Python code if you need to "
                            "compute something, or put your final integer answer "
                            "in \\boxed{}."
                        ),
                    })
                continue

            all_outputs = []
            for code in code_blocks:
                result = self.sandbox.execute(code, timeout=self.code_timeout)
                all_outputs.append(result)

            exec_output = "\n---\n".join(all_outputs)
            messages.append({
                "role": "user",
                "content": (
                    f"Code execution output:\n```\n{exec_output}\n```\n"
                    "Continue solving based on these results. "
                    "Put your final integer answer (0-99999) in \\boxed{}."
                ),
            })

        answer = extract_answer(full_response)
        return {"answer": answer, "messages": messages, "turns": turns_used}

    def solve(self, problem: str) -> Dict[str, Any]:
        """Solve with multiple attempts and majority voting.

        Returns dict with 'answer', 'attempts' (list of per-attempt traces).
        """
        answers: List[int] = []
        attempts_data: List[Dict[str, Any]] = []

        for i in range(self.attempts):
            print(f"    Attempt {i + 1}/{self.attempts} ...", end=" ", flush=True)
            try:
                result = self._solve_once(problem, attempt_idx=i)
                ans = result["answer"]
                print(f"-> {ans}")
                attempts_data.append({
                    "attempt": i + 1,
                    "answer": ans,
                    "turns": result["turns"],
                    "messages": result["messages"],
                })
                if ans is not None:
                    answers.append(ans)
            except Exception as e:
                print(f"-> Error: {e}")
                attempts_data.append({
                    "attempt": i + 1,
                    "answer": None,
                    "error": str(e),
                    "turns": 0,
                    "messages": [],
                })
            finally:
                self.sandbox.reset()

        if not answers:
            print("    No valid answer found — defaulting to 0")
            return {"answer": 0, "attempts": attempts_data}

        counter = Counter(answers)
        best = counter.most_common(1)[0][0]
        print(f"    Votes: {dict(counter)} -> Selected: {best}")
        return {"answer": best, "attempts": attempts_data}

    def shutdown(self):
        self.sandbox.close()


# ---------- OpenAI-compatible API Solver ----------

class OpenAISolver:
    """Solver using any OpenAI-compatible API (llama-server, vLLM server, etc.)."""

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:8080/v1",
        attempts: int = 1,
        max_turns: int = 16,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        code_timeout: float = 30.0,
    ):
        import requests as _requests
        self._requests = _requests

        self.model_name = model
        self.api_base = api_base.rstrip("/")
        self.attempts = attempts
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.code_timeout = code_timeout

        # Verify server is reachable
        print(f"[Step 2] Using OpenAI-compatible API at {self.api_base}")
        print(f"  Model: {model}")
        try:
            r = _requests.get(f"{self.api_base}/models", timeout=10)
            r.raise_for_status()
            models = r.json().get("data", [])
            if models:
                print(f"  Available models: {[m['id'] for m in models]}")
            print(f"  Server is reachable.\n")
        except Exception as e:
            print(f"  WARNING: Could not reach server: {e}\n")

        print(f"[Step 3] Creating Jupyter sandbox...")
        self.sandbox = Sandbox(timeout=code_timeout)
        print(f"  Sandbox ready.\n")

    def _chat_completion(self, messages: List[Dict[str, str]], seed: int = 42) -> str:
        """Call the /v1/chat/completions endpoint.

        Handles models with thinking/reasoning mode (e.g. Qwen3) where the
        response may include a 'reasoning_content' field alongside 'content'.
        Returns the combined text so answer/code extraction works on both.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 0.95,
            "seed": seed,
        }
        r = self._requests.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            timeout=600,
        )
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        # Wrap reasoning in <think> tags so strip_think_tags() can handle it
        if reasoning:
            return f"<think>{reasoning}</think>\n{content}"
        return content

    def _solve_once(self, problem: str, attempt_idx: int) -> Dict[str, Any]:
        """Single attempt. Returns dict with 'answer', 'messages', 'turns'."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]

        full_response = ""
        turns_used = 0

        for turn in range(self.max_turns):
            turns_used = turn + 1
            reply = self._chat_completion(messages, seed=42 + attempt_idx)

            reply_clean = strip_think_tags(reply)
            full_response += reply + "\n"
            messages.append({"role": "assistant", "content": reply})

            answer = extract_answer(reply_clean) or extract_answer(reply)
            if answer is not None:
                return {"answer": answer, "messages": messages, "turns": turns_used}

            code_blocks = extract_code_blocks(reply_clean) or extract_code_blocks(reply)
            if not code_blocks:
                if turn < self.max_turns - 1:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Continue solving. Write Python code if you need to "
                            "compute something, or put your final integer answer "
                            "in \\boxed{}."
                        ),
                    })
                continue

            all_outputs = []
            for code in code_blocks:
                result = self.sandbox.execute(code, timeout=self.code_timeout)
                all_outputs.append(result)

            exec_output = "\n---\n".join(all_outputs)
            messages.append({
                "role": "user",
                "content": (
                    f"Code execution output:\n```\n{exec_output}\n```\n"
                    "Continue solving based on these results. "
                    "Put your final integer answer (0-99999) in \\boxed{}."
                ),
            })

        answer = extract_answer(full_response)
        return {"answer": answer, "messages": messages, "turns": turns_used}

    def solve(self, problem: str) -> Dict[str, Any]:
        """Solve with multiple attempts and majority voting.

        Returns dict with 'answer', 'attempts' (list of per-attempt traces).
        """
        answers: List[int] = []
        attempts_data: List[Dict[str, Any]] = []

        for i in range(self.attempts):
            print(f"    Attempt {i + 1}/{self.attempts} ...", end=" ", flush=True)
            try:
                result = self._solve_once(problem, attempt_idx=i)
                ans = result["answer"]
                print(f"-> {ans}")
                attempts_data.append({
                    "attempt": i + 1,
                    "answer": ans,
                    "turns": result["turns"],
                    "messages": result["messages"],
                })
                if ans is not None:
                    answers.append(ans)
            except Exception as e:
                print(f"-> Error: {e}")
                attempts_data.append({
                    "attempt": i + 1,
                    "answer": None,
                    "error": str(e),
                    "turns": 0,
                    "messages": [],
                })
            finally:
                self.sandbox.reset()

        if not answers:
            print("    No valid answer found — defaulting to 0")
            return {"answer": 0, "attempts": attempts_data}

        counter = Counter(answers)
        best = counter.most_common(1)[0][0]
        print(f"    Votes: {dict(counter)} -> Selected: {best}")
        return {"answer": best, "attempts": attempts_data}

    def shutdown(self):
        self.sandbox.close()


# ==============================================================================
# CLI & Main
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="AIMO3 Demo — solve reference problems with a local LLM"
    )
    p.add_argument(
        "--backend", choices=["vllm", "openai"], default="vllm",
        help="Inference backend: 'vllm' (direct) or 'openai' (API-compatible server)",
    )
    p.add_argument(
        "--api-base", default="http://localhost:8080/v1",
        help="Base URL for OpenAI-compatible API (used with --backend openai)",
    )
    p.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Model name (HuggingFace ID for vllm, or model name for openai backend)",
    )
    p.add_argument(
        "--input-csv", default=None,
        help="Input CSV with 'id' and 'problem' columns (default: auto from PDF)",
    )
    p.add_argument(
        "--output-dir", default="output",
        help="Base directory for run outputs (default: output/)",
    )
    p.add_argument(
        "--output-csv", default=None,
        help="Output CSV path (default: auto inside --output-dir)",
    )
    p.add_argument(
        "--pdf",
        default="data/ai-mathematical-olympiad-progress-prize-3/AIMO3_Reference_Problems.pdf",
        help="Path to AIMO3 reference problems PDF",
    )
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--quantization", default=None, help="e.g. awq, gptq")
    p.add_argument(
        "--attempts", type=int, default=3,
        help="Solution attempts per problem (majority vote)",
    )
    p.add_argument(
        "--max-turns", type=int, default=16,
        help="Max conversation turns per attempt",
    )
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--code-timeout", type=float, default=30.0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("logs", exist_ok=True)

    # ── Create timestamped run directory ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1].replace(".gguf", "")
    run_name = f"{timestamp}_{model_short}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    if args.output_csv is None:
        args.output_csv = os.path.join(run_dir, "submission.csv")

    # Save run config
    config = {k: v for k, v in vars(args).items()}
    config["run_dir"] = run_dir
    config["run_name"] = run_name
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Tee stdout to a log file in the run directory
    log_path = os.path.join(run_dir, "run.log")
    log_file = open(log_path, "w")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"Run directory: {run_dir}")
    print(f"Run config saved to: {run_dir}/config.json\n")

    # ── GPU check (skip for openai backend) ──
    if args.backend == "vllm":
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU: {name} ({mem:.1f} GB)\n")
            else:
                print("WARNING: No GPU detected by PyTorch/ROCm\n")
        except Exception as e:
            print(f"WARNING: Could not query GPU: {e}\n")

    # ── Step 1: Get problems ──
    if args.input_csv and os.path.exists(args.input_csv):
        problems_csv = args.input_csv
    else:
        problems_csv = extract_problems_from_pdf(
            args.pdf,
            os.path.join(run_dir, "problems_extracted.csv"),
        )

    print(f"Reading problems from: {problems_csv}")
    df = pd.read_csv(problems_csv)
    print(f"Found {len(df)} problems\n")

    # Check for known answers (for scoring)
    known_answers: Dict[str, int] = {}
    if "answer" in df.columns:
        known_answers = {
            str(row["id"]): int(row["answer"]) for _, row in df.iterrows()
        }

    # ── Step 2-3: Initialize solver ──
    if args.backend == "openai":
        solver = OpenAISolver(
            model=args.model,
            api_base=args.api_base,
            attempts=args.attempts,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            code_timeout=args.code_timeout,
        )
    else:
        solver = Solver(
            model=args.model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            quantization=args.quantization,
            attempts=args.attempts,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            code_timeout=args.code_timeout,
        )

    # ── Step 4: Solve each problem ──
    results: List[Dict[str, Any]] = []
    traces_dir = os.path.join(run_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    total_t0 = time.time()

    try:
        for idx, row in df.iterrows():
            prob_id = str(row.get("id", f"prob_{idx}"))
            problem_text = row.get("problem") or row.get("question", "")

            print(f"{'=' * 70}")
            print(f"Problem {idx + 1}/{len(df)}  (ID: {prob_id})")
            print(f"{'=' * 70}")
            preview = problem_text[:300]
            if len(problem_text) > 300:
                preview += "..."
            print(f"{preview}\n")

            t0 = time.time()
            solve_result = solver.solve(problem_text)
            elapsed = time.time() - t0
            answer = solve_result["answer"]

            results.append({"id": prob_id, "answer": answer})

            tag = ""
            expected = known_answers.get(prob_id)
            if expected is not None:
                verdict = "CORRECT" if answer == expected else "WRONG"
                tag = f"  [{verdict}, expected={expected}]"

            print(f"  => Answer: {answer}{tag}  ({elapsed:.1f}s)\n")

            # Save per-problem trace
            trace = {
                "id": prob_id,
                "problem": problem_text,
                "answer": answer,
                "expected": expected,
                "correct": answer == expected if expected is not None else None,
                "time_s": round(elapsed, 1),
                "attempts": solve_result["attempts"],
            }
            trace_path = os.path.join(traces_dir, f"{prob_id}.json")
            with open(trace_path, "w") as f:
                json.dump(trace, f, indent=2, ensure_ascii=False)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # ── Step 5: Write submission CSV + results summary ──
        if results:
            out_df = pd.DataFrame(results)
            out_df.to_csv(args.output_csv, index=False)
            total_elapsed = time.time() - total_t0

            correct = 0
            if known_answers and results:
                correct = sum(
                    1 for r in results
                    if r["id"] in known_answers and r["answer"] == known_answers[r["id"]]
                )

            print(f"\n{'=' * 70}")
            print(f"RESULTS")
            print(f"{'=' * 70}")
            print(f"Run directory:       {run_dir}")
            print(f"Submission saved to: {args.output_csv}")
            print(f"Problems attempted:  {len(results)}/{len(df)}")
            if known_answers:
                print(f"Accuracy:            {correct}/{len(results)} "
                      f"({100 * correct / len(results):.0f}%)")
            print(f"Total time:          {total_elapsed:.1f}s")
            print(f"{'=' * 70}")

            # Save machine-readable results summary
            summary = {
                "model": args.model,
                "backend": args.backend,
                "attempts": args.attempts,
                "max_turns": args.max_turns,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "problems_attempted": len(results),
                "problems_total": len(df),
                "correct": correct if known_answers else None,
                "accuracy": correct / len(results) if known_answers else None,
                "total_time_s": round(total_elapsed, 1),
                "results": results,
            }
            summary_path = os.path.join(run_dir, "results.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Results summary:     {summary_path}")
        else:
            print("No results to save.")

        solver.shutdown()
        log_file.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
