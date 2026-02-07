"""
AIMO Progress Prize 3 — Kaggle Submission Notebook
===================================================
Strategy: TIR (Tool-Integrated Reasoning) + Majority Voting
Model: Qwen3-30B-A3B-Thinking-2507 (MoE, 3B active — fast & strong reasoning)

Kaggle setup:
  1. Add model as input: Qwen/Qwen3-30B-A3B (or Thinking-2507 variant)
     Path will be like /kaggle/input/qwen3-30b-a3b/transformers/default/1
  2. Enable GPU (H100)
  3. Disable Internet
  4. Submit

To test locally (no vLLM needed, uses llama-server or any OpenAI-compatible API):
    python kaggle_submission.py --api-base http://localhost:8080/v1 --model-name my-model
"""

# ── Cell 1: Environment Setup ──────────────────────────────────────────────

import os
import sys
import re
import gc
import time
import tempfile
import subprocess
import warnings
from typing import Optional
from collections import Counter, defaultdict

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# NVIDIA-specific (Kaggle H100)
if os.path.exists("/usr/local/cuda/bin/ptxas"):
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

warnings.simplefilter("ignore")

# ── Cell 2: Imports ────────────────────────────────────────────────────────

import pandas as pd
import polars as pl

try:
    import kaggle_evaluation.aimo_3_inference_server
    HAS_KAGGLE_EVAL = True
except ImportError:
    HAS_KAGGLE_EVAL = False

# ── Cell 3: Configuration ─────────────────────────────────────────────────

# --- Model ---
# Change MODEL_PATH to your Kaggle model input path.
# The model is auto-discovered if you don't set it explicitly.
MODEL_PATH = None  # Set below by find_model_path()
MODEL_NAME = "Qwen3-30B-A3B"

# --- Inference ---
MAX_TOKENS = 32768          # Max tokens per generation (thinking + answer)
MAX_ROUNDS = 3              # Max TIR rounds (code→output→continue)
N_PROMPTS = 10              # Number of diverse prompts (= number of samples)
TEMPERATURE = 0.6
MIN_P = 0.05
SEED = 42
CODE_TIMEOUT = 10           # Seconds per code execution

# --- Time management ---
CUTOFF_TIME = time.time() + (8 * 60 + 45) * 60  # 8h45m (leave 15min buffer for 9h limit)

# --- System prompts (diverse for majority voting) ---
SYSTEM_PROMPTS = [
    (
        "You are a world-class IMO competitor. Think step by step. "
        "You may write Python code in ```python ... ``` blocks — the environment has "
        "sympy, numpy, mpmath pre-imported. Always use print() for results. "
        "Your final answer must be an integer 0-99999. Put it in \\boxed{}."
    ),
    (
        "You are an expert mathematical problem solver for olympiad-style problems. "
        "Break the problem into cases, verify each step, and use code to check. "
        "Write Python in ```python ... ``` blocks (sympy, numpy available). "
        "Final answer as integer 0-99999 in \\boxed{}."
    ),
    (
        "Solve this competition math problem carefully. Show your reasoning step by step. "
        "Use Python code (```python ... ```) to verify computations — sympy and numpy are available. "
        "Double-check your answer. Put the final integer (0-99999) in \\boxed{}."
    ),
    (
        "You are a careful competition mathematician. Think deeply about this problem. "
        "Use algebraic reasoning and verify with Python code when helpful. "
        "The environment has sympy, numpy, mpmath. Use print() for output. "
        "Final answer: integer 0-99999 in \\boxed{}."
    ),
    (
        "Approach this olympiad problem systematically. Consider multiple methods. "
        "Write Python code in ```python ... ``` blocks to compute and verify. "
        "sympy, numpy, mpmath, itertools are available. "
        "Put your final integer answer (0-99999) in \\boxed{}."
    ),
] * 2  # Repeat to get 10 prompts; vLLM batching handles diversity via temperature


# ── Cell 4: Find Model Path ───────────────────────────────────────────────

def find_model_path():
    """Auto-discover model path from Kaggle inputs."""
    candidates = [
        # Qwen3-30B-A3B variants
        "/kaggle/input/qwen3-30b-a3b-thinking-2507/transformers/default/1",
        "/kaggle/input/qwen3-30b-a3b/transformers/default/1",
        "/kaggle/input/qwen-3/transformers/30b-a3b/1",
        # Fallback: search all inputs
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "config.json")):
            return path
    # Walk /kaggle/input to find any model
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and "tokenizer.json" in files:
                return root
    return None


if MODEL_PATH is None:
    MODEL_PATH = find_model_path()
    if MODEL_PATH:
        print(f"Found model: {MODEL_PATH}")
    else:
        print("Warning: No model found. Will need --api-base for local testing.")


# ── Cell 5: Code Execution (subprocess-based, simple & Kaggle-safe) ──────

class PythonREPL:
    """Execute Python code in a subprocess with timeout."""
    def __init__(self, timeout=CODE_TIMEOUT):
        self.timeout = timeout

    def __call__(self, code: str) -> tuple[bool, str]:
        # Prepend common imports
        full_code = (
            "import math, numpy as np, sympy as sp, mpmath, itertools, collections\n"
            "from sympy import *\n"
            "mpmath.mp.dps = 64\n"
        ) + code
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "run.py")
            with open(path, "w") as f:
                f.write(full_code)
            try:
                result = subprocess.run(
                    [sys.executable, path],
                    capture_output=True, text=True, timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return False, f"Timed out after {self.timeout}s"
            if result.returncode == 0:
                return True, result.stdout.strip()
            return False, result.stderr.strip()[-500:]  # Truncate long errors

repl = PythonREPL()


# ── Cell 6: Answer Extraction ─────────────────────────────────────────────

def extract_boxed_answers(text: str) -> list[int]:
    """Extract integer answers from \\boxed{...} patterns."""
    answers = []
    # Find all \boxed{...} with balanced braces
    for needle in [r"\boxed{", "boxed{"]:
        i = 0
        while True:
            j = text.find(needle, i)
            if j < 0:
                break
            k = j + len(needle)
            depth = 1
            buf = []
            while k < len(text) and depth > 0:
                ch = text[k]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
                buf.append(ch)
                k += 1
            payload = "".join(buf).replace(",", "").replace("_", "").strip()
            for num_str in re.findall(r"\b\d{1,5}\b", payload):
                n = int(num_str)
                if 0 <= n <= 99999:
                    answers.append(n)
            i = max(k, j + 1)
    return answers


def extract_python_code(text: str) -> list[str]:
    """Extract ```python ... ``` code blocks."""
    return re.findall(r"```python\s*\n?(.*?)```", text, re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def select_answer(answers: list) -> int:
    """Majority vote over valid answers."""
    valid = [int(a) for a in answers if 0 <= int(a) <= 99999]
    if not valid:
        return 0
    return Counter(valid).most_common(1)[0][0]


# ── Cell 7: Model Wrapper ─────────────────────────────────────────────────

class Model:
    """Wraps vLLM (Kaggle) or OpenAI-compatible API (local testing)."""

    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None
        self.chat_mode = None
        self._api_client = None  # For local testing with external server
        self._api_model = None

    def load_vllm(self):
        """Load model with vLLM Python API (for Kaggle)."""
        from vllm import LLM, SamplingParams
        import torch

        print(f"Loading model from {MODEL_PATH}...")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.llm = LLM(
            MODEL_PATH,
            dtype="bfloat16",
            max_num_seqs=64,
            max_model_len=MAX_TOKENS,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.94,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            min_p=MIN_P,
            skip_special_tokens=True,
            max_tokens=MAX_TOKENS,
            seed=SEED,
        )
        # Detect thinking mode support
        self.chat_mode = self._detect_chat_mode()
        print(f"Chat mode: {self.chat_mode}")
        print("Model loaded!")

    def load_api(self, base_url: str, model_name: str):
        """Use an external OpenAI-compatible API (for local testing)."""
        import requests
        self._api_client = requests
        self._api_base = base_url.rstrip("/")
        self._api_model = model_name
        print(f"Using API: {self._api_base}, model: {self._api_model}")

    def _detect_chat_mode(self) -> str:
        """Detect whether tokenizer supports enable_thinking."""
        test = [{"role": "user", "content": "test"}]
        for mode in ["enable_thinking_true", "enable_thinking_false", "no_kwargs"]:
            try:
                self._apply_template(test, mode)
                return mode
            except Exception:
                continue
        return "no_kwargs"

    def _apply_template(self, messages: list, mode: str = None) -> str:
        mode = mode or self.chat_mode or "no_kwargs"
        kwargs = dict(conversation=messages, tokenize=False, add_generation_prompt=True)
        if mode == "enable_thinking_true":
            kwargs["enable_thinking"] = True
        elif mode == "enable_thinking_false":
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(**kwargs)

    def generate_batch(self, messages_batch: list) -> list[str]:
        """Generate responses for a batch of conversation histories."""
        t0 = time.time()
        n = len(messages_batch)

        if self._api_client is not None:
            # Sequential API calls for local testing
            results = []
            for idx, messages in enumerate(messages_batch):
                call_t0 = time.time()
                resp = self._api_client.post(
                    f"{self._api_base}/chat/completions",
                    json={
                        "model": self._api_model,
                        "messages": messages,
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                        "seed": SEED,
                    },
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]["message"]
                content = choice.get("content") or ""
                reasoning = choice.get("reasoning_content") or ""
                if reasoning:
                    content = f"<think>{reasoning}</think>\n{content}"

                # Log: token counts and preview
                usage = data.get("usage", {})
                tok_in = usage.get("prompt_tokens", "?")
                tok_out = usage.get("completion_tokens", "?")
                preview = strip_think(content).replace("\n", " ")[:120]
                dt = time.time() - call_t0
                print(f"    [{idx+1}/{n}] {tok_in}→{tok_out} tok, {dt:.1f}s | {preview}...")

                results.append(content)
            print(f"  Batch done: {n} calls in {time.time()-t0:.1f}s")
            return results

        # vLLM batch generation
        prompts = [self._apply_template(msgs) for msgs in messages_batch]
        outputs = self.llm.generate(prompts, self.sampling_params)
        results = []
        for idx, o in enumerate(outputs):
            text = o.outputs[0].text
            tok_out = len(o.outputs[0].token_ids)
            preview = strip_think(text).replace("\n", " ")[:120]
            print(f"    [{idx+1}/{n}] {tok_out} tok | {preview}...")
            results.append(text)
        print(f"  Batch done: {n} generations in {time.time()-t0:.1f}s")
        return results

    def predict(self, problem: str) -> int:
        """Solve one problem with TIR + majority voting."""
        if time.time() > CUTOFF_TIME:
            return 0

        # Create batch of diverse prompts
        msgs_batch = [
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": problem},
            ]
            for prompt in SYSTEM_PROMPTS[:N_PROMPTS]
        ]

        all_answers = []

        for round_idx in range(MAX_ROUNDS):
            if time.time() > CUTOFF_TIME:
                break

            print(f"  Round {round_idx+1}/{MAX_ROUNDS}: generating {len(msgs_batch)} samples...")
            # Generate responses for all conversations in batch
            responses = self.generate_batch(msgs_batch)

            # Process each response
            next_batch = []
            for i, resp in enumerate(responses):
                msgs_batch[i].append({"role": "assistant", "content": resp})

                cleaned = strip_think(resp)

                # Extract boxed answers
                boxed = extract_boxed_answers(cleaned)
                if not boxed:
                    boxed = extract_boxed_answers(resp)
                all_answers.extend(boxed)

                # Extract and execute code
                codes = extract_python_code(cleaned)
                if not codes:
                    codes = extract_python_code(resp)

                code_answers = []
                code_output_text = ""
                for code in codes:
                    if time.time() > CUTOFF_TIME:
                        break
                    success, output = repl(code)
                    status = "OK" if success else "ERR"
                    print(f"      Code exec [{status}]: {output[:100]}{'...' if len(output)>100 else ''}")
                    if success and output:
                        code_output_text += output + "\n"
                        # Extract numbers from code output
                        for num_str in re.findall(r"\b\d{1,5}\b", output):
                            n = int(num_str)
                            if 0 <= n <= 99999:
                                code_answers.append(n)

                # If no boxed answer yet and we have code output, continue the conversation
                if not boxed and code_output_text.strip():
                    msgs_batch[i].append({
                        "role": "user",
                        "content": (
                            f"Code output:\n```\n{code_output_text.strip()}\n```\n"
                            "Continue solving. Put your final answer in \\boxed{}."
                        ),
                    })
                    next_batch.append(msgs_batch[i])
                elif not boxed and round_idx < MAX_ROUNDS - 1:
                    msgs_batch[i].append({
                        "role": "user",
                        "content": "Continue solving. Put your final answer in \\boxed{}.",
                    })
                    next_batch.append(msgs_batch[i])

            # Update batch to only include unresolved conversations
            msgs_batch = next_batch

            # Early stop if we have enough answers with consensus
            if len(all_answers) >= 5:
                top_count = Counter(all_answers).most_common(1)[0][1]
                if top_count >= 3:
                    break

            if not msgs_batch:
                break

        # Final: force remaining conversations to give an answer
        if msgs_batch and time.time() <= CUTOFF_TIME:
            for msgs in msgs_batch:
                msgs.append({"role": "user", "content": "Output your final answer now as \\boxed{N}."})
            try:
                final_responses = self.generate_batch(msgs_batch)
                for resp in final_responses:
                    cleaned = strip_think(resp)
                    all_answers.extend(extract_boxed_answers(cleaned))
                    all_answers.extend(extract_boxed_answers(resp))
            except Exception:
                pass

        if not all_answers:
            return 0

        answer = select_answer(all_answers)
        print(f"  Votes: {dict(Counter(all_answers))}, Selected: {answer}")
        return answer


model = Model()


# ── Cell 8: Predict Function (Kaggle API) ─────────────────────────────────

def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """Kaggle evaluation API entry point."""
    pid = id_.item(0)
    question_text = question.item(0)

    print(f"\n{'='*50}")
    print(f"Problem {pid}")
    print(f"{'='*50}")

    result = model.predict(question_text)
    print(f"Final answer: {result}")

    return pl.DataFrame({"id": pid, "answer": result})


# ── Cell 9: Main Entry Point ──────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AIMO3 Solver")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL (for local testing)")
    parser.add_argument("--model-name", default="Qwen3-30B-A3B-Q4_K_M.gguf", help="Model name for API calls")
    parser.add_argument("--input-csv", default=None, help="Input CSV for local testing")
    parser.add_argument("--output-csv", default="submission.csv")
    args = parser.parse_args()

    # Load model
    if args.api_base:
        # Local testing with external server
        model.load_api(args.api_base, args.model_name)
    elif MODEL_PATH:
        # Kaggle: load vLLM
        model.load_vllm()
    else:
        print("Error: No model path found and no --api-base specified.")
        sys.exit(1)

    # --- Kaggle competition mode ---
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN") and HAS_KAGGLE_EVAL:
        inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
        inference_server.serve()

    # --- Local gateway mode (simulates Kaggle evaluation) ---
    elif HAS_KAGGLE_EVAL and os.path.exists("/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv"):
        inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
        inference_server.run_local_gateway(
            ("/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv",)
        )

    # --- Local CSV mode (for development) ---
    else:
        input_csv = args.input_csv
        if not input_csv:
            for candidate in ["data/aimo3/reference.csv", "data/aimo3/test.csv"]:
                if os.path.exists(candidate):
                    input_csv = candidate
                    break
        if not input_csv or not os.path.exists(input_csv):
            print("No input CSV found. Use --input-csv.")
            sys.exit(1)

        df = pd.read_csv(input_csv)
        n = len(df)
        has_answers = "answer" in df.columns
        correct = 0
        results = []

        for idx, row in df.iterrows():
            pid = row["id"]
            question = row.get("problem") or row.get("question")
            expected = int(row["answer"]) if has_answers else None

            print(f"\n{'='*50}")
            print(f"Problem {idx+1}/{n} [{pid}] (expected: {expected})")
            print(f"{'='*50}")

            answer = model.predict(question)
            tag = ""
            if expected is not None:
                is_correct = answer == expected
                if is_correct:
                    correct += 1
                tag = f" [{'CORRECT' if is_correct else 'WRONG'}]"
            print(f"  >> Answer: {answer}{tag}")
            results.append({"id": pid, "answer": answer})

        pd.DataFrame(results).to_csv(args.output_csv, index=False)
        print(f"\nSaved: {args.output_csv}")
        if has_answers:
            print(f"Score: {correct}/{n} ({correct/n*100:.0f}%)")


if __name__ == "__main__":
    main()
