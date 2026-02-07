"""
AIMO Progress Prize 3 — 2nd Place (imagination-research) Kaggle Submission
==========================================================================
Strategy: Dual Prompting (CoT + Code) × TIR × Early Stopping × Majority Voting
Model: DeepSeek-R1-Distill-Qwen-14B (SFT+DPO fine-tuned by imagination-research)

Reference: https://github.com/imagination-research/aimo2

Kaggle setup:
  1. Add model input: imagination-research/deepseek-14b-sft-dpo2
     (or deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-AWQ as fallback)
  2. Enable GPU (H100)
  3. Disable Internet
  4. Submit

Local testing (RTX 4090 or any GPU with >=16GB):
  # Option A: Start vLLM server separately, then run:
  python kaggle_submission.py --api-base http://localhost:8080/v1 --model-name my-model

  # Option B: Let the script start vLLM directly:
  python kaggle_submission.py --local-model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-AWQ
"""

# ── Cell 1: Environment Setup ──────────────────────────────────────────────

import os
import sys
import re
import gc
import time
import json
import tempfile
import subprocess
import warnings
from typing import Optional
from collections import Counter, defaultdict

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# ── Cell 3: Model Configuration ───────────────────────────────────────────
# === MODEL CONFIGURATION ===
# Option A: imagination-research fine-tuned model (default — best accuracy)
# Upload from HuggingFace: imagination-research/deepseek-14b-sft-dpo2
MODEL_PATH = None  # Auto-discovered below
MODEL_CANDIDATES = [
    # Option A: Pre-fine-tuned SFT+DPO model (imagination-research)
    "/kaggle/input/deepseek-14b-sft-dpo2/transformers/default/1",
    "/kaggle/input/deepseek-14b-sft-dpo2",
    # Option B: Custom fine-tuned model (upload as Kaggle dataset)
    # "/kaggle/input/my-custom-finetuned-model",
    # Option C: Off-the-shelf AWQ quantized (no training needed)
    "/kaggle/input/deepseek-r1-distill-qwen-14b-awq/transformers/default/1",
    "/kaggle/input/deepseek-r1-distill-qwen-14b-awq",
    # Option D: Full-precision base model
    "/kaggle/input/deepseek-r1-distill-qwen-14b/transformers/default/1",
]

# ── Cell 4: Inference Configuration ──────────────────────────────────────

# --- Sampling ---
N_COT_SAMPLES = 7           # Chain-of-thought reasoning samples
N_CODE_SAMPLES = 8          # Code-first solving samples
TOTAL_SAMPLES = N_COT_SAMPLES + N_CODE_SAMPLES  # 15 total per problem
TEMPERATURE = 0.7
MAX_TOKENS = 32768          # Max tokens per generation
MAX_TIR_ROUNDS = 3          # Max TIR rounds per sample
CODE_TIMEOUT = 30           # Seconds per code execution
SEED = 42

# --- Early stopping ---
EARLY_STOP_CONSENSUS = 5    # Stop when N answers agree

# --- Time management ---
TOTAL_TIME_LIMIT = 9 * 3600       # 9 hours
TIME_BUFFER = 15 * 60             # 15 min buffer
CUTOFF_TIME = time.time() + TOTAL_TIME_LIMIT - TIME_BUFFER
TOTAL_PROBLEMS = 110               # AIMO3 has 110 problems

# --- Speed levels ---
# Speed 3 (default): 15 samples (7 CoT + 8 Code)
# Speed 2 (medium):  10 samples (5 CoT + 5 Code)
# Speed 1 (fast):     5 samples (2 CoT + 3 Code)
SPEED_THRESHOLDS = {
    3: {"min_time_per_q": 180, "n_cot": 7, "n_code": 8},
    2: {"min_time_per_q": 90,  "n_cot": 5, "n_code": 5},
    1: {"min_time_per_q": 0,   "n_cot": 2, "n_code": 3},
}

# ── Cell 5: Dual Prompts ─────────────────────────────────────────────────

COT_SYSTEM_PROMPT = """\
You are a world-class mathematical problem solver.
Reason step by step to solve the problem. Show all your work.
You may write Python code in ```python ... ``` blocks to verify computations — \
the environment has sympy, numpy, mpmath pre-imported. Always use print() for results.
The final answer must be a non-negative integer between 0 and 99999.
Place ONLY the final answer inside \\boxed{} — nothing else after it."""

CODE_SYSTEM_PROMPT = """\
You are a world-class computational mathematician.
Write Python code to solve the given math problem programmatically.
The code runs with sympy, numpy, mpmath, itertools, collections pre-imported.
Always use print() to output the final numerical result.
After seeing the code output, state the final answer in \\boxed{}.
The answer must be a non-negative integer between 0 and 99999."""


# ── Cell 6: Find Model Path ─────────────────────────────────────────────

def find_model_path():
    """Auto-discover model path from Kaggle inputs or local paths."""
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            # Check if it's a model directory (has config.json)
            if os.path.isfile(os.path.join(path, "config.json")):
                return path
            # Maybe the model is in a subdirectory
            for root, dirs, files in os.walk(path):
                if "config.json" in files and ("tokenizer.json" in files or "tokenizer_config.json" in files):
                    return root
    # Generic search in /kaggle/input
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and ("tokenizer.json" in files or "tokenizer_config.json" in files):
                return root
    return None


MODEL_PATH = find_model_path()
if MODEL_PATH:
    print(f"Found model: {MODEL_PATH}")
else:
    print("No model found. Will need --api-base or --local-model for testing.")


# ── Cell 7: Code Execution ───────────────────────────────────────────────

class PythonREPL:
    """Execute Python code in a subprocess with timeout."""
    def __init__(self, timeout=CODE_TIMEOUT):
        self.timeout = timeout

    def __call__(self, code: str) -> tuple[bool, str]:
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
            return False, result.stderr.strip()[-500:]

repl = PythonREPL()


# ── Cell 8: Answer & Code Extraction ────────────────────────────────────

def extract_boxed_answers(text: str) -> list[int]:
    """Extract integer answers from \\boxed{...} patterns."""
    answers = []
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


# ── Cell 9: Model Wrapper ───────────────────────────────────────────────

class Model:
    """Wraps vLLM (Kaggle) or OpenAI-compatible API (local testing)."""

    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None
        self.chat_mode = None
        self._api_client = None
        self._api_base = None
        self._api_model = None
        self._problems_solved = 0
        self._global_start = time.time()

    def load_vllm(self, model_path: str = None):
        """Load model with vLLM Python API."""
        from vllm import LLM, SamplingParams
        import torch

        path = model_path or MODEL_PATH
        print(f"Loading model from {path}...")
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")

        self.llm = LLM(
            path,
            dtype="bfloat16",
            max_num_seqs=32,
            max_model_len=MAX_TOKENS,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            skip_special_tokens=True,
            max_tokens=MAX_TOKENS,
            seed=SEED,
        )
        self.chat_mode = self._detect_chat_mode()
        print(f"Chat mode: {self.chat_mode}")
        print("Model loaded!")

    def load_api(self, base_url: str, model_name: str):
        """Use an external OpenAI-compatible API."""
        import requests
        self._api_client = requests
        self._api_base = base_url.rstrip("/")
        self._api_model = model_name
        print(f"Using API: {self._api_base}, model: {self._api_model}")

    def _detect_chat_mode(self) -> str:
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

    def _get_speed(self) -> tuple[int, int]:
        """Dynamic speed control based on remaining time budget."""
        elapsed = time.time() - self._global_start
        remaining = max(0, (CUTOFF_TIME - time.time()))
        problems_left = max(1, TOTAL_PROBLEMS - self._problems_solved)
        avg_per_q = remaining / problems_left

        for speed in [3, 2, 1]:
            cfg = SPEED_THRESHOLDS[speed]
            if avg_per_q >= cfg["min_time_per_q"]:
                return cfg["n_cot"], cfg["n_code"]
        cfg = SPEED_THRESHOLDS[1]
        return cfg["n_cot"], cfg["n_code"]

    def _chat_api(self, messages: list, seed: int = SEED) -> str:
        """Single API call."""
        resp = self._api_client.post(
            f"{self._api_base}/chat/completions",
            json={
                "model": self._api_model,
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "seed": seed,
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
        return content

    def generate_batch(self, messages_batch: list, seeds: list[int] = None) -> list[str]:
        """Generate responses for a batch of conversations."""
        t0 = time.time()
        n = len(messages_batch)
        if seeds is None:
            seeds = [SEED] * n

        if self._api_client is not None:
            results = []
            for idx, (messages, seed) in enumerate(zip(messages_batch, seeds)):
                call_t0 = time.time()
                content = self._chat_api(messages, seed=seed)
                preview = strip_think(content).replace("\n", " ")[:120]
                dt = time.time() - call_t0
                print(f"    [{idx+1}/{n}] {dt:.1f}s | {preview}...")
                results.append(content)
            print(f"  Batch done: {n} calls in {time.time()-t0:.1f}s")
            return results

        # vLLM batch generation — use per-sample seeds for diversity
        from vllm import SamplingParams
        prompts = [self._apply_template(msgs) for msgs in messages_batch]
        params_list = [
            SamplingParams(
                temperature=TEMPERATURE,
                skip_special_tokens=True,
                max_tokens=MAX_TOKENS,
                seed=seed,
            )
            for seed in seeds
        ]
        outputs = self.llm.generate(prompts, params_list)
        results = []
        for idx, o in enumerate(outputs):
            text = o.outputs[0].text
            tok_out = len(o.outputs[0].token_ids)
            preview = strip_think(text).replace("\n", " ")[:120]
            print(f"    [{idx+1}/{n}] {tok_out} tok | {preview}...")
            results.append(text)
        print(f"  Batch done: {n} generations in {time.time()-t0:.1f}s")
        return results

    def _solve_single(self, problem: str, system_prompt: str, mode: str, seed: int) -> dict:
        """Solve one sample with TIR loop. Returns {answer, mode, turns}."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ]

        for turn in range(MAX_TIR_ROUNDS):
            if time.time() > CUTOFF_TIME:
                break

            # Generate
            if self._api_client is not None:
                resp = self._chat_api(messages, seed=seed)
            else:
                prompts = [self._apply_template(messages)]
                from vllm import SamplingParams
                sp = SamplingParams(temperature=TEMPERATURE, skip_special_tokens=True,
                                    max_tokens=MAX_TOKENS, seed=seed)
                outputs = self.llm.generate(prompts, sp)
                resp = outputs[0].outputs[0].text

            messages.append({"role": "assistant", "content": resp})
            cleaned = strip_think(resp)

            # Check for boxed answer (sample-level early stop)
            boxed = extract_boxed_answers(cleaned) or extract_boxed_answers(resp)
            if boxed:
                return {"answer": boxed[-1], "mode": mode, "turns": turn + 1}

            # Execute code blocks
            codes = extract_python_code(cleaned) or extract_python_code(resp)
            if codes:
                outputs_text = []
                for code in codes:
                    if time.time() > CUTOFF_TIME:
                        break
                    success, output = repl(code)
                    status = "OK" if success else "ERR"
                    print(f"      Code [{status}]: {output[:100]}{'...' if len(output)>100 else ''}")
                    if success and output:
                        outputs_text.append(output)

                if outputs_text:
                    exec_output = "\n".join(outputs_text)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Code output:\n```\n{exec_output}\n```\n"
                            "Based on this result, state your final answer in \\boxed{}."
                        ),
                    })
                    continue

            # No code, no answer — prompt to continue
            if turn < MAX_TIR_ROUNDS - 1:
                if mode == "code":
                    messages.append({
                        "role": "user",
                        "content": "Write Python code to solve this, or state the answer in \\boxed{}.",
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": "Continue. Put your final answer in \\boxed{}.",
                    })

        # Final attempt: force an answer
        messages.append({"role": "user", "content": "Output your final answer now as \\boxed{N}."})
        if self._api_client is not None:
            resp = self._chat_api(messages, seed=seed)
        else:
            prompts = [self._apply_template(messages)]
            from vllm import SamplingParams
            sp = SamplingParams(temperature=TEMPERATURE, skip_special_tokens=True,
                                max_tokens=MAX_TOKENS, seed=seed)
            outs = self.llm.generate(prompts, sp)
            resp = outs[0].outputs[0].text

        cleaned = strip_think(resp)
        boxed = extract_boxed_answers(cleaned) or extract_boxed_answers(resp)
        return {"answer": boxed[-1] if boxed else None, "mode": mode, "turns": MAX_TIR_ROUNDS}

    def predict(self, problem: str) -> int:
        """Solve one problem with dual prompting + TIR + early stopping + majority voting."""
        if time.time() > CUTOFF_TIME:
            return 0

        n_cot, n_code = self._get_speed()
        total = n_cot + n_code
        print(f"  Speed: {n_cot} CoT + {n_code} Code = {total} samples")

        # Build schedule: interleave CoT and Code samples
        schedule = (
            [("cot", COT_SYSTEM_PROMPT, i) for i in range(n_cot)]
            + [("code", CODE_SYSTEM_PROMPT, i) for i in range(n_code)]
        )

        all_answers = []
        stopped_early = False

        for idx, (mode, prompt, sample_i) in enumerate(schedule):
            if time.time() > CUTOFF_TIME:
                break

            seed = SEED + idx * 13  # Varied seeds for diversity
            t0 = time.time()

            result = self._solve_single(problem, prompt, mode, seed)
            elapsed = time.time() - t0

            answer = result["answer"]
            print(f"    [{mode.upper()} {idx+1}/{total}] answer={answer}, "
                  f"turns={result['turns']}, {elapsed:.1f}s")

            if answer is not None:
                all_answers.append(answer)

                # Question-level early stopping
                counter = Counter(all_answers)
                top_count = counter.most_common(1)[0][1]
                if top_count >= EARLY_STOP_CONSENSUS:
                    print(f"    Early stop: {top_count} answers agree!")
                    stopped_early = True
                    break

        if not all_answers:
            self._problems_solved += 1
            return 0

        answer = select_answer(all_answers)
        print(f"  Votes: {dict(Counter(all_answers))}, Selected: {answer}")
        self._problems_solved += 1
        return answer


model = Model()


# ── Cell 10: Predict Function (Kaggle API) ──────────────────────────────

def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """Kaggle evaluation API entry point."""
    pid = id_.item(0)
    question_text = question.item(0)

    print(f"\n{'='*60}")
    print(f"Problem {pid} ({model._problems_solved + 1}/{TOTAL_PROBLEMS})")
    print(f"{'='*60}")

    result = model.predict(question_text)
    print(f"Final answer: {result}")

    return pl.DataFrame({"id": pid, "answer": result})


# ── Cell 11: Main Entry Point ───────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AIMO3 Solver — 2nd Place (imagination-research)")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                        help="Model name for API calls")
    parser.add_argument("--local-model", default=None,
                        help="Local model path for vLLM (auto-downloads from HF if needed)")
    parser.add_argument("--input-csv", default=None, help="Input CSV for local testing")
    parser.add_argument("--output-csv", default="submission.csv")
    parser.add_argument("--n-problems", type=int, default=None, help="Limit number of problems")
    args = parser.parse_args()

    # Load model
    if args.api_base:
        model.load_api(args.api_base, args.model_name)
    elif args.local_model:
        global MODEL_PATH
        MODEL_PATH = args.local_model
        model.load_vllm(args.local_model)
    elif MODEL_PATH:
        model.load_vllm()
    else:
        print("Error: No model found. Use --api-base or --local-model.")
        sys.exit(1)

    # --- Kaggle competition mode ---
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN") and HAS_KAGGLE_EVAL:
        inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
        inference_server.serve()

    # --- Local gateway mode ---
    elif HAS_KAGGLE_EVAL and os.path.exists("/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv"):
        inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
        inference_server.run_local_gateway(
            ("/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv",)
        )

    # --- Local CSV mode ---
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
        if args.n_problems:
            df = df.head(args.n_problems)
        n = len(df)
        has_answers = "answer" in df.columns
        correct = 0
        results = []

        for idx, row in df.iterrows():
            pid = row["id"]
            question = row.get("problem") or row.get("question")
            expected = int(row["answer"]) if has_answers else None

            print(f"\n{'='*60}")
            print(f"Problem {idx+1}/{n} [{pid}] (expected: {expected})")
            print(f"{'='*60}")

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
