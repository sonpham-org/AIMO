"""
Solution 3: Aliev (3rd Place, AIMO Progress Prize 2)
=====================================================
Team: Aliev (notebook by mavicbf)
Score: ~30/50

Key ideas recreated here:
  1. Straightforward approach: Use DeepSeek-R1-Distill-Qwen-14B-AWQ
     as-is (no fine-tuning) with code execution and majority voting.

  2. AWQ quantization for fitting 14B model on 4x L4 GPUs.

  3. Simple but effective pipeline: prompt -> reason -> execute code ->
     majority vote across N samples.

  This represents the "strong baseline" approach — showing that a good
  off-the-shelf reasoning model with proper inference setup can place
  top-3 even without custom training.

Reference: https://www.kaggle.com/code/mavicbf/3rd-place-solution-aliev
Model: deepseek-r1-distill-qwen-14b-awq
"""

import time
from collections import Counter
from typing import Any, Dict, List, Optional

from recreate_solutions.common import (
    Sandbox,
    extract_answer,
    extract_code_blocks,
    ensure_print,
    strip_think_tags,
)

SYSTEM_PROMPT = """\
You are an expert mathematician. Solve the following problem step by step.
You can write Python code in ```python ... ``` blocks to help with calculations.
The environment has sympy, numpy, mpmath pre-imported. Use print() for output.
Your final answer must be a non-negative integer (0-99999).
Write your final answer inside \\boxed{}."""


class AlievSolver:
    """Recreates the Aliev 3rd-place baseline: simple TIR + majority voting."""

    def __init__(
        self,
        api_base: str = "http://localhost:8080/v1",
        model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        n_samples: int = 16,
        max_turns: int = 10,
        temperature: float = 0.6,
        max_tokens: int = 8192,
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
        print(f"[Aliev] Initialized: model={model}, n_samples={n_samples}")

    def _chat(self, messages: List[Dict], seed: int = 42) -> str:
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
        msg = r.json()["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if reasoning:
            return f"<think>{reasoning}</think>\n{content}"
        return content

    def _solve_once(self, problem: str, seed: int) -> Dict[str, Any]:
        """Single attempt: reason + code execution loop."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        full_response = ""

        for turn in range(self.max_turns):
            reply = self._chat(messages, seed=seed)
            reply_clean = strip_think_tags(reply)
            full_response += reply + "\n"
            messages.append({"role": "assistant", "content": reply})

            answer = extract_answer(reply_clean) or extract_answer(reply)
            if answer is not None:
                return {"answer": answer, "turns": turn + 1}

            code_blocks = extract_code_blocks(reply_clean) or extract_code_blocks(reply)
            if code_blocks:
                outputs = []
                for code in code_blocks:
                    outputs.append(self.sandbox.execute(ensure_print(code)))
                exec_output = "\n---\n".join(outputs)
                messages.append({
                    "role": "user",
                    "content": (
                        f"Code output:\n```\n{exec_output}\n```\n"
                        "Continue solving. Put your final answer in \\boxed{}."
                    ),
                })
                self.sandbox.reset()
            elif turn < self.max_turns - 1:
                messages.append({
                    "role": "user",
                    "content": "Continue solving or put your final answer in \\boxed{}.",
                })

        return {"answer": extract_answer(full_response), "turns": self.max_turns}

    def solve(self, problem: str) -> Dict[str, Any]:
        """Solve with N samples and majority voting."""
        print(f"  [Aliev] Generating {self.n_samples} samples...")
        answers: List[int] = []
        candidates: List[Dict] = []

        for i in range(self.n_samples):
            t0 = time.time()
            result = self._solve_once(problem, seed=42 + i * 11)
            elapsed = time.time() - t0
            print(f"    Sample {i + 1}/{self.n_samples}: "
                  f"answer={result['answer']}, turns={result['turns']}, time={elapsed:.1f}s")
            candidates.append(result)
            if result["answer"] is not None:
                answers.append(result["answer"])

        if not answers:
            return {"answer": 0, "candidates": candidates}

        counter = Counter(answers)
        answer = counter.most_common(1)[0][0]
        print(f"  [Aliev] Votes: {dict(counter)}, Final: {answer}")

        return {"answer": answer, "candidates": candidates}

    def shutdown(self):
        self.sandbox.close()
