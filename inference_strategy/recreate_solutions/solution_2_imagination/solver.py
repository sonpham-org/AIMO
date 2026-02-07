"""
Solution 2: imagination-research (2nd Place, AIMO Progress Prize 2)
===================================================================
Team: Tsinghua University + Microsoft Research
Score: 34/50 public (1st), 31/50 private (2nd)

Key ideas recreated here:
  1. Dual Prompting: Two prompt variants run in parallel —
     - CoT variant: pure reasoning ("reason step by step")
     - Code variant: code-first solving ("write Python to solve")
     This doubles the diversity of solutions from the same model.

  2. SFT + DPO Training: They fine-tuned DeepSeek-R1-Distill-Qwen-14B
     with SFT then DPO to reduce output length while maintaining accuracy.
     (We can't recreate the training, but we use the same prompt patterns.)

  3. Sample-level Early Stopping: Stop generating once a boxed answer or
     executable code is found — don't waste tokens on redundant reasoning.

  4. Question-level Early Stopping: If N out of M answers agree, stop
     generating more samples for that question.

  5. Dynamic Speed Adjustment: Adapt the number of samples based on
     remaining time budget per question.

  6. lmdeploy + AWQ + KV8 quantization for throughput.

Reference: https://github.com/imagination-research/aimo2
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

# -- Dual Prompt Templates --
COT_SYSTEM_PROMPT = """\
You are a world-class mathematical problem solver.
Reason step by step to solve the problem. Show all your work.
The final answer must be a non-negative integer between 0 and 99999.
Place ONLY the final answer inside \\boxed{} — nothing else after it.
If the answer could exceed 999, compute the answer modulo the specified modulus."""

CODE_SYSTEM_PROMPT = """\
You are a world-class computational mathematician.
Write Python code to solve the given math problem programmatically.
The code runs in a Jupyter notebook with sympy, numpy, mpmath pre-imported.
Always use print() to output the final numerical result.
After seeing the code output, state the final answer in \\boxed{}.
The answer must be a non-negative integer between 0 and 99999."""


class ImaginationSolver:
    """Recreates the imagination-research dual-prompt pipeline."""

    def __init__(
        self,
        api_base: str = "http://localhost:8080/v1",
        model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        n_cot_samples: int = 7,
        n_code_samples: int = 8,
        max_turns: int = 10,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        code_timeout: float = 30.0,
        early_stop_consensus: int = 5,
        time_budget_per_question: float = 300.0,
        min_samples_fast: int = 10,
    ):
        import requests
        self._requests = requests

        self.api_base = api_base.rstrip("/")
        self.model = model
        self.n_cot_samples = n_cot_samples
        self.n_code_samples = n_code_samples
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.code_timeout = code_timeout
        self.early_stop_consensus = early_stop_consensus
        self.time_budget_per_question = time_budget_per_question
        self.min_samples_fast = min_samples_fast

        self.sandbox = Sandbox(timeout=code_timeout)
        self._global_start = time.time()
        self._problems_remaining = 50

        total = n_cot_samples + n_code_samples
        print(f"[Imagination] Initialized: {n_cot_samples} CoT + {n_code_samples} Code "
              f"= {total} samples, early_stop={early_stop_consensus}")

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

    def _solve_cot(self, problem: str, seed: int) -> Dict[str, Any]:
        """CoT-only attempt: pure reasoning, but execute code if model produces it."""
        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
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
                return {"answer": answer, "mode": "cot", "turns": turn + 1}

            # Even in CoT mode, if the model writes code, execute it
            code_blocks = extract_code_blocks(reply_clean) or extract_code_blocks(reply)
            if code_blocks:
                outputs = []
                for code in code_blocks:
                    outputs.append(self.sandbox.execute(ensure_print(code)))
                exec_output = "\n---\n".join(outputs)
                messages.append({
                    "role": "user",
                    "content": f"Output:\n```\n{exec_output}\n```\nState your final answer in \\boxed{{}}.",
                })
                self.sandbox.reset()
            elif turn < self.max_turns - 1:
                messages.append({
                    "role": "user",
                    "content": "Continue. Put your final answer in \\boxed{}.",
                })

        return {"answer": extract_answer(full_response), "mode": "cot", "turns": self.max_turns}

    def _solve_code(self, problem: str, seed: int) -> Dict[str, Any]:
        """Code-first attempt: instruct model to solve via Python."""
        messages = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        full_response = ""

        for turn in range(self.max_turns):
            reply = self._chat(messages, seed=seed)
            reply_clean = strip_think_tags(reply)
            full_response += reply + "\n"
            messages.append({"role": "assistant", "content": reply})

            # Check for answer first (sample-level early stop)
            answer = extract_answer(reply_clean) or extract_answer(reply)
            if answer is not None:
                return {"answer": answer, "mode": "code", "turns": turn + 1}

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
                        "Based on this result, state the final answer in \\boxed{}."
                    ),
                })
                self.sandbox.reset()
            elif turn < self.max_turns - 1:
                messages.append({
                    "role": "user",
                    "content": "Write Python code to solve this, or state the answer in \\boxed{}.",
                })

        return {"answer": extract_answer(full_response), "mode": "code", "turns": self.max_turns}

    def _adjust_speed(self) -> tuple:
        """Dynamic speed adjustment based on remaining time budget.

        Returns (n_cot, n_code) samples to generate.
        """
        elapsed = time.time() - self._global_start
        remaining = max(0, self.time_budget_per_question * self._problems_remaining - elapsed)
        avg_per_q = remaining / max(1, self._problems_remaining)

        if avg_per_q < 120:
            # Fast mode: fewer samples
            total = self.min_samples_fast
            n_cot = total // 2
            n_code = total - n_cot
            return n_cot, n_code

        return self.n_cot_samples, self.n_code_samples

    def solve(self, problem: str) -> Dict[str, Any]:
        """Solve with dual prompting, early stopping, and dynamic speed."""
        n_cot, n_code = self._adjust_speed()
        total = n_cot + n_code
        print(f"  [Imagination] Running {n_cot} CoT + {n_code} Code samples...")

        all_answers: List[int] = []
        candidates: List[Dict] = []
        stopped_early = False

        # Interleave CoT and Code samples for diversity
        schedule = (
            [("cot", i) for i in range(n_cot)]
            + [("code", i) for i in range(n_code)]
        )

        for idx, (mode, sample_i) in enumerate(schedule):
            t0 = time.time()
            seed = 42 + idx * 13

            if mode == "cot":
                result = self._solve_cot(problem, seed=seed)
            else:
                result = self._solve_code(problem, seed=seed)

            elapsed = time.time() - t0
            print(f"    [{mode.upper()}] Sample {idx + 1}/{total}: "
                  f"answer={result['answer']}, turns={result['turns']}, time={elapsed:.1f}s")
            candidates.append(result)

            if result["answer"] is not None:
                all_answers.append(result["answer"])

                # Question-level early stopping
                counter = Counter(all_answers)
                top_count = counter.most_common(1)[0][1]
                if top_count >= self.early_stop_consensus:
                    print(f"    Early stop: {top_count} answers agree!")
                    stopped_early = True
                    break

        if not all_answers:
            return {"answer": 0, "candidates": candidates, "early_stopped": False}

        counter = Counter(all_answers)
        answer = counter.most_common(1)[0][0]
        print(f"  [Imagination] Votes: {dict(counter)}, Final: {answer}")

        self._problems_remaining = max(0, self._problems_remaining - 1)

        return {
            "answer": answer,
            "candidates": candidates,
            "early_stopped": stopped_early,
        }

    def shutdown(self):
        self.sandbox.close()
