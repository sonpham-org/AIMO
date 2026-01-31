"""
Solution 1: NemoSkills (1st Place, AIMO Progress Prize 2)
=========================================================
Team: NVIDIA NemoSkills (7 members)
Score: 34/50 public, 1st place final
Paper: arXiv:2504.16891

Key ideas recreated here:
  1. Tool-Integrated Reasoning (TIR): The model interleaves natural language
     reasoning with Python code execution. Unlike simple "write code to solve",
     TIR has the model reason step-by-step and call code mid-thought to verify
     computations, then continue reasoning with the result.

  2. GenSelect: Instead of majority voting, a separate LLM call evaluates all
     candidate solutions and picks the best one. This outperforms maj@N voting
     because it can assess solution quality, not just answer frequency.

  3. Multiple sampling with high temperature for diversity.

Original model: OpenMath-Nemotron-14B (fine-tuned Qwen2.5-14B)
Recreation: Works with any OpenAI-compatible API (DeepSeek-R1, Qwen, etc.)

Reference: https://github.com/NVIDIA-NeMo/Skills
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

# -- TIR System Prompt --
# The key insight: explicitly instruct the model to use code *during* reasoning,
# not as a separate "code-only" mode.
TIR_SYSTEM_PROMPT = """\
You are a world-class mathematical problem solver. Solve the problem step by step.

IMPORTANT RULES:
1. Think carefully about each step before writing anything.
2. When you need to compute, verify, or enumerate something, write Python code
   in a ```python ... ``` block. The code runs in a Jupyter notebook with
   sympy, numpy, mpmath, itertools pre-imported.
3. After seeing the code output, CONTINUE your reasoning using the result.
4. You may interleave reasoning and code multiple times.
5. Always use print() to display computation results.
6. When confident in your final answer, write it as \\boxed{ANSWER}.
   The answer must be a non-negative integer (0-99999).
7. Double-check your answer with a verification code block before boxing it."""

# -- GenSelect Prompt --
# Given multiple candidate solutions, pick the best one.
GENSELECT_PROMPT_TEMPLATE = """\
You are a mathematical judge. Below are {n} candidate solutions to the same problem.
Each solution may contain reasoning and code execution results.

PROBLEM:
{problem}

{solutions_text}

TASK: Analyze each solution for correctness. Consider:
- Is the mathematical reasoning sound?
- Are the computations correct?
- Does the final answer follow logically from the work shown?
- Are there any errors, invalid assumptions, or skipped steps?

Output ONLY the number of the best solution (1-{n}), followed by the answer.
Format: BEST: <solution_number>, ANSWER: <integer>"""


class NemoSkillsSolver:
    """Recreates the NemoSkills TIR + GenSelect inference pipeline."""

    def __init__(
        self,
        api_base: str = "http://localhost:8080/v1",
        model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        n_samples: int = 16,
        max_turns: int = 16,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        code_timeout: float = 30.0,
        use_genselect: bool = True,
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
        self.use_genselect = use_genselect

        self.sandbox = Sandbox(timeout=code_timeout)
        print(f"[NemoSkills] Initialized with model={model}, n_samples={n_samples}")

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

    def _tir_solve_once(self, problem: str, seed: int) -> Dict[str, Any]:
        """Single TIR attempt: interleave reasoning with code execution."""
        messages = [
            {"role": "system", "content": TIR_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        full_response = ""
        code_calls = 0

        for turn in range(self.max_turns):
            reply = self._chat(messages, seed=seed)
            reply_clean = strip_think_tags(reply)
            full_response += reply + "\n"
            messages.append({"role": "assistant", "content": reply})

            # Check for final answer
            answer = extract_answer(reply_clean) or extract_answer(reply)
            if answer is not None:
                return {
                    "answer": answer,
                    "response": full_response,
                    "code_calls": code_calls,
                    "turns": turn + 1,
                }

            # Execute any code blocks (TIR: code interleaved with reasoning)
            code_blocks = extract_code_blocks(reply_clean) or extract_code_blocks(reply)
            if code_blocks:
                outputs = []
                for code in code_blocks:
                    code_calls += 1
                    result = self.sandbox.execute(ensure_print(code), timeout=self.code_timeout)
                    outputs.append(result)
                exec_output = "\n---\n".join(outputs)
                messages.append({
                    "role": "user",
                    "content": (
                        f"Code execution output:\n```\n{exec_output}\n```\n"
                        "Continue your reasoning using these results. "
                        "If you have the final answer, put it in \\boxed{}."
                    ),
                })
            else:
                # No code and no answer — nudge the model
                if turn < self.max_turns - 1:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Continue solving. Use Python code to verify your work, "
                            "or state your final answer in \\boxed{}."
                        ),
                    })

            self.sandbox.reset()

        # Last resort: scan entire response
        answer = extract_answer(full_response)
        return {
            "answer": answer,
            "response": full_response,
            "code_calls": code_calls,
            "turns": self.max_turns,
        }

    def _genselect(self, problem: str, candidates: List[Dict]) -> int:
        """Use GenSelect: have the model pick the best solution."""
        valid = [c for c in candidates if c["answer"] is not None]
        if not valid:
            return 0

        solutions_text = ""
        for i, c in enumerate(valid):
            # Truncate long responses for the judge
            resp = c["response"][:3000]
            solutions_text += f"\n--- SOLUTION {i + 1} (Answer: {c['answer']}) ---\n{resp}\n"

        prompt = GENSELECT_PROMPT_TEMPLATE.format(
            n=len(valid),
            problem=problem,
            solutions_text=solutions_text,
        )

        try:
            reply = self._chat(
                [{"role": "user", "content": prompt}],
                seed=0,
            )
            reply_clean = strip_think_tags(reply)
            # Parse "ANSWER: <int>"
            import re
            m = re.search(r"ANSWER:\s*(\d+)", reply_clean)
            if m:
                val = int(m.group(1))
                if 0 <= val <= 99999:
                    return val
            # Fallback: parse "BEST: <n>"
            m = re.search(r"BEST:\s*(\d+)", reply_clean)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(valid):
                    return valid[idx]["answer"]
        except Exception:
            pass

        # Fallback to majority voting
        return self._majority_vote(valid)

    def _majority_vote(self, candidates: List[Dict]) -> int:
        answers = [c["answer"] for c in candidates if c["answer"] is not None]
        if not answers:
            return 0
        return Counter(answers).most_common(1)[0][0]

    def solve(self, problem: str) -> Dict[str, Any]:
        """Solve a problem using N TIR samples + GenSelect/majority vote."""
        print(f"  [NemoSkills] Generating {self.n_samples} TIR samples...")
        candidates = []
        for i in range(self.n_samples):
            t0 = time.time()
            result = self._tir_solve_once(problem, seed=42 + i * 7)
            elapsed = time.time() - t0
            print(f"    Sample {i + 1}/{self.n_samples}: answer={result['answer']}, "
                  f"turns={result['turns']}, code_calls={result['code_calls']}, "
                  f"time={elapsed:.1f}s")
            candidates.append(result)

        if self.use_genselect:
            print("  [NemoSkills] Running GenSelect...")
            answer = self._genselect(problem, candidates)
        else:
            answer = self._majority_vote(candidates)

        answers = [c["answer"] for c in candidates if c["answer"] is not None]
        print(f"  [NemoSkills] Votes: {dict(Counter(answers))}, Final: {answer}")

        return {
            "answer": answer,
            "candidates": candidates,
            "method": "genselect" if self.use_genselect else "majority_vote",
        }

    def shutdown(self):
        self.sandbox.close()
