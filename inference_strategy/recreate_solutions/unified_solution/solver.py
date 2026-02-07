"""
Unified Solution: Best-of-All-Worlds for AIMO Progress Prize 3
================================================================
Combines the strongest ideas from all three top AIMO2 solutions:

  From NemoSkills (1st):  TIR with multi-step code interleaving + GenSelect
  From Imagination (2nd): Dual prompting (CoT + Code) + early stopping + speed control
  From Aliev (3rd):       Simple, robust baseline with majority voting fallback

Architecture:
  1. Dual-mode TIR sampling: Half samples use CoT-TIR prompt, half use Code-TIR prompt
  2. Question-level early stopping: Stop once consensus is reached
  3. GenSelect for final answer selection (with majority voting fallback)
  4. Dynamic time management: Adjust sample count per remaining budget
  5. Verification round: Top answer gets a verification code-execution check

This solver works with any OpenAI-compatible API backend.
Switch between AIMO2 and AIMO3 problems via --competition flag.
"""

import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from recreate_solutions.common import (
    Sandbox,
    extract_answer,
    extract_code_blocks,
    strip_think_tags,
)
from recreate_solutions.common.answer_extraction import ensure_print


# ── Prompt Templates ──

COT_TIR_PROMPT = """\
You are a world-class IMO competitor. Solve the problem step by step with rigorous reasoning.

RULES:
1. Think carefully about each step. Show all mathematical reasoning.
2. When you need to compute, verify, or enumerate, write Python code in ```python ... ``` blocks.
3. The code runs in Jupyter with sympy, numpy, mpmath, itertools pre-imported.
4. After seeing code output, CONTINUE reasoning with the result — do NOT repeat the code.
5. You may interleave reasoning and code as many times as needed.
6. Always use print() to display results.
7. VERIFY your answer with a final computation before boxing it.
8. Final answer must be an integer 0-99999 inside \\boxed{}."""

CODE_TIR_PROMPT = """\
You are a computational mathematician. Solve the problem primarily through Python code.

RULES:
1. Break the problem into computational steps.
2. Write Python code in ```python ... ``` blocks to solve each step.
3. The code runs in Jupyter with sympy, numpy, mpmath, itertools pre-imported.
4. Use print() to display intermediate and final results.
5. After each code execution, briefly reason about the result before the next step.
6. For the final answer, write a verification script that checks the answer.
7. Final answer must be an integer 0-99999 inside \\boxed{}."""

GENSELECT_PROMPT = """\
You are a mathematical judge evaluating {n} candidate solutions.

PROBLEM:
{problem}

{solutions}

Evaluate each solution:
1. Is the mathematical reasoning correct?
2. Are computations verified?
3. Does the answer follow from the work?
4. Any errors or invalid assumptions?

Output format (exactly):
BEST: <solution_number>
REASON: <brief explanation>
ANSWER: <integer>"""

VERIFY_PROMPT = """\
A candidate answer to the following problem is {answer}. Verify this is correct.

PROBLEM:
{problem}

Write Python code to verify the answer {answer} is correct.
If correct, respond: VERIFIED \\boxed{{{answer}}}
If wrong, solve the problem yourself and give the correct answer in \\boxed{{}}."""


class UnifiedSolver:
    """Best-of-all-worlds AIMO solver combining top-3 AIMO2 techniques."""

    def __init__(
        self,
        api_base: str = "http://localhost:8080/v1",
        model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        n_cot_samples: int = 8,
        n_code_samples: int = 8,
        max_turns: int = 16,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        code_timeout: float = 30.0,
        early_stop_consensus: int = 5,
        use_genselect: bool = True,
        use_verification: bool = True,
        total_time_budget: float = 17400.0,  # 4h50m like Kaggle
        n_problems: int = 50,
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
        self.use_genselect = use_genselect
        self.use_verification = use_verification
        self.total_time_budget = total_time_budget

        self.sandbox = Sandbox(timeout=code_timeout)
        self._global_start = time.time()
        self._problems_remaining = n_problems

        total = n_cot_samples + n_code_samples
        print(f"[Unified] {n_cot_samples} CoT + {n_code_samples} Code = {total} samples")
        print(f"[Unified] GenSelect={'ON' if use_genselect else 'OFF'}, "
              f"Verify={'ON' if use_verification else 'OFF'}, "
              f"EarlyStop@{early_stop_consensus}")

    def _chat(self, messages: List[Dict], seed: int = 42, temperature: float = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
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

    def _tir_attempt(self, problem: str, system_prompt: str, seed: int) -> Dict[str, Any]:
        """Single TIR attempt (works for both CoT and Code modes)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ]
        full_response = ""
        code_calls = 0

        for turn in range(self.max_turns):
            reply = self._chat(messages, seed=seed)
            reply_clean = strip_think_tags(reply)
            full_response += reply + "\n"
            messages.append({"role": "assistant", "content": reply})

            answer = extract_answer(reply_clean) or extract_answer(reply)
            if answer is not None:
                return {
                    "answer": answer,
                    "response": full_response,
                    "code_calls": code_calls,
                    "turns": turn + 1,
                }

            code_blocks = extract_code_blocks(reply_clean) or extract_code_blocks(reply)
            if code_blocks:
                outputs = []
                for code in code_blocks:
                    code_calls += 1
                    outputs.append(self.sandbox.execute(ensure_print(code)))
                exec_output = "\n---\n".join(outputs)
                messages.append({
                    "role": "user",
                    "content": (
                        f"Code output:\n```\n{exec_output}\n```\n"
                        "Continue reasoning with these results. "
                        "If ready, put your final answer in \\boxed{}."
                    ),
                })
                self.sandbox.reset()
            elif turn < self.max_turns - 1:
                messages.append({
                    "role": "user",
                    "content": "Continue. Use code if needed, or state your answer in \\boxed{}.",
                })

        return {
            "answer": extract_answer(full_response),
            "response": full_response,
            "code_calls": code_calls,
            "turns": self.max_turns,
        }

    def _genselect(self, problem: str, candidates: List[Dict]) -> Optional[int]:
        """GenSelect: have the model judge and pick the best solution."""
        valid = [c for c in candidates if c["answer"] is not None]
        if len(valid) <= 1:
            return valid[0]["answer"] if valid else None

        solutions_text = ""
        for i, c in enumerate(valid[:8]):  # Cap at 8 to fit context
            resp = c["response"][:2500]
            solutions_text += (
                f"\n--- SOLUTION {i + 1} (Answer: {c['answer']}) ---\n{resp}\n"
            )

        prompt = GENSELECT_PROMPT.format(
            n=min(len(valid), 8),
            problem=problem,
            solutions=solutions_text,
        )

        try:
            reply = self._chat([{"role": "user", "content": prompt}], seed=0, temperature=0.1)
            reply_clean = strip_think_tags(reply)

            m = re.search(r"ANSWER:\s*(\d+)", reply_clean)
            if m:
                val = int(m.group(1))
                if 0 <= val <= 99999:
                    return val

            m = re.search(r"BEST:\s*(\d+)", reply_clean)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(valid):
                    return valid[idx]["answer"]
        except Exception:
            pass

        return None

    def _verify_answer(self, problem: str, answer: int) -> int:
        """Verification round: have the model check the proposed answer."""
        prompt = VERIFY_PROMPT.format(problem=problem, answer=answer)
        messages = [{"role": "user", "content": prompt}]

        try:
            reply = self._chat(messages, seed=0, temperature=0.1)
            reply_clean = strip_think_tags(reply)

            code_blocks = extract_code_blocks(reply_clean) or extract_code_blocks(reply)
            if code_blocks:
                outputs = []
                for code in code_blocks:
                    outputs.append(self.sandbox.execute(ensure_print(code)))
                exec_output = "\n---\n".join(outputs)
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "user",
                    "content": f"Output:\n```\n{exec_output}\n```\nIs the answer correct? Give final answer in \\boxed{{}}.",
                })
                reply2 = self._chat(messages, seed=0, temperature=0.1)
                verified_answer = extract_answer(strip_think_tags(reply2)) or extract_answer(reply2)
                if verified_answer is not None:
                    return verified_answer
                self.sandbox.reset()

            verified_answer = extract_answer(reply_clean) or extract_answer(reply)
            if verified_answer is not None:
                return verified_answer
        except Exception:
            pass

        return answer  # Keep original if verification fails

    def _compute_budget(self) -> tuple:
        """Dynamic speed adjustment. Returns (n_cot, n_code)."""
        elapsed = time.time() - self._global_start
        remaining = max(0, self.total_time_budget - elapsed)
        avg_per_q = remaining / max(1, self._problems_remaining)

        if avg_per_q < 60:
            return 3, 3  # Panic mode
        elif avg_per_q < 120:
            return 4, 4  # Fast mode
        elif avg_per_q < 240:
            return 5, 5  # Normal-fast
        return self.n_cot_samples, self.n_code_samples

    def solve(self, problem: str) -> Dict[str, Any]:
        """Full pipeline: dual TIR sampling -> early stop -> GenSelect -> verify."""
        n_cot, n_code = self._compute_budget()
        total = n_cot + n_code
        print(f"  [Unified] {n_cot} CoT + {n_code} Code = {total} samples")

        # Build interleaved schedule
        schedule = []
        for i in range(max(n_cot, n_code)):
            if i < n_cot:
                schedule.append(("cot", i))
            if i < n_code:
                schedule.append(("code", i))

        all_answers: List[int] = []
        candidates: List[Dict] = []
        stopped_early = False

        for idx, (mode, sample_i) in enumerate(schedule):
            t0 = time.time()
            seed = 42 + idx * 17
            prompt = COT_TIR_PROMPT if mode == "cot" else CODE_TIR_PROMPT

            result = self._tir_attempt(problem, prompt, seed=seed)
            result["mode"] = mode
            elapsed = time.time() - t0

            print(f"    [{mode.upper():4s}] {idx + 1}/{total}: "
                  f"ans={result['answer']}, code={result['code_calls']}, "
                  f"turns={result['turns']}, {elapsed:.1f}s")

            candidates.append(result)
            if result["answer"] is not None:
                all_answers.append(result["answer"])

                # Question-level early stopping
                counter = Counter(all_answers)
                top = counter.most_common(1)[0]
                if top[1] >= self.early_stop_consensus:
                    print(f"    Early stop: {top[1]}x answer={top[0]}")
                    stopped_early = True
                    break

        # Answer selection
        if not all_answers:
            self._problems_remaining = max(0, self._problems_remaining - 1)
            return {"answer": 0, "candidates": candidates, "method": "none"}

        # Try GenSelect first
        method = "majority_vote"
        counter = Counter(all_answers)

        if self.use_genselect and len(set(all_answers)) > 1:
            genselect_answer = self._genselect(problem, candidates)
            if genselect_answer is not None:
                answer = genselect_answer
                method = "genselect"
            else:
                answer = counter.most_common(1)[0][0]
        else:
            answer = counter.most_common(1)[0][0]

        # Verification round
        if self.use_verification and not stopped_early:
            print(f"  [Unified] Verifying answer {answer}...")
            verified = self._verify_answer(problem, answer)
            if verified != answer:
                print(f"  [Unified] Verification changed answer: {answer} -> {verified}")
                answer = verified
                method += "+verified"

        print(f"  [Unified] Votes: {dict(counter)}, Method: {method}, Final: {answer}")
        self._problems_remaining = max(0, self._problems_remaining - 1)

        return {
            "answer": answer,
            "candidates": candidates,
            "method": method,
            "early_stopped": stopped_early,
            "votes": dict(counter),
        }

    def shutdown(self):
        self.sandbox.close()
