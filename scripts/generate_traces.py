"""Generate rich TIR traces for offline replay of selection strategies.

Runs an LLM with Tool-Integrated Reasoning on reference problems, saving all raw
data per attempt (logprobs, code executions, answers, entropy) as JSON.

Works with any OpenAI-compatible API (llama-server, vLLM, etc.).

Usage:
    python scripts/generate_traces.py \
        --api-base http://localhost:8080/v1 \
        --model Qwen3-4B \
        --n-samples 12 \
        --max-turns 16 \
        --temperature 0.6 \
        --competition aimo3 \
        --prompt-mix "reasoning:8,code_first:2,case_analysis:2" \
        --logprobs 5 \
        --output-dir output/traces
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import requests

# Add repo root to path so we can import recreate_solutions.common
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from recreate_solutions.common import (
    Sandbox,
    extract_answer,
    extract_code_blocks,
    strip_think_tags,
    ensure_print,
    load_problems,
)
from recreate_solutions.common.data_loader import get_known_answers
from scripts.prompts import (
    PREFERENCE_PROMPT,
    get_prompt_sequence,
    parse_prompt_mix,
)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def chat_completion(
    api_base: str,
    model: str,
    messages: List[Dict],
    temperature: float = 0.6,
    max_tokens: int = 16384,
    seed: int = 42,
    logprobs_n: int = 5,
    extra_params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Call OpenAI-compatible chat/completions, return parsed response + logprobs.

    Returns:
        {
            "content": str,           # assistant message content
            "reasoning_content": str,  # reasoning content (if any, e.g. DeepSeek-R1)
            "logprobs": list | None,   # list of top-logprob dicts per token
            "total_tokens": int,       # completion token count
            "finish_reason": str,
        }
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "logprobs": True,
        "top_logprobs": logprobs_n,
    }
    if extra_params:
        payload.update(extra_params)

    r = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        json=payload,
        timeout=600,
    )
    r.raise_for_status()
    data = r.json()

    choice = data["choices"][0]
    msg = choice["message"]
    usage = data.get("usage", {})

    # Extract logprobs — format varies by server
    raw_logprobs = choice.get("logprobs")
    token_logprobs = None
    if raw_logprobs and isinstance(raw_logprobs, dict):
        content_lp = raw_logprobs.get("content")
        if content_lp and isinstance(content_lp, list):
            token_logprobs = []
            for entry in content_lp:
                top = entry.get("top_logprobs", [])
                if top:
                    token_logprobs.append({
                        t.get("token", ""): t.get("logprob", 0.0) for t in top
                    })

    return {
        "content": msg.get("content") or "",
        "reasoning_content": msg.get("reasoning_content") or "",
        "logprobs": token_logprobs,
        "total_tokens": usage.get("completion_tokens", 0),
        "finish_reason": choice.get("finish_reason", ""),
    }


# ---------------------------------------------------------------------------
# Entropy calculation
# ---------------------------------------------------------------------------

def compute_mean_entropy(logprobs: Optional[List[Dict[str, float]]]) -> float:
    """Shannon entropy in bits from top-K logprobs, averaged over tokens.

    Matches the formula from kaggle_submissions/improv1_entropy_plus:
        H_token = -sum(exp(lp) * log2(exp(lp))) for each lp in top logprobs
        mean_entropy = sum(H_token) / num_tokens
    """
    if not logprobs:
        return float("inf")
    total, count = 0.0, 0
    for top_lp in logprobs:
        if not isinstance(top_lp, dict) or not top_lp:
            continue
        ent = 0.0
        for lp in top_lp.values():
            p = math.exp(lp)
            if p > 0:
                ent -= p * math.log2(p)
        total += ent
        count += 1
    return total / count if count else float("inf")


def summarize_logprobs(logprobs: Optional[List[Dict[str, float]]]) -> Dict[str, Any]:
    """Compute summary statistics over per-token entropy values."""
    if not logprobs:
        return {"mean": float("inf"), "min": float("inf"), "max": float("inf"), "count": 0}

    entropies = []
    for top_lp in logprobs:
        if not isinstance(top_lp, dict) or not top_lp:
            continue
        ent = 0.0
        for lp in top_lp.values():
            p = math.exp(lp)
            if p > 0:
                ent -= p * math.log2(p)
        entropies.append(ent)

    if not entropies:
        return {"mean": float("inf"), "min": float("inf"), "max": float("inf"), "count": 0}

    return {
        "mean": sum(entropies) / len(entropies),
        "min": min(entropies),
        "max": max(entropies),
        "count": len(entropies),
    }


# ---------------------------------------------------------------------------
# Code-output answer fallback
# ---------------------------------------------------------------------------

def extract_code_output_answer(text: str) -> Optional[int]:
    """Extract last integer (0-99999) from code output as fallback answer."""
    candidates = []
    for num_str in re.findall(r"\b(\d{1,5})\b", text):
        val = int(num_str)
        if 0 <= val <= 99999:
            candidates.append(val)
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Single TIR attempt
# ---------------------------------------------------------------------------

def solve_once(
    api_base: str,
    model: str,
    problem_text: str,
    system_prompt: str,
    seed: int,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    sandbox: Sandbox,
    logprobs_n: int = 5,
    extra_params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run one TIR attempt: reason -> code -> verify -> answer.

    Returns a rich attempt dict with all data needed for offline replay.
    """
    user_message = f"{problem_text} {PREFERENCE_PROMPT}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    full_text = ""
    all_logprobs = []
    code_executions = []
    total_response_tokens = 0
    n_python_calls = 0
    n_python_errors = 0
    answer = None
    answer_source = None
    final_answer_turn = None
    last_code_output = ""
    all_answers_extracted = []  # Track all answers found across turns

    for turn in range(max_turns):
        try:
            resp = chat_completion(
                api_base=api_base,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed + turn,
                logprobs_n=logprobs_n,
                extra_params=extra_params,
            )
        except Exception as e:
            # Save partial trace on API error - compute entropy from what we have
            result = _build_attempt_result(
                answer=answer,
                answer_source=answer_source,
                turns_used=turn,
                final_answer_turn=final_answer_turn,
                full_text=full_text,
                code_executions=code_executions,
                n_python_calls=n_python_calls,
                n_python_errors=n_python_errors,
                total_response_tokens=total_response_tokens,
                all_logprobs=all_logprobs,
                all_answers_extracted=all_answers_extracted,
                temperature=temperature,
                seed=seed,
            )
            result["error"] = str(e)
            return result

        content = resp["content"]
        reasoning = resp["reasoning_content"]
        if reasoning:
            reply = f"<think>{reasoning}</think>\n{content}"
        else:
            reply = content

        full_text += reply + "\n"
        total_response_tokens += resp["total_tokens"]
        if resp["logprobs"]:
            all_logprobs.extend(resp["logprobs"])

        messages.append({"role": "assistant", "content": reply})
        clean = strip_think_tags(reply)

        # Check for boxed answer
        ans = extract_answer(clean) or extract_answer(reply)
        if ans is not None:
            all_answers_extracted.append({"turn": turn, "answer": ans, "source": "boxed"})
            answer = ans
            answer_source = "boxed"
            final_answer_turn = turn
            return _build_attempt_result(
                answer=answer,
                answer_source=answer_source,
                turns_used=turn + 1,
                final_answer_turn=final_answer_turn,
                full_text=full_text,
                code_executions=code_executions,
                n_python_calls=n_python_calls,
                n_python_errors=n_python_errors,
                total_response_tokens=total_response_tokens,
                all_logprobs=all_logprobs,
                all_answers_extracted=all_answers_extracted,
                temperature=temperature,
                seed=seed,
            )

        # Check for code to execute
        code_blocks = extract_code_blocks(clean) or extract_code_blocks(reply)
        if code_blocks:
            outputs = []
            for code in code_blocks:
                n_python_calls += 1
                out = sandbox.execute(ensure_print(code))
                is_error = out.startswith("[ERROR]") or "Traceback" in out or "Error" in out
                if is_error:
                    n_python_errors += 1
                else:
                    last_code_output = out
                    # Track potential answers from code output
                    code_ans = extract_code_output_answer(out)
                    if code_ans is not None:
                        all_answers_extracted.append({
                            "turn": turn, "answer": code_ans, "source": "code_output"
                        })
                outputs.append(out)
                code_executions.append({
                    "turn": turn,
                    "code": code,
                    "output": out,
                    "is_error": is_error,
                })

            exec_text = "\n---\n".join(outputs)
            messages.append({
                "role": "user",
                "content": (
                    f"Code output:\n```\n{exec_text}\n```\n"
                    "Continue solving. Put your final answer in \\boxed{}."
                ),
            })
            sandbox.reset()
        elif turn < max_turns - 1:
            messages.append({
                "role": "user",
                "content": "Continue solving. Put your final answer in \\boxed{}.",
            })

    # Last-ditch: scan full text for answer
    answer = extract_answer(full_text)
    if answer is not None:
        answer_source = "boxed_lastditch"
        final_answer_turn = max_turns - 1
        all_answers_extracted.append({
            "turn": max_turns - 1, "answer": answer, "source": "boxed_lastditch"
        })

    # Code-output fallback
    if answer is None and last_code_output:
        answer = extract_code_output_answer(last_code_output)
        if answer is not None:
            answer_source = "code_output_fallback"
            final_answer_turn = max_turns - 1

    return _build_attempt_result(
        answer=answer,
        answer_source=answer_source,
        turns_used=max_turns,
        final_answer_turn=final_answer_turn,
        full_text=full_text,
        code_executions=code_executions,
        n_python_calls=n_python_calls,
        n_python_errors=n_python_errors,
        total_response_tokens=total_response_tokens,
        all_logprobs=all_logprobs,
        all_answers_extracted=all_answers_extracted,
        temperature=temperature,
        seed=seed,
    )


def _build_attempt_result(
    answer, answer_source, turns_used, final_answer_turn, full_text, code_executions,
    n_python_calls, n_python_errors, total_response_tokens, all_logprobs,
    all_answers_extracted, temperature, seed,
) -> Dict[str, Any]:
    """Build attempt result dict with full trace data for offline replay.

    Saves raw token_logprobs to enable custom entropy calculations later.
    """
    entropy = compute_mean_entropy(all_logprobs)
    lp_summary = summarize_logprobs(all_logprobs)

    # Convert logprobs to serializable format (limit size if huge)
    # Each entry is {token: logprob, ...} for top-K tokens
    # For very long generations, we sample to keep file sizes reasonable
    raw_logprobs = all_logprobs
    if len(all_logprobs) > 2000:
        # Sample every Nth token to keep ~2000 entries
        step = len(all_logprobs) // 2000
        raw_logprobs = all_logprobs[::step]

    return {
        "answer": answer,
        "answer_source": answer_source,
        "turns_used": turns_used,
        "final_answer_turn": final_answer_turn,
        "full_response_text": full_text,
        "code_executions": code_executions,
        "n_python_calls": n_python_calls,
        "n_python_errors": n_python_errors,
        "total_response_tokens": total_response_tokens,
        # Entropy data
        "entropy": entropy,
        "entropy_min": lp_summary["min"],
        "entropy_max": lp_summary["max"],
        "entropy_count": lp_summary["count"],
        # Raw logprobs for custom entropy calculations
        "token_logprobs": raw_logprobs,
        # All answers found across turns (for early-stopping simulation)
        "all_answers_extracted": all_answers_extracted,
        # Generation params (for reproducibility)
        "temperature": temperature,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all(args):
    """Main orchestrator: load problems, run TIR, save traces."""

    # Parse prompt mix
    if args.prompt_mix:
        mix = parse_prompt_mix(args.prompt_mix)
    else:
        mix = None  # uses DEFAULT_MIX

    prompt_seq = get_prompt_sequence(mix)

    # If n_samples differs from prompt_seq length, cycle/truncate
    if args.n_samples != len(prompt_seq):
        full_seq = prompt_seq
        prompt_seq = []
        for i in range(args.n_samples):
            prompt_seq.append(full_seq[i % len(full_seq)])

    # Load problems
    df = load_problems(csv_path=args.csv_path, competition=args.competition)
    known = get_known_answers(df)
    print(f"Loaded {len(df)} problems ({len(known)} with known answers)")

    # Filter problems if specified
    if args.problems:
        ids = [p.strip() for p in args.problems.split(",")]
        df = df[df["id"].astype(str).isin(ids)]
        print(f"Filtered to {len(df)} problems: {ids}")

    if len(df) == 0:
        print("No problems to solve. Exiting.")
        return

    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join("output", "traces", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}")

    # Build extra API params
    extra_params = {}
    if args.min_p is not None:
        extra_params["min_p"] = args.min_p

    # Save config
    config = {
        "api_base": args.api_base,
        "model": args.model,
        "n_samples": args.n_samples,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "logprobs": args.logprobs,
        "competition": args.competition,
        "prompt_mix": {name: sum(1 for n, _ in prompt_seq if n == name)
                       for name in set(n for n, _ in prompt_seq)},
        "extra_params": extra_params,
        "timestamp": timestamp,
        "n_problems": len(df),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Initialize sandbox
    sandbox = Sandbox(timeout=args.code_timeout)
    print("Sandbox initialized")

    # Solve each problem
    all_results = []
    total_correct = 0
    total_with_answer = 0
    total_start = time.time()

    for prob_idx, (_, row) in enumerate(df.iterrows()):
        pid = str(row["id"])
        problem_text = row["problem"]
        ground_truth = known.get(pid)

        # Resume support: skip if trace file already exists
        trace_path = os.path.join(out_dir, f"problem_{pid}.json")
        if os.path.exists(trace_path):
            print(f"\n[SKIP] Problem {prob_idx+1}/{len(df)} [{pid}] - trace exists")
            continue

        print(f"\n{'='*60}")
        print(f"Problem {prob_idx+1}/{len(df)} [{pid}]")
        if ground_truth is not None:
            print(f"  Ground truth: {ground_truth}")
        print(f"  {problem_text[:100]}...")
        print(f"{'='*60}")

        prob_start = time.time()
        attempts = []

        for i in range(args.n_samples):
            prompt_type, system_prompt = prompt_seq[i]
            seed = args.seed + i * 37

            t0 = time.time()
            attempt = solve_once(
                api_base=args.api_base,
                model=args.model,
                problem_text=problem_text,
                system_prompt=system_prompt,
                seed=seed,
                max_turns=args.max_turns,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                sandbox=sandbox,
                logprobs_n=args.logprobs,
                extra_params=extra_params if extra_params else None,
            )
            dt = time.time() - t0

            attempt["attempt_idx"] = i
            attempt["seed"] = seed
            attempt["prompt_type"] = prompt_type
            attempt["system_prompt"] = system_prompt
            attempt["wall_time_s"] = round(dt, 2)

            ans = attempt["answer"]
            ent = attempt.get("entropy", float("inf"))  # fallback for edge cases
            src = attempt["answer_source"] or "none"
            err = attempt.get("error", "")
            err_msg = f" ERROR={err}" if err else ""
            print(f"  Sample {i+1}/{args.n_samples}: answer={ans} "
                  f"(src={src}, entropy={ent:.3f}, turns={attempt['turns_used']}, "
                  f"time={dt:.1f}s){err_msg}")

            attempts.append(attempt)

        prob_wall_time = time.time() - prob_start

        # Default answer selection: majority vote
        valid_answers = [a["answer"] for a in attempts if a["answer"] is not None]
        if valid_answers:
            counter = Counter(valid_answers)
            default_answer = counter.most_common(1)[0][0]
            default_method = "majority_vote"
        else:
            default_answer = 0
            default_method = "no_valid_answers"

        is_correct = (default_answer == ground_truth) if ground_truth is not None else None
        if ground_truth is not None:
            total_with_answer += 1
            if is_correct:
                total_correct += 1

        tag = "CORRECT" if is_correct else ("WRONG" if is_correct is False else "?")
        print(f"\n  >> [{tag}] Default={default_answer}, Expected={ground_truth}, "
              f"Time={prob_wall_time:.1f}s")
        print(f"  >> Votes: {dict(Counter(valid_answers))}")

        # Build problem trace
        problem_trace = {
            "problem_id": pid,
            "problem_text": problem_text,
            "ground_truth": ground_truth,
            "wall_time_s": round(prob_wall_time, 2),
            "attempts": attempts,
            "default_answer": default_answer,
            "default_method": default_method,
            "default_votes": dict(Counter(valid_answers)),
        }

        # Save per-problem trace (trace_path defined at start of loop for resume check)
        with open(trace_path, "w") as f:
            json.dump(problem_trace, f, indent=2, default=_json_safe)

        all_results.append({
            "problem_id": pid,
            "ground_truth": ground_truth,
            "default_answer": default_answer,
            "correct": is_correct,
            "wall_time_s": round(prob_wall_time, 2),
            "n_attempts": len(attempts),
            "n_valid_answers": len(valid_answers),
        })

    total_time = time.time() - total_start

    # Summary
    accuracy = total_correct / total_with_answer if total_with_answer else 0
    print(f"\n{'='*60}")
    print(f"RESULTS (majority vote): {total_correct}/{total_with_answer} "
          f"({accuracy*100:.1f}%)")
    print(f"Total time: {total_time:.0f}s "
          f"({total_time/len(df):.0f}s per problem)")
    print(f"{'='*60}")

    summary = {
        "model": args.model,
        "n_samples": args.n_samples,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "correct": total_correct,
        "total": total_with_answer,
        "accuracy": round(accuracy, 4),
        "total_time_s": round(total_time, 1),
        "per_problem": all_results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    sandbox.close()
    print(f"\nTraces saved to: {out_dir}")
    return out_dir


def _json_safe(obj):
    """Handle inf/nan for JSON serialization."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return "NaN"
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate rich TIR traces for offline replay of selection strategies"
    )
    parser.add_argument("--api-base", default="http://localhost:8080/v1",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default="Qwen3-4B",
                        help="Model name for the API")
    parser.add_argument("--n-samples", type=int, default=12,
                        help="Number of attempts per problem")
    parser.add_argument("--max-turns", type=int, default=16,
                        help="Max TIR turns per attempt")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--min-p", type=float, default=None,
                        help="min_p sampling parameter")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed (each attempt gets seed + i*37)")
    parser.add_argument("--logprobs", type=int, default=5,
                        help="Number of top logprobs to request")
    parser.add_argument("--code-timeout", type=float, default=30.0,
                        help="Timeout for code execution in seconds")
    parser.add_argument("--competition", default="aimo3",
                        choices=["aimo2", "aimo3"])
    parser.add_argument("--csv-path", default=None,
                        help="Explicit CSV path (overrides --competition)")
    parser.add_argument("--problems", type=str, default=None,
                        help="Comma-separated problem IDs to solve")
    parser.add_argument("--prompt-mix", type=str, default=None,
                        help="Prompt mix, e.g. 'reasoning:8,code_first:2,case_analysis:2'")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for traces")

    args = parser.parse_args()
    generate_all(args)


if __name__ == "__main__":
    main()
