#!/usr/bin/env python3
"""
Trace Shortening via Gemini API Distillation

Takes curated math samples and produces shorter but accurate solutions using
Google's Gemini models. Two strategies:
  1. Distill: Given problem + answer, generate concise solution from scratch
  2. Condense: Given problem + existing solution, rewrite concisely

Output saves each result immediately to disk (one JSON line per problem).
Resume with --resume to skip already-completed UIDs.

Usage:
  # Distill all 5000 samples with Gemini 2.0 Flash (~$6)
  python dataset_generation/scripts/shorten_traces.py \
      --input dataset_generation/output/packaged/all.jsonl \
      --output dataset_generation/output/shortened_flash.jsonl \
      --model gemini-2.0-flash --strategy distill

  # Test with 5 samples first
  python dataset_generation/scripts/shorten_traces.py \
      --input dataset_generation/output/packaged/all.jsonl \
      --output dataset_generation/output/shortened_test.jsonl \
      --model gemini-2.0-flash --max-samples 5

  # Resume after interruption
  python dataset_generation/scripts/shorten_traces.py \
      --input dataset_generation/output/packaged/all.jsonl \
      --output dataset_generation/output/shortened_flash.jsonl \
      --model gemini-2.0-flash --resume
"""

import argparse
import json
import os
import re
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------

def get_client():
    """Initialize Gemini client. Reads key from env or .env file."""
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("GOOGLE_API_KEY=") and not api_key:
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not api_key:
        print("Error: Set GEMINI_API_KEY in environment or .env file")
        sys.exit(1)
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DISTILL_SYSTEM = """You are an expert competition mathematics solver. Write concise, rigorous solutions.

Rules:
- Show essential reasoning steps clearly — no filler, no hedging, no restating the problem
- Use Python code blocks (```python ... ```) for computation and verification when it saves space vs manual calculation
- Always verify your final answer matches the expected answer
- End with \\boxed{ANSWER}
- Target: under 2000 words total"""

DISTILL_USER = """Solve this math problem. The correct answer is {answer}.

Write a SHORT, clean solution — just the key mathematical insights and necessary computation. Use Python code to verify.

Problem:
{problem}"""

CONDENSE_SYSTEM = """You are an expert at condensing verbose mathematical solutions into concise ones.

Rules:
- Preserve the core mathematical reasoning and key insights
- Remove false starts, redundant explanations, excessive hedging
- Keep Python code blocks that verify key computations, remove trivial ones
- End with \\boxed{ANSWER}
- Target: under 2000 words total"""

CONDENSE_USER = """Rewrite this solution more concisely. Keep the mathematical rigor but remove verbosity.

Problem:
{problem}

Expected answer: {answer}

Original solution (verbose):
{solution}

Write a concise version:"""


# ---------------------------------------------------------------------------
# Harmony parser — extract readable text from Harmony format
# ---------------------------------------------------------------------------

def strip_harmony(text: str) -> str:
    """Convert Harmony format to readable plain text."""
    if "<|start|>" not in text:
        return text

    parts = re.split(r'<\|end\|>(?:<\|start\|>)?', text)
    result = []
    for part in parts:
        if part.startswith('system') or part.startswith('user'):
            continue
        msg_match = re.match(
            r'assistant(?:\s+to=python)?<\|channel\|>[^<]*<\|message\|>(.+?)(?:<\|return\|>)?$',
            part, re.DOTALL
        )
        if msg_match:
            content = msg_match.group(1).strip()
            if content and content != '\\boxed{' and len(content) > 5:
                result.append(content)
            continue
        call_match = re.search(r'<\|message\|>(.+?)$', part, re.DOTALL)
        if call_match and '<|call|>' in part:
            output = call_match.group(1).strip()
            if output:
                result.append(f"[Output: {output[:500]}]")

    return "\n\n".join(result)


# ---------------------------------------------------------------------------
# Core shortening logic
# ---------------------------------------------------------------------------

def shorten_one(client, model: str, sample: dict, strategy: str,
                max_output_tokens: int, temperature: float) -> dict:
    """Shorten a single sample using Gemini API. Returns result dict."""
    problem = sample["problem_text"]
    answer = sample.get("answer", "")
    uid = sample.get("uid", "unknown")

    if strategy == "distill":
        prompt = DISTILL_USER.format(problem=problem, answer=answer)
        system = DISTILL_SYSTEM
    else:
        readable = strip_harmony(sample["solution_text"])
        if len(readable) > 30000:
            readable = readable[:15000] + "\n...[truncated]...\n" + readable[-15000:]
        prompt = CONDENSE_USER.format(
            problem=problem, answer=answer, solution=readable
        )
        system = CONDENSE_SYSTEM

    from google.genai import types

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                ),
            )
            text = response.text or ""

            # Verify answer is present
            answer_present = False
            if answer:
                boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
                if boxed and str(answer).strip() in [b.strip() for b in boxed]:
                    answer_present = True
                elif str(answer) in text:
                    answer_present = True

            # Token usage
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                um = response.usage_metadata
                usage = {
                    "input_tokens": getattr(um, 'prompt_token_count', 0),
                    "output_tokens": getattr(um, 'candidates_token_count', 0),
                    "total_tokens": getattr(um, 'total_token_count', 0),
                }

            return {
                "uid": uid,
                "status": "ok",
                "shortened_solution": text,
                "original_length": len(sample["solution_text"]),
                "shortened_length": len(text),
                "answer_verified": answer_present,
                "usage": usage,
            }

        except Exception as e:
            err_str = str(e)
            # Retry on rate limits (429) and server errors (500/503)
            if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                    or "500" in err_str or "503" in err_str):
                wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    continue
            # Non-retryable error or final attempt
            return {
                "uid": uid,
                "status": "error",
                "error": err_str,
                "shortened_solution": "",
                "original_length": len(sample["solution_text"]),
                "shortened_length": 0,
                "answer_verified": False,
                "usage": {},
            }

    # Should not reach here, but just in case
    return {
        "uid": uid, "status": "error", "error": "max retries exceeded",
        "shortened_solution": "", "original_length": len(sample["solution_text"]),
        "shortened_length": 0, "answer_verified": False, "usage": {},
    }


# ---------------------------------------------------------------------------
# Prices for cost estimation
# ---------------------------------------------------------------------------

PRICES = {
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-3-flash": (0.50, 3.0),
    "gemini-3-pro": (2.0, 12.0),
}


# ---------------------------------------------------------------------------
# Checkpoint & output management
# ---------------------------------------------------------------------------

class OutputWriter:
    """Thread-safe, append-only JSONL writer with checkpoint tracking.

    Saves each result immediately. On resume, reads existing output to find
    already-processed UIDs so we never duplicate or lose work.
    """

    def __init__(self, output_path: str, resume: bool = False):
        self.output_path = output_path
        self.lock = threading.Lock()
        self.done_uids = set()

        if resume and os.path.exists(output_path):
            # Scan existing output for completed UIDs
            with open(output_path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        self.done_uids.add(d.get("uid"))
                    except (json.JSONDecodeError, KeyError):
                        continue
            print(f"  Checkpoint: {len(self.done_uids)} already completed")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        mode = "a" if resume and os.path.exists(output_path) else "w"
        self._file = open(output_path, mode)

    def write_result(self, sample: dict, result: dict, model: str, strategy: str):
        """Write one result to output. Thread-safe, flushes immediately."""
        # Build output record — keep all original metadata, replace solution
        output = {}
        for k, v in sample.items():
            if k == "solution_text":
                continue  # Don't store the huge original
            output[k] = v
        output["solution_text"] = result["shortened_solution"]
        output["shortening"] = {
            "model": model,
            "strategy": strategy,
            "status": result["status"],
            "error": result.get("error"),
            "original_chars": result["original_length"],
            "shortened_chars": result["shortened_length"],
            "compression_ratio": round(
                result["shortened_length"] / max(result["original_length"], 1), 3
            ),
            "answer_verified": result["answer_verified"],
            "usage": result["usage"],
        }

        with self.lock:
            self._file.write(json.dumps(output) + "\n")
            self._file.flush()
            os.fsync(self._file.fileno())  # Force to disk
            self.done_uids.add(sample.get("uid"))

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def load_samples(input_path: str, filter_min_pr: float, filter_max_pr: float) -> list:
    """Load and optionally filter samples."""
    samples = []
    with open(input_path) as f:
        for line in f:
            d = json.loads(line)
            pr = d.get("pass_rate")
            if pr is not None:
                if pr < filter_min_pr or pr > filter_max_pr:
                    continue
            samples.append(d)
    return samples


def run(args):
    # Load samples first (before client init, so dry-run works without key)
    print(f"Loading samples from {args.input}...")
    samples = load_samples(args.input, args.filter_min_pass_rate, args.filter_max_pass_rate)
    print(f"  Loaded {len(samples)} samples")

    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        print(f"  Limited to {len(samples)} samples")

    # Init output writer (handles resume/checkpoint)
    writer = OutputWriter(args.output, resume=args.resume)
    if args.resume:
        before = len(samples)
        samples = [s for s in samples if s.get("uid") not in writer.done_uids]
        print(f"  Skipping {before - len(samples)} already done, {len(samples)} remaining")

    if not samples:
        print("Nothing to process.")
        return

    # Estimate cost
    avg_input_tokens = 12000 if args.strategy == "condense" else 800
    avg_output_tokens = min(args.max_output_tokens, 3000)
    if args.model in PRICES:
        ip, op = PRICES[args.model]
        est_cost = len(samples) * (avg_input_tokens * ip / 1e6 + avg_output_tokens * op / 1e6)
        print(f"\n  Estimated cost: ${est_cost:.2f} ({args.model})")
        print(f"  Input: ~{avg_input_tokens} tok × ${ip}/1M")
        print(f"  Output: ~{avg_output_tokens} tok × ${op}/1M")

    if args.dry_run:
        print("\n  --dry-run: exiting without API calls")
        return

    client = get_client()
    print(f"\n  Starting {args.strategy} with {args.model}...")
    print(f"  RPM limit: {args.rpm}, workers: {args.workers}")

    total = len(samples)
    stats = {"ok": 0, "err": 0, "verified": 0, "input_tok": 0, "output_tok": 0}
    start_time = time.time()
    rate_lock = threading.Lock()
    last_request = [0.0]  # mutable for closure
    request_interval = 60.0 / args.rpm if args.rpm > 0 else 0

    def process_one(sample):
        """Process a single sample with rate limiting."""
        # Rate limit
        if request_interval > 0:
            with rate_lock:
                now = time.time()
                wait = request_interval - (now - last_request[0])
                if wait > 0:
                    time.sleep(wait)
                last_request[0] = time.time()

        result = shorten_one(
            client, args.model, sample, args.strategy,
            args.max_output_tokens, args.temperature
        )
        # Write immediately
        writer.write_result(sample, result, args.model, args.strategy)
        return result

    # Process (sequential or parallel)
    workers = min(args.workers, len(samples))
    completed = 0

    def log_progress(result):
        nonlocal completed
        completed += 1
        if result["status"] == "ok":
            stats["ok"] += 1
            if result["answer_verified"]:
                stats["verified"] += 1
            stats["input_tok"] += result["usage"].get("input_tokens", 0)
            stats["output_tok"] += result["usage"].get("output_tokens", 0)
        else:
            stats["err"] += 1

        if completed % 10 == 0 or completed <= 3 or completed == total:
            elapsed = time.time() - start_time
            rate = completed / elapsed * 60
            cost = (stats["input_tok"] * PRICES.get(args.model, (0, 0))[0] / 1e6 +
                    stats["output_tok"] * PRICES.get(args.model, (0, 0))[1] / 1e6)
            cr = result["shortened_length"] / max(result["original_length"], 1)
            print(f"  [{completed}/{total}] ok={stats['ok']} err={stats['err']} "
                  f"verified={stats['verified']} ratio={cr:.2f} "
                  f"rate={rate:.0f}/min cost=${cost:.4f}")

    if workers <= 1:
        for sample in samples:
            result = process_one(sample)
            log_progress(result)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, s): s for s in samples}
            for future in as_completed(futures):
                result = future.result()
                log_progress(result)

    writer.close()

    # Final report
    elapsed = time.time() - start_time
    cost = (stats["input_tok"] * PRICES.get(args.model, (0, 0))[0] / 1e6 +
            stats["output_tok"] * PRICES.get(args.model, (0, 0))[1] / 1e6)

    print(f"\n{'='*60}")
    print(f"DONE: {stats['ok']} shortened, {stats['err']} errors, "
          f"{stats['verified']} answer-verified ({stats['verified']/max(stats['ok'],1):.0%})")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Tokens: {stats['input_tok']:,} in + {stats['output_tok']:,} out")
    print(f"  Actual cost: ${cost:.4f}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")

    # Compression summary
    if os.path.exists(args.output):
        orig, short = [], []
        with open(args.output) as f:
            for line in f:
                d = json.loads(line)
                s = d.get("shortening", {})
                if s.get("status") == "ok":
                    orig.append(s["original_chars"])
                    short.append(s["shortened_chars"])
        if orig:
            print(f"\n  Avg original:  {sum(orig)/len(orig):,.0f} chars")
            print(f"  Avg shortened: {sum(short)/len(short):,.0f} chars")
            print(f"  Compression:   {sum(short)/sum(orig):.1%} of original")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Shorten math traces using Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Input JSONL (from packaged dataset)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL with shortened solutions")
    parser.add_argument("--model", default="gemini-2.0-flash",
                        help="Gemini model (default: gemini-2.0-flash)")
    parser.add_argument("--strategy", default="distill",
                        choices=["distill", "condense"],
                        help="distill=from scratch (cheap), condense=rewrite (costly)")
    parser.add_argument("--max-output-tokens", type=int, default=4096,
                        help="Max output tokens per response (default: 4096)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (default: 0.3)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--rpm", type=int, default=200,
                        help="Rate limit: requests per minute (default: 200)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to process (0=all)")
    parser.add_argument("--filter-min-pass-rate", type=float, default=0.0,
                        help="Only samples with pass_rate >= this")
    parser.add_argument("--filter-max-pass-rate", type=float, default=1.0,
                        help="Only samples with pass_rate <= this")
    parser.add_argument("--resume", action="store_true",
                        help="Resume: skip already-processed UIDs in output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show cost estimate only, no API calls")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
