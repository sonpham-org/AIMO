#!/usr/bin/env python3
"""
Fast Verification Pipeline for Math Solution Datasets

Zero-cost, instant checks for filtering math solution datasets:
  1. Outcome verification: Extract \\boxed{} answers, compare to ground truth
  2. TIR integrity: Verify code blocks, count tool calls, detect failures
  3. Hard filters: Length bounds, repetition detection, answer format

Accepts JSONL or JSON input with flexible field mapping.

Usage:
  # Verify a JSONL file (auto-detects field names)
  python dataset_generation/scripts/verify_solutions.py input.jsonl

  # Specify output path
  python dataset_generation/scripts/verify_solutions.py input.jsonl -o verified.jsonl

  # Only keep verified-correct solutions
  python dataset_generation/scripts/verify_solutions.py input.jsonl --correct-only

  # Custom field mapping
  python dataset_generation/scripts/verify_solutions.py input.jsonl \\
      --field-solution completion --field-answer ground_truth

  # AIMO3 Hard format (nested metadata)
  python dataset_generation/scripts/verify_solutions.py input.jsonl --format aimo3_hard

  # Reject-file: save rejected samples separately
  python dataset_generation/scripts/verify_solutions.py input.jsonl \\
      -o verified.jsonl --rejects rejected.jsonl

  # Adjust thresholds
  python dataset_generation/scripts/verify_solutions.py input.jsonl \\
      --min-length 1000 --max-length 80000 --max-tool-calls 15

  # JSON array input
  python dataset_generation/scripts/verify_solutions.py input.json

  # Stats only (no output file)
  python dataset_generation/scripts/verify_solutions.py input.jsonl --stats-only
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...}, handling nested braces. Takes last match."""
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return None

    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth == 0:
        return text[start:i - 1].strip()
    return None


def extract_answer_fallback(text: str) -> str | None:
    """Extract answer from common text patterns (fallback if no \\boxed{})."""
    patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(\S+)',
        r'(?:the\s+)?(?:final\s+)?answer\s*[:=]\s*(\S+)',
        r'(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?answer\s+is\s*[:\s]*(\S+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip('.')
    return None


def extract_answer(text: str) -> tuple[str | None, str]:
    """Extract answer, return (answer, method)."""
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed, "boxed"
    fallback = extract_answer_fallback(text)
    if fallback is not None:
        return fallback, "text_pattern"
    return None, "none"


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    s = str(answer).strip()
    # Remove LaTeX formatting
    s = s.replace('\\,', '').replace('\\;', '').replace('\\!', '')
    s = s.replace('\\text{', '').replace('\\mathrm{', '').replace('\\mathbf{', '')
    s = s.replace('\\dfrac', '\\frac')
    # Remove surrounding \( \) or $ $
    s = re.sub(r'^\\\(|\\\)$', '', s)
    s = s.replace('$', '').replace(' ', '').lower()
    # Strip variable assignments like "m=99" -> "99", "x=42" -> "42"
    m = re.match(r'^[a-z_]\s*=\s*(.+)$', s)
    if m:
        s = m.group(1)
    # Remove trailing/leading braces that aren't part of content
    s = s.strip('{}')
    return s


def answers_match(extracted: str, ground_truth: str) -> bool:
    """Check if two answers match (exact or numeric equivalence)."""
    if extracted is None or ground_truth is None:
        return False

    norm_ext = normalize_answer(extracted)
    norm_gt = normalize_answer(ground_truth)

    # Exact match
    if norm_ext == norm_gt:
        return True

    # Numeric equivalence
    try:
        val_ext = float(norm_ext)
        val_gt = float(norm_gt)
        if math.isclose(val_ext, val_gt, rel_tol=1e-9, abs_tol=1e-9):
            return True
        if val_ext == int(val_ext) and val_gt == int(val_gt):
            return int(val_ext) == int(val_gt)
    except (ValueError, OverflowError):
        pass

    # Fraction equivalence
    frac_pattern = r'^(-?\d+)/(\d+)$'
    for a, b in [(norm_ext, norm_gt), (norm_gt, norm_ext)]:
        m = re.match(frac_pattern, a)
        if m:
            try:
                frac_val = int(m.group(1)) / int(m.group(2))
                other_val = float(b)
                if math.isclose(frac_val, other_val, rel_tol=1e-9):
                    return True
            except (ValueError, ZeroDivisionError):
                pass

    # Try SymPy if available (for symbolic equivalence)
    try:
        from sympy import simplify, sympify
        sym_ext = sympify(norm_ext)
        sym_gt = sympify(norm_gt)
        if simplify(sym_ext - sym_gt) == 0:
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# TIR integrity checks
# ---------------------------------------------------------------------------

def count_tool_calls(text: str) -> int:
    """Count code execution blocks in the solution."""
    count = 0
    # Harmony format
    count += len(re.findall(r'<\|start\|>assistant to=python', text))
    # Markdown code blocks
    if count == 0:
        count = len(re.findall(r'```python', text))
    return count


def count_code_errors(text: str) -> int:
    """Count code execution errors/tracebacks in tool outputs."""
    errors = 0
    errors += len(re.findall(r'Traceback \(most recent call last\)', text))
    errors += len(re.findall(r'Error:', text))
    errors += len(re.findall(r'TimeoutError', text))
    errors += len(re.findall(r'MemoryError', text))
    # Subtract false positives from problem statements mentioning "Error"
    errors -= len(re.findall(r'(?:margin of |standard |percent )error', text, re.IGNORECASE))
    return max(errors, 0)


def detect_repetition(text: str, min_length: int = 80) -> bool:
    """Detect repetitive patterns (model stuck in loop)."""
    pattern = r'(.{' + str(min_length) + r',})\1'
    return bool(re.search(pattern, text))


def check_tir_integrity(text: str) -> dict:
    """Run all TIR integrity checks. Returns dict of results."""
    n_tool_calls = count_tool_calls(text)
    n_code_errors = count_code_errors(text)
    has_repetition = detect_repetition(text)

    return {
        "n_tool_calls": n_tool_calls,
        "n_code_errors": n_code_errors,
        "has_repetition": has_repetition,
        "code_error_rate": n_code_errors / max(n_tool_calls, 1),
    }


# ---------------------------------------------------------------------------
# Hard filters
# ---------------------------------------------------------------------------

def apply_hard_filters(
    text: str,
    tir: dict,
    min_length: int = 500,
    max_length: int = 100_000,
    max_tool_calls: int = 20,
    max_code_error_rate: float = 0.5,
    require_boxed: bool = True,
) -> tuple[bool, str]:
    """Apply hard filters. Returns (passed, rejection_reason)."""
    # Length checks
    if len(text) < min_length:
        return False, "too_short"
    if len(text) > max_length:
        return False, "too_long"

    # Answer format
    if require_boxed:
        has_boxed = "\\boxed{" in text or "boxed{" in text
        has_text_answer = (
            "final answer" in text.lower()
            or "the answer is" in text.lower()
        )
        if not has_boxed and not has_text_answer:
            return False, "no_answer_marker"

    # Repetition
    if tir["has_repetition"]:
        return False, "repetitive"

    # Tool call bounds
    if tir["n_tool_calls"] > max_tool_calls:
        return False, "too_many_tool_calls"

    # Code error rate
    if tir["n_tool_calls"] > 0 and tir["code_error_rate"] > max_code_error_rate:
        return False, "high_code_error_rate"

    return True, ""


# ---------------------------------------------------------------------------
# Input loading (flexible field mapping)
# ---------------------------------------------------------------------------

# Common field name aliases
SOLUTION_FIELDS = ["solution", "completion", "solution_text", "text", "response", "output"]
PROBLEM_FIELDS = ["problem", "prompt", "problem_text", "question", "input", "instruction"]
ANSWER_FIELDS = ["answer", "ground_truth", "expected_answer", "standard", "gold", "label"]


def find_field(record: dict, candidates: list[str], override: str | None = None) -> str | None:
    """Find the first matching field name in a record."""
    if override and override in record:
        return record[override]
    for c in candidates:
        if c in record:
            return record[c]
    return None


def extract_record_aimo3_hard(raw: dict) -> dict:
    """Extract fields from AIMO3 Hard format (nested metadata_infos)."""
    meta = raw.get("metadata_infos", {})
    rht = meta.get("reason_high_with_tool", {})
    count = rht.get("count", 8)
    passed = rht.get("pass", 0)
    return {
        "problem": meta.get("problem", ""),
        "solution": raw.get("text", ""),
        "answer": str(meta.get("standard", "")) if meta.get("standard") is not None else None,
        "pass_rate": passed / count if count > 0 else None,
        "source": meta.get("data_source", "aimo3_hard"),
    }


def extract_record_generic(raw: dict, field_solution: str | None, field_problem: str | None,
                           field_answer: str | None) -> dict:
    """Extract fields from generic JSONL using flexible field mapping."""
    solution = find_field(raw, SOLUTION_FIELDS, field_solution) or ""
    problem = find_field(raw, PROBLEM_FIELDS, field_problem) or ""
    answer = find_field(raw, ANSWER_FIELDS, field_answer)
    return {
        "problem": problem,
        "solution": solution,
        "answer": str(answer) if answer is not None else None,
        "pass_rate": raw.get("pass_rate"),
        "source": raw.get("source", ""),
    }


def load_input(path: str, fmt: str, max_rows: int,
               field_solution: str | None, field_problem: str | None,
               field_answer: str | None):
    """Load input file, yielding (line_number, record) tuples."""
    with open(path) as f:
        content = f.read(1).strip()

    if content == '[':
        # JSON array
        with open(path) as f:
            data = json.load(f)
        for i, raw in enumerate(data):
            if max_rows and i >= max_rows:
                break
            if fmt == "aimo3_hard":
                yield i, extract_record_aimo3_hard(raw)
            else:
                yield i, extract_record_generic(raw, field_solution, field_problem, field_answer)
    else:
        # JSONL
        with open(path) as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if fmt == "aimo3_hard":
                    yield i, extract_record_aimo3_hard(raw)
                else:
                    yield i, extract_record_generic(raw, field_solution, field_problem, field_answer)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_verification(args):
    """Run the full verification pipeline."""
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Stats
    total = 0
    passed_all = 0
    stats = {
        "answer_extracted": 0,
        "answer_verified": 0,
        "answer_wrong": 0,
        "answer_no_ground_truth": 0,
        "no_answer_found": 0,
    }
    filter_reasons = Counter()
    extraction_methods = Counter()

    # Output setup
    out_file = None
    reject_file = None
    if not args.stats_only and args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_file = open(args.output, "w")
    if args.rejects:
        os.makedirs(os.path.dirname(args.rejects) or ".", exist_ok=True)
        reject_file = open(args.rejects, "w")

    try:
        for line_num, record in load_input(
            input_path, args.format, args.max_rows,
            args.field_solution, args.field_problem, args.field_answer
        ):
            total += 1
            solution = record["solution"]
            ground_truth = record["answer"]

            # --- Step 1: TIR integrity checks ---
            tir = check_tir_integrity(solution)

            # --- Step 2: Hard filters ---
            passed, reason = apply_hard_filters(
                solution, tir,
                min_length=args.min_length,
                max_length=args.max_length,
                max_tool_calls=args.max_tool_calls,
                max_code_error_rate=args.max_code_error_rate,
                require_boxed=not args.no_require_boxed,
            )

            if not passed:
                filter_reasons[reason] += 1
                if reject_file:
                    reject_record = {
                        "line": line_num,
                        "reason": reason,
                        "problem": record["problem"][:200],
                    }
                    reject_file.write(json.dumps(reject_record) + "\n")
                continue

            # --- Step 3: Answer extraction ---
            extracted_answer, method = extract_answer(solution)
            extraction_methods[method] += 1

            if extracted_answer is None:
                stats["no_answer_found"] += 1
                if args.correct_only:
                    filter_reasons["no_answer_extracted"] += 1
                    continue
            else:
                stats["answer_extracted"] += 1

            # --- Step 4: Answer verification ---
            verified = None
            if ground_truth is not None and ground_truth != "" and ground_truth != "None":
                if extracted_answer is not None:
                    if answers_match(extracted_answer, ground_truth):
                        stats["answer_verified"] += 1
                        verified = True
                    else:
                        stats["answer_wrong"] += 1
                        verified = False
                        if args.correct_only:
                            filter_reasons["wrong_answer"] += 1
                            if reject_file:
                                reject_record = {
                                    "line": line_num,
                                    "reason": "wrong_answer",
                                    "extracted": extracted_answer,
                                    "expected": ground_truth,
                                }
                                reject_file.write(json.dumps(reject_record) + "\n")
                            continue
            else:
                stats["answer_no_ground_truth"] += 1

            # --- Output ---
            passed_all += 1
            if out_file:
                output_record = {
                    "uid": record.get("uid", f"row_{line_num}"),
                    "problem_text": record["problem"],
                    "solution_text": solution,
                    "extracted_answer": extracted_answer,
                    "answer_method": method,
                    "ground_truth": ground_truth,
                    "answer_verified": verified,
                    "n_tool_calls": tir["n_tool_calls"],
                    "n_code_errors": tir["n_code_errors"],
                    "solution_length": len(solution),
                }
                if record.get("pass_rate") is not None:
                    output_record["pass_rate"] = record["pass_rate"]
                if record.get("source"):
                    output_record["source"] = record["source"]
                out_file.write(json.dumps(output_record) + "\n")

            # Progress
            if total % 10000 == 0:
                print(f"  Processed {total:,}...", file=sys.stderr)

    finally:
        if out_file:
            out_file.close()
        if reject_file:
            reject_file.close()

    # --- Report ---
    print_report(
        input_path, total, passed_all, stats, filter_reasons,
        extraction_methods, args.output
    )


def print_report(input_path, total, passed, stats, filter_reasons,
                 extraction_methods, output_path):
    """Print verification summary report."""
    print(f"\n{'=' * 70}")
    print(f"VERIFICATION REPORT")
    print(f"{'=' * 70}")
    print(f"Input:  {input_path}")
    print(f"Total:  {total:,}")
    print()

    # Hard filter results
    rejected = sum(filter_reasons.values())
    print(f"--- HARD FILTERS ---")
    print(f"  Passed:   {total - sum(v for k, v in filter_reasons.items() if k not in ('wrong_answer', 'no_answer_extracted')):,}")
    if filter_reasons:
        for reason, count in filter_reasons.most_common():
            print(f"  Rejected [{reason}]: {count:,}")
    print()

    # Answer extraction
    print(f"--- ANSWER EXTRACTION ---")
    print(f"  Extracted:    {stats['answer_extracted']:,} / {total:,} "
          f"({100 * stats['answer_extracted'] / max(total, 1):.1f}%)")
    print(f"  No answer:    {stats['no_answer_found']:,}")
    print(f"  Methods:      " + ", ".join(
        f"{k}={v}" for k, v in extraction_methods.most_common()
    ))
    print()

    # Answer verification
    verifiable = stats['answer_verified'] + stats['answer_wrong']
    print(f"--- ANSWER VERIFICATION ---")
    if verifiable > 0:
        print(f"  Correct:      {stats['answer_verified']:,} / {verifiable:,} "
              f"({100 * stats['answer_verified'] / verifiable:.1f}%)")
        print(f"  Wrong:        {stats['answer_wrong']:,} / {verifiable:,} "
              f"({100 * stats['answer_wrong'] / verifiable:.1f}%)")
    print(f"  No ground truth: {stats['answer_no_ground_truth']:,}")
    print()

    # Final
    print(f"--- RESULT ---")
    print(f"  Output samples: {passed:,} / {total:,} "
          f"({100 * passed / max(total, 1):.1f}%)")
    if output_path:
        print(f"  Output file:    {output_path}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fast verification pipeline for math solution datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.jsonl -o verified.jsonl
  %(prog)s input.jsonl --correct-only -o correct_only.jsonl
  %(prog)s input.jsonl --format aimo3_hard --stats-only
  %(prog)s input.jsonl --field-solution completion --field-answer gold
""",
    )
    parser.add_argument("input", help="Input JSONL or JSON file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSONL file with verified samples")
    parser.add_argument("--rejects", default=None,
                        help="Output JSONL file with rejected samples and reasons")
    parser.add_argument("--correct-only", action="store_true",
                        help="Only output samples with verified-correct answers")
    parser.add_argument("--stats-only", action="store_true",
                        help="Print stats only, do not write output file")

    # Format
    parser.add_argument("--format", default="auto",
                        choices=["auto", "aimo3_hard", "generic"],
                        help="Input format (default: auto-detect)")

    # Field mapping
    parser.add_argument("--field-solution", default=None,
                        help="Field name for solution text")
    parser.add_argument("--field-problem", default=None,
                        help="Field name for problem text")
    parser.add_argument("--field-answer", default=None,
                        help="Field name for ground truth answer")

    # Thresholds
    parser.add_argument("--min-length", type=int, default=500,
                        help="Minimum solution length in chars (default: 500)")
    parser.add_argument("--max-length", type=int, default=100_000,
                        help="Maximum solution length in chars (default: 100000)")
    parser.add_argument("--max-tool-calls", type=int, default=20,
                        help="Maximum allowed tool calls (default: 20)")
    parser.add_argument("--max-code-error-rate", type=float, default=0.5,
                        help="Maximum code error rate (default: 0.5)")
    parser.add_argument("--no-require-boxed", action="store_true",
                        help="Do not require \\boxed{} or text answer marker")

    # Limits
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Max rows to process (0=all)")

    args = parser.parse_args()

    # Auto-detect format
    if args.format == "auto":
        # Peek at first record
        with open(args.input) as f:
            content = f.read(1)
            f.seek(0)
            if content == '[':
                first_line = f.read()
                first_record = json.loads(first_line)[0]
            else:
                first_line = f.readline()
                first_record = json.loads(first_line)

        if "metadata_infos" in first_record:
            args.format = "aimo3_hard"
            print(f"Auto-detected format: aimo3_hard")
        else:
            args.format = "generic"
            # Show detected fields
            detected = []
            for name, candidates in [
                ("solution", SOLUTION_FIELDS),
                ("problem", PROBLEM_FIELDS),
                ("answer", ANSWER_FIELDS),
            ]:
                found = next((c for c in candidates if c in first_record), None)
                if found:
                    detected.append(f"{name}={found}")
            print(f"Auto-detected format: generic ({', '.join(detected)})")

    run_verification(args)


if __name__ == "__main__":
    main()
