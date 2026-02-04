#!/usr/bin/env python3
"""
Ablation testing script for AIMO answer selection strategies.

Tests different combinations of improvements:
- Baseline: Simple majority vote / entropy-gated consensus
- Option C: Self-consistency boost (track all \boxed{} answers, boost repeated ones)
- Option B: Code verification (verify when top-2 candidates are close)
- Combined: B + C together

Usage:
    python scripts/ablation_test.py --traces-dir output/traces/gpt_oss_120b_full
    python scripts/ablation_test.py --traces-dir output/traces/aime_amd_20260204_073853
"""

import argparse
import json
import re
import os
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class SelectionConfig:
    """Configuration for answer selection strategy."""
    name: str

    # Entropy gating
    entropy_threshold: float = 0.3
    min_consensus_votes: int = 2

    # Option C: Self-consistency boost
    enable_self_consistency: bool = False
    repeat_answer_boost: float = 1.3
    code_answer_boost: float = 1.5

    # Option B: Code verification
    enable_code_verification: bool = False
    verify_threshold: float = 1.5  # Verify if top/second ratio < this

    # Error penalty
    error_penalty: float = 0.6


# Define ablation configurations
ABLATION_CONFIGS = [
    # Simple majority vote (no entropy filter)
    SelectionConfig(
        name="Simple majority vote (no filter)",
        entropy_threshold=1.0,  # Effectively no filter
        min_consensus_votes=1,
        enable_self_consistency=False,
        enable_code_verification=False,
    ),
    # Entropy-filtered baseline
    SelectionConfig(
        name="Entropy-gated (threshold 0.5)",
        entropy_threshold=0.5,
        enable_self_consistency=False,
        enable_code_verification=False,
    ),
    # Option C only
    SelectionConfig(
        name="Option C (self-consistency, ent 0.5)",
        entropy_threshold=0.5,
        enable_self_consistency=True,
        enable_code_verification=False,
    ),
    # Option B only
    SelectionConfig(
        name="Option B (code verify, ent 0.5)",
        entropy_threshold=0.5,
        enable_self_consistency=False,
        enable_code_verification=True,
    ),
    # Combined B + C
    SelectionConfig(
        name="B + C combined (ent 0.5)",
        entropy_threshold=0.5,
        enable_self_consistency=True,
        enable_code_verification=True,
    ),
    # Ablation: different entropy thresholds for B+C
    SelectionConfig(
        name="B + C (ent 0.3, strict)",
        entropy_threshold=0.3,
        enable_self_consistency=True,
        enable_code_verification=True,
    ),
    SelectionConfig(
        name="B + C (ent 0.7, relaxed)",
        entropy_threshold=0.7,
        enable_self_consistency=True,
        enable_code_verification=True,
    ),
    # Ablation: different boost values
    SelectionConfig(
        name="B + C (repeat boost 1.5)",
        entropy_threshold=0.5,
        enable_self_consistency=True,
        enable_code_verification=True,
        repeat_answer_boost=1.5,
    ),
    SelectionConfig(
        name="B + C (code boost 2.0)",
        entropy_threshold=0.5,
        enable_self_consistency=True,
        enable_code_verification=True,
        code_answer_boost=2.0,
    ),
]


def extract_boxed_answers(text: str) -> list[int]:
    """Extract all \\boxed{...} answers from text."""
    answers = []
    pattern = r'\\boxed\{([^{}]*)\}'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        # Clean up the match
        cleaned = match.strip()
        # Try to extract a number
        num_match = re.search(r'-?\d+', cleaned)
        if num_match:
            try:
                ans = int(num_match.group())
                answers.append(ans)
            except ValueError:
                pass
    return answers


def extract_code_outputs(text: str) -> list[int]:
    """Extract numbers from code execution outputs in the response."""
    outputs = []

    # Look for output markers like "Output:", "Result:", "Answer:", etc.
    output_patterns = [
        r'(?:Output|Result|Answer):\s*(\d+)',
        r'```output\s*(\d+)',
        r'>>> .*?\n(\d+)\n',
        r'print\([^)]+\)\s*#?\s*(\d+)',
    ]

    for pattern in output_patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            try:
                outputs.append(int(match))
            except ValueError:
                pass

    return outputs


def analyze_attempt(attempt: dict) -> dict:
    """Analyze a single attempt to extract all relevant signals."""
    text = attempt.get('full_response_text', '')

    # Extract all boxed answers throughout the response
    all_boxed = extract_boxed_answers(text)

    # Extract code outputs
    code_outputs = extract_code_outputs(text)

    # Final answer from the attempt
    final_answer = attempt.get('answer')

    return {
        'final_answer': final_answer,
        'answer_source': attempt.get('answer_source', 'unknown'),
        'all_boxed_answers': all_boxed,
        'code_outputs': code_outputs,
        'entropy': attempt.get('entropy', 0),
        'n_python_errors': attempt.get('n_python_errors', 0),
        'n_python_calls': attempt.get('n_python_calls', 0),
    }


def select_answer(analyzed_attempts: list[dict], cfg: SelectionConfig) -> tuple[int, str]:
    """
    Select final answer using the given strategy configuration.

    Returns: (selected_answer, method_used)
    """
    if not analyzed_attempts:
        return 0, "no_attempts"

    # Filter valid attempts (with numeric answers and low entropy)
    valid_attempts = []
    for a in analyzed_attempts:
        if a['final_answer'] is not None and isinstance(a['final_answer'], (int, float)):
            if a['entropy'] <= cfg.entropy_threshold:
                valid_attempts.append(a)

    if not valid_attempts:
        # Fallback: use all attempts with answers, ignoring entropy
        valid_attempts = [a for a in analyzed_attempts if a['final_answer'] is not None]
        if not valid_attempts:
            return 0, "no_valid_answers"

    # Count votes per answer
    answer_scores = defaultdict(float)
    answer_details = defaultdict(list)

    for a in valid_attempts:
        ans = int(a['final_answer'])
        answer_details[ans].append(a)

        # Base score is 1 vote
        score = 1.0

        if cfg.enable_self_consistency:
            # Apply error penalty
            if a['n_python_errors'] > 0:
                score *= cfg.error_penalty

            # Boost if answer appeared in code output
            if ans in a['code_outputs']:
                score *= cfg.code_answer_boost

            # Boost based on repeated appearances in boxed answers
            appearances = sum(1 for b in a['all_boxed_answers'] if b == ans)
            if appearances > 1:
                score *= (cfg.repeat_answer_boost ** (appearances - 1))

        answer_scores[ans] += score

    if not answer_scores:
        return 0, "no_scores"

    # Sort by score
    sorted_answers = sorted(answer_scores.items(), key=lambda x: -x[1])
    top_answer, top_score = sorted_answers[0]

    # Check consensus requirement
    raw_votes = len(answer_details[top_answer])
    if raw_votes < cfg.min_consensus_votes:
        return 0, "no_consensus"

    method = "consensus"

    # Option B: Code verification when uncertain
    if cfg.enable_code_verification and len(sorted_answers) >= 2:
        second_answer, second_score = sorted_answers[1]

        if second_score > 0 and top_score / second_score < cfg.verify_threshold:
            # Uncertain - check if we can verify with code
            top_has_code_support = any(
                top_answer in a['code_outputs']
                for a in answer_details[top_answer]
            )
            second_has_code_support = any(
                second_answer in a['code_outputs']
                for a in answer_details[second_answer]
            )

            if not top_has_code_support and second_has_code_support:
                # Second answer has code support, switch to it
                top_answer = second_answer
                method = "code_verified_switch"
            elif top_has_code_support:
                method = "code_verified"
            else:
                method = "uncertain_no_code"

    return top_answer, method


def run_ablation(traces_dir: Path, configs: list[SelectionConfig]) -> dict:
    """Run ablation testing on a set of traces."""

    # Load all problem traces
    problem_files = list(traces_dir.glob("problem_*.json"))
    print(f"Found {len(problem_files)} problem traces in {traces_dir}")

    if not problem_files:
        return {"error": "No problem traces found"}

    problems = []
    for pf in problem_files:
        with open(pf) as f:
            problems.append(json.load(f))

    # Run each configuration
    results = {}

    for cfg in configs:
        correct = 0
        total = 0
        details = []
        method_counts = Counter()

        for prob in problems:
            ground_truth = prob.get('ground_truth')
            if ground_truth is None:
                continue

            # Analyze all attempts
            analyzed = [analyze_attempt(a) for a in prob.get('attempts', [])]

            # Select answer using this config
            selected, method = select_answer(analyzed, cfg)
            method_counts[method] += 1

            is_correct = selected == ground_truth
            if is_correct:
                correct += 1
            total += 1

            details.append({
                'problem_id': prob['problem_id'],
                'ground_truth': ground_truth,
                'selected': selected,
                'method': method,
                'correct': is_correct,
            })

        accuracy = correct / total if total > 0 else 0
        results[cfg.name] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'method_counts': dict(method_counts),
            'details': details,
        }

        print(f"\n{cfg.name}:")
        print(f"  Accuracy: {correct}/{total} = {accuracy:.1%}")
        print(f"  Methods: {dict(method_counts)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Ablation testing for AIMO selection strategies")
    parser.add_argument("--traces-dir", type=Path, required=True, help="Directory containing trace JSON files")
    parser.add_argument("--output", type=Path, help="Output file for detailed results (JSON)")
    args = parser.parse_args()

    if not args.traces_dir.exists():
        print(f"Error: Traces directory does not exist: {args.traces_dir}")
        return 1

    print("=" * 60)
    print("AIMO Ablation Testing")
    print("=" * 60)

    results = run_ablation(args.traces_dir, ABLATION_CONFIGS)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<45} {'Accuracy':>12}")
    print("-" * 60)
    for name, data in results.items():
        if 'error' not in data:
            acc_str = f"{data['correct']}/{data['total']} ({data['accuracy']:.1%})"
            print(f"{name:<45} {acc_str:>12}")

    # Save detailed results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
