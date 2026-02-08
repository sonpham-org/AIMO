#!/usr/bin/env python3
"""
Format Converter for AIMO3 Fine-Tuning Datasets

Converts curated datasets into formats ready for different training frameworks:
  1. Harmony (native gpt-oss-120b format) - no conversion needed
  2. ChatML (OpenAI-style) - for Unsloth QLoRA SFT
  3. ShareGPT / Alpaca - for common SFT frameworks
  4. Tinker (Fireworks) - for Tinker GRPO RL training

Input: Curated JSONL from quality_filter.py
Output: Training-ready format files (JSONL)

Usage:
  # Convert curated dataset to ChatML for Unsloth SFT
  python dataset_generation/scripts/format_converter.py \
      --input dataset_generation/output/curated_3000.jsonl \
      --format chatml \
      --output dataset_generation/output/train_chatml.jsonl

  # Convert to ShareGPT format (problem -> solution conversations)
  python dataset_generation/scripts/format_converter.py \
      --input dataset_generation/output/curated_3000.jsonl \
      --format sharegpt \
      --output dataset_generation/output/train_sharegpt.jsonl

  # Convert to Tinker format for GRPO RL (problems only)
  python dataset_generation/scripts/format_converter.py \
      --input dataset_generation/output/curated_3000.jsonl \
      --format tinker_rl \
      --output dataset_generation/output/train_tinker.jsonl

  # Convert AIMO3 Hard directly (with Harmony parsing)
  python dataset_generation/scripts/format_converter.py \
      --input-dataset aimo3_hard \
      --format chatml \
      --max-rows 1000 \
      --output dataset_generation/output/hard_chatml_1k.jsonl
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Harmony parser (based on data/downloads/aimo3_tir/convert_harmony_format.py)
# ---------------------------------------------------------------------------

def parse_harmony_segments(harmony_text: str) -> list[dict[str, Any]]:
    """Parse Harmony format into structured segments."""
    segments = []
    parts = re.split(r'<\|end\|><\|start\|>', harmony_text)

    for part in parts:
        if not part.strip():
            continue

        # Tool call: assistant to=python
        tool_match = re.match(
            r'assistant to=python<\|channel\|>([^<]+)<\|message\|>(.+?)(?:<\|end\|>|$)',
            part, re.DOTALL
        )
        if tool_match:
            segments.append({
                'type': 'tool_call',
                'role': 'assistant',
                'content': tool_match.group(2).strip(),
                'tool': 'python',
            })
            continue

        # Tool response: <|call|>
        if part.startswith('<|call|>'):
            resp_match = re.match(
                r'<\|call\|><\|start\|>assistant<\|channel\|>([^<]+)<\|message\|>(.+?)(?:<\|end\|>|$)',
                part, re.DOTALL
            )
            if resp_match:
                segments.append({
                    'type': 'tool_response',
                    'role': 'tool',
                    'content': resp_match.group(2).strip(),
                })
                continue

        # Regular assistant message
        msg_match = re.match(
            r'assistant<\|channel\|>([^<]+)<\|message\|>(.+?)(?:<\|end\|>|$)',
            part, re.DOTALL
        )
        if msg_match:
            segments.append({
                'type': 'message',
                'role': 'assistant',
                'content': msg_match.group(2).strip(),
            })
            continue

    return segments


# ---------------------------------------------------------------------------
# Output format converters
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant that solves problems step by step, "
    "using Python code execution when helpful for computation and verification. "
    "Always provide your final answer in \\boxed{} format."
)

TOOL_DESCRIPTION = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Execute Python code for mathematical computation and verification",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    }
}


def to_chatml(problem: str, solution: str, is_harmony: bool = False) -> dict:
    """Convert to ChatML format (OpenAI-style messages).

    For Harmony format solutions, parses tool calls into proper message structure.
    For plain text solutions, wraps as single assistant message.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    if is_harmony and ("<|start|>" in solution or "<|end|>" in solution):
        segments = parse_harmony_segments(solution)
        tool_id = 0
        for seg in segments:
            if seg["type"] == "tool_call":
                call_id = f"call_python_{tool_id}"
                tool_id += 1
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": json.dumps({"code": seg["content"]})
                        }
                    }]
                })
            elif seg["type"] == "tool_response":
                # Use the most recent tool call ID
                prev_id = f"call_python_{tool_id - 1}" if tool_id > 0 else "call_python_0"
                messages.append({
                    "role": "tool",
                    "content": seg["content"],
                    "tool_call_id": prev_id,
                })
            elif seg["type"] == "message":
                messages.append({
                    "role": "assistant",
                    "content": seg["content"],
                })
    else:
        # Plain text solution
        messages.append({
            "role": "assistant",
            "content": solution,
        })

    return {"messages": messages}


def to_sharegpt(problem: str, solution: str, **kwargs) -> dict:
    """Convert to ShareGPT format (conversations list).

    This format is widely supported by Axolotl, LLaMA-Factory, etc.
    """
    conversations = [
        {"from": "system", "value": SYSTEM_PROMPT},
        {"from": "human", "value": problem},
        {"from": "gpt", "value": solution},
    ]
    return {"conversations": conversations}


def to_alpaca(problem: str, solution: str, **kwargs) -> dict:
    """Convert to Alpaca format (instruction/input/output)."""
    return {
        "instruction": SYSTEM_PROMPT,
        "input": problem,
        "output": solution,
    }


def to_tinker_sft(problem: str, solution: str, is_harmony: bool = False, **kwargs) -> dict:
    """Convert to Tinker SFT format (messages list for supervised fine-tuning)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    if is_harmony and ("<|start|>" in solution or "<|end|>" in solution):
        segments = parse_harmony_segments(solution)
        for seg in segments:
            if seg["type"] == "tool_call":
                messages.append({
                    "role": "assistant",
                    "content": f"```python\n{seg['content']}\n```",
                })
            elif seg["type"] == "tool_response":
                messages.append({
                    "role": "user",
                    "content": f"[Tool output]\n{seg['content']}",
                })
            elif seg["type"] == "message":
                messages.append({
                    "role": "assistant",
                    "content": seg["content"],
                })
    else:
        messages.append({"role": "assistant", "content": solution})

    return {"messages": messages}


def to_tinker_rl(problem: str, answer: str = None, **kwargs) -> dict:
    """Convert to Tinker RL format (problems only, with optional ground truth for reward)."""
    record = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
    }
    if answer:
        record["metadata"] = {"ground_truth": answer}
    return record


FORMAT_CONVERTERS = {
    "chatml": to_chatml,
    "sharegpt": to_sharegpt,
    "alpaca": to_alpaca,
    "tinker_sft": to_tinker_sft,
    "tinker_rl": to_tinker_rl,
}


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/downloads")


def load_curated_jsonl(path: str, max_rows: int = 0):
    """Load curated JSONL (output of quality_filter.py)."""
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            d = json.loads(line)
            yield {
                "problem": d.get("problem_text", ""),
                "solution": d.get("solution_text", ""),
                "answer": d.get("answer"),
                "source": d.get("source", ""),
                "is_harmony": d.get("source", "") in ("aimo3_hard", "aimo3_tir"),
            }


def load_aimo3_hard(max_rows: int = 0):
    """Load AIMO3 Hard directly."""
    path = DATA_DIR / "aimo3_hard" / "AIMO3-High-Difficulty-Tool-Calling-Dataset.jsonl"
    if not path.exists():
        return
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            d = json.loads(line)
            meta = d.get("metadata_infos", {})
            yield {
                "problem": meta.get("problem", ""),
                "solution": d.get("text", ""),
                "answer": str(meta.get("standard", "")),
                "source": "aimo3_hard",
                "is_harmony": True,
            }


def load_aimo3_tir(max_rows: int = 0):
    """Load AIMO3 TIR directly."""
    path = DATA_DIR / "aimo3_tir" / "data.csv"
    if not path.exists():
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            yield {
                "problem": row.get("prompt", ""),
                "solution": row.get("completion", ""),
                "answer": None,
                "source": "aimo3_tir",
                "is_harmony": True,
            }


DATASET_LOADERS = {
    "aimo3_hard": load_aimo3_hard,
    "aimo3_tir": load_aimo3_tir,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(args):
    fmt = args.format
    converter = FORMAT_CONVERTERS[fmt]

    # Load input
    if args.input:
        samples = load_curated_jsonl(args.input, max_rows=args.max_rows)
        input_name = Path(args.input).stem
    elif args.input_dataset:
        if args.input_dataset not in DATASET_LOADERS:
            print(f"Error: unknown dataset '{args.input_dataset}'. "
                  f"Available: {list(DATASET_LOADERS.keys())}")
            sys.exit(1)
        samples = DATASET_LOADERS[args.input_dataset](max_rows=args.max_rows)
        input_name = args.input_dataset
    else:
        print("Error: provide --input or --input-dataset")
        sys.exit(1)

    output_path = args.output or f"dataset_generation/output/{input_name}_{fmt}.jsonl"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Converting to {fmt} format...")
    print(f"Output: {output_path}")

    count = 0
    with open(output_path, "w") as out:
        for sample in samples:
            kwargs = {
                "problem": sample["problem"],
                "solution": sample["solution"],
                "answer": sample.get("answer"),
                "is_harmony": sample.get("is_harmony", False),
            }

            record = converter(**kwargs)

            # Add metadata if requested
            if args.include_metadata:
                record["_source"] = sample.get("source", "")
                record["_answer"] = sample.get("answer")

            out.write(json.dumps(record) + "\n")
            count += 1

    print(f"Converted {count:,} samples to {fmt} format -> {output_path}")

    # Print sample
    if args.show_sample:
        print(f"\n--- Sample output (first record) ---")
        with open(output_path) as f:
            first = json.loads(f.readline())
            print(json.dumps(first, indent=2)[:2000])
            if len(json.dumps(first)) > 2000:
                print("... (truncated)")


def main():
    parser = argparse.ArgumentParser(description="Format Converter for AIMO3 Datasets")
    parser.add_argument("--input", type=str, default=None,
                        help="Input curated JSONL file")
    parser.add_argument("--input-dataset", type=str, default=None,
                        choices=list(DATASET_LOADERS.keys()),
                        help="Load directly from a named dataset")
    parser.add_argument("--format", type=str, required=True,
                        choices=list(FORMAT_CONVERTERS.keys()),
                        help="Output format")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Max rows to convert (0=all)")
    parser.add_argument("--include-metadata", action="store_true",
                        help="Include source metadata in output")
    parser.add_argument("--show-sample", action="store_true",
                        help="Print a sample record after conversion")

    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
