#!/usr/bin/env python3
"""
Example: Run DeepSeek-R1-Distill-Llama-70B-AWQ on AMD Strix Halo via vLLM.

Usage:
    python example.py
    python example.py --model "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"
    python example.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --dtype float16
    python example.py --prompt "Prove that sqrt(2) is irrational."
"""

import os
import sys
import argparse

# ── AMD Strix Halo (gfx1151/gfx1105) environment ──────────────────────────
# These must be set BEFORE importing torch/vllm.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("VLLM_TARGET_DEVICE", "rocm")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Suppress harmless amdgpu.ids warning
os.environ.setdefault("AMD_LOG_LEVEL", "0")

from vllm import LLM, SamplingParams


DEFAULT_MODEL = "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"
DEFAULT_PROMPT = (
    "What is the sum of all positive integers n such that n^2 + n + 1 "
    "divides n^4 + 3n^3 + 4n^2 + 3n + 1? Show your reasoning step by step."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a model with vLLM on AMD ROCm")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_PROMPT,
        help="Prompt to send to the model",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192,
        help="Maximum context length (reduce if OOM)",
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        help="Model dtype (float16 recommended for RDNA GPUs)",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90,
        help="Fraction of GPU memory to use (0.0-1.0)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--quantization", type=str, default=None,
        help="Quantization method override (auto-detected for AWQ/GPTQ models)",
    )
    parser.add_argument(
        "--server", action="store_true",
        help="Start an OpenAI-compatible API server instead of one-shot inference",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for the API server (with --server)",
    )
    return parser.parse_args()


def run_one_shot(args):
    """Load model and run a single prompt."""
    print(f"Loading model: {args.model}")
    print(f"dtype={args.dtype}, gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f"max_model_len={args.max_model_len}")
    print()

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=True,  # More compatible on RDNA GPUs
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.95,
    )

    print(f"Prompt: {args.prompt}\n")
    print("=" * 60)

    outputs = llm.generate([args.prompt], sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        print(generated_text)
        print("=" * 60)
        print(f"\nTokens generated: {len(output.outputs[0].token_ids)}")


def run_server(args):
    """Start an OpenAI-compatible API server."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", "1",
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--enforce-eager",
        "--trust-remote-code",
    ]
    if args.quantization:
        cmd.extend(["--quantization", args.quantization])

    print(f"Starting vLLM server on port {args.port}...")
    print(f"Model: {args.model}")
    print(f"API will be at: http://localhost:{args.port}/v1")
    print()
    os.execvp(cmd[0], cmd)


def main():
    args = parse_args()

    # Verify GPU is available
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: No GPU detected by PyTorch/ROCm.")
            print("Make sure HSA_OVERRIDE_GFX_VERSION=11.0.0 is set.")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    except Exception as e:
        print(f"WARNING: Could not query GPU: {e}")

    if args.server:
        run_server(args)
    else:
        run_one_shot(args)


if __name__ == "__main__":
    main()
