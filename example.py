#!/usr/bin/env python3
"""
Example: Run DeepSeek-R1-Distill-Llama-70B-AWQ on AMD Strix Halo via vLLM.

Requires:
    - Python 3.12 venv at .venv/ (created with uv)
    - vLLM 0.15.0+rocm700 wheel
    - TheRock SDK at /tmp/therock-sdk/ (ROCm 7.x runtime libs)
    - libmpi_cxx.so.40 stub at /tmp/ (OpenMPI 5.x compat shim)

Usage:
    .venv/bin/python example.py
    .venv/bin/python example.py --model "facebook/opt-125m" --max-model-len 512
    .venv/bin/python example.py --model "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"
    .venv/bin/python example.py --prompt "Prove that sqrt(2) is irrational."
"""

import os
import sys
import time
import argparse

# ── AMD Strix Halo (gfx1151) environment ─────────────────────────────────
# These must be set BEFORE importing torch/vllm.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")
os.environ.setdefault("VLLM_TARGET_DEVICE", "rocm")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Suppress harmless amdgpu.ids warning
os.environ.setdefault("AMD_LOG_LEVEL", "0")
# TheRock SDK library paths (needed for ROCm 7.0 runtime + amdsmi)
for lib_dir in ["/tmp/amdsmi-lib", "/tmp/therock-sdk/lib", "/tmp"]:
    if os.path.isdir(lib_dir):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dir not in ld:
            os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld}" if ld else lib_dir

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


def print_perf_summary(label, outputs, elapsed, prompt_tokens):
    """Print performance metrics for a generation run."""
    total_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    gen_tok_per_sec = total_gen_tokens / elapsed if elapsed > 0 else 0
    total_tokens = prompt_tokens + total_gen_tokens
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    print(f"  Prompt tokens:      {prompt_tokens}")
    print(f"  Generated tokens:   {total_gen_tokens}")
    print(f"  Total tokens:       {total_tokens}")
    print(f"  Wall time:          {elapsed:.2f} s")
    print(f"  Generation speed:   {gen_tok_per_sec:.2f} tok/s")
    if prompt_tokens > 0:
        # Rough TTFT is hard to measure with offline API; report overall instead
        print(f"  Overall throughput: {total_tokens / elapsed:.2f} tok/s (prompt + gen)")
    print(f"{'─' * 50}")


def run_one_shot(args):
    """Load model and run a single prompt."""
    print(f"Loading model: {args.model}")
    print(f"dtype={args.dtype}, gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f"max_model_len={args.max_model_len}")
    print()

    t_load_start = time.perf_counter()
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
    t_load = time.perf_counter() - t_load_start
    print(f"\nModel loaded in {t_load:.2f} s")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.95,
    )

    # --- Warmup run (short) ---
    print("\nWarmup run...")
    warmup_params = SamplingParams(temperature=0, max_tokens=8)
    t_warmup_start = time.perf_counter()
    llm.generate(["Hi"], warmup_params)
    t_warmup = time.perf_counter() - t_warmup_start
    print(f"Warmup done in {t_warmup:.2f} s")

    # --- Main generation ---
    print(f"\nPrompt: {args.prompt}\n")
    print("=" * 60)

    prompt_token_ids = llm.get_tokenizer().encode(args.prompt)
    prompt_tokens = len(prompt_token_ids)

    t_gen_start = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling_params)
    t_gen = time.perf_counter() - t_gen_start

    for output in outputs:
        generated_text = output.outputs[0].text
        print(generated_text)
    print("=" * 60)

    print_perf_summary("Performance", outputs, t_gen, prompt_tokens)


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
