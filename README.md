# AIMO Solver

This repository contains a local inference script and environment setup for the AI Mathematical Olympiad (AIMO) competition. The system utilizes large language models to solve complex mathematical problems by iteratively generating and executing Python code in a sandboxed environment.

## Features
- Scalable inference using vLLM backend.
- Sandboxed Python execution via Jupyter kernels.
- Support for multiple solution attempts with early-stopping consensus.
- Configuration for both NVIDIA and AMD GPUs.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your environment and run:
   ```bash
   python main.py --model_path /path/to/your/model
   ```

## Hardware Requirements
- High-VRAM GPU (recommended 24GB+)
- Support for FP8/BF16/FP16 precision
