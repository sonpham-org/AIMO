#!/bin/bash
# NVIDIA Trace Generation Job
# Run on NVIDIA machine (4090 / CUDA)
# Uses vLLM or llama-server with GGUF models

set -e
cd "$(dirname "$0")/.."

# Activate venv if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Configuration - adjust for your setup
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-8B-Q4_K_M.gguf}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
API_PORT="${API_PORT:-8080}"
N_SAMPLES="${N_SAMPLES:-12}"
MAX_TURNS="${MAX_TURNS:-16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
CTX_SIZE="${CTX_SIZE:-32768}"

# Use vLLM for HF models, llama-server for GGUF
USE_VLLM="${USE_VLLM:-false}"  # Set to true if using HF model with vLLM

echo "=== NVIDIA Trace Generation Job ==="
echo "Model: $MODEL_PATH"
echo "Samples per problem: $N_SAMPLES"
echo "Max turns: $MAX_TURNS"
echo "Temperature: $TEMPERATURE"
echo "Using vLLM: $USE_VLLM"
echo ""

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not found"

# Check if server is running
if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
    echo "Server already running on port $API_PORT"
else
    mkdir -p logs

    if [ "$USE_VLLM" = "true" ]; then
        echo "Starting vLLM server..."
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --served-model-name "$MODEL_NAME" \
            --port "$API_PORT" \
            --host 0.0.0.0 \
            --gpu-memory-utilization 0.9 \
            --max-model-len "$CTX_SIZE" \
            > logs/vllm_server_nvidia.log 2>&1 &
    else
        echo "Starting llama-server..."
        llama-server \
            --model "$MODEL_PATH" \
            --port "$API_PORT" \
            --n-gpu-layers "$N_GPU_LAYERS" \
            --ctx-size "$CTX_SIZE" \
            --flash-attn auto \
            --host 0.0.0.0 \
            > logs/llama_server_nvidia.log 2>&1 &
    fi

    SERVER_PID=$!
    echo "Server started (PID: $SERVER_PID)"

    # Wait for server to be ready
    echo "Waiting for server to be ready..."
    for i in {1..120}; do
        if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
            echo "Server ready!"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "Server failed to start. Check logs."
            exit 1
        fi
        sleep 2
    done
fi

# Create output directory
OUTPUT_DIR="output/traces/nvidia_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Starting trace generation..."
echo "Output: $OUTPUT_DIR"
echo ""

# Run trace generation
python scripts/generate_traces.py \
    --api-base "http://localhost:$API_PORT/v1" \
    --model "$MODEL_NAME" \
    --n-samples "$N_SAMPLES" \
    --max-turns "$MAX_TURNS" \
    --temperature "$TEMPERATURE" \
    --competition aimo3 \
    --prompt-mix "reasoning:8,code_first:2,case_analysis:2" \
    --logprobs 5 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== NVIDIA Job Complete ==="
echo "Traces saved to: $OUTPUT_DIR"
