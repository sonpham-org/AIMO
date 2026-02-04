#!/bin/bash
# AMD Trace Generation Job
# Run on AMD machine (Radeon 8060S / Vulkan)
# Uses llama-server with GGUF models

set -e
cd "$(dirname "$0")/.."

# Activate venv if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-8B-Q4_K_M.gguf}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
API_PORT="${API_PORT:-8081}"  # Use 8081 to avoid conflict with other servers
N_SAMPLES="${N_SAMPLES:-8}"
MAX_TURNS="${MAX_TURNS:-12}"
TEMPERATURE="${TEMPERATURE:-0.7}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
CTX_SIZE="${CTX_SIZE:-16384}"

echo "=== AMD Trace Generation Job ==="
echo "Model: $MODEL_PATH"
echo "Samples per problem: $N_SAMPLES"
echo "Max turns: $MAX_TURNS"
echo "Temperature: $TEMPERATURE"
echo ""

# Check if llama-server is running
if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
    echo "llama-server already running on port $API_PORT"
else
    echo "Starting llama-server..."
    # Start llama-server in background
    llama-server \
        --model "$MODEL_PATH" \
        --port "$API_PORT" \
        --n-gpu-layers "$N_GPU_LAYERS" \
        --ctx-size "$CTX_SIZE" \
        --flash-attn auto \
        --host 0.0.0.0 \
        > logs/llama_server_amd.log 2>&1 &

    LLAMA_PID=$!
    echo "llama-server started (PID: $LLAMA_PID)"

    # Wait for server to be ready
    echo "Waiting for server to be ready..."
    for i in {1..60}; do
        if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
            echo "Server ready!"
            break
        fi
        sleep 2
    done
fi

# Create output directory
OUTPUT_DIR="output/traces/amd_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
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
    --prompt-mix "reasoning:6,code_first:2" \
    --logprobs 5 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== AMD Job Complete ==="
echo "Traces saved to: $OUTPUT_DIR"
