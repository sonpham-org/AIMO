#!/bin/bash
# =============================================================================
# AIMO3 Dataset Generation Pipeline — Full Run
#
# Runs the complete pipeline from raw downloaded datasets to packaged output:
#   1. Profile all datasets
#   2. Verify answers
#   3. Run quality filtering (SFT mode)
#   4. Build DPO pairs
#   5. Package for submission
#
# Usage:
#   # Full pipeline with defaults (3000 SFT samples)
#   bash dataset_generation/scripts/run_pipeline.sh
#
#   # Custom target size
#   TARGET=5000 bash dataset_generation/scripts/run_pipeline.sh
#
#   # Quick test (100 rows per dataset)
#   QUICK=1 bash dataset_generation/scripts/run_pipeline.sh
# =============================================================================

set -euo pipefail

# Configuration (override with env vars)
TARGET="${TARGET:-3000}"
DATASET_NAME="${DATASET_NAME:-aimo3-curated-sft}"
OUTPUT_DIR="${OUTPUT_DIR:-dataset_generation/output}"
SCRIPTS_DIR="dataset_generation/scripts"
MAX_ROWS="${MAX_ROWS:-0}"  # 0 = all
QUICK="${QUICK:-0}"

if [ "$QUICK" = "1" ]; then
    MAX_ROWS=1000
    TARGET=500
    echo "=== QUICK MODE: max_rows=1000, target=500 ==="
fi

echo "============================================================"
echo "AIMO3 Dataset Generation Pipeline"
echo "============================================================"
echo "Target: $TARGET samples"
echo "Output: $OUTPUT_DIR"
echo "Max rows per dataset: ${MAX_ROWS:-all}"
echo ""

# Step 1: Profile datasets
echo "============================================================"
echo "STEP 1/5: Profiling datasets"
echo "============================================================"
PROFILE_ARGS="--datasets aimo3_hard aimo3_tir limo s1k"
if [ "$MAX_ROWS" != "0" ]; then
    PROFILE_ARGS="$PROFILE_ARGS --max-rows $MAX_ROWS"
fi
python3 "$SCRIPTS_DIR/dataset_profiler.py" $PROFILE_ARGS \
    --output "$OUTPUT_DIR/profile_report.json"
echo ""

# Step 2: Verify answers
echo "============================================================"
echo "STEP 2/5: Verifying answers"
echo "============================================================"
VERIFY_ARGS="--datasets aimo3_hard aimo3_tir limo"
if [ "$MAX_ROWS" != "0" ]; then
    VERIFY_ARGS="$VERIFY_ARGS --max-rows $MAX_ROWS"
fi
python3 "$SCRIPTS_DIR/answer_verification.py" $VERIFY_ARGS
echo ""

# Step 3: Quality filtering (SFT mode)
echo "============================================================"
echo "STEP 3/5: Quality filtering (SFT mode, target=$TARGET)"
echo "============================================================"
FILTER_ARGS="--datasets aimo3_hard aimo3_tir limo s1k --target $TARGET"
FILTER_ARGS="$FILTER_ARGS --min-pass-rate 0.01 --max-pass-rate 0.90"
FILTER_ARGS="$FILTER_ARGS --output $OUTPUT_DIR/curated_sft_${TARGET}.jsonl"
FILTER_ARGS="$FILTER_ARGS --iw-sft --profile"
if [ "$MAX_ROWS" != "0" ]; then
    FILTER_ARGS="$FILTER_ARGS --max-rows $MAX_ROWS"
fi
python3 "$SCRIPTS_DIR/quality_filter.py" $FILTER_ARGS
echo ""

# Step 4: Build DPO pairs
echo "============================================================"
echo "STEP 4/5: Building DPO pairs"
echo "============================================================"
DPO_ARGS="--source aimo3_hard --output $OUTPUT_DIR/dpo_pairs.jsonl"
if [ "$MAX_ROWS" != "0" ]; then
    DPO_ARGS="$DPO_ARGS --max-problems $MAX_ROWS"
fi
python3 "$SCRIPTS_DIR/build_dpo_pairs.py" $DPO_ARGS
echo ""

# Step 5: Package
echo "============================================================"
echo "STEP 5/5: Packaging dataset"
echo "============================================================"
python3 "$SCRIPTS_DIR/package_dataset.py" \
    --input "$OUTPUT_DIR/curated_sft_${TARGET}.jsonl" \
    --dpo-input "$OUTPUT_DIR/dpo_pairs.jsonl" \
    --name "$DATASET_NAME" \
    --output-dir "$OUTPUT_DIR/packaged" \
    --kaggle
echo ""

# Final summary
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Output directory: $OUTPUT_DIR/packaged/"
echo ""
echo "Files:"
ls -lh "$OUTPUT_DIR/packaged/" 2>/dev/null || echo "(no files found)"
echo ""
echo "To upload to Kaggle:"
echo "  export \$(grep KAGGLE_API_TOKEN /home/son/GitHub/AIMO/.env) && \\"
echo "  kaggle datasets create -p $OUTPUT_DIR/packaged"
