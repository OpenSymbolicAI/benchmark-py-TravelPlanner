#!/usr/bin/env bash
# Run TravelPlanner benchmark across 12 models on 4 providers (15 hard queries each)
# Usage: ./run_all_models.sh
#   - Runs all models sequentially by default
#   - Each model gets 15 hard tasks from the train split, parallelism=3
#   - Results go to logs/ directory with timestamped folders

set -euo pipefail

NUM=15
SPLIT="train"
LEVEL="hard"
PARALLEL=1
MAX_ITER=10

run_model() {
    local provider="$1"
    local model="$2"
    local label="${provider}/${model}"
    echo ""
    echo "================================================================"
    echo "  Running: ${label}  (${NUM} ${LEVEL} tasks, p=${PARALLEL})"
    echo "================================================================"
    uv run travelplanner-bench \
        --model "$model" \
        --provider "$provider" \
        --split "$SPLIT" \
        --level "$LEVEL" \
        -n "$NUM" \
        -p "$PARALLEL" \
        --max-iterations "$MAX_ITER" \
    || echo "  !! ${label} FAILED with exit code $?"
}

echo "Starting TravelPlanner multi-model benchmark (re-run rate-limited models)"
echo "Tasks per model: ${NUM} | Split: ${SPLIT} | Level: ${LEVEL} | Parallel: ${PARALLEL}"
echo ""

# ── Re-run rate-limited models with p=1 ──
run_model groq meta-llama/llama-4-scout-17b-16e-instruct
run_model anthropic claude-sonnet-4-20250514
run_model anthropic claude-haiku-4-5-20251001
run_model openai gpt-4.1
run_model openai gpt-4o

echo ""
echo "================================================================"
echo "  All models complete! Results in logs/"
echo "================================================================"
echo ""
echo "To generate plots, run:"
echo "  uv run python plot_model_comparison.py"
