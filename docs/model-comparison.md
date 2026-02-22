# Multi-Model Comparison Guide

Run the TravelPlanner benchmark across 12 LLMs on 4 providers (Fireworks, Groq, Anthropic, OpenAI) and generate cost/token/latency comparison plots.

## Prerequisites

1. API keys in `.env`:

```
FIREWORKS_API_KEY=fw_...
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
```

2. Dependencies installed:

```bash
uv sync
```

## Quick Start

```bash
# 1. Run all 12 models (15 tasks each, ~30-60 min)
./run_all_models.sh

# 2. Generate comparison plots
uv run python plot_model_comparison.py
```

Plots are saved to `plots/model_comparison.png`.

## Models Tested

### Passing (12 models)

| Provider | Model ID | Display Name | Pricing (in/out per 1M tokens) |
|----------|----------|-------------|-------------------------------|
| **Fireworks** | `gpt-oss-120b` | GPT-OSS-120B | $0.90 / $0.90 |
| **Fireworks** | `gpt-oss-20b` | GPT-OSS-20B | $0.20 / $0.20 |
| **Fireworks** | `deepseek-v3p2` | DeepSeek V3.2 | $0.90 / $0.90 |
| **Fireworks** | `kimi-k2p5` | Kimi K2.5 | $0.90 / $0.90 |
| **Fireworks** | `mixtral-8x22b-instruct` | Mixtral 8x22B | $0.90 / $0.90 |
| **Groq** | `llama-3.3-70b-versatile` | Llama 3.3 70B | $0.59 / $0.79 |
| **Groq** | `qwen/qwen3-32b` | Qwen3 32B | $0.34 / $0.40 |
| **Groq** | `meta-llama/llama-4-scout-17b-16e-instruct` | Llama 4 Scout | $0.20 / $0.60 |
| **Anthropic** | `claude-sonnet-4-20250514` | Claude Sonnet 4 | $3.00 / $15.00 |
| **Anthropic** | `claude-haiku-4-5-20251001` | Claude Haiku 4.5 | $0.80 / $4.00 |
| **OpenAI** | `gpt-4.1` | GPT-4.1 | $2.00 / $8.00 |
| **OpenAI** | `gpt-4.1-mini` | GPT-4.1 Mini | $0.40 / $1.60 |
| **OpenAI** | `gpt-4o` | GPT-4o | $2.50 / $10.00 |

### Known Failures

| Provider | Model ID | Failure Reason |
|----------|----------|---------------|
| **Groq** | `llama-3.1-8b-instant` | HTTP 429 rate limits + model too small to follow the structured plan format |
| **Fireworks** | `deepseek-coder-7b-instruct-v1p5` | HTTP 404 — model deprecated/removed from Fireworks serverless |

## Running Individual Models

```bash
# Single model, 1 task (smoke test)
uv run travelplanner-bench --model gpt-4.1 --provider openai --split train --level easy -n 1

# Single model, 15 tasks with parallelism
uv run travelplanner-bench --model llama-3.3-70b-versatile --provider groq --split train -n 15 -p 3

# Specific task IDs
uv run travelplanner-bench --model gpt-4o --provider openai --split train --task-ids tp_0003,tp_0010
```

## Customizing the Benchmark Script

Edit `run_all_models.sh` to change:

- `NUM=15` — number of tasks per model
- `SPLIT="train"` — dataset split (`train`=45, `validation`=180, `test`=1000)
- `PARALLEL=3` — concurrent tasks per model
- `MAX_ITER=10` — max agent iterations per task

To add a new model, add a line like:

```bash
run_model <provider> <model-id>
```

## Generating Plots

```bash
# Default: requires at least 5 tasks per run
uv run python plot_model_comparison.py

# Include smoke tests (1+ tasks)
uv run python plot_model_comparison.py --min-tasks 1

# Custom logs/output directories
uv run python plot_model_comparison.py --logs-dir logs --output-dir plots
```

The script:
1. Scans `logs/` for the most recent run of each model
2. Extracts per-task token counts, cost, and latency from `results.json`
3. Generates three plots in the `plots/` directory:
   - **`model_comparison.png`** — 4-panel overview (cost, tokens, latency, pass rate)
   - **`model_selection_scatter.png`** — Cost vs Latency scatter with bubble size = token usage and Pareto frontier
   - **`model_selection_heatmap.png`** — Ranked heatmap across all 5 metrics (cost, latency, tokens, LLM calls, pass rate)
4. Prints a summary table to the console, including failed models

## Plot Output Structure

All plots are saved to the `--output-dir` directory (default: `plots/`):

```
plots/
  model_comparison.png          # 4-panel overview
  model_selection_scatter.png   # Cost vs Latency (Pareto frontier)
  model_selection_heatmap.png   # Ranked heatmap (5 metrics)
```

Additional plots can be added to the same directory by extending `plot_model_comparison.py`.

## Logs Structure

After running, the `logs/` directory contains:

```
logs/
  20260221_212010_gpt-oss-120b/
    summary.json       # Aggregate metrics (pass rates, timing)
    results.json       # Per-task results with iteration_logs
    agent_debug.log    # Detailed agent execution log
    task_0001_tp_0000.md  # Per-task markdown report
    ...
```

Key fields in `results.json` per task:
- `final_pass` — whether the plan passed all constraints
- `wall_time_seconds` — end-to-end latency
- `iteration_logs[].input_tokens` / `output_tokens` — per-LLM-call token counts
- `iteration_logs[].time_seconds` — per-LLM-call latency
- `commonsense_micro` / `hard_micro` — constraint pass rates

## Adding a New Provider

1. Add the provider to `travelplanner_bench/runner.py` in the `provider_map` dict
2. Add model pricing to `travelplanner_bench/token_tracking.py` in `MODEL_PRICING`
3. Add a display name to `plot_model_comparison.py` in `MODEL_SHORT_NAMES`
4. Add a provider color to `plot_model_comparison.py` in `PROVIDER_COLORS`
5. Add the model to `run_all_models.sh`
