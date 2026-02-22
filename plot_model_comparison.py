"""Plot cost, tokens, and latency comparison across models from benchmark logs.

Usage:
    # From exported data file (recommended — check benchmark_data.json into git):
    uv run python plot_model_comparison.py --data benchmark_data.json

    # From logs directory (scans for latest runs):
    uv run python plot_model_comparison.py --logs-dir logs [--min-tasks 5]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from travelplanner_bench.token_tracking import MODEL_PRICING

# Provider colors for consistent styling
PROVIDER_COLORS = {
    "fireworks": "#FF6B35",
    "groq": "#6C5CE7",
    "anthropic": "#D4A574",
    "openai": "#10A37F",
}

# Short display names for models
MODEL_SHORT_NAMES = {
    "gpt-oss-120b": "GPT-OSS-120B",
    "gpt-oss-20b": "GPT-OSS-20B",
    "deepseek-v3p2": "DeepSeek V3.2",
    "kimi-k2p5": "Kimi K2.5",
    "mixtral-8x22b-instruct": "Mixtral 8x22B",
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "qwen3-32b": "Qwen3 32B",
    "llama-4-scout-17b-16e-instruct": "Llama 4 Scout",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "gpt-4.1": "GPT-4.1",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4o": "GPT-4o",
}


def _detect_provider(summary: dict) -> str:
    return summary.get("config", {}).get("provider", "unknown")


def _normalize_model_name(model: str) -> str:
    """Strip provider prefixes to get the short model name."""
    # accounts/fireworks/models/X -> X
    if "accounts/" in model:
        model = model.rsplit("/", 1)[-1]
    # meta-llama/llama-4-scout... -> llama-4-scout...
    # qwen/qwen3-32b -> qwen3-32b
    # moonshotai/kimi... -> kimi...
    for prefix in ("meta-llama/", "qwen/", "moonshotai/", "openai/"):
        if model.startswith(prefix):
            model = model[len(prefix):]
    return model


def _get_pricing(model_short: str) -> dict[str, float]:
    return MODEL_PRICING.get(model_short, {"input": 1.0, "output": 1.0})


def load_from_data_file(
    data_path: Path, level: str = "hard", min_tasks: int = 5,
) -> tuple[dict[str, dict], list[dict]]:
    """Load runs from an exported benchmark_data.json file.

    Returns (runs, failed_models) in the same format as load_latest_runs().
    """
    entries = json.loads(data_path.read_text())
    runs: dict[str, dict] = {}
    failed_models: list[dict] = []

    for entry in entries:
        if entry["level"] != level:
            continue
        if entry["total"] < min_tasks:
            continue

        model_short = entry["model"]
        provider = entry["provider"]
        delivered = entry["delivered"]

        if delivered == 0:
            error_msgs = [t.get("error", "") for t in entry["task_metrics"] if t.get("error")]
            first_error = (error_msgs[0][:200] if error_msgs else "No plans delivered")
            failed_models.append({
                "model_short": model_short,
                "model_raw": entry.get("model_raw", model_short),
                "provider": provider,
                "tasks": entry["total"],
                "errors": entry["errors"],
                "reason": first_error,
                "log_dir": entry.get("log_dir", ""),
            })
            continue

        task_metrics = []
        for t in entry["task_metrics"]:
            # Skip error tasks (e.g. 429 rate limits) — they have 0 tokens
            # and would skew cost/token/latency averages
            if t.get("error"):
                continue
            pricing = _get_pricing(model_short)
            cost = (
                t["input_tokens"] * pricing["input"] / 1_000_000
                + t["output_tokens"] * pricing["output"] / 1_000_000
            )
            task_metrics.append({
                "input_tokens": t["input_tokens"],
                "output_tokens": t["output_tokens"],
                "total_tokens": t["total_tokens"],
                "llm_calls": t["llm_calls"],
                "wall_time": t["wall_time_seconds"],
                "cost": cost,
                "passed": t["passed"],
            })

        if not task_metrics:
            continue

        runs[model_short] = {
            "model_short": model_short,
            "model_raw": entry.get("model_raw", model_short),
            "provider": provider,
            "summary": entry,
            "task_metrics": task_metrics,
            "log_dir": entry.get("log_dir", ""),
        }

    return runs, failed_models


def load_latest_runs(
    logs_dir: Path, min_tasks: int = 5, include_failed: bool = True,
) -> tuple[dict[str, dict], list[dict]]:
    """Find the most recent run per model with at least min_tasks tasks.

    Returns:
        (runs, failed_models) where runs maps model_short -> run data
        and failed_models is a list of dicts describing models that
        produced 0 deliveries or had 100% errors.
    """
    runs: dict[str, dict] = {}  # model_short -> run data
    failed_models: list[dict] = []
    seen: set[str] = set()

    # Sort directories by name (timestamp) descending
    log_dirs = sorted(logs_dir.iterdir(), reverse=True)

    for d in log_dirs:
        if not d.is_dir():
            continue
        summary_path = d / "summary.json"
        results_path = d / "results.json"
        if not summary_path.exists() or not results_path.exists():
            continue

        summary = json.loads(summary_path.read_text())
        config = summary.get("config", {})
        model_raw = config.get("model", "")
        model_short = _normalize_model_name(model_raw)

        # Skip if we already have a newer run for this model
        if model_short in seen:
            continue
        seen.add(model_short)

        results = json.loads(results_path.read_text())
        if len(results) < min_tasks:
            continue

        provider = _detect_provider(summary)
        delivered = sum(1 for r in results if r.get("plan_delivered"))
        errors = sum(1 for r in results if r.get("error"))

        # Track models where nothing was delivered (broken / incompatible)
        if delivered == 0:
            error_msgs = [r.get("error", "") for r in results if r.get("error")]
            first_error = error_msgs[0][:200] if error_msgs else "No plans delivered"
            failed_models.append({
                "model_short": model_short,
                "model_raw": model_raw,
                "provider": provider,
                "tasks": len(results),
                "errors": errors,
                "reason": first_error,
                "log_dir": str(d),
            })
            continue

        # Extract per-task metrics from iteration_logs (skip error tasks)
        task_metrics = []
        for r in results:
            if r.get("error"):
                continue
            logs = r.get("iteration_logs", [])
            input_tokens = sum(l.get("input_tokens", 0) for l in logs)
            output_tokens = sum(l.get("output_tokens", 0) for l in logs)
            total_tokens = input_tokens + output_tokens
            llm_calls = len(logs)
            wall_time = r.get("wall_time_seconds", 0)
            passed = r.get("final_pass", False)

            pricing = _get_pricing(model_short)
            cost = (
                input_tokens * pricing["input"] / 1_000_000
                + output_tokens * pricing["output"] / 1_000_000
            )

            task_metrics.append({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "llm_calls": llm_calls,
                "wall_time": wall_time,
                "cost": cost,
                "passed": passed,
            })

        if not task_metrics:
            continue

        runs[model_short] = {
            "model_short": model_short,
            "model_raw": model_raw,
            "provider": provider,
            "summary": summary,
            "task_metrics": task_metrics,
            "log_dir": str(d),
        }

    return runs, failed_models


def _sort_models(runs: dict[str, dict]) -> list[str]:
    """Sort models by provider then by avg cost (ascending)."""
    provider_order = {"fireworks": 0, "groq": 1, "anthropic": 2, "openai": 3}
    return sorted(
        runs.keys(),
        key=lambda m: (
            provider_order.get(runs[m]["provider"], 99),
            np.mean([t["cost"] for t in runs[m]["task_metrics"]]),
        ),
    )


def plot_comparison(
    runs: dict[str, dict], output_dir: Path, failed_models: list[dict] | None = None,
) -> None:
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models = _sort_models(runs)
    n = len(models)
    if n == 0:
        print("No runs found to plot.")
        return

    short_names = [MODEL_SHORT_NAMES.get(m, m) for m in models]
    providers = [runs[m]["provider"] for m in models]
    colors = [PROVIDER_COLORS.get(p, "#888888") for p in providers]

    # Aggregate metrics
    avg_tokens = [np.mean([t["total_tokens"] for t in runs[m]["task_metrics"]]) for m in models]
    avg_input = [np.mean([t["input_tokens"] for t in runs[m]["task_metrics"]]) for m in models]
    avg_output = [np.mean([t["output_tokens"] for t in runs[m]["task_metrics"]]) for m in models]
    avg_cost = [np.mean([t["cost"] for t in runs[m]["task_metrics"]]) for m in models]
    total_cost = [sum(t["cost"] for t in runs[m]["task_metrics"]) for m in models]
    avg_latency = [np.mean([t["wall_time"] for t in runs[m]["task_metrics"]]) for m in models]
    p95_latency = [np.percentile([t["wall_time"] for t in runs[m]["task_metrics"]], 95) for m in models]
    avg_calls = [np.mean([t["llm_calls"] for t in runs[m]["task_metrics"]]) for m in models]
    pass_rates = [
        np.mean([t["passed"] for t in runs[m]["task_metrics"]]) * 100 for m in models
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("TravelPlanner Benchmark: Multi-Model Comparison", fontsize=16, fontweight="bold")

    x = np.arange(n)
    bar_width = 0.6

    # ── Plot 1: Cost per task ──
    ax = axes[0, 0]
    bars = ax.bar(x, [c * 100 for c in avg_cost], bar_width, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Avg Cost per Task (cents)", fontsize=11)
    ax.set_title("Cost per Task", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    for bar, val in zip(bars, avg_cost):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"${val:.4f}", ha="center", va="bottom", fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)

    # ── Plot 2: Total tokens per task (stacked input/output) ──
    ax = axes[0, 1]
    bars_in = ax.bar(x, avg_input, bar_width, color=colors, alpha=0.7, label="Input", edgecolor="white", linewidth=0.5)
    bars_out = ax.bar(x, avg_output, bar_width, bottom=avg_input, color=colors, alpha=0.4, label="Output", edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Avg Tokens per Task", fontsize=11)
    ax.set_title("Token Usage per Task (Input + Output)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    for i, (inp, out) in enumerate(zip(avg_input, avg_output)):
        ax.text(i, inp + out + 50, f"{int(inp + out):,}", ha="center", va="bottom", fontsize=7.5)
    ax.legend(["Input tokens", "Output tokens"], loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Plot 3: Latency ──
    ax = axes[1, 0]
    bars = ax.bar(x, avg_latency, bar_width, color=colors, edgecolor="white", linewidth=0.5)
    # Add p95 markers
    ax.scatter(x, p95_latency, color="red", zorder=5, s=30, marker="v", label="P95")
    ax.set_ylabel("Seconds", fontsize=11)
    ax.set_title("Latency per Task (avg + P95)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    for bar, val in zip(bars, avg_latency):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=7.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Plot 4: Pass rate + LLM calls ──
    ax = axes[1, 1]
    bars = ax.bar(x, pass_rates, bar_width, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Pass Rate (%)", fontsize=11)
    ax.set_title("Final Pass Rate & Avg LLM Calls", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 115)
    for bar, rate, calls in zip(bars, pass_rates, avg_calls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%\n({calls:.1f} calls)", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.3, linewidth=1)
    ax.grid(axis="y", alpha=0.3)

    # Provider legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PROVIDER_COLORS[p], label=p.title())
        for p in ["fireworks", "groq", "anthropic", "openai"]
        if any(runs[m]["provider"] == p for m in models)
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=10,
               bbox_to_anchor=(0.98, 0.98), ncol=1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / "model_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

    # ── Summary table to console ──
    print(f"\n{'Model':<25} {'Provider':<12} {'Pass%':>7} {'Avg Cost':>10} {'Avg Tokens':>12} {'Avg Time':>10} {'LLM Calls':>10}")
    print("─" * 90)
    for m in models:
        name = MODEL_SHORT_NAMES.get(m, m)
        prov = runs[m]["provider"]
        metrics = runs[m]["task_metrics"]
        pr = np.mean([t["passed"] for t in metrics]) * 100
        ac = np.mean([t["cost"] for t in metrics])
        at = np.mean([t["total_tokens"] for t in metrics])
        al = np.mean([t["wall_time"] for t in metrics])
        calls = np.mean([t["llm_calls"] for t in metrics])
        print(f"{name:<25} {prov:<12} {pr:>6.1f}% ${ac:>8.4f} {at:>11,.0f} {al:>9.1f}s {calls:>9.1f}")

    # ── Failed models ──
    if failed_models:
        print(f"\n{'=' * 90}")
        print("FAILED MODELS (0% delivery rate)")
        print(f"{'=' * 90}")
        print(f"{'Model':<30} {'Provider':<12} {'Tasks':>6} {'Errors':>7}  {'Reason'}")
        print("─" * 90)
        for fm in failed_models:
            name = MODEL_SHORT_NAMES.get(fm["model_short"], fm["model_short"])
            print(f"{name:<30} {fm['provider']:<12} {fm['tasks']:>6} {fm['errors']:>7}  {fm['reason'][:50]}")


def plot_selection_scatter(runs: dict[str, dict], output_dir: Path) -> None:
    """Scatter plot: Cost vs Latency with bubble size = tokens, Pareto frontier."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models = _sort_models(runs)
    if len(models) < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    costs, latencies, tokens, clrs, names = [], [], [], [], []
    for m in models:
        metrics = runs[m]["task_metrics"]
        costs.append(np.mean([t["cost"] for t in metrics]) * 100)  # cents
        latencies.append(np.mean([t["wall_time"] for t in metrics]))
        tokens.append(np.mean([t["total_tokens"] for t in metrics]))
        clrs.append(PROVIDER_COLORS.get(runs[m]["provider"], "#888888"))
        names.append(MODEL_SHORT_NAMES.get(m, m))

    costs_arr = np.array(costs)
    latencies_arr = np.array(latencies)
    tokens_arr = np.array(tokens)

    # Bubble size: scale tokens to reasonable marker area
    size_min, size_max = 80, 800
    if tokens_arr.max() > tokens_arr.min():
        sizes = size_min + (tokens_arr - tokens_arr.min()) / (tokens_arr.max() - tokens_arr.min()) * (size_max - size_min)
    else:
        sizes = np.full_like(tokens_arr, (size_min + size_max) / 2)

    ax.scatter(costs_arr, latencies_arr, s=sizes, c=clrs, alpha=0.7, edgecolors="white", linewidths=1.5, zorder=5)

    # Label each bubble
    for i, name in enumerate(names):
        ax.annotate(name, (costs_arr[i], latencies_arr[i]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8, fontweight="bold")

    # Pareto frontier (lower cost AND lower latency is better)
    indices = np.argsort(costs_arr)
    pareto_x, pareto_y = [], []
    min_lat = float("inf")
    for idx in indices:
        if latencies_arr[idx] < min_lat:
            pareto_x.append(costs_arr[idx])
            pareto_y.append(latencies_arr[idx])
            min_lat = latencies_arr[idx]
    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, "g--", alpha=0.6, linewidth=2, label="Pareto frontier", zorder=3)
        ax.fill_between(pareto_x, pareto_y, alpha=0.05, color="green", zorder=2)

    ax.set_xlabel("Avg Cost per Task (cents)", fontsize=12)
    ax.set_ylabel("Avg Latency per Task (seconds)", fontsize=12)
    ax.set_title("Model Selection: Cost vs Latency\n(bubble size = token usage)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Provider legend + Pareto
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=PROVIDER_COLORS[p], label=p.title())
        for p in ["fireworks", "groq", "anthropic", "openai"]
        if any(runs[m]["provider"] == p for m in models)
    ]
    legend_elements.append(Line2D([0], [0], color="green", linestyle="--", alpha=0.6, linewidth=2, label="Pareto frontier"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "model_selection_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_selection_heatmap(runs: dict[str, dict], output_dir: Path) -> None:
    """Ranked heatmap: models × metrics, colored by rank (green=best, red=worst)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models = _sort_models(runs)
    n = len(models)
    if n < 2:
        return

    metric_names = ["Cost", "Latency", "Tokens", "LLM Calls", "Pass Rate"]
    # Raw values per model
    raw = np.zeros((n, 5))
    for i, m in enumerate(models):
        metrics = runs[m]["task_metrics"]
        raw[i, 0] = np.mean([t["cost"] for t in metrics]) * 100  # cents
        raw[i, 1] = np.mean([t["wall_time"] for t in metrics])
        raw[i, 2] = np.mean([t["total_tokens"] for t in metrics])
        raw[i, 3] = np.mean([t["llm_calls"] for t in metrics])
        raw[i, 4] = np.mean([t["passed"] for t in metrics]) * 100

    # Rank each column (1 = best). For pass rate, higher is better; for others, lower is better.
    ranks = np.zeros_like(raw)
    for col in range(5):
        if col == 4:  # pass rate: higher = better → rank descending
            ranks[:, col] = n + 1 - np.argsort(np.argsort(-raw[:, col])) - 1 + 1
            # simpler: use scipy-style ranking
            order = np.argsort(-raw[:, col])
        else:  # lower = better
            order = np.argsort(raw[:, col])
        for rank_pos, idx in enumerate(order):
            ranks[idx, col] = rank_pos + 1

    short_names = [MODEL_SHORT_NAMES.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.6)))

    # Normalize ranks to [0, 1] for colormap (1=best→green, n=worst→red)
    norm_ranks = (ranks - 1) / max(n - 1, 1)  # 0 = best, 1 = worst
    cmap = plt.cm.RdYlGn_r  # green (low) to red (high)

    im = ax.imshow(norm_ranks, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Annotate cells with rank and raw value
    fmt_funcs = [
        lambda v: f"${v:.3f}",     # cost (cents → dollar-like)
        lambda v: f"{v:.1f}s",     # latency
        lambda v: f"{int(v):,}",   # tokens
        lambda v: f"{v:.1f}",      # llm calls
        lambda v: f"{v:.0f}%",     # pass rate
    ]
    for i in range(n):
        for j in range(5):
            rank_val = int(ranks[i, j])
            raw_val = fmt_funcs[j](raw[i, j])
            ax.text(j, i, f"#{rank_val}\n{raw_val}",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white" if norm_ranks[i, j] > 0.6 else "black")

    ax.set_xticks(range(5))
    ax.set_xticklabels(metric_names, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_title("Model Comparison: Ranked Heatmap\n(#1 = best per metric, green = better)", fontsize=14, fontweight="bold")

    # Add composite score (avg rank) as right-side labels
    avg_ranks = ranks.mean(axis=1)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([f"Avg: {avg_ranks[i]:.1f}" for i in range(n)], fontsize=9)
    ax2.tick_params(length=0)

    plt.tight_layout()
    out_path = output_dir / "model_selection_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot model comparison from benchmark logs")
    parser.add_argument("--data", type=str, default=None, help="Path to benchmark_data.json (preferred over --logs-dir)")
    parser.add_argument("--level", type=str, default="hard", help="Filter by level when using --data (default: hard)")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Logs directory (default: logs)")
    parser.add_argument("--min-tasks", type=int, default=5, help="Min tasks per run to include (default: 5)")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory for plots (default: plots)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.data:
        data_path = Path(args.data)
        print(f"Loading from {data_path} (level={args.level}, min_tasks={args.min_tasks})...")
        runs, failed_models = load_from_data_file(data_path, level=args.level, min_tasks=args.min_tasks)
    else:
        logs_dir = Path(args.logs_dir)
        print(f"Scanning {logs_dir} for benchmark runs (min {args.min_tasks} tasks)...")
        runs, failed_models = load_latest_runs(logs_dir, min_tasks=args.min_tasks)

    print(f"Found {len(runs)} passing model runs: {', '.join(sorted(runs.keys()))}")
    if failed_models:
        print(f"Found {len(failed_models)} failed models: {', '.join(fm['model_short'] for fm in failed_models)}")

    if not runs:
        print("No qualifying runs found. Run the benchmark first with: ./run_all_models.sh")
        return

    plot_comparison(runs, output_dir, failed_models=failed_models)
    plot_selection_scatter(runs, output_dir)
    plot_selection_heatmap(runs, output_dir)


if __name__ == "__main__":
    main()
