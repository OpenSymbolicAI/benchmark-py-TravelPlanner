#!/usr/bin/env python3
"""Analyze token usage patterns from TravelPlanner benchmark data."""

import json
import os
from collections import defaultdict
from pathlib import Path

# ─── 1. Load benchmark_data.json ────────────────────────────────────────────

with open("benchmark_data.json") as f:
    runs = json.load(f)

# ─── 2. Aggregate per-model stats (passed=true, total_tokens>0) ─────────────

ModelStats = lambda: {
    "input_tokens": [],
    "output_tokens": [],
    "total_tokens": [],
    "llm_calls": [],
    "wall_time": [],
}

# model -> level -> stats
by_model_level = defaultdict(lambda: defaultdict(ModelStats))
# model -> stats (all levels combined)
by_model_all = defaultdict(ModelStats)

for run in runs:
    model = run["model"]
    level = run.get("level", "all")
    for t in run["task_metrics"]:
        if not t["passed"] or t["total_tokens"] <= 0:
            continue
        for bucket in [by_model_level[model][level], by_model_all[model]]:
            bucket["input_tokens"].append(t["input_tokens"])
            bucket["output_tokens"].append(t["output_tokens"])
            bucket["total_tokens"].append(t["total_tokens"])
            bucket["llm_calls"].append(t["llm_calls"])
            bucket["wall_time"].append(t["wall_time_seconds"])


def avg(lst):
    return sum(lst) / len(lst) if lst else 0


def fmt(n):
    """Format large numbers with commas."""
    return f"{n:,.0f}"


# ─── 3. Print overall model summary table ───────────────────────────────────

print("=" * 120)
print("TOKEN USAGE ANALYSIS — TravelPlanner Benchmark (successful tasks only)")
print("=" * 120)

print()
print("─" * 120)
print(
    f"{'Model':<42s} {'Tasks':>5s}  {'Avg Input':>10s}  {'Avg Output':>10s}  "
    f"{'Avg Total':>10s}  {'In/Out':>6s}  {'Avg LLM':>7s}  {'Avg Time':>8s}  "
    f"{'Tok/Call':>8s}"
)
print("─" * 120)

# Sort by total tokens ascending
sorted_models = sorted(by_model_all.keys(), key=lambda m: avg(by_model_all[m]["total_tokens"]))

for model in sorted_models:
    s = by_model_all[model]
    n = len(s["total_tokens"])
    if n == 0:
        continue
    a_in = avg(s["input_tokens"])
    a_out = avg(s["output_tokens"])
    a_tot = avg(s["total_tokens"])
    ratio = a_in / a_out if a_out > 0 else 0
    a_calls = avg(s["llm_calls"])
    a_time = avg(s["wall_time"])
    tok_per_call = a_tot / a_calls if a_calls > 0 else 0

    print(
        f"{model:<42s} {n:>5d}  {fmt(a_in):>10s}  {fmt(a_out):>10s}  "
        f"{fmt(a_tot):>10s}  {ratio:>6.1f}x  {a_calls:>7.1f}  {a_time:>7.1f}s  "
        f"{fmt(tok_per_call):>8s}"
    )

print("─" * 120)

# ─── 4. Per-model breakdown by level ────────────────────────────────────────

print()
print("=" * 120)
print("BREAKDOWN BY DIFFICULTY LEVEL")
print("=" * 120)

for model in sorted_models:
    levels_data = by_model_level[model]
    if not levels_data:
        continue

    # Only show models with multiple levels
    available_levels = [l for l in ["easy", "medium", "hard", "all"] if l in levels_data and levels_data[l]["total_tokens"]]
    if not available_levels:
        continue

    print()
    print(f"  {model}")
    print(f"  {'Level':<10s} {'Tasks':>5s}  {'Avg Input':>10s}  {'Avg Output':>10s}  {'Avg Total':>10s}  {'In/Out':>6s}  {'Avg Calls':>9s}  {'Avg Time':>8s}")
    print(f"  {'-'*75}")

    for level in available_levels:
        s = levels_data[level]
        n = len(s["total_tokens"])
        if n == 0:
            continue
        a_in = avg(s["input_tokens"])
        a_out = avg(s["output_tokens"])
        a_tot = avg(s["total_tokens"])
        ratio = a_in / a_out if a_out > 0 else 0
        a_calls = avg(s["llm_calls"])
        a_time = avg(s["wall_time"])
        print(
            f"  {level:<10s} {n:>5d}  {fmt(a_in):>10s}  {fmt(a_out):>10s}  "
            f"{fmt(a_tot):>10s}  {ratio:>6.1f}x  {a_calls:>9.1f}  {a_time:>7.1f}s"
        )

# ─── 5. Phase-level analysis from detailed results.json ─────────────────────

print()
print("=" * 120)
print("PHASE-LEVEL TOKEN BREAKDOWN (Retrieval vs Assembly)")
print("=" * 120)

# Find all results.json files that have iteration_logs
log_dir = Path("logs")
results_files = {
    "gpt-4o (hard/train)": "20260222_020548_gpt-4o",
    "claude-sonnet-4 (hard/train)": "20260222_015745_claude-sonnet-4-20250514",
    "gpt-4.1 (hard/train)": "20260222_020429_gpt-4.1",
    "gpt-4.1-mini (hard/train)": "20260222_003837_gpt-4.1-mini",
    "kimi-k2p5 (hard/train)": "20260221_225939_kimi-k2p5",
    "llama-3.3-70b (hard/train)": "20260221_234356_llama-3.3-70b-versatile",
    "llama-4-scout (hard/train)": "20260222_014830_llama-4-scout-17b-16e-instruct",
    "mixtral-8x22b (hard/train)": "20260221_232931_mixtral-8x22b-instruct",
    "qwen3-32b (hard/train)": "20260221_234422_qwen3-32b",
}

for label, dirname in sorted(results_files.items()):
    rpath = log_dir / dirname / "results.json"
    if not rpath.exists():
        continue

    with open(rpath) as f:
        tasks = json.load(f)

    # Filter to passed tasks with iteration_logs
    passed_tasks = [t for t in tasks if t.get("final_pass") and t.get("iteration_logs")]
    if not passed_tasks:
        continue

    retrieval_in, retrieval_out, retrieval_time = [], [], []
    assembly_in, assembly_out, assembly_time = [], [], []
    retrieval_attempts, assembly_attempts = [], []

    for task in passed_tasks:
        r_in, r_out, r_t, r_count = 0, 0, 0.0, 0
        a_in, a_out, a_t, a_count = 0, 0, 0.0, 0
        for log in task["iteration_logs"]:
            if log["phase"] == "retrieval":
                r_in += log["input_tokens"]
                r_out += log["output_tokens"]
                r_t += log["time_seconds"]
                r_count += 1
            elif log["phase"] == "assembly":
                a_in += log["input_tokens"]
                a_out += log["output_tokens"]
                a_t += log["time_seconds"]
                a_count += 1

        retrieval_in.append(r_in)
        retrieval_out.append(r_out)
        retrieval_time.append(r_t)
        retrieval_attempts.append(r_count)
        assembly_in.append(a_in)
        assembly_out.append(a_out)
        assembly_time.append(a_t)
        assembly_attempts.append(a_count)

    n = len(passed_tasks)
    total_in = avg(retrieval_in) + avg(assembly_in)
    total_out = avg(retrieval_out) + avg(assembly_out)
    total_tok = total_in + total_out

    r_pct = (avg(retrieval_in) + avg(retrieval_out)) / total_tok * 100 if total_tok > 0 else 0
    a_pct = (avg(assembly_in) + avg(assembly_out)) / total_tok * 100 if total_tok > 0 else 0

    print(f"\n  {label}  ({n} passed tasks)")
    print(f"  {'Phase':<12s} {'Avg In':>8s}  {'Avg Out':>8s}  {'Avg Total':>10s}  {'% of Total':>10s}  {'Avg Time':>8s}  {'Avg Iters':>9s}")
    print(f"  {'-'*75}")
    print(
        f"  {'Retrieval':<12s} {fmt(avg(retrieval_in)):>8s}  {fmt(avg(retrieval_out)):>8s}  "
        f"{fmt(avg(retrieval_in)+avg(retrieval_out)):>10s}  {r_pct:>9.1f}%  "
        f"{avg(retrieval_time):>7.1f}s  {avg(retrieval_attempts):>9.1f}"
    )
    print(
        f"  {'Assembly':<12s} {fmt(avg(assembly_in)):>8s}  {fmt(avg(assembly_out)):>8s}  "
        f"{fmt(avg(assembly_in)+avg(assembly_out)):>10s}  {a_pct:>9.1f}%  "
        f"{avg(assembly_time):>7.1f}s  {avg(assembly_attempts):>9.1f}"
    )
    print(
        f"  {'TOTAL':<12s} {fmt(total_in):>8s}  {fmt(total_out):>8s}  "
        f"{fmt(total_tok):>10s}  {'100.0%':>10s}  "
        f"{avg(retrieval_time)+avg(assembly_time):>7.1f}s  "
        f"{avg(retrieval_attempts)+avg(assembly_attempts):>9.1f}"
    )


# ─── 6. Efficiency comparison: tokens per passed task ───────────────────────

print()
print("=" * 120)
print("EFFICIENCY: TOKENS PER SUCCESSFUL TASK (sorted by total tokens)")
print("=" * 120)
print()
print(
    f"{'Model':<42s} {'Passed':>6s} {'Failed':>6s} {'Pass%':>6s}  "
    f"{'Total Tok/Task':>14s}  {'Total Tok Used':>14s}"
)
print("─" * 100)

model_efficiency = {}
for model in sorted_models:
    s = by_model_all[model]
    n = len(s["total_tokens"])
    if n == 0:
        continue
    # Count total tasks attempted for this model
    total_attempted = 0
    total_failed = 0
    for run in runs:
        if run["model"] == model:
            total_attempted += len(run["task_metrics"])
            total_failed += sum(1 for t in run["task_metrics"] if not t["passed"] or t["total_tokens"] <= 0)

    a_tot = avg(s["total_tokens"])
    total_used = sum(s["total_tokens"])
    pass_rate = n / total_attempted * 100 if total_attempted > 0 else 0

    model_efficiency[model] = {
        "passed": n,
        "failed": total_failed,
        "pass_rate": pass_rate,
        "avg_total": a_tot,
        "sum_total": total_used,
    }

for model in sorted(model_efficiency, key=lambda m: model_efficiency[m]["avg_total"]):
    e = model_efficiency[model]
    print(
        f"{model:<42s} {e['passed']:>6d} {e['failed']:>6d} {e['pass_rate']:>5.1f}%  "
        f"{fmt(e['avg_total']):>14s}  {fmt(e['sum_total']):>14s}"
    )

print("─" * 100)

# ─── 7. Input vs Output token distribution ──────────────────────────────────

print()
print("=" * 120)
print("INPUT vs OUTPUT TOKEN DISTRIBUTION (all successful tasks)")
print("=" * 120)
print()
print(
    f"{'Model':<42s} {'Input %':>8s}  {'Output %':>8s}  "
    f"{'Min Total':>10s}  {'Max Total':>10s}  {'Median':>10s}"
)
print("─" * 100)

for model in sorted_models:
    s = by_model_all[model]
    n = len(s["total_tokens"])
    if n == 0:
        continue
    a_in = avg(s["input_tokens"])
    a_out = avg(s["output_tokens"])
    a_tot = a_in + a_out
    in_pct = a_in / a_tot * 100 if a_tot > 0 else 0
    out_pct = a_out / a_tot * 100 if a_tot > 0 else 0

    sorted_totals = sorted(s["total_tokens"])
    min_t = sorted_totals[0]
    max_t = sorted_totals[-1]
    mid = len(sorted_totals) // 2
    median_t = sorted_totals[mid] if len(sorted_totals) % 2 == 1 else (sorted_totals[mid - 1] + sorted_totals[mid]) / 2

    print(
        f"{model:<42s} {in_pct:>7.1f}%  {out_pct:>7.1f}%  "
        f"{fmt(min_t):>10s}  {fmt(max_t):>10s}  {fmt(median_t):>10s}"
    )

print("─" * 100)
print()
print("Analysis complete.")
