"""TravelPlanner Benchmark runner: evaluation loop, metrics, logging, CLI."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travelplanner_bench.models import IterationLog, TravelPlannerResult, TravelPlannerTask

LOGS_DIR = Path(__file__).parent.parent / "logs"

log = logging.getLogger(__name__)


def _create_run_dir(model: str) -> Path:
    """Create a timestamped run directory under logs/."""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_short = model.rsplit("/", 1)[-1]
    run_dir = LOGS_DIR / f"{timestamp}_{model_short}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_single_task(
    task: TravelPlannerTask,
    llm_config: Any,  # LLMConfig - keep Any to avoid opensymbolicai import at module level
    max_iterations: int,
    obs_config: Any | None = None,  # ObservabilityConfig
) -> dict[str, Any]:
    """Run a single TravelPlanner task with the OpenSymbolicAI agent."""
    from travelplanner_bench.agent import TravelPlannerAgent
    from travelplanner_bench.evaluation import evaluate_plan
    from travelplanner_bench.tools import ReferenceDatabase

    agent = TravelPlannerAgent(
        llm=llm_config,
        max_iterations=max_iterations,
        observability=obs_config,
    )

    start = time.perf_counter()
    try:
        plan, iterations, iteration_logs = agent.solve(task)
        elapsed = time.perf_counter() - start

        db = ReferenceDatabase(task.reference_information)
        eval_result = evaluate_plan(plan, task, db)
        eval_result.iterations = iterations
        eval_result.wall_time_seconds = elapsed
        eval_result.model = str(llm_config.model) if hasattr(llm_config, "model") else ""

        result = eval_result.model_dump()
        result["iteration_logs"] = [il.model_dump() for il in iteration_logs]
        return result
    except Exception as e:
        elapsed = time.perf_counter() - start
        log.exception("Task %s failed: %s", task.task_id, e)
        result = TravelPlannerResult(
            task_id=task.task_id,
            query=task.query,
            level=task.level,
            days=task.days,
            wall_time_seconds=elapsed,
            error=str(e),
        ).model_dump()
        result["iteration_logs"] = []
        return result


def _write_task_log(run_dir: Path, index: int, result: dict[str, Any]) -> None:
    """Write a per-task log file."""
    log_path = run_dir / f"task_{index:04d}_{result['task_id']}.md"
    lines = [
        f"# Task {index}: {result['task_id']}",
        "",
        f"**Query**: {result.get('query', '')}",
        f"**Level**: {result.get('level', '')}",
        f"**Days**: {result.get('days', '')}",
        "",
        "## Result",
        f"**Plan delivered**: {result.get('plan_delivered', False)}",
        f"**Final pass**: {result.get('final_pass', False)}",
        f"**Iterations**: {result.get('iterations', 0)}",
        f"**Time**: {result.get('wall_time_seconds', 0):.2f}s",
        "",
        "## Commonsense Constraints",
        f"- Within sandbox: {result.get('within_sandbox', False)}",
        f"- Complete info: {result.get('complete_info', False)}",
        f"- Within current city: {result.get('within_current_city', False)}",
        f"- Reasonable city route: {result.get('reasonable_city_route', False)}",
        f"- Diverse restaurants: {result.get('diverse_restaurants', False)}",
        f"- Diverse attractions: {result.get('diverse_attractions', False)}",
        f"- Non-conflicting transport: {result.get('non_conflicting_transport', False)}",
        f"- Valid accommodation: {result.get('valid_accommodation', False)}",
        f"- **Micro**: {result.get('commonsense_micro', 0):.2f}",
        f"- **Macro**: {result.get('commonsense_macro', False)}",
        "",
        "## Hard Constraints",
        f"- Budget: {result.get('budget_ok')}",
        f"- Room rule: {result.get('room_rule_ok')}",
        f"- Room type: {result.get('room_type_ok')}",
        f"- Cuisine: {result.get('cuisine_ok')}",
        f"- Transportation: {result.get('transportation_ok')}",
        f"- **Micro**: {result.get('hard_micro', 0):.2f}",
        f"- **Macro**: {result.get('hard_macro', False)}",
        "",
    ]

    if result.get("error"):
        lines.extend(["## Error", f"```\n{result['error']}\n```", ""])

    plan = result.get("plan")
    if plan:
        lines.extend([
            "## Plan",
            "```json",
            json.dumps(plan, indent=2, default=str),
            "```",
            "",
        ])

    # Full iteration logs with LLM inputs/outputs and tool calls
    iteration_logs = result.get("iteration_logs", [])

    # Aggregate token/time totals per phase
    phase_totals: dict[str, dict[str, float]] = {}
    for il in iteration_logs:
        phase = il.get("phase", "orchestrator")
        if phase not in phase_totals:
            phase_totals[phase] = {"input_tokens": 0, "output_tokens": 0, "time": 0.0, "llm_calls": 0}
        phase_totals[phase]["input_tokens"] += il.get("input_tokens", 0)
        phase_totals[phase]["output_tokens"] += il.get("output_tokens", 0)
        phase_totals[phase]["time"] += il.get("time_seconds", 0.0)
        phase_totals[phase]["llm_calls"] += 1

    if phase_totals:
        lines.extend(["## Token & Time Summary", ""])
        for phase, totals in phase_totals.items():
            lines.append(
                f"- **{phase}**: {totals['llm_calls']} LLM calls, "
                f"{totals['input_tokens']} in / {totals['output_tokens']} out tokens, "
                f"{totals['time']:.1f}s"
            )
        lines.append("")

    for iter_log in iteration_logs:
        phase = iter_log.get("phase", "orchestrator")
        iter_num = iter_log.get("iteration", iter_log.get("attempt", "?"))
        phase_label = phase.replace("_", " ").title()
        header = f"## {phase_label} — Iteration {iter_num}"

        lines.extend([
            header,
            "",
            f"**Model**: {iter_log.get('model', '')}",
            f"**Tokens**: {iter_log.get('input_tokens', 0)} in / {iter_log.get('output_tokens', 0)} out",
            f"**Time**: {iter_log.get('time_seconds', 0):.2f}s",
            f"**Goal achieved**: {iter_log.get('goal_achieved', False)}",
            "",
            "### LLM Prompt",
            "```",
            iter_log.get("prompt", ""),
            "```",
            "",
            "### LLM Response",
            "```",
            iter_log.get("response", ""),
            "```",
            "",
            "### Extracted Code",
            "```python",
            iter_log.get("extracted_code", ""),
            "```",
            "",
            "### Tool Calls",
            "",
        ])
        for step in iter_log.get("steps", []):
            status = "OK" if step.get("success") else f"FAIL: {step.get('error', '')}"
            lines.extend([
                f"**Step {step.get('step', '?')}**: `{step.get('primitive', '?')}({step.get('args', '')})`",
                f"- Status: {status}",
                f"- Time: {step.get('time_seconds', 0):.2f}s",
                f"- Result:",
                "```",
                step.get("result", ""),
                "```",
                "",
            ])

    log_path.write_text("\n".join(lines))


def _compute_summary(
    results: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Compute aggregate metrics from results."""
    from travelplanner_bench.evaluation import compute_aggregate_metrics

    # Reconstruct TravelPlannerResult objects
    result_objects = []
    for r in results:
        try:
            obj = TravelPlannerResult(**{
                k: v for k, v in r.items()
                if k in TravelPlannerResult.model_fields
            })
            result_objects.append(obj)
        except Exception:
            log.warning("Could not reconstruct result for %s", r.get("task_id"))

    metrics = compute_aggregate_metrics(result_objects)
    metrics["config"] = config
    return metrics


def run_benchmark(
    model: str,
    provider: str,
    split: str = "validation",
    level: str | None = None,
    num: int | None = None,
    max_iterations: int = 10,
    parallel: int = 3,
    shuffle: bool = False,
    seed: int = 42,
    task_ids: list[str] | None = None,
    observe: bool = False,
) -> dict[str, Any]:
    """Run the full TravelPlanner benchmark."""
    from dotenv import load_dotenv

    load_dotenv()

    from opensymbolicai.llm import LLMConfig, Provider

    from travelplanner_bench.data import get_level_counts, load_travelplanner

    if task_ids:
        # Load enough tasks to cover the requested IDs
        max_index = max(int(tid.split("_")[1]) for tid in task_ids) + 1
        all_tasks = load_travelplanner(split=split, level=None, num=max_index, shuffle=False)
        requested = {tid for tid in task_ids}
        tasks = [t for t in all_tasks if t.task_id in requested]
    else:
        tasks = load_travelplanner(split=split, level=level, num=num, shuffle=shuffle, seed=seed)
    level_counts = get_level_counts(tasks)

    print(f"\n{'=' * 60}")
    print("TravelPlanner Benchmark")
    print(f"{'=' * 60}")
    print(f"Model: {provider}/{model}")
    print(f"Tasks: {len(tasks)} ({', '.join(f'{k}={v}' for k, v in level_counts.items())})")
    print(f"Max iterations: {max_iterations}")

    run_dir = _create_run_dir(model)
    print(f"Logs: {run_dir}")

    # Set up observability (send traces to local collector)
    obs_config = None
    if observe:
        from opensymbolicai.observability import ObservabilityConfig

        session_id = uuid.uuid4().hex
        collector_url = "http://localhost:8100/events"
        obs_config = ObservabilityConfig(
            enabled=True,
            collector_url=collector_url,
            session_id=session_id,
            capture_llm_prompts=True,
            capture_llm_responses=True,
            capture_execution_steps=True,
            capture_plan_source=True,
            tags={"benchmark": "travelplanner", "split": split, "model": model},
        )
        print(f"Traces: {collector_url}")
        print(f"Session: {session_id}")
        print("Dashboard: http://localhost:8101")

    # Set up file logging
    agent_log_path = run_dir / "agent_debug.log"
    file_handler = logging.FileHandler(agent_log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    logging.getLogger("travelplanner_bench.agent").addHandler(file_handler)
    logging.getLogger("travelplanner_bench.agent").setLevel(logging.DEBUG)

    provider_map = {
        "ollama": Provider.OLLAMA,
        "openai": Provider.OPENAI,
        "anthropic": Provider.ANTHROPIC,
        "fireworks": Provider.FIREWORKS,
        "groq": Provider.GROQ,
    }
    # Fireworks requires fully-qualified model paths
    if provider == "fireworks" and not model.startswith("accounts/"):
        model = f"accounts/fireworks/models/{model}"
    llm_config = LLMConfig(provider=provider_map[provider], model=model)

    config = {
        "model": model,
        "provider": provider,
        "split": split,
        "level": level,
        "num": num,
        "max_iterations": max_iterations,
        "parallel": parallel,
    }

    results: list[dict[str, Any]] = []
    total = len(tasks)
    completed = 0

    workers = min(parallel, total)
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{total}] {task.query[:80]}...")
            result = _run_single_task(task, llm_config, max_iterations, obs_config)
            results.append(result)
            _write_task_log(run_dir, i, result)
            status = "PASS" if result.get("final_pass") else "FAIL"
            delivered = "delivered" if result.get("plan_delivered") else "no plan"
            print(
                f"  -> {status} ({delivered}) | "
                f"cs={result.get('commonsense_micro', 0):.0%} | "
                f"hard={result.get('hard_micro', 0):.0%} | "
                f"time={result.get('wall_time_seconds', 0):.1f}s"
            )
    else:
        futures_map: dict[Any, tuple[int, Any]] = {}
        indexed_results: dict[int, dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i, task in enumerate(tasks, 1):
                future = executor.submit(_run_single_task, task, llm_config, max_iterations, obs_config)
                futures_map[future] = (i, task)

            for future in as_completed(futures_map):
                idx, task = futures_map[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as e:
                    result = TravelPlannerResult(
                        task_id=task.task_id,
                        query=task.query,
                        level=task.level,
                        days=task.days,
                        error=str(e),
                    ).model_dump()
                indexed_results[idx] = result
                _write_task_log(run_dir, idx, result)
                status = "PASS" if result.get("final_pass") else "FAIL"
                print(
                    f"  [{completed}/{total}] {status} | "
                    f"cs={result.get('commonsense_micro', 0):.0%} | "
                    f"{task.query[:60]}..."
                )

        for i in range(1, total + 1):
            results.append(indexed_results[i])

    # Compute and save summary
    summary = _compute_summary(results, config)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # Print summary
    print(f"\n{'=' * 60}")
    print("Results Summary")
    print(f"{'=' * 60}")
    print(f"Model: {provider}/{model}")
    print(f"Delivery rate: {summary.get('delivery_rate', 0):.1%} ({summary.get('delivered', 0)}/{summary.get('total', 0)})")
    print(f"\nCommonsense Constraints:")
    print(f"  Micro avg: {summary.get('commonsense_micro_avg', 0):.1%}")
    print(f"  Macro rate: {summary.get('commonsense_macro_rate', 0):.1%}")
    print(f"\nHard Constraints:")
    print(f"  Micro avg: {summary.get('hard_micro_avg', 0):.1%}")
    print(f"  Macro rate: {summary.get('hard_macro_rate', 0):.1%}")
    print(f"\nFinal Pass Rate: {summary.get('final_pass_rate', 0):.1%}")

    per_level = summary.get("per_level", {})
    if per_level:
        print(f"\nPer-Level Breakdown:")
        for lvl in sorted(per_level.keys()):
            stats = per_level[lvl]
            print(
                f"  {lvl}: delivery={stats.get('delivery_rate', 0):.0%} | "
                f"cs_macro={stats.get('commonsense_macro_rate', 0):.0%} | "
                f"hard_macro={stats.get('hard_macro_rate', 0):.0%} | "
                f"final={stats.get('final_pass_rate', 0):.0%} "
                f"({stats.get('total', 0)} tasks)"
            )

    timing = summary.get("timing", {})
    print(f"\nTiming: {timing.get('total_seconds', 0):.1f}s total, {timing.get('avg_per_task', 0):.1f}s avg/task")
    print(f"Errors: {summary.get('errors', 0)}")
    print(f"\nLogs saved to: {run_dir}")

    return summary


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TravelPlanner Benchmark: Multi-constraint travel planning evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 5 easy validation tasks with GPT-4o
  uv run travelplanner-bench --model gpt-4o --provider openai --level easy -n 5

  # Run all validation tasks with Fireworks
  uv run travelplanner-bench --model gpt-oss-120b --provider fireworks

  # Run on training set
  uv run travelplanner-bench --model gpt-4o --provider openai --split train --num 10
        """,
    )
    parser.add_argument(
        "--model",
        default="gpt-oss-120b",
        help="Model name/ID (default: gpt-oss-120b)",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "anthropic", "fireworks", "groq"],
        default="fireworks",
        help="LLM provider (default: fireworks)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
        help="Dataset split (default: validation)",
    )
    parser.add_argument(
        "--level", "-l",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Filter by difficulty level (default: all)",
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=None,
        help="Number of tasks to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Max agent iterations per task (default: 10)",
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle tasks before evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated task IDs to run (e.g., tp_0003,tp_0010,tp_0015)",
    )
    parser.add_argument(
        "--observe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable observability traces (default: enabled, use --no-observe to disable)",
    )
    args = parser.parse_args()

    task_ids = args.task_ids.split(",") if args.task_ids else None

    run_benchmark(
        model=args.model,
        provider=args.provider,
        split=args.split,
        level=args.level,
        num=args.num,
        max_iterations=args.max_iterations,
        parallel=args.parallel,
        shuffle=args.shuffle,
        seed=args.seed,
        task_ids=task_ids,
        observe=args.observe,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
