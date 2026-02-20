"""Data loading for the TravelPlanner benchmark.

Loads from HuggingFace dataset `osunlp/TravelPlanner` and parses
JSON string fields into structured types.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)


def parse_reference_information(ref_info: Any) -> list[dict[str, str]]:
    """Parse reference_information into a list of {Description, Content} dicts.

    The field may be a JSON string, a Python literal string, or already a list.
    """
    if isinstance(ref_info, list):
        return ref_info
    if not isinstance(ref_info, str) or not ref_info.strip():
        return []
    try:
        parsed = json.loads(ref_info)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(ref_info)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return []


def parse_local_constraint(constraint: Any) -> dict[str, Any]:
    """Parse local_constraint into a dict.

    Expected keys: cuisine, room_type, room_rule, transportation.
    """
    if isinstance(constraint, dict):
        return constraint
    if not isinstance(constraint, str) or not constraint.strip():
        return {}
    try:
        parsed = json.loads(constraint)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(constraint)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return {}


def parse_annotated_plan(plan: Any) -> list[dict[str, Any]] | None:
    """Parse annotated_plan JSON string into list of day entries.

    The train split stores annotated_plan as [metadata_dict, [day1, day2, ...]].
    We normalise this to a flat list of day-plan dicts.
    """
    parsed: Any = plan
    if isinstance(parsed, str):
        if not parsed.strip():
            return None
        try:
            parsed = json.loads(parsed)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(parsed)
            except (ValueError, SyntaxError):
                return None

    if not isinstance(parsed, list):
        return None

    # Normalise [metadata_dict, [day_plans...]] → [day_plans...]
    if (
        len(parsed) == 2
        and isinstance(parsed[0], dict)
        and isinstance(parsed[1], list)
        and all(isinstance(d, dict) for d in parsed[1])
    ):
        return parsed[1]

    # Already a flat list of dicts
    if all(isinstance(d, dict) for d in parsed):
        return parsed

    return parsed


def parse_date_field(date_val: Any) -> list[str]:
    """Parse the date field which may be a string repr of a list or an actual list."""
    if isinstance(date_val, list):
        return date_val
    if not isinstance(date_val, str) or not date_val.strip():
        return []
    try:
        parsed = json.loads(date_val)
        if isinstance(parsed, list):
            return [str(d) for d in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(date_val)
        if isinstance(parsed, list):
            return [str(d) for d in parsed]
    except (ValueError, SyntaxError):
        pass
    # Try regex extraction of date patterns
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", date_val)
    return dates


def _extract_budget_from_query(query: str) -> int:
    """Try to extract budget from the query text."""
    match = re.search(r"budget[^$]*\$\s*([\d,]+)", query, re.IGNORECASE)
    if match:
        return int(match.group(1).replace(",", ""))
    match = re.search(r"\$([\d,]+)\s*(?:budget|total)", query, re.IGNORECASE)
    if match:
        return int(match.group(1).replace(",", ""))
    return 0


def _extract_people_from_query(query: str) -> int:
    """Try to extract number of people from the query text."""
    match = re.search(r"for\s+(\d+)\s+people", query, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1


def load_travelplanner(
    split: str = "validation",
    level: str | None = None,
    num: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list:
    """Load TravelPlanner tasks from HuggingFace.

    Args:
        split: "train", "validation", or "test"
        level: Filter by "easy", "medium", or "hard" (None = all)
        num: Max tasks to return (None = all)
        shuffle: Shuffle before slicing
        seed: RNG seed for shuffling

    Returns:
        List of TravelPlannerTask objects.
    """
    from datasets import load_dataset

    from travelplanner_bench.models import TravelPlannerTask

    ds = load_dataset("osunlp/TravelPlanner", split, split=split)

    if shuffle:
        ds = ds.shuffle(seed=seed)

    tasks: list[TravelPlannerTask] = []
    for i, row in enumerate(ds):
        task_level = row.get("level", "easy")
        if level is not None and task_level != level:
            continue

        ref_info = parse_reference_information(row.get("reference_information", []))
        local_constraint = parse_local_constraint(row.get("local_constraint", {}))
        annotated_plan = parse_annotated_plan(row.get("annotated_plan"))
        date = parse_date_field(row.get("date", []))

        query = row.get("query", "")
        budget = row.get("budget", 0)
        if not budget:
            budget = _extract_budget_from_query(query)
        people_number = row.get("people_number", 0)
        if not people_number:
            people_number = _extract_people_from_query(query)

        task = TravelPlannerTask(
            task_id=f"tp_{i:04d}",
            query=query,
            org=row.get("org", ""),
            dest=row.get("dest", ""),
            days=row.get("days", 3),
            date=date,
            level=task_level,
            visiting_city_number=row.get("visiting_city_number", 1),
            people_number=people_number,
            local_constraint=local_constraint,
            budget=budget,
            reference_information=ref_info,
            annotated_plan=annotated_plan,
        )
        tasks.append(task)

        if num is not None and len(tasks) >= num:
            break

    log.info("Loaded %d TravelPlanner tasks (split=%s, level=%s)", len(tasks), split, level)
    return tasks


def get_level_counts(tasks: list) -> dict[str, int]:
    """Count tasks per difficulty level."""
    counts: dict[str, int] = {}
    for task in tasks:
        lvl = task.level
        counts[lvl] = counts.get(lvl, 0) + 1
    return counts
