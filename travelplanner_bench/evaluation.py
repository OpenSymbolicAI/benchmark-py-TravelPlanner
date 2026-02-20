"""Evaluation for the TravelPlanner benchmark.

Implements 8 commonsense constraint checks, 5 hard constraint checks,
and aggregate scoring following the original TravelPlanner paper.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from travelplanner_bench.models import TravelPlannerResult, TravelPlannerTask
from travelplanner_bench.tools import ReferenceDatabase

log = logging.getLogger(__name__)


# ===========================================================================
# Helper utilities
# ===========================================================================


def _extract_entity_name(field_value: str) -> str | None:
    """Extract entity name from a plan field, stripping city suffix and cost."""
    if not field_value or field_value.strip() == "-":
        return None
    # Remove cost suffixes like ", Cost: 120"
    name = re.sub(r",\s*[Cc]ost:?\s*\$?\d+", "", field_value).strip()
    # Remove trailing city: "Restaurant Name, Chicago" -> "Restaurant Name"
    # But be careful: some names have commas (e.g., "Jim's Steaks, South St.")
    # Only strip if the last segment is a known city pattern
    return name


def _extract_restaurant_name(field_value: str) -> str | None:
    """Extract restaurant name, stripping city and cost info."""
    if not field_value or field_value.strip() == "-":
        return None
    name = field_value.strip()
    # Remove cost info
    name = re.sub(r",\s*[Cc]ost:?\s*\$?\d+", "", name).strip()
    return name if name and name != "-" else None


def _extract_attraction_names(field_value: str) -> list[str]:
    """Extract attraction names from semicolon-separated list."""
    if not field_value or field_value.strip() == "-":
        return []
    names = []
    for part in field_value.split(";"):
        name = part.strip()
        if name and name != "-":
            # Remove trailing cost info
            name = re.sub(r",\s*[Cc]ost:?\s*\$?\d+", "", name).strip()
            if name:
                names.append(name)
    return names


def _extract_accommodation_name(field_value: str) -> str | None:
    """Extract accommodation name."""
    if not field_value or field_value.strip() == "-":
        return None
    name = field_value.strip()
    name = re.sub(r",\s*[Cc]ost:?\s*\$?\d+", "", name).strip()
    return name if name and name != "-" else None


def _get_current_city(day: dict[str, Any]) -> str | None:
    """Extract the current city from a day entry.

    Handles both "CityName" and "from X to Y" formats.
    For "from X to Y", returns Y (the destination city for that day).
    """
    val = day.get("current_city", "")
    if not val or val.strip() == "-":
        return None
    val = val.strip()
    match = re.match(r"from\s+(.+?)\s+to\s+(.+)", val, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return val


def _normalize_name(name: str) -> str:
    """Normalize a name for comparison."""
    return name.lower().strip()


def _name_in_set(name: str, name_set: set[str]) -> bool:
    """Check if a name matches any entry in a set (case-insensitive)."""
    n = _normalize_name(name)
    for entry in name_set:
        if _normalize_name(entry) == n:
            return True
    return False


def _parse_cost(val: str | Any) -> float:
    """Parse a cost value, handling various formats."""
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return 0.0
    val = val.strip().replace("$", "").replace(",", "")
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ===========================================================================
# Commonsense Constraints (8 checks)
# ===========================================================================


def check_within_sandbox(
    plan: list[dict[str, Any]],
    db: ReferenceDatabase,
    task: TravelPlannerTask,
) -> bool:
    """Check that all entities in the plan exist in the reference database."""
    for day in plan:
        # Check transportation (flights)
        transport = day.get("transportation", "-")
        if transport and transport.strip() != "-":
            flight_match = re.search(r"Flight\s+Number:\s*([A-Za-z0-9]+)", transport)
            if flight_match:
                fn = flight_match.group(1).strip()
                if fn and not _name_in_set(fn, db.all_flight_numbers):
                    log.debug("Sandbox fail: flight %s not found", fn)
                    return False

        # Check restaurants (breakfast, lunch, dinner)
        for meal_key in ("breakfast", "lunch", "dinner"):
            name = _extract_restaurant_name(day.get(meal_key, "-"))
            if name and not _name_in_set(name, db.all_restaurant_names):
                # Try stripping city suffix: "Name, City" -> "Name"
                parts = name.rsplit(",", 1)
                base_name = parts[0].strip() if len(parts) > 1 else name
                if not _name_in_set(base_name, db.all_restaurant_names):
                    log.debug("Sandbox fail: restaurant %r not found", name)
                    return False

        # Check accommodations
        acc_name = _extract_accommodation_name(day.get("accommodation", "-"))
        if acc_name and not _name_in_set(acc_name, db.all_accommodation_names):
            parts = acc_name.rsplit(",", 1)
            base_name = parts[0].strip() if len(parts) > 1 else acc_name
            if not _name_in_set(base_name, db.all_accommodation_names):
                log.debug("Sandbox fail: accommodation %r not found", acc_name)
                return False

        # Check attractions
        for attr_name in _extract_attraction_names(day.get("attraction", "-")):
            if not _name_in_set(attr_name, db.all_attraction_names):
                parts = attr_name.rsplit(",", 1)
                base_name = parts[0].strip() if len(parts) > 1 else attr_name
                if not _name_in_set(base_name, db.all_attraction_names):
                    log.debug("Sandbox fail: attraction %r not found", attr_name)
                    return False

    return True


def check_complete_info(
    plan: list[dict[str, Any]],
    task: TravelPlannerTask,
) -> bool:
    """Check that plan has complete information.

    Each day should have reasonable coverage of required fields.
    """
    if len(plan) != task.days:
        return False

    for day in plan:
        fields = ["transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"]
        filled = sum(
            1 for f in fields
            if day.get(f, "-").strip() not in ("", "-")
        )
        # Allow first/last day to have fewer (travel days)
        day_num = day.get("days", 0)
        if day_num == task.days:
            # Last day: may only have transportation (return flight)
            if filled < 1:
                return False
        elif day_num == 1:
            # First day: arrival, may skip breakfast
            if filled < 2:
                return False
        else:
            if filled < 3:
                return False

    return True


def check_within_current_city(
    plan: list[dict[str, Any]],
    db: ReferenceDatabase,
    task: TravelPlannerTask,
) -> bool:
    """Check that daily activities are in the designated city."""
    for day in plan:
        city = _get_current_city(day)
        if not city:
            continue

        city_lower = _normalize_name(city)

        # Check restaurants
        for meal_key in ("breakfast", "lunch", "dinner"):
            name = _extract_restaurant_name(day.get(meal_key, "-"))
            if not name:
                continue
            # Strip city suffix for lookup
            base_name = name.rsplit(",", 1)[0].strip() if "," in name else name
            db_city = db.restaurant_city.get(_normalize_name(base_name), "")
            if db_city and _normalize_name(db_city) != city_lower:
                log.debug(
                    "City fail: restaurant %r in %r, expected %r",
                    base_name, db_city, city,
                )
                return False

        # Check attractions
        for attr_name in _extract_attraction_names(day.get("attraction", "-")):
            base_name = attr_name.rsplit(",", 1)[0].strip() if "," in attr_name else attr_name
            db_city = db.attraction_city.get(_normalize_name(base_name), "")
            if db_city and _normalize_name(db_city) != city_lower:
                log.debug(
                    "City fail: attraction %r in %r, expected %r",
                    base_name, db_city, city,
                )
                return False

        # Check accommodation
        acc_name = _extract_accommodation_name(day.get("accommodation", "-"))
        if acc_name:
            base_name = acc_name.rsplit(",", 1)[0].strip() if "," in acc_name else acc_name
            db_city = db.accommodation_city.get(_normalize_name(base_name), "")
            if db_city and _normalize_name(db_city) != city_lower:
                log.debug(
                    "City fail: accommodation %r in %r, expected %r",
                    base_name, db_city, city,
                )
                return False

    return True


def check_reasonable_city_route(
    plan: list[dict[str, Any]],
    task: TravelPlannerTask,
) -> bool:
    """Check that the city visiting sequence is reasonable."""
    if not plan:
        return False

    # Extract city sequence
    cities_visited: list[str] = []
    for day in plan:
        cc = day.get("current_city", "")
        if not cc:
            return False
        match = re.match(r"from\s+(.+?)\s+to\s+(.+)", cc, re.IGNORECASE)
        if match:
            if not cities_visited:
                cities_visited.append(match.group(1).strip())
            cities_visited.append(match.group(2).strip())
        else:
            cities_visited.append(cc.strip())

    if not cities_visited:
        return False

    # Must start from origin
    if _normalize_name(cities_visited[0]) != _normalize_name(task.org):
        log.debug("Route fail: starts from %r, expected %r", cities_visited[0], task.org)
        return False

    # Last day must return to origin
    last_city = day.get("current_city", "")
    match = re.match(r"from\s+(.+?)\s+to\s+(.+)", last_city, re.IGNORECASE)
    if match:
        if _normalize_name(match.group(2).strip()) != _normalize_name(task.org):
            log.debug("Route fail: doesn't return to origin")
            return False
    else:
        # If last day is just staying in a city, that's only OK for single-city trips
        # where the city IS the origin (i.e., they drove/taxied back same day)
        pass

    return True


def check_diverse_restaurants(
    plan: list[dict[str, Any]],
) -> bool:
    """Check that no restaurant appears more than once across all days."""
    seen: set[str] = set()
    for day in plan:
        for meal_key in ("breakfast", "lunch", "dinner"):
            name = _extract_restaurant_name(day.get(meal_key, "-"))
            if not name:
                continue
            base_name = name.rsplit(",", 1)[0].strip() if "," in name else name
            key = _normalize_name(base_name)
            if key in seen:
                log.debug("Diversity fail: duplicate restaurant %r", base_name)
                return False
            seen.add(key)
    return True


def check_diverse_attractions(
    plan: list[dict[str, Any]],
) -> bool:
    """Check that no attraction appears more than once across all days."""
    seen: set[str] = set()
    for day in plan:
        for attr_name in _extract_attraction_names(day.get("attraction", "-")):
            base_name = attr_name.rsplit(",", 1)[0].strip() if "," in attr_name else attr_name
            key = _normalize_name(base_name)
            if key in seen:
                log.debug("Diversity fail: duplicate attraction %r", base_name)
                return False
            seen.add(key)
    return True


def check_non_conflicting_transport(
    plan: list[dict[str, Any]],
) -> bool:
    """Check for conflicting transportation modes.

    If self-driving is used for inter-city travel, flights should not
    also be used (and vice versa).
    """
    has_flight = False
    has_self_driving = False

    for day in plan:
        transport = day.get("transportation", "-")
        if not transport or transport.strip() == "-":
            continue
        t_lower = transport.lower()
        if "flight" in t_lower:
            has_flight = True
        if "self-driving" in t_lower or "self driving" in t_lower:
            has_self_driving = True

    if has_flight and has_self_driving:
        log.debug("Transport conflict: both flight and self-driving used")
        return False
    return True


def check_valid_accommodation(
    plan: list[dict[str, Any]],
    db: ReferenceDatabase,
) -> bool:
    """Check that accommodation minimum nights requirements are met."""
    # Count consecutive nights at each accommodation
    acc_nights: dict[str, int] = {}
    for day in plan:
        acc_name = _extract_accommodation_name(day.get("accommodation", "-"))
        if acc_name:
            base_name = acc_name.rsplit(",", 1)[0].strip() if "," in acc_name else acc_name
            key = _normalize_name(base_name)
            acc_nights[key] = acc_nights.get(key, 0) + 1

    # Check against minimum_nights in database
    for acc_key, nights in acc_nights.items():
        # Find the accommodation in the database
        for city_accs in db.accommodations.values():
            for acc in city_accs:
                name = acc.get("NAME", acc.get("Name", "")).strip()
                if _normalize_name(name) == acc_key:
                    min_nights_str = acc.get("minimum nights", acc.get("minimum_nights", "1"))
                    try:
                        min_nights = int(min_nights_str)
                    except (ValueError, TypeError):
                        min_nights = 1
                    if nights < min_nights:
                        log.debug(
                            "Accommodation fail: %r stayed %d nights, minimum %d",
                            name, nights, min_nights,
                        )
                        return False
    return True


# ===========================================================================
# Hard Constraints (5 checks)
# ===========================================================================


def check_budget(
    plan: list[dict[str, Any]],
    db: ReferenceDatabase,
    task: TravelPlannerTask,
) -> bool | None:
    """Check that total trip cost is within budget."""
    if not task.budget:
        return None

    total_cost = 0.0
    people = task.people_number or 1

    for day in plan:
        # Flight costs
        transport = day.get("transportation", "-")
        if transport and transport.strip() != "-":
            flight_match = re.search(r"Flight\s+Number:\s*([A-Za-z0-9]+)", transport)
            if flight_match:
                fn = flight_match.group(1).strip()
                flight_cost = _find_flight_cost(db, fn)
                total_cost += flight_cost * people

            # Self-driving/taxi cost from transport description
            cost_match = re.search(r"[Cc]ost:?\s*\$?([\d,.]+)", transport)
            if cost_match and not flight_match:
                total_cost += _parse_cost(cost_match.group(1))

        # Meal costs
        for meal_key in ("breakfast", "lunch", "dinner"):
            name = _extract_restaurant_name(day.get(meal_key, "-"))
            if name:
                base_name = name.rsplit(",", 1)[0].strip() if "," in name else name
                meal_cost = _find_restaurant_cost(db, base_name)
                total_cost += meal_cost * people

        # Accommodation costs
        acc_name = _extract_accommodation_name(day.get("accommodation", "-"))
        if acc_name:
            base_name = acc_name.rsplit(",", 1)[0].strip() if "," in acc_name else acc_name
            acc_cost = _find_accommodation_cost(db, base_name)
            total_cost += acc_cost

    return total_cost <= task.budget


def _find_flight_cost(db: ReferenceDatabase, flight_number: str) -> float:
    """Find cost of a flight by flight number."""
    fn_lower = _normalize_name(flight_number)
    for flights in db.flights.values():
        for flight in flights:
            if _normalize_name(flight.get("Flight Number", "")) == fn_lower:
                return _parse_cost(flight.get("Price", "0"))
    return 0.0


def _find_restaurant_cost(db: ReferenceDatabase, name: str) -> float:
    """Find average cost of a restaurant by name."""
    n = _normalize_name(name)
    for restaurants in db.restaurants.values():
        for rest in restaurants:
            if _normalize_name(rest.get("Name", "")) == n:
                return _parse_cost(rest.get("Average Cost", "0"))
    return 0.0


def _find_accommodation_cost(db: ReferenceDatabase, name: str) -> float:
    """Find per-night cost of an accommodation by name."""
    n = _normalize_name(name)
    for accs in db.accommodations.values():
        for acc in accs:
            acc_name = acc.get("NAME", acc.get("Name", "")).strip()
            if _normalize_name(acc_name) == n:
                return _parse_cost(acc.get("price", "0"))
    return 0.0


def check_room_rule(
    plan: list[dict[str, Any]],
    db: ReferenceDatabase,
    task: TravelPlannerTask,
) -> bool | None:
    """Check accommodation house rules compliance."""
    rule = task.local_constraint.get("room_rule")
    if not rule:
        return None

    rule_lower = rule.lower().strip()

    for day in plan:
        acc_name = _extract_accommodation_name(day.get("accommodation", "-"))
        if not acc_name:
            continue
        base_name = acc_name.rsplit(",", 1)[0].strip() if "," in acc_name else acc_name
        house_rules = _find_accommodation_field(db, base_name, "house_rules")
        if house_rules:
            rules_lower = house_rules.lower()
            # Check if the required rule is present
            if rule_lower not in rules_lower:
                log.debug(
                    "Room rule fail: %r rules=%r, need %r",
                    base_name, house_rules, rule,
                )
                return False
    return True


def check_room_type(
    plan: list[dict[str, Any]],
    db: ReferenceDatabase,
    task: TravelPlannerTask,
) -> bool | None:
    """Check that accommodation room type matches requirement."""
    required_type = task.local_constraint.get("room_type")
    if not required_type:
        return None

    required_lower = required_type.lower().strip()
    # Handle "not shared room" as a special case
    is_negation = required_lower.startswith("not ")

    for day in plan:
        acc_name = _extract_accommodation_name(day.get("accommodation", "-"))
        if not acc_name:
            continue
        base_name = acc_name.rsplit(",", 1)[0].strip() if "," in acc_name else acc_name
        actual_type = _find_accommodation_field(db, base_name, "room type")
        if not actual_type:
            continue
        actual_lower = actual_type.lower().strip()

        if is_negation:
            forbidden = required_lower[4:].strip()  # strip "not "
            if forbidden in actual_lower:
                log.debug(
                    "Room type fail: %r is %r, forbidden %r",
                    base_name, actual_type, forbidden,
                )
                return False
        else:
            if required_lower not in actual_lower:
                log.debug(
                    "Room type fail: %r is %r, need %r",
                    base_name, actual_type, required_type,
                )
                return False
    return True


def _find_accommodation_field(
    db: ReferenceDatabase, name: str, field: str
) -> str | None:
    """Find a specific field value for an accommodation by name."""
    n = _normalize_name(name)
    for accs in db.accommodations.values():
        for acc in accs:
            acc_name = acc.get("NAME", acc.get("Name", "")).strip()
            if _normalize_name(acc_name) == n:
                return acc.get(field, acc.get(field.replace(" ", "_"), ""))
    return None


def check_cuisine(
    plan: list[dict[str, Any]],
    db: ReferenceDatabase,
    task: TravelPlannerTask,
) -> bool | None:
    """Check that all required cuisines are represented in meals."""
    required_cuisines = task.local_constraint.get("cuisine")
    if not required_cuisines:
        return None

    if isinstance(required_cuisines, str):
        required_cuisines = [c.strip() for c in required_cuisines.split(",")]

    required_set = {c.lower().strip() for c in required_cuisines}

    # Collect all cuisines from restaurants in the plan
    found_cuisines: set[str] = set()
    for day in plan:
        for meal_key in ("breakfast", "lunch", "dinner"):
            name = _extract_restaurant_name(day.get(meal_key, "-"))
            if not name:
                continue
            base_name = name.rsplit(",", 1)[0].strip() if "," in name else name
            cuisines = _find_restaurant_cuisines(db, base_name)
            for c in cuisines:
                found_cuisines.add(c.lower().strip())

    # Check all required cuisines are found
    for req in required_set:
        if not any(req in fc for fc in found_cuisines):
            log.debug("Cuisine fail: %r not found in meals", req)
            return False
    return True


def _find_restaurant_cuisines(db: ReferenceDatabase, name: str) -> list[str]:
    """Find cuisines offered by a restaurant."""
    n = _normalize_name(name)
    for restaurants in db.restaurants.values():
        for rest in restaurants:
            if _normalize_name(rest.get("Name", "")) == n:
                cuisines_str = rest.get("Cuisines", "")
                return [c.strip() for c in cuisines_str.split(",") if c.strip()]
    return []


def check_transportation_constraint(
    plan: list[dict[str, Any]],
    task: TravelPlannerTask,
) -> bool | None:
    """Check hard transportation constraint (e.g., 'no flight', 'no self-driving')."""
    transport_constraint = task.local_constraint.get("transportation")
    if not transport_constraint:
        return None

    constraint_lower = transport_constraint.lower().strip()

    for day in plan:
        transport = day.get("transportation", "-")
        if not transport or transport.strip() == "-":
            continue
        t_lower = transport.lower()

        if "no flight" in constraint_lower and "flight" in t_lower:
            log.debug("Transport constraint fail: flight used but forbidden")
            return False
        if "no self-driving" in constraint_lower and (
            "self-driving" in t_lower or "self driving" in t_lower
        ):
            log.debug("Transport constraint fail: self-driving used but forbidden")
            return False

    return True


# ===========================================================================
# Aggregate Scoring
# ===========================================================================


def evaluate_plan(
    plan: list[dict[str, Any]] | None,
    task: TravelPlannerTask,
    db: ReferenceDatabase,
) -> TravelPlannerResult:
    """Run all constraint checks and compute aggregate scores."""
    result = TravelPlannerResult(
        task_id=task.task_id,
        query=task.query,
        level=task.level,
        days=task.days,
    )

    if plan is None or len(plan) == 0:
        result.plan_delivered = False
        return result

    result.plan_delivered = True
    result.plan = plan

    # Commonsense checks (8)
    result.within_sandbox = check_within_sandbox(plan, db, task)
    result.complete_info = check_complete_info(plan, task)
    result.within_current_city = check_within_current_city(plan, db, task)
    result.reasonable_city_route = check_reasonable_city_route(plan, task)
    result.diverse_restaurants = check_diverse_restaurants(plan)
    result.diverse_attractions = check_diverse_attractions(plan)
    result.non_conflicting_transport = check_non_conflicting_transport(plan)
    result.valid_accommodation = check_valid_accommodation(plan, db)

    commonsense_checks = [
        result.within_sandbox,
        result.complete_info,
        result.within_current_city,
        result.reasonable_city_route,
        result.diverse_restaurants,
        result.diverse_attractions,
        result.non_conflicting_transport,
        result.valid_accommodation,
    ]
    result.commonsense_micro = sum(commonsense_checks) / len(commonsense_checks)
    result.commonsense_macro = all(commonsense_checks)

    # Hard checks (5)
    result.budget_ok = check_budget(plan, db, task)
    result.room_rule_ok = check_room_rule(plan, db, task)
    result.room_type_ok = check_room_type(plan, db, task)
    result.cuisine_ok = check_cuisine(plan, db, task)
    result.transportation_ok = check_transportation_constraint(plan, task)

    hard_checks = [
        v
        for v in [
            result.budget_ok,
            result.room_rule_ok,
            result.room_type_ok,
            result.cuisine_ok,
            result.transportation_ok,
        ]
        if v is not None
    ]

    if hard_checks:
        result.hard_micro = sum(hard_checks) / len(hard_checks)
        result.hard_macro = all(hard_checks)
    else:
        result.hard_micro = 1.0
        result.hard_macro = True

    result.final_pass = result.commonsense_macro and result.hard_macro
    return result


def compute_aggregate_metrics(
    results: list[TravelPlannerResult],
) -> dict[str, Any]:
    """Compute benchmark-level aggregate metrics."""
    total = len(results)
    if total == 0:
        return {"total": 0}

    delivered = [r for r in results if r.plan_delivered]
    delivery_rate = len(delivered) / total

    # Commonsense metrics (over delivered plans)
    cs_micro_avg = (
        sum(r.commonsense_micro for r in delivered) / len(delivered)
        if delivered
        else 0.0
    )
    cs_macro_rate = (
        sum(1 for r in delivered if r.commonsense_macro) / len(delivered)
        if delivered
        else 0.0
    )

    # Hard metrics (over delivered plans)
    hard_micro_avg = (
        sum(r.hard_micro for r in delivered) / len(delivered)
        if delivered
        else 0.0
    )
    hard_macro_rate = (
        sum(1 for r in delivered if r.hard_macro) / len(delivered)
        if delivered
        else 0.0
    )

    # Final pass rate (over all tasks)
    final_pass_rate = sum(1 for r in results if r.final_pass) / total

    # Per-level breakdown
    per_level: dict[str, dict[str, Any]] = {}
    for r in results:
        lvl = r.level
        if lvl not in per_level:
            per_level[lvl] = {
                "total": 0,
                "delivered": 0,
                "commonsense_macro": 0,
                "hard_macro": 0,
                "final_pass": 0,
            }
        per_level[lvl]["total"] += 1
        if r.plan_delivered:
            per_level[lvl]["delivered"] += 1
        if r.commonsense_macro:
            per_level[lvl]["commonsense_macro"] += 1
        if r.hard_macro:
            per_level[lvl]["hard_macro"] += 1
        if r.final_pass:
            per_level[lvl]["final_pass"] += 1

    for lvl, stats in per_level.items():
        n = stats["total"]
        stats["delivery_rate"] = stats["delivered"] / n if n else 0.0
        stats["commonsense_macro_rate"] = stats["commonsense_macro"] / n if n else 0.0
        stats["hard_macro_rate"] = stats["hard_macro"] / n if n else 0.0
        stats["final_pass_rate"] = stats["final_pass"] / n if n else 0.0

    # Timing
    all_times = [r.wall_time_seconds for r in results]

    return {
        "total": total,
        "delivered": len(delivered),
        "delivery_rate": delivery_rate,
        "commonsense_micro_avg": cs_micro_avg,
        "commonsense_macro_rate": cs_macro_rate,
        "hard_micro_avg": hard_micro_avg,
        "hard_macro_rate": hard_macro_rate,
        "final_pass_rate": final_pass_rate,
        "per_level": per_level,
        "timing": {
            "total_seconds": sum(all_times),
            "avg_per_task": sum(all_times) / total if total else 0.0,
        },
        "errors": sum(1 for r in results if r.error),
    }
