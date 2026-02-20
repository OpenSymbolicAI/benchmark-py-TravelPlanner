"""Plan Assembler Agent: DesignExecute agent for deterministic plan assembly.

Takes gathered travel data + constraints and produces a valid day-by-day plan.
All primitives are pure deterministic functions - the LLM only plans the
sequence of operations, while filtering, optimization, and cost math are
handled entirely by code.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from opensymbolicai.blueprints.design_execute import DesignExecute
from opensymbolicai.core import decomposition, primitive
from opensymbolicai.llm import LLM, LLMConfig
from opensymbolicai.models import DesignExecuteConfig

from travelplanner_bench.models import GatheredData, TravelPlannerTask

log = logging.getLogger(__name__)


def _parse_cost(val: str | Any) -> float:
    """Parse a cost value from various formats."""
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return 0.0
    val = val.strip().replace("$", "").replace(",", "")
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _normalize_costs(d: dict[str, Any], cost_keys: list[str]) -> dict[str, Any]:
    """Return a shallow copy of dict with specified cost fields parsed to floats."""
    result = dict(d)
    for key in cost_keys:
        if key in result:
            result[key] = _parse_cost(result[key])
    return result


class PlanAssemblerAgent(DesignExecute):
    """Deterministic plan assembler for travel itinerary construction.

    Takes gathered data (flights, restaurants, accommodations, attractions)
    and task constraints (budget, room type, room rule, cuisine, transportation)
    and assembles a valid day-by-day plan using deterministic primitives.

    All cost fields are PRE-PARSED to floats — you can safely do arithmetic
    on dict values like flight["Price"] or acc["price"].

    PREFERRED WORKFLOW (use compound primitives to minimize calls):
    1. plan_transport() → selects + formats flights or driving in one call
    2. select_accommodation() → filters by room_type/room_rule + picks optimal
    3. prepare_meals() → selects restaurants + assigns to meal slots
    4. pick_diverse_attractions() → picks unique attractions
    5. total_trip_cost() + check_budget() → verify budget
    6. Loop over days calling build_day() with meals[day_idx]
    7. set_plan() → submit the assembled days

    RULES:
    - All entity names MUST come from the gathered data variables
    - Use compound primitives (plan_transport, select_accommodation,
      prepare_meals) instead of lower-level calls when possible
    - NEVER manually assign restaurants to meal slots — use prepare_meals()
    - Always call check_budget() before set_plan()
    - For multi-city trips, loop over cities and allocate days per city
    """

    def __init__(
        self,
        llm: LLMConfig | LLM,
        max_plan_retries: int = 3,
    ) -> None:
        super().__init__(
            llm=llm,
            name="PlanAssemblerAgent",
            description=(
                "Deterministic plan assembler for travel itinerary construction. "
                "All cost fields are PRE-PARSED to floats. Use compound primitives "
                "to minimize calls.\n\n"
                "WORKFLOW (compound primitives):\n"
                "1. plan_transport(outbound, return, distances, constraint, org, dest)\n"
                "2. select_accommodation(accs, nights, budget, room_type, room_rule)\n"
                "3. prepare_meals(restaurants, num_days, city, required_cuisines)\n"
                "4. pick_diverse_attractions(attractions, count)\n"
                "5. total_trip_cost() + check_budget()\n"
                "6. Loop: build_day() using meals[day_idx] for each day\n"
                "7. set_plan()\n\n"
                "RULES:\n"
                "- Use compound primitives (plan_transport, select_accommodation, "
                "prepare_meals) instead of lower-level calls\n"
                "- NEVER manually assign restaurants to meal slots\n"
                "- Always check_budget() before set_plan()"
            ),
            config=DesignExecuteConfig(
                max_plan_retries=max_plan_retries,
                max_loop_iterations=50,
                max_total_primitive_calls=200,
                multi_turn=True,
            ),
        )
        self._submitted_plan: list[dict[str, Any]] | None = None
        self._last_error: str | None = None

    # =========================================================================
    # FILTERING PRIMITIVES
    # =========================================================================

    @primitive(read_only=True)
    def filter_by_room_type(
        self, accommodations: list[dict], required_type: str
    ) -> list[dict]:
        """Filter accommodations by room type.

        Args:
            accommodations: List of accommodation dicts.
            required_type: Required room type (e.g., "entire room", "private room").
                           Prefix with "not " to exclude a type (e.g., "not shared room").

        Returns:
            Filtered list of accommodations matching the room type.
        """
        required_lower = required_type.lower().strip()
        is_negation = required_lower.startswith("not ")

        result = []
        for acc in accommodations:
            actual = acc.get("room type", acc.get("room_type", "")).lower().strip()
            if is_negation:
                forbidden = required_lower[4:].strip()
                if forbidden not in actual:
                    result.append(acc)
            else:
                if required_lower in actual:
                    result.append(acc)
        return result

    @primitive(read_only=True)
    def filter_by_room_rule(
        self, accommodations: list[dict], required_rule: str
    ) -> list[dict]:
        """Filter accommodations by house rule.

        Args:
            accommodations: List of accommodation dicts.
            required_rule: Required rule (e.g., "No smoking", "No pets").

        Returns:
            Filtered list of accommodations whose house_rules contain the required rule.
        """
        rule_lower = required_rule.lower().strip()
        result = []
        for acc in accommodations:
            rules = acc.get("house_rules", "").lower()
            if rule_lower in rules:
                result.append(acc)
        return result

    @primitive(read_only=True)
    def filter_by_min_nights(
        self, accommodations: list[dict], num_nights: int
    ) -> list[dict]:
        """Filter accommodations that allow stays of num_nights or fewer.

        Args:
            accommodations: List of accommodation dicts.
            num_nights: Number of nights you plan to stay.

        Returns:
            Accommodations where minimum_nights <= num_nights.
        """
        result = []
        for acc in accommodations:
            min_nights_str = acc.get("minimum nights", acc.get("minimum_nights", "1"))
            try:
                min_nights = int(min_nights_str)
            except (ValueError, TypeError):
                min_nights = 1
            if min_nights <= num_nights:
                result.append(acc)
        return result

    @primitive(read_only=True)
    def filter_by_cuisine(
        self, restaurants: list[dict], required_cuisines: list[str]
    ) -> list[dict]:
        """Filter restaurants that serve at least one of the required cuisines.

        Args:
            restaurants: List of restaurant dicts.
            required_cuisines: List of required cuisine types (e.g., ["Chinese", "Italian"]).

        Returns:
            Restaurants that serve at least one required cuisine.
        """
        required_lower = {c.lower().strip() for c in required_cuisines}
        result = []
        for rest in restaurants:
            cuisines_str = rest.get("Cuisines", "")
            cuisines = {c.strip().lower() for c in cuisines_str.split(",") if c.strip()}
            if cuisines & required_lower:
                result.append(rest)
        return result

    @primitive(read_only=True)
    def filter_valid_transport(
        self,
        flights: list[dict],
        distances: list[dict],
        constraint: str,
    ) -> dict:
        """Determine valid transport options based on constraint.

        Args:
            flights: List of available flight dicts.
            distances: List of available distance/driving dicts.
            constraint: Transportation constraint (e.g., "no flight", "no self-driving", or "").

        Returns:
            Dict with "flights" and "distances" keys containing valid options.
        """
        constraint_lower = constraint.lower().strip() if constraint else ""
        valid_flights = flights if "no flight" not in constraint_lower else []
        valid_distances = distances if "no self-driving" not in constraint_lower else []
        return {"flights": valid_flights, "distances": valid_distances}

    # =========================================================================
    # OPTIMIZATION PRIMITIVES
    # =========================================================================

    @primitive(read_only=True)
    def cheapest_flights(self, flights: list[dict], n: int = 1) -> list[dict]:
        """Return the n cheapest flights sorted by price.

        Args:
            flights: List of flight dicts.
            n: Number of cheapest flights to return.

        Returns:
            Up to n flights sorted by ascending price.
        """
        sorted_flights = sorted(
            flights, key=lambda f: _parse_cost(f.get("Price", "0"))
        )
        return sorted_flights[:n]

    @primitive(read_only=True)
    def cheapest_accommodations(
        self, accommodations: list[dict], n: int = 1
    ) -> list[dict]:
        """Return the n cheapest accommodations sorted by price.

        Args:
            accommodations: List of accommodation dicts.
            n: Number of cheapest to return.

        Returns:
            Up to n accommodations sorted by ascending price per night.
        """
        sorted_accs = sorted(
            accommodations, key=lambda a: _parse_cost(a.get("price", "0"))
        )
        return sorted_accs[:n]

    @primitive(read_only=True)
    def cheapest_restaurant_set(
        self,
        restaurants: list[dict],
        count: int,
        required_cuisines: list[str] | None = None,
    ) -> list[dict]:
        """Find the cheapest set of unique restaurants, optionally covering required cuisines.

        First ensures all required cuisines are covered (picking the cheapest
        restaurant per cuisine), then fills remaining slots with cheapest overall.

        Args:
            restaurants: List of restaurant dicts.
            count: Total number of unique restaurants needed.
            required_cuisines: Optional list of cuisines that must be covered.

        Returns:
            List of up to `count` unique restaurants, cheapest first,
            covering all required cuisines if possible.
        """
        selected: list[dict] = []
        selected_names: set[str] = set()

        # Phase 1: Cover required cuisines
        if required_cuisines:
            for cuisine in required_cuisines:
                cuisine_lower = cuisine.lower().strip()
                candidates = []
                for r in restaurants:
                    name = r.get("Name", "")
                    if name in selected_names:
                        continue
                    cuisines_str = r.get("Cuisines", "")
                    r_cuisines = {
                        c.strip().lower() for c in cuisines_str.split(",") if c.strip()
                    }
                    if cuisine_lower in r_cuisines:
                        candidates.append(r)
                if candidates:
                    cheapest = min(
                        candidates, key=lambda r: _parse_cost(r.get("Average Cost", "0"))
                    )
                    selected.append(cheapest)
                    selected_names.add(cheapest.get("Name", ""))

        # Phase 2: Fill remaining with cheapest unselected
        remaining = count - len(selected)
        if remaining > 0:
            sorted_rest = sorted(
                restaurants, key=lambda r: _parse_cost(r.get("Average Cost", "0"))
            )
            for r in sorted_rest:
                if len(selected) >= count:
                    break
                name = r.get("Name", "")
                if name not in selected_names:
                    selected.append(r)
                    selected_names.add(name)

        return selected

    @primitive(read_only=True)
    def optimal_accommodation(
        self,
        accommodations: list[dict],
        nights: int,
        budget_remaining: float,
    ) -> dict | None:
        """Find the cheapest accommodation that fits within remaining budget.

        Args:
            accommodations: List of accommodation dicts (should be pre-filtered).
            nights: Number of nights staying.
            budget_remaining: Remaining budget in dollars.

        Returns:
            Cheapest accommodation where (price * nights) <= budget_remaining,
            or None if no valid option.
        """
        sorted_accs = sorted(
            accommodations, key=lambda a: _parse_cost(a.get("price", "0"))
        )
        for acc in sorted_accs:
            total = _parse_cost(acc.get("price", "0")) * nights
            if total <= budget_remaining:
                return acc
        # If nothing fits budget, return cheapest anyway (better than nothing)
        return sorted_accs[0] if sorted_accs else None

    @primitive(read_only=True)
    def assign_meals(
        self,
        restaurants: list[dict],
        num_days: int,
        city: str,
    ) -> list[dict[str, str]]:
        """Deterministically assign unique restaurants to meal slots across all days.

        Day 1: no breakfast (arrival day), has lunch and dinner.
        Last day: no meals (departure day).
        Middle days: breakfast, lunch, and dinner.

        Each restaurant is used AT MOST ONCE across the entire trip. This
        guarantees the diverse restaurants constraint is satisfied.

        Args:
            restaurants: List of unique restaurant dicts (from cheapest_restaurant_set).
            num_days: Total number of days in the trip.
            city: City name for formatting.

        Returns:
            List of dicts (one per day), each with "breakfast", "lunch", "dinner" keys.
            Values are formatted "Restaurant Name, City" or "-".
        """
        meals: list[dict[str, str]] = []
        r_idx = 0

        for day in range(1, num_days + 1):
            day_meals = {"breakfast": "-", "lunch": "-", "dinner": "-"}

            if day == num_days:
                # Last day: departure, no meals
                pass
            elif day == 1:
                # First day: no breakfast (arrival)
                for slot in ["lunch", "dinner"]:
                    if r_idx < len(restaurants):
                        name = restaurants[r_idx].get("Name", "")
                        day_meals[slot] = f"{name}, {city}"
                        r_idx += 1
            else:
                # Middle days: all three meals
                for slot in ["breakfast", "lunch", "dinner"]:
                    if r_idx < len(restaurants):
                        name = restaurants[r_idx].get("Name", "")
                        day_meals[slot] = f"{name}, {city}"
                        r_idx += 1

            meals.append(day_meals)

        return meals

    @primitive(read_only=True)
    def pick_diverse_attractions(
        self, attractions: list[dict], count: int
    ) -> list[dict]:
        """Pick up to `count` unique attractions.

        Args:
            attractions: List of attraction dicts.
            count: Number of unique attractions to pick.

        Returns:
            Up to `count` unique attractions.
        """
        seen: set[str] = set()
        result: list[dict] = []
        for attr in attractions:
            name = attr.get("Name", "")
            if name not in seen:
                result.append(attr)
                seen.add(name)
            if len(result) >= count:
                break
        return result

    # =========================================================================
    # COMPOUND PRIMITIVES (reduce call count, enforce type safety)
    # =========================================================================

    @primitive(read_only=True)
    def select_accommodation(
        self,
        accommodations: list[dict],
        nights: int,
        budget: float,
        room_type: str | None = None,
        room_rule: str | None = None,
    ) -> dict | None:
        """Select the best accommodation after applying all filters.

        Combines room_type filtering, room_rule filtering, min_nights filtering,
        and budget-optimal selection into a single call.

        Args:
            accommodations: List of accommodation dicts for a city.
            nights: Number of nights to stay.
            budget: Total remaining budget (used to prefer affordable options).
            room_type: Required room type (e.g., "entire room") or None.
            room_rule: Required house rule (e.g., "No smoking") or None.

        Returns:
            Best accommodation dict, or None if no options available.
        """
        filtered = list(accommodations)
        if room_type:
            filtered = self.filter_by_room_type(filtered, room_type)
        if room_rule:
            filtered = self.filter_by_room_rule(filtered, room_rule)
        filtered = self.filter_by_min_nights(filtered, nights)
        return self.optimal_accommodation(filtered, nights, budget)

    @primitive(read_only=True)
    def prepare_meals(
        self,
        restaurants: list[dict],
        num_days: int,
        city: str,
        required_cuisines: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Select restaurants and assign them to meal slots in one call.

        Combines cheapest_restaurant_set + assign_meals into a single operation.
        Automatically calculates the number of meal slots needed.

        Day 1: lunch + dinner (arrival). Last day: no meals (departure).
        Middle days: breakfast + lunch + dinner.

        Args:
            restaurants: All available restaurant dicts for the city.
            num_days: Total trip days (or days in this city for multi-city).
            city: City name for formatting.
            required_cuisines: Cuisine types that must be covered, or None.

        Returns:
            List of dicts (one per day), each with "breakfast", "lunch", "dinner"
            keys. Values are "Restaurant Name, City" or "-".
        """
        # Calculate meal slots needed
        if num_days <= 1:
            slots = 0
        elif num_days == 2:
            slots = 2  # day 1: lunch + dinner
        else:
            slots = 2 + (num_days - 2) * 3  # day1=2, middle=3 each, last=0

        selected = self.cheapest_restaurant_set(
            restaurants, slots, required_cuisines=required_cuisines
        )
        return self.assign_meals(selected, num_days, city)

    @primitive(read_only=True)
    def plan_transport(
        self,
        outbound_flights: list[dict],
        return_flights: list[dict],
        distances: list[dict] | None,
        constraint: str,
        origin: str,
        destination: str,
    ) -> dict[str, Any]:
        """Select and format transport for the trip in one call.

        Handles flight vs self-driving selection based on constraint, picks
        cheapest options, and returns pre-formatted transport strings.

        Args:
            outbound_flights: Available outbound flight dicts.
            return_flights: Available return flight dicts.
            distances: Available driving/distance dicts, or None.
            constraint: Transport constraint ("no flight", "no self-driving", or "").
            origin: Origin city name.
            destination: Destination city name.

        Returns:
            Dict with keys:
              - "outbound_str": formatted transport string for day 1
              - "return_str": formatted transport string for last day
              - "outbound_flight": flight dict if flying (for cost calc), or None
              - "return_flight": flight dict if flying (for cost calc), or None
              - "driving_costs": list of driving costs as floats (empty if flying)
              - "mode": "flight" or "self-driving" or "taxi"
        """
        transport = self.filter_valid_transport(
            outbound_flights, distances or [], constraint
        )

        valid_flights = transport["flights"]
        valid_distances = transport["distances"]

        if valid_flights:
            out_list = self.cheapest_flights(valid_flights, n=1)
            # For return, filter return_flights by same constraint
            ret_transport = self.filter_valid_transport(
                return_flights, [], constraint
            )
            ret_list = self.cheapest_flights(ret_transport["flights"], n=1)

            out_f = out_list[0] if out_list else None
            ret_f = ret_list[0] if ret_list else None

            return {
                "outbound_str": self.format_flight(out_f) if out_f else "-",
                "return_str": self.format_flight(ret_f) if ret_f else "-",
                "outbound_flight": out_f,
                "return_flight": ret_f,
                "driving_costs": [],
                "mode": "flight",
            }
        elif valid_distances:
            # Use the first distance entry (typically one per route)
            dist = valid_distances[0] if isinstance(valid_distances, list) else valid_distances
            drive_cost = _parse_cost(dist.get("cost", 0))
            return {
                "outbound_str": self.format_driving(dist, origin, destination),
                "return_str": self.format_driving(dist, destination, origin),
                "outbound_flight": None,
                "return_flight": None,
                "driving_costs": [drive_cost, drive_cost],  # out + return
                "mode": dist.get("mode", "self-driving"),
            }
        else:
            return {
                "outbound_str": "-",
                "return_str": "-",
                "outbound_flight": None,
                "return_flight": None,
                "driving_costs": [],
                "mode": "unknown",
            }

    # =========================================================================
    # COST CALCULATION PRIMITIVES
    # =========================================================================

    @primitive(read_only=True)
    def get_cost(self, entity: dict) -> float:
        """Get the numeric cost from any travel entity dict.

        Automatically detects the cost field based on entity type:
        - Flights: "Price"
        - Restaurants: "Average Cost"
        - Accommodations: "price"
        - Distances: "cost"

        Args:
            entity: Any travel data dict (flight, restaurant, accommodation, distance).

        Returns:
            Cost as a float. Returns 0.0 if no cost field found.
        """
        for key in ("Price", "Average Cost", "price", "cost"):
            if key in entity:
                return _parse_cost(entity[key])
        return 0.0

    @primitive(read_only=True)
    def flight_cost(self, flight: dict) -> float:
        """Get the per-person cost of a flight.

        Args:
            flight: A single flight dict.

        Returns:
            Price as a float.
        """
        return _parse_cost(flight.get("Price", "0"))

    @primitive(read_only=True)
    def accommodation_cost(self, accommodation: dict, nights: int) -> float:
        """Get the total cost of an accommodation for given nights.

        Args:
            accommodation: A single accommodation dict.
            nights: Number of nights staying.

        Returns:
            Total cost (price per night * nights).
        """
        return _parse_cost(accommodation.get("price", "0")) * nights

    @primitive(read_only=True)
    def restaurant_cost(self, restaurant: dict) -> float:
        """Get the per-person average cost of a restaurant.

        Args:
            restaurant: A single restaurant dict.

        Returns:
            Average cost as a float.
        """
        return _parse_cost(restaurant.get("Average Cost", "0"))

    @primitive(read_only=True)
    def total_trip_cost(
        self,
        flights: list[dict],
        accommodations: list[dict],
        accommodation_nights: list[int],
        restaurants: list[dict],
        people: int,
        driving_costs: list[float] | None = None,
    ) -> float:
        """Calculate total trip cost across all components.

        Args:
            flights: List of flight dicts used in the trip.
            accommodations: List of accommodation dicts used (one per city).
            accommodation_nights: Number of nights at each accommodation.
            restaurants: List of restaurant dicts used for meals.
            people: Number of travelers.
            driving_costs: Optional list of driving/taxi costs.

        Returns:
            Total trip cost.
        """
        total = 0.0

        # Flights: per person
        for f in flights:
            total += _parse_cost(f.get("Price", "0")) * people

        # Accommodations: per night (not per person)
        for acc, nights in zip(accommodations, accommodation_nights):
            total += _parse_cost(acc.get("price", "0")) * nights

        # Restaurants: per person
        for r in restaurants:
            total += _parse_cost(r.get("Average Cost", "0")) * people

        # Driving costs
        if driving_costs:
            for cost in driving_costs:
                total += cost

        return total

    @primitive(read_only=True)
    def check_budget(self, total_cost: float, budget: float) -> bool:
        """Check if total cost is within budget.

        Args:
            total_cost: Calculated total trip cost.
            budget: Maximum allowed budget.

        Returns:
            True if within budget, False otherwise.
        """
        return total_cost <= budget

    @primitive(read_only=True)
    def remaining_budget(self, budget: float, spent_so_far: float) -> float:
        """Calculate remaining budget.

        Args:
            budget: Total budget.
            spent_so_far: Amount already spent.

        Returns:
            Remaining budget (may be negative if over budget).
        """
        return budget - spent_so_far

    # =========================================================================
    # PLAN ASSEMBLY PRIMITIVES
    # =========================================================================

    @primitive(read_only=True)
    def format_flight(self, flight: dict) -> str:
        """Format a flight dict into the standard transport string.

        Args:
            flight: A flight dict with Flight Number, OriginCityName, etc.

        Returns:
            Formatted string like "Flight Number: F123, from X to Y, Departure Time: ..., Arrival Time: ..."
        """
        fn = flight.get("Flight Number", "")
        origin = flight.get("OriginCityName", "")
        dest = flight.get("DestCityName", "")
        dep = flight.get("DepTime", "")
        arr = flight.get("ArrTime", "")
        return (
            f"Flight Number: {fn}, from {origin} to {dest}, "
            f"Departure Time: {dep}, Arrival Time: {arr}"
        )

    @primitive(read_only=True)
    def format_driving(self, distance_info: dict, origin: str, destination: str) -> str:
        """Format driving info into transport string.

        Args:
            distance_info: Dict with duration, distance, cost fields.
            origin: Origin city name.
            destination: Destination city name.

        Returns:
            Formatted string like "Self-driving, from X to Y, Duration: ..., Distance: ..., Cost: ..."
        """
        duration = distance_info.get("duration", "")
        distance = distance_info.get("distance", "")
        cost = distance_info.get("cost", "")
        mode = distance_info.get("mode", "self-driving")
        mode_label = "Self-driving" if "self" in mode.lower() else "Taxi"
        return (
            f"{mode_label}, from {origin} to {destination}, "
            f"Duration: {duration}, Distance: {distance}, Cost: {cost}"
        )

    @primitive(read_only=True)
    def format_restaurant(self, restaurant: dict, city: str) -> str:
        """Format a restaurant for plan output.

        Args:
            restaurant: Restaurant dict with Name field.
            city: City name.

        Returns:
            Formatted string like "Restaurant Name, City".
        """
        name = restaurant.get("Name", "")
        return f"{name}, {city}"

    @primitive(read_only=True)
    def format_attractions(self, attractions: list[dict], city: str) -> str:
        """Format attractions for plan output (semicolon-separated).

        Args:
            attractions: List of attraction dicts.
            city: City name.

        Returns:
            Formatted string like "Attraction1, City;Attraction2, City" or "-".
        """
        if not attractions:
            return "-"
        parts = [f"{a.get('Name', '')}, {city}" for a in attractions]
        return ";".join(parts)

    @primitive(read_only=True)
    def format_accommodation(self, accommodation: dict, city: str) -> str:
        """Format an accommodation for plan output.

        Args:
            accommodation: Accommodation dict with NAME field.
            city: City name.

        Returns:
            Formatted string like "Accommodation Name, City".
        """
        name = accommodation.get("NAME", accommodation.get("Name", ""))
        return f"{name}, {city}"

    @primitive(read_only=False)
    def build_day(
        self,
        day_num: int,
        current_city: str,
        transportation: str,
        breakfast: str,
        attraction: str,
        lunch: str,
        dinner: str,
        accommodation: str,
    ) -> dict:
        """Build a single day entry for the plan.

        Args:
            day_num: Day number (1, 2, 3, ...).
            current_city: "from X to Y" on travel days, or just city name.
            transportation: Formatted transport string or "-".
            breakfast: "Restaurant Name, City" or "-".
            attraction: "Attr1, City;Attr2, City" or "-".
            lunch: "Restaurant Name, City" or "-".
            dinner: "Restaurant Name, City" or "-".
            accommodation: "Accommodation Name, City" or "-".

        Returns:
            Dict with all day fields.
        """
        return {
            "days": day_num,
            "current_city": current_city,
            "transportation": transportation,
            "breakfast": breakfast,
            "attraction": attraction,
            "lunch": lunch,
            "dinner": dinner,
            "accommodation": accommodation,
        }

    @primitive(read_only=False)
    def set_plan(self, plan: list[dict]) -> str:
        """Submit the final travel plan.

        Args:
            plan: List of day entries (each from build_day()).

        Returns:
            Confirmation string.
        """
        self._submitted_plan = plan
        return f"Plan submitted with {len(plan)} days."

    # =========================================================================
    # DECOMPOSITION EXAMPLES
    # =========================================================================

    @decomposition(
        intent=(
            "Build a 3-day single-city trip plan from Sarasota to Chicago, "
            "budget $1900, 1 person, no special constraints"
        ),
        expanded_intent=(
            "Simple case: use compound primitives to reduce call count. "
            "plan_transport() handles flight selection + formatting. "
            "select_accommodation() handles filtering + budget optimization. "
            "prepare_meals() handles restaurant selection + meal assignment. "
            "Then loop over days building each day entry."
        ),
    )
    def _ex_simple_3day(self) -> str:
        # Transport: select + format in one call
        transport = self.plan_transport(
            outbound_flights, return_flights, distances, "", "Sarasota", "Chicago"
        )

        # Accommodation: filter + select in one call (2 nights for 3-day)
        best_acc = self.select_accommodation(accommodations, 2, 1900.0)

        # Meals: select restaurants + assign to slots in one call
        meals = self.prepare_meals(restaurants, 3, "Chicago")

        # Attractions
        day_attractions = self.pick_diverse_attractions(attractions, 4)

        # Budget check (costs are pre-parsed floats, safe to use directly)
        flights_used = [f for f in [transport["outbound_flight"], transport["return_flight"]] if f]
        cost = self.total_trip_cost(
            flights_used, [best_acc], [2], restaurants[:5], 1,
            driving_costs=transport["driving_costs"],
        )
        ok = self.check_budget(cost, 1900.0)

        # Build days
        acc_str = self.format_accommodation(best_acc, "Chicago")
        day1 = self.build_day(
            1, "from Sarasota to Chicago", transport["outbound_str"],
            meals[0]["breakfast"],
            self.format_attractions(day_attractions[:2], "Chicago"),
            meals[0]["lunch"], meals[0]["dinner"], acc_str,
        )
        day2 = self.build_day(
            2, "Chicago", "-",
            meals[1]["breakfast"],
            self.format_attractions(day_attractions[2:4], "Chicago"),
            meals[1]["lunch"], meals[1]["dinner"], acc_str,
        )
        day3 = self.build_day(
            3, "from Chicago to Sarasota", transport["return_str"],
            meals[2]["breakfast"], "-",
            meals[2]["lunch"], meals[2]["dinner"], "-",
        )

        result = self.set_plan([day1, day2, day3])
        return result

    @decomposition(
        intent=(
            "Build a 3-day trip with budget $1200, room type 'entire room', "
            "cuisine constraint ['Chinese', 'Italian'], 2 people"
        ),
        expanded_intent=(
            "Hard constraints: select_accommodation handles room_type filtering. "
            "prepare_meals handles cuisine coverage via required_cuisines. "
            "plan_transport handles flight/driving selection. "
            "All cost fields are pre-parsed floats — safe for arithmetic."
        ),
    )
    def _ex_constrained_3day(self) -> str:
        # Accommodation: filter by room type + select in one call
        best_acc = self.select_accommodation(
            accommodations, 2, 1200.0, room_type="entire room"
        )

        # Meals: select restaurants covering cuisines + assign in one call
        meals = self.prepare_meals(
            restaurants, 3, "CityName",
            required_cuisines=["Chinese", "Italian"],
        )

        # Transport
        transport = self.plan_transport(
            outbound_flights, return_flights, distances, "",
            "Origin", "CityName",
        )

        # Attractions
        attrs = self.pick_diverse_attractions(attractions, 4)

        # Budget check
        flights_used = [f for f in [transport["outbound_flight"], transport["return_flight"]] if f]
        cost = self.total_trip_cost(
            flights_used, [best_acc], [2],
            restaurants[:5], 2,
            driving_costs=transport["driving_costs"],
        )
        ok = self.check_budget(cost, 1200.0)

        # Build days
        acc_str = self.format_accommodation(best_acc, "CityName")
        day1 = self.build_day(
            1, "from Origin to CityName", transport["outbound_str"],
            meals[0]["breakfast"],
            self.format_attractions(attrs[:2], "CityName"),
            meals[0]["lunch"], meals[0]["dinner"], acc_str,
        )
        day2 = self.build_day(
            2, "CityName", "-",
            meals[1]["breakfast"],
            self.format_attractions(attrs[2:4], "CityName"),
            meals[1]["lunch"], meals[1]["dinner"], acc_str,
        )
        day3 = self.build_day(
            3, "from CityName to Origin", transport["return_str"],
            meals[2]["breakfast"], "-",
            meals[2]["lunch"], meals[2]["dinner"], "-",
        )
        result = self.set_plan([day1, day2, day3])
        return result

    @decomposition(
        intent=(
            "Build a 5-day multi-city trip plan visiting 2 cities: "
            "Orlando -> San Antonio (2 days) -> Houston (2 days) -> Orlando, "
            "budget $3100, 1 person, no special constraints"
        ),
        expanded_intent=(
            "Multi-city: handle 3 flight legs (outbound, intercity, return). "
            "Use per-city restaurant/accommodation/attraction variables. "
            "prepare_meals per city, select_accommodation per city. "
            "Day 1 = arrival at city1, Day N = return from city2. "
            "Every day must have meals (breakfast, lunch, dinner) and accommodation "
            "(except last day which may skip accommodation)."
        ),
    )
    def _ex_multi_city_5day(self) -> str:
        # --- Flight legs ---
        out_sorted = self.cheapest_flights(outbound_flights, n=1)
        out_f = out_sorted[0] if out_sorted else None
        out_str = self.format_flight(out_f) if out_f else "-"

        inter_sorted = self.cheapest_flights(intercity_flights, n=1)
        inter_f = inter_sorted[0] if inter_sorted else None
        inter_str = self.format_flight(inter_f) if inter_f else "-"

        ret_sorted = self.cheapest_flights(return_flights, n=1)
        ret_f = ret_sorted[0] if ret_sorted else None
        ret_str = self.format_flight(ret_f) if ret_f else "-"

        # --- City 1: San Antonio (2 nights) ---
        acc1 = self.select_accommodation(san_antonio_accommodations, 2, 3100.0)
        meals1 = self.prepare_meals(san_antonio_restaurants, 2, "San Antonio")
        attrs1 = self.pick_diverse_attractions(san_antonio_attractions, 4)
        acc1_str = self.format_accommodation(acc1, "San Antonio")

        # --- City 2: Houston (2 nights) ---
        acc2 = self.select_accommodation(houston_accommodations, 2, 3100.0)
        meals2 = self.prepare_meals(houston_restaurants, 3, "Houston")
        attrs2 = self.pick_diverse_attractions(houston_attractions, 4)
        acc2_str = self.format_accommodation(acc2, "Houston")

        # --- Build all 5 days ---
        day1 = self.build_day(
            1, "from Orlando to San Antonio", out_str,
            meals1[0]["breakfast"],
            self.format_attractions(attrs1[:2], "San Antonio"),
            meals1[0]["lunch"], meals1[0]["dinner"], acc1_str,
        )
        day2 = self.build_day(
            2, "San Antonio", "-",
            meals1[1]["breakfast"],
            self.format_attractions(attrs1[2:4], "San Antonio"),
            meals1[1]["lunch"], meals1[1]["dinner"], acc1_str,
        )
        day3 = self.build_day(
            3, "from San Antonio to Houston", inter_str,
            meals2[0]["breakfast"],
            self.format_attractions(attrs2[:2], "Houston"),
            meals2[0]["lunch"], meals2[0]["dinner"], acc2_str,
        )
        day4 = self.build_day(
            4, "Houston", "-",
            meals2[1]["breakfast"],
            self.format_attractions(attrs2[2:4], "Houston"),
            meals2[1]["lunch"], meals2[1]["dinner"], acc2_str,
        )
        day5 = self.build_day(
            5, "from Houston to Orlando", ret_str,
            meals2[2]["breakfast"], "-",
            meals2[2]["lunch"], meals2[2]["dinner"], "-",
        )

        result = self.set_plan([day1, day2, day3, day4, day5])
        return result

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    @property
    def last_error(self) -> str | None:
        """Error message from the most recent failed assembly attempt."""
        return self._last_error

    def assemble_plan(
        self,
        gathered: GatheredData,
        task: TravelPlannerTask,
        previous_error: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Assemble a valid day-by-day plan from gathered data.

        Args:
            gathered: All gathered travel data.
            task: The task with constraints.
            previous_error: Error from a previous attempt, included in the
                task description so the LLM can adapt its approach.

        Returns:
            List of day dicts forming the plan, or None on failure.
        """
        self._submitted_plan = None
        self._last_error = None

        # Build the task description with structured data
        task_str = self._build_task_string(gathered, task)

        if previous_error:
            task_str += (
                "\n\n⚠️ PREVIOUS ATTEMPT FAILED with error:\n"
                f"{previous_error}\n"
                "Please fix the issue and try a different approach."
            )

        # Inject gathered data into execution namespace
        self._gathered = gathered
        self._task = task

        result = self.run(task_str)

        if self._submitted_plan is not None:
            log.info("POST-PROC: calling _fill_missing_fields on %d-day plan", len(self._submitted_plan))
            self._fill_missing_fields(self._submitted_plan, gathered, task)
            return self._submitted_plan

        # Capture error for caller
        self._last_error = result.error or "set_plan() was never called"
        return None

    def _fill_missing_fields(
        self,
        plan: list[dict[str, Any]],
        gathered: GatheredData,
        task: TravelPlannerTask,
    ) -> None:
        """Post-process plan to fill in missing meals/transport/accommodation.

        1. Ensures transport mode consistency (no mixing flights & self-driving).
        2. Clears the last/return day (meals/attractions/accommodation → "-").
        3. Fixes transition-day meals that are in the origin city instead of
           the destination city.
        4. Fills missing meals from available restaurant data.
        5. Fills missing accommodation from available data.
        6. Deduplicates attractions across all days.
        """
        # --- Phase 1: Fill missing transport on transition days ---
        self._fill_missing_transport(plan, gathered)

        # --- Phase 1b: Fix transport mode conflicts ---
        self._fix_transport_conflicts(plan, gathered)

        # --- Phase 2: Clear last/return day ---
        self._clear_return_day(plan, task)

        # --- Phase 3: Fix meals referencing wrong city ---
        self._fix_wrong_city_meals(plan, gathered, task)

        # --- Phase 4: Fill missing meals and accommodation ---
        used_restaurants: set[str] = set()
        for day in plan:
            for meal_key in ("breakfast", "lunch", "dinner"):
                name = day.get(meal_key, "-").strip()
                if name and name != "-":
                    used_restaurants.add(name.split(",")[0].strip().lower())

        for day in plan:
            day_num = day.get("days", 0)
            city = self._infer_stay_city(day, task, gathered)
            if not city:
                continue

            city_restaurants = self._find_city_data(gathered.restaurants, city)
            city_accs = self._find_city_data(gathered.accommodations, city)

            # Fill missing meals
            for meal_key in ("breakfast", "lunch", "dinner"):
                val = day.get(meal_key, "-").strip()
                if not val or val == "-":
                    rest = self._pick_unused_restaurant(
                        city_restaurants, used_restaurants
                    )
                    if rest:
                        name = rest.get("Name", "").strip()
                        day[meal_key] = f"{name}, {city}"
                        used_restaurants.add(name.lower())

            # Fill missing accommodation (skip last day)
            if day_num != task.days:
                val = day.get("accommodation", "-").strip()
                if (not val or val == "-") and city_accs:
                    acc = min(
                        city_accs,
                        key=lambda a: _parse_cost(a.get("price", 9999)),
                    )
                    acc_name = acc.get("NAME", acc.get("Name", "")).strip()
                    day["accommodation"] = f"{acc_name}, {city}"

        # --- Phase 5: Deduplicate attractions ---
        self._deduplicate_attractions(plan, gathered)

    def _fix_transport_conflicts(
        self, plan: list[dict[str, Any]], gathered: GatheredData
    ) -> None:
        """Ensure all transport in the plan uses a single mode (flight or driving).

        If both modes are detected, convert all transport to self-driving
        (which is always available in the reference data).
        """
        has_flight = False
        has_driving = False
        for day in plan:
            trans = day.get("transportation", "-").strip()
            if not trans or trans == "-":
                continue
            t_lower = trans.lower()
            if "flight" in t_lower:
                has_flight = True
            if "self-driving" in t_lower or "self driving" in t_lower:
                has_driving = True

        if not (has_flight and has_driving):
            return  # No conflict

        # Conflict detected: convert all flights to self-driving
        for day in plan:
            trans = day.get("transportation", "-").strip()
            if not trans or trans == "-":
                continue
            if "flight" not in trans.lower():
                continue
            # Find a matching driving distance for this leg
            current_city = day.get("current_city", "")
            if "from" in current_city.lower() and "to" in current_city.lower():
                parts = current_city.split("to")
                if len(parts) >= 2:
                    origin_part = parts[0].replace("from", "").strip()
                    dest_part = parts[-1].strip()
                    # Find matching distance
                    for route_key, dist in gathered.distances.items():
                        rk_lower = route_key.lower()
                        if (
                            origin_part.lower() in rk_lower
                            and dest_part.lower() in rk_lower
                            and "self-driving" in rk_lower
                        ):
                            day["transportation"] = self.format_driving(
                                dist, origin_part, dest_part
                            )
                            break
                    else:
                        # No distance data; build a generic self-driving string
                        day["transportation"] = (
                            f"Self-driving, from {origin_part} to {dest_part}"
                        )

    def _fill_missing_transport(
        self, plan: list[dict[str, Any]], gathered: GatheredData
    ) -> None:
        """Fill missing transportation on transition days ('from X to Y')."""
        for day in plan:
            trans = day.get("transportation", "-").strip()
            if trans and trans != "-":
                continue
            current = day.get("current_city", "")
            match = re.match(r"from\s+(.+?)\s+to\s+(.+)", current, re.IGNORECASE)
            if not match:
                continue
            origin = match.group(1).strip()
            dest = match.group(2).strip()
            # Try flights first
            for fk, flights in gathered.flights.items():
                if origin.lower() in fk.lower() and dest.lower() in fk.lower() and flights:
                    day["transportation"] = self.format_flight(flights[0])
                    break
            else:
                # Try self-driving distance
                for dk, dist in gathered.distances.items():
                    dk_lower = dk.lower()
                    if (
                        origin.lower() in dk_lower
                        and dest.lower() in dk_lower
                        and "self-driving" in dk_lower
                    ):
                        day["transportation"] = self.format_driving(dist, origin, dest)
                        break
                else:
                    day["transportation"] = f"Self-driving, from {origin} to {dest}"

    @staticmethod
    def _clear_return_day(
        plan: list[dict[str, Any]], task: TravelPlannerTask
    ) -> None:
        """Clear meals/attractions/accommodation on the last (return) day.

        The last day of a multi-city trip is typically a travel-only day
        returning to the origin.  The evaluation expects all activities on
        a "from X to Y" day to be in city Y, but there is usually no
        restaurant/attraction data for the origin city.  Clearing these
        fields to "-" is the standard pattern for the return day.
        """
        if not plan:
            return
        last_day = plan[-1]
        current = last_day.get("current_city", "")
        # Only clear if this is a "from X to Origin" transition
        if "from" not in current.lower() or "to" not in current.lower():
            return
        match = re.match(r"from\s+(.+?)\s+to\s+(.+)", current, re.IGNORECASE)
        if not match:
            return
        dest = match.group(2).strip().lower()
        # Verify destination is the trip origin (return leg)
        log.info("RETURN_DAY: dest=%r, task.org=%r, match=%s", dest, task.org.lower().strip(), dest == task.org.lower().strip())
        if dest != task.org.lower().strip():
            return
        log.info("RETURN_DAY: clearing last day fields")
        for key in ("breakfast", "lunch", "dinner", "attraction", "accommodation"):
            last_day[key] = "-"

    def _fix_wrong_city_meals(
        self,
        plan: list[dict[str, Any]],
        gathered: GatheredData,
        task: TravelPlannerTask,
    ) -> None:
        """Fix meals that reference the wrong city.

        On any day, the evaluation expects all activities to be in the
        current city.  For "from X to Y" days that's city Y; for stay
        days it's the city itself.  Replace any meal whose city suffix
        doesn't match.
        """
        used_restaurants: set[str] = set()
        for day in plan:
            for mk in ("breakfast", "lunch", "dinner"):
                v = day.get(mk, "-").strip()
                if v and v != "-":
                    used_restaurants.add(v.split(",")[0].strip().lower())

        for day in plan:
            current = day.get("current_city", "")
            if not current or current.strip() == "-":
                continue
            # Determine the expected city
            match = re.match(r"from\s+(.+?)\s+to\s+(.+)", current, re.IGNORECASE)
            if match:
                expected_city_raw = match.group(2).strip()
            else:
                expected_city_raw = current.strip()
            expected_city = self._match_gathered_city(expected_city_raw, gathered)
            city_restaurants = self._find_city_data(gathered.restaurants, expected_city)
            if not city_restaurants:
                continue

            for meal_key in ("breakfast", "lunch", "dinner"):
                val = day.get(meal_key, "-").strip()
                if not val or val == "-":
                    continue
                # Check if the meal's city suffix matches the expected city
                if "," not in val:
                    continue
                meal_city = val.rsplit(",", 1)[1].strip()
                if meal_city.lower() == expected_city.lower():
                    continue
                # Wrong city — replace with a restaurant from the expected city
                rest = self._pick_unused_restaurant(
                    city_restaurants, used_restaurants
                )
                if rest:
                    name = rest.get("Name", "").strip()
                    day[meal_key] = f"{name}, {expected_city}"
                    used_restaurants.add(name.lower())

    def _deduplicate_attractions(
        self,
        plan: list[dict[str, Any]],
        gathered: GatheredData,
    ) -> None:
        """Ensure no attraction appears more than once across all days.

        If a duplicate is found, replace it with an unused attraction
        from the same city.
        """
        seen: set[str] = set()
        for day in plan:
            raw = day.get("attraction", "-").strip()
            if not raw or raw == "-":
                continue
            city = None
            new_parts: list[str] = []
            changed = False
            for attr in raw.split(";"):
                attr = attr.strip()
                if not attr or attr == "-":
                    continue
                # Parse "AttrName, City"
                if "," in attr:
                    base_name = attr.rsplit(",", 1)[0].strip()
                    attr_city = attr.rsplit(",", 1)[1].strip()
                    if not city:
                        city = attr_city
                else:
                    base_name = attr
                key = base_name.lower()
                if key not in seen:
                    seen.add(key)
                    new_parts.append(attr)
                else:
                    # Duplicate — replace with an unseen attraction
                    replacement = self._pick_unseen_attraction(
                        gathered, city or "", seen
                    )
                    if replacement:
                        seen.add(replacement.lower())
                        suffix = f", {city}" if city else ""
                        new_parts.append(f"{replacement}{suffix}")
                        changed = True
                    # else: just drop the duplicate
            if new_parts:
                day["attraction"] = ";".join(new_parts)
            elif raw != "-":
                day["attraction"] = "-"

    @staticmethod
    def _pick_unseen_attraction(
        gathered: GatheredData, city: str, seen: set[str]
    ) -> str | None:
        """Pick an attraction from the given city that hasn't been used."""
        city_lower = city.lower()
        for city_key, attractions in gathered.attractions.items():
            if city_key.lower() != city_lower:
                continue
            for attr in attractions:
                name = attr.get("Name", "").strip()
                if name and name.lower() not in seen:
                    return name
        return None

    @staticmethod
    def _infer_stay_city(
        day: dict[str, Any],
        task: TravelPlannerTask,
        gathered: GatheredData,
    ) -> str | None:
        """Infer the main city for a day entry.

        Uses fuzzy matching against gathered data keys to resolve city names
        that may differ in casing or punctuation (e.g., Devil's Lake vs Devils Lake).
        """
        current = day.get("current_city", "")
        if not current or current == "-":
            return None
        # "from X to Y" → destination city Y
        match = re.match(r"from\s+(.+?)\s+to\s+(.+)", current, re.IGNORECASE)
        if match:
            raw_city = match.group(2).strip()
            return PlanAssemblerAgent._match_gathered_city(raw_city, gathered)
        return PlanAssemblerAgent._match_gathered_city(current.strip(), gathered)

    @staticmethod
    def _match_gathered_city(city: str, gathered: GatheredData) -> str:
        """Match a city name against gathered data keys, handling variations."""
        city_lower = city.lower()
        # Check all data sources for a matching city key
        for data_dict in (gathered.restaurants, gathered.accommodations, gathered.attractions):
            for key in data_dict:
                if key.lower() == city_lower:
                    return key  # Return the canonical key form
                # Fuzzy: strip punctuation for comparison
                key_clean = key.lower().replace("'", "").replace("'", "")
                city_clean = city_lower.replace("'", "").replace("'", "")
                if key_clean == city_clean:
                    return key
                # Partial match
                if city_clean in key_clean or key_clean in city_clean:
                    return key
        return city  # Fallback to original

    @staticmethod
    def _find_city_data(data_dict: dict[str, list], city: str) -> list:
        """Find data for a city with fuzzy matching."""
        if city in data_dict:
            return data_dict[city]
        city_lower = city.lower()
        city_clean = city_lower.replace("'", "").replace("'", "")
        for k, v in data_dict.items():
            k_lower = k.lower()
            if k_lower == city_lower:
                return v
            k_clean = k_lower.replace("'", "").replace("'", "")
            if k_clean == city_clean:
                return v
            if city_clean in k_clean or k_clean in city_clean:
                return v
        return []

    @staticmethod
    def _pick_unused_restaurant(
        restaurants: list[dict], used: set[str]
    ) -> dict | None:
        """Pick a restaurant not yet used in the plan."""
        for r in restaurants:
            name = r.get("Name", "").strip()
            if name and name.lower() not in used:
                return r
        return restaurants[0] if restaurants else None

    def _build_task_string(
        self, gathered: GatheredData, task: TravelPlannerTask
    ) -> str:
        """Build structured task description for the LLM."""
        parts = [
            f"Build a {task.days}-day travel plan from {task.org} to {task.dest}.",
            f"People: {task.people_number}",
            f"Dates: {task.date}",
        ]

        if task.budget:
            parts.append(f"Budget: ${task.budget}")

        # Constraints
        constraints = []
        if task.local_constraint.get("room_type"):
            constraints.append(f"Room type: {task.local_constraint['room_type']}")
        if task.local_constraint.get("room_rule"):
            constraints.append(f"Room rule: {task.local_constraint['room_rule']}")
        if task.local_constraint.get("cuisine"):
            constraints.append(f"Cuisine: {task.local_constraint['cuisine']}")
        if task.local_constraint.get("transportation"):
            constraints.append(
                f"Transportation: {task.local_constraint['transportation']}"
            )
        if constraints:
            parts.append("Constraints: " + ", ".join(constraints))

        parts.append("")
        parts.append("AVAILABLE DATA (use these variable names in your plan):")
        parts.append("")

        # Flights
        for route_key, flights in gathered.flights.items():
            var_name = _safe_var_name(f"flights_{route_key}")
            prices = [_parse_cost(f.get("Price", "0")) for f in flights]
            parts.append(
                f"  {var_name} = <{len(flights)} flights, "
                f"prices ${min(prices):.0f}-${max(prices):.0f}>"
            )

        # Restaurants per city
        for city, restaurants in gathered.restaurants.items():
            var_name = _safe_var_name(f"restaurants_{city}")
            costs = [_parse_cost(r.get("Average Cost", "0")) for r in restaurants]
            cuisines_set: set[str] = set()
            for r in restaurants:
                for c in r.get("Cuisines", "").split(","):
                    c = c.strip()
                    if c:
                        cuisines_set.add(c)
            parts.append(
                f"  {var_name} = <{len(restaurants)} restaurants, "
                f"costs ${min(costs):.0f}-${max(costs):.0f}, "
                f"cuisines: {', '.join(sorted(cuisines_set)[:10])}>"
            )

        # Accommodations per city
        for city, accs in gathered.accommodations.items():
            var_name = _safe_var_name(f"accommodations_{city}")
            prices = [_parse_cost(a.get("price", "0")) for a in accs]
            parts.append(
                f"  {var_name} = <{len(accs)} accommodations, "
                f"prices ${min(prices):.0f}-${max(prices):.0f}/night>"
            )

        # Attractions per city
        for city, attrs in gathered.attractions.items():
            var_name = _safe_var_name(f"attractions_{city}")
            parts.append(f"  {var_name} = <{len(attrs)} attractions>")

        # Distances
        for route_key, dist in gathered.distances.items():
            var_name = _safe_var_name(f"distance_{route_key}")
            parts.append(
                f"  {var_name} = <cost: {dist.get('cost', '?')}, "
                f"duration: {dist.get('duration', '?')}>"
            )

        parts.append("")
        parts.append(
            "IMPORTANT: The data variables above are pre-loaded in your "
            "execution environment. Use them directly in primitive calls."
        )

        return "\n".join(parts)

    def _build_execution_namespace(self) -> dict[str, Any]:
        """Inject gathered data as variables into the execution namespace.

        All cost fields are pre-parsed to floats so the LLM can safely do
        arithmetic on raw dict values without hitting str+float type errors.

        In addition to the canonical ``_safe_var_name`` keys (e.g.
        ``flights_chicago_nyc_on_2022_03_16``), we inject short convenience
        aliases that the LLM commonly generates from the decomposition
        examples:

        * ``outbound_flights`` / ``return_flights`` – first and second
          flight route respectively
        * ``flights`` – alias for outbound_flights
        * ``{city}_restaurants``, ``{city}_accommodations``,
          ``{city}_attractions`` – per-city aliases without the long
          safe-name prefix
        * ``restaurants``, ``accommodations``, ``attractions`` – point to
          the *first* city's data (covers the common single-city case)
        * ``distances`` – list of all distance dicts
        """
        ns: dict[str, Any] = {}
        if not hasattr(self, "_gathered"):
            return ns

        gathered = self._gathered

        # ------------------------------------------------------------------
        # Flights  (canonical + convenience aliases)
        # ------------------------------------------------------------------
        flight_lists: list[list[dict]] = []
        for route_key, flights in gathered.flights.items():
            var_name = _safe_var_name(f"flights_{route_key}")
            normalised = [_normalize_costs(f, ["Price"]) for f in flights]
            ns[var_name] = normalised
            flight_lists.append(normalised)

        if flight_lists:
            ns["outbound_flights"] = flight_lists[0]
            ns["flights"] = flight_lists[0]
            # return_flights is always the LAST leg (back to origin)
            ns["return_flights"] = flight_lists[-1] if len(flight_lists) > 1 else []
            # For multi-city: intercity_flights is the middle leg(s)
            if len(flight_lists) > 2:
                ns["intercity_flights"] = flight_lists[1]
            else:
                ns["intercity_flights"] = []
            # Numbered leg aliases for multi-city trips
            for i, fl in enumerate(flight_lists):
                ns[f"leg{i + 1}_flights"] = fl
        else:
            ns["outbound_flights"] = []
            ns["return_flights"] = []
            ns["flights"] = []
            ns["intercity_flights"] = []

        # ------------------------------------------------------------------
        # Restaurants  (canonical + convenience aliases)
        # ------------------------------------------------------------------
        first_restaurants: list[dict] | None = None
        for city, restaurants in gathered.restaurants.items():
            normalised = [
                _normalize_costs(r, ["Average Cost"]) for r in restaurants
            ]
            ns[_safe_var_name(f"restaurants_{city}")] = normalised
            city_alias = _safe_var_name(city)
            ns[f"{city_alias}_restaurants"] = normalised
            if first_restaurants is None:
                first_restaurants = normalised
        ns["restaurants"] = first_restaurants or []

        # ------------------------------------------------------------------
        # Accommodations  (canonical + convenience aliases)
        # ------------------------------------------------------------------
        first_accommodations: list[dict] | None = None
        for city, accs in gathered.accommodations.items():
            normalised = [_normalize_costs(a, ["price"]) for a in accs]
            ns[_safe_var_name(f"accommodations_{city}")] = normalised
            city_alias = _safe_var_name(city)
            ns[f"{city_alias}_accommodations"] = normalised
            if first_accommodations is None:
                first_accommodations = normalised
        ns["accommodations"] = first_accommodations or []

        # ------------------------------------------------------------------
        # Attractions  (canonical + convenience aliases)
        # ------------------------------------------------------------------
        first_attractions: list[dict] | None = None
        for city, attrs in gathered.attractions.items():
            ns[_safe_var_name(f"attractions_{city}")] = attrs
            city_alias = _safe_var_name(city)
            ns[f"{city_alias}_attractions"] = attrs
            if first_attractions is None:
                first_attractions = attrs
        ns["attractions"] = first_attractions or []

        # ------------------------------------------------------------------
        # Distances  (canonical + convenience aliases)
        # ------------------------------------------------------------------
        distance_list: list[dict] = []
        for route_key, dist in gathered.distances.items():
            normalised = _normalize_costs(dist, ["cost"])
            ns[_safe_var_name(f"distance_{route_key}")] = normalised
            distance_list.append(normalised)
        ns["distances"] = distance_list

        return ns

    def execute(self, plan: str) -> Any:
        """Execute with gathered data injected into namespace."""
        # Inject gathered data variables into persisted namespace so
        # DesignExecute.execute() picks them up (multi_turn=True).
        extra_ns = self._build_execution_namespace()
        if extra_ns:
            self._persisted_namespace.update(extra_ns)
        return super().execute(plan)


def _safe_var_name(s: str) -> str:
    """Convert a string to a valid Python variable name."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    if s and s[0].isdigit():
        s = "_" + s
    return s.lower()
