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
from opensymbolicai.observability import ObservabilityConfig

from travelplanner_bench.constants import (
    ACCOMMODATION,
    ATTRACTION,
    BREAKFAST,
    CURRENT_CITY,
    DINNER,
    FLIGHT,
    LUNCH,
    MEAL_KEYS,
    NO_DATA,
    SELF_DRIVING,
    TAXI,
    TRANSPORTATION,
)
from travelplanner_bench.models import (
    Accommodation,
    Attraction,
    DistanceInfo,
    Flight,
    GatheredData,
    Restaurant,
    TransportPlan,
    TravelPlannerTask,
    ValidTransport,
)
from travelplanner_bench.utils import normalize_name, parse_cost, strip_city_suffix

log = logging.getLogger(__name__)


class PlanAssemblerAgent(DesignExecute):
    """Deterministic plan assembler using filtering, optimization, and cost primitives."""

    def __init__(
        self,
        llm: LLMConfig | LLM,
        max_plan_retries: int = 3,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            name="PlanAssemblerAgent",
            description="Deterministic plan assembler using filtering, optimization, and cost primitives.",
            config=DesignExecuteConfig(
                max_plan_retries=max_plan_retries,
                max_loop_iterations=50,
                max_total_primitive_calls=200,
                multi_turn=True,
                observability=observability,
            ),
        )
        self._submitted_plan: list[dict[str, Any]] | None = None
        self._last_error: str | None = None

    # =========================================================================
    # FILTERING PRIMITIVES
    # =========================================================================

    @primitive(read_only=True)
    def filter_by_room_type(
        self, accommodations: list[Accommodation], required_type: str
    ) -> list[Accommodation]:
        """Filter accommodations by room type.

        Args:
            accommodations: List of Accommodation objects.
            required_type: Required room type (e.g., "entire room", "private room").
                           Prefix with "not " to exclude a type (e.g., "not shared room").

        Returns:
            Filtered list of accommodations matching the room type.
        """
        required_lower = required_type.lower().strip()
        is_negation = required_lower.startswith("not ")

        result = []
        for acc in accommodations:
            actual = acc.room_type.lower().strip()
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
        self, accommodations: list[Accommodation], required_rule: str
    ) -> list[Accommodation]:
        """Filter accommodations by house rule.

        Args:
            accommodations: List of Accommodation objects.
            required_rule: Required rule (e.g., "No smoking", "No pets").

        Returns:
            Filtered list of accommodations whose house_rules contain the required rule.
        """
        rule_lower = required_rule.lower().strip()
        return [acc for acc in accommodations if rule_lower in acc.house_rules.lower()]

    @primitive(read_only=True)
    def filter_by_min_nights(
        self, accommodations: list[Accommodation], num_nights: int
    ) -> list[Accommodation]:
        """Filter accommodations that allow stays of num_nights or fewer.

        Args:
            accommodations: List of Accommodation objects.
            num_nights: Number of nights you plan to stay.

        Returns:
            Accommodations where min_nights <= num_nights.
        """
        return [acc for acc in accommodations if acc.min_nights <= num_nights]

    @primitive(read_only=True)
    def filter_by_cuisine(
        self, restaurants: list[Restaurant], required_cuisines: list[str]
    ) -> list[Restaurant]:
        """Filter restaurants that serve at least one of the required cuisines.

        Args:
            restaurants: List of Restaurant objects.
            required_cuisines: List of required cuisine types (e.g., ["Chinese", "Italian"]).

        Returns:
            Restaurants that serve at least one required cuisine.
        """
        required_lower = {c.lower().strip() for c in required_cuisines}
        return [r for r in restaurants if r.cuisine_set() & required_lower]

    @primitive(read_only=True)
    def filter_valid_transport(
        self,
        flights: list[Flight],
        distances: list[DistanceInfo],
        constraint: str,
    ) -> ValidTransport:
        """Determine valid transport options based on constraint.

        Args:
            flights: List of available Flight objects.
            distances: List of available DistanceInfo objects.
            constraint: Transportation constraint (e.g., "no flight", "no self-driving", or "").

        Returns:
            ValidTransport with filtered flights and distances.
        """
        constraint_lower = constraint.lower().strip() if constraint else ""
        valid_flights = flights if "no flight" not in constraint_lower else []
        valid_distances = distances if "no self-driving" not in constraint_lower else []
        return ValidTransport(flights=valid_flights, distances=valid_distances)

    # =========================================================================
    # OPTIMIZATION PRIMITIVES
    # =========================================================================

    @primitive(read_only=True)
    def cheapest_flights(self, flights: list[Flight], n: int = 1) -> list[Flight]:
        """Return the n cheapest flights sorted by price.

        Args:
            flights: List of Flight objects.
            n: Number of cheapest flights to return.

        Returns:
            Up to n flights sorted by ascending price.
        """
        return sorted(flights, key=lambda f: f.price)[:n]

    @primitive(read_only=True)
    def cheapest_accommodations(
        self, accommodations: list[Accommodation], n: int = 1
    ) -> list[Accommodation]:
        """Return the n cheapest accommodations sorted by price.

        Args:
            accommodations: List of Accommodation objects.
            n: Number of cheapest to return.

        Returns:
            Up to n accommodations sorted by ascending price per night.
        """
        return sorted(accommodations, key=lambda a: a.price)[:n]

    @primitive(read_only=True)
    def cheapest_restaurant_set(
        self,
        restaurants: list[Restaurant],
        count: int,
        required_cuisines: list[str] | None = None,
    ) -> list[Restaurant]:
        """Find the cheapest set of unique restaurants, optionally covering required cuisines.

        First ensures all required cuisines are covered (picking the cheapest
        restaurant per cuisine), then fills remaining slots with cheapest overall.

        Args:
            restaurants: List of Restaurant objects.
            count: Total number of unique restaurants needed.
            required_cuisines: Optional list of cuisines that must be covered.

        Returns:
            List of up to `count` unique restaurants, cheapest first,
            covering all required cuisines if possible.
        """
        selected: list[Restaurant] = []
        selected_names: set[str] = set()

        # Phase 1: Cover required cuisines
        if required_cuisines:
            for cuisine in required_cuisines:
                cuisine_lower = cuisine.lower().strip()
                candidates = [
                    r for r in restaurants
                    if r.name not in selected_names
                    and cuisine_lower in r.cuisine_set()
                ]
                if candidates:
                    cheapest = min(candidates, key=lambda r: r.average_cost)
                    selected.append(cheapest)
                    selected_names.add(cheapest.name)

        # Phase 2: Fill remaining with cheapest unselected
        if len(selected) < count:
            for r in sorted(restaurants, key=lambda r: r.average_cost):
                if len(selected) >= count:
                    break
                if r.name not in selected_names:
                    selected.append(r)
                    selected_names.add(r.name)

        return selected

    @primitive(read_only=True)
    def optimal_accommodation(
        self,
        accommodations: list[Accommodation],
        nights: int,
        budget_remaining: float,
    ) -> Accommodation | None:
        """Find the cheapest accommodation that fits within remaining budget.

        Args:
            accommodations: List of Accommodation objects (should be pre-filtered).
            nights: Number of nights staying.
            budget_remaining: Remaining budget in dollars.

        Returns:
            Cheapest accommodation where (price * nights) <= budget_remaining,
            or None if no valid option.
        """
        sorted_accs = sorted(accommodations, key=lambda a: a.price)
        for acc in sorted_accs:
            if acc.price * nights <= budget_remaining:
                return acc
        # If nothing fits budget, return cheapest anyway (better than nothing)
        return sorted_accs[0] if sorted_accs else None

    @primitive(read_only=True)
    def assign_meals(
        self,
        restaurants: list[Restaurant],
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
            restaurants: List of unique Restaurant objects (from cheapest_restaurant_set).
            num_days: Total number of days in the trip.
            city: City name for formatting.

        Returns:
            List of dicts (one per day), each with "breakfast", "lunch", "dinner" keys.
            Values are formatted "Restaurant Name, City" or "-".
        """
        meals: list[dict[str, str]] = []
        r_idx = 0

        for day in range(1, num_days + 1):
            day_meals = {BREAKFAST: NO_DATA, LUNCH: NO_DATA, DINNER: NO_DATA}

            if day == num_days:
                # Last day: departure, no meals
                pass
            elif day == 1:
                # First day: no breakfast (arrival)
                for slot in [LUNCH, DINNER]:
                    if r_idx < len(restaurants):
                        day_meals[slot] = f"{restaurants[r_idx].name}, {city}"
                        r_idx += 1
            else:
                # Middle days: all three meals
                for slot in MEAL_KEYS:
                    if r_idx < len(restaurants):
                        day_meals[slot] = f"{restaurants[r_idx].name}, {city}"
                        r_idx += 1

            meals.append(day_meals)

        return meals

    @primitive(read_only=True)
    def pick_diverse_attractions(
        self, attractions: list[Attraction], count: int
    ) -> list[Attraction]:
        """Pick up to `count` unique attractions.

        Args:
            attractions: List of Attraction objects.
            count: Number of unique attractions to pick.

        Returns:
            Up to `count` unique attractions.
        """
        seen: set[str] = set()
        result: list[Attraction] = []
        for attr in attractions:
            if attr.name not in seen:
                result.append(attr)
                seen.add(attr.name)
            if len(result) >= count:
                break
        return result

    # =========================================================================
    # COMPOUND PRIMITIVES (reduce call count, enforce type safety)
    # =========================================================================

    @primitive(read_only=True)
    def select_accommodation(
        self,
        accommodations: list[Accommodation],
        nights: int,
        budget: float,
        room_type: str | None = None,
        room_rule: str | None = None,
    ) -> Accommodation | None:
        """Select the best accommodation after applying all filters.

        Combines room_type filtering, room_rule filtering, min_nights filtering,
        and budget-optimal selection into a single call.

        Args:
            accommodations: List of Accommodation objects for a city.
            nights: Number of nights to stay.
            budget: Total remaining budget (used to prefer affordable options).
            room_type: Required room type (e.g., "entire room") or None.
            room_rule: Required house rule (e.g., "No smoking") or None.

        Returns:
            Best Accommodation, or None if no options available.
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
        restaurants: list[Restaurant],
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
            restaurants: All available Restaurant objects for the city.
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
        outbound_flights: list[Flight],
        return_flights: list[Flight],
        distances: list[DistanceInfo] | None,
        constraint: str,
        origin: str,
        destination: str,
    ) -> TransportPlan:
        """Select and format transport for the trip in one call.

        Handles flight vs self-driving selection based on constraint, picks
        cheapest options, and returns pre-formatted transport strings.

        Args:
            outbound_flights: Available outbound Flight objects.
            return_flights: Available return Flight objects.
            distances: Available DistanceInfo objects, or None.
            constraint: Transport constraint ("no flight", "no self-driving", or "").
            origin: Origin city name.
            destination: Destination city name.

        Returns:
            TransportPlan with formatted strings, selected flights, and costs.
        """
        valid = self.filter_valid_transport(
            outbound_flights, distances or [], constraint
        )

        if valid.flights:
            out_list = self.cheapest_flights(valid.flights, n=1)
            ret_valid = self.filter_valid_transport(
                return_flights, [], constraint
            )
            ret_list = self.cheapest_flights(ret_valid.flights, n=1)

            out_f = out_list[0] if out_list else None
            ret_f = ret_list[0] if ret_list else None

            return TransportPlan(
                outbound_str=self.format_flight(out_f) if out_f else "-",
                return_str=self.format_flight(ret_f) if ret_f else "-",
                outbound_flight=out_f,
                return_flight=ret_f,
                mode=FLIGHT,
            )
        elif valid.distances:
            dist = valid.distances[0] if isinstance(valid.distances, list) else valid.distances
            return TransportPlan(
                outbound_str=self.format_driving(dist, origin, destination),
                return_str=self.format_driving(dist, destination, origin),
                driving_costs=[dist.cost, dist.cost],
                mode=dist.mode,
            )
        else:
            return TransportPlan(mode="unknown")

    # =========================================================================
    # COST CALCULATION PRIMITIVES
    # =========================================================================

    @primitive(read_only=True)
    def get_cost(self, entity: Flight | Restaurant | Accommodation | DistanceInfo) -> float:
        """Get the numeric cost from any travel entity.

        Args:
            entity: Any travel entity (Flight, Restaurant, Accommodation, DistanceInfo).

        Returns:
            Cost as a float. Returns 0.0 if no cost field found.
        """
        if isinstance(entity, Flight):
            return entity.price
        if isinstance(entity, Restaurant):
            return entity.average_cost
        if isinstance(entity, Accommodation):
            return entity.price
        if isinstance(entity, DistanceInfo):
            return entity.cost
        return 0.0

    @primitive(read_only=True)
    def flight_cost(self, flight: Flight) -> float:
        """Get the per-person cost of a flight.

        Args:
            flight: A Flight object.

        Returns:
            Price as a float.
        """
        return flight.price

    @primitive(read_only=True)
    def accommodation_cost(self, accommodation: Accommodation, nights: int) -> float:
        """Get the total cost of an accommodation for given nights.

        Args:
            accommodation: An Accommodation object.
            nights: Number of nights staying.

        Returns:
            Total cost (price per night * nights).
        """
        return accommodation.price * nights

    @primitive(read_only=True)
    def restaurant_cost(self, restaurant: Restaurant) -> float:
        """Get the per-person average cost of a restaurant.

        Args:
            restaurant: A Restaurant object.

        Returns:
            Average cost as a float.
        """
        return restaurant.average_cost

    @primitive(read_only=True)
    def total_trip_cost(
        self,
        flights: list[Flight],
        accommodations: list[Accommodation],
        accommodation_nights: list[int],
        restaurants: list[Restaurant],
        people: int,
        driving_costs: list[float] | None = None,
    ) -> float:
        """Calculate total trip cost across all components.

        Args:
            flights: List of Flight objects used in the trip.
            accommodations: List of Accommodation objects used (one per city).
            accommodation_nights: Number of nights at each accommodation.
            restaurants: List of Restaurant objects used for meals.
            people: Number of travelers.
            driving_costs: Optional list of driving/taxi costs.

        Returns:
            Total trip cost.
        """
        total = 0.0

        # Flights: per person
        for f in flights:
            total += f.price * people

        # Accommodations: per night (not per person)
        for acc, nights in zip(accommodations, accommodation_nights):
            total += acc.price * nights

        # Restaurants: per person
        for r in restaurants:
            total += r.average_cost * people

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
    def format_flight(self, flight: Flight) -> str:
        """Format a Flight into the standard transport string.

        Args:
            flight: A Flight object.

        Returns:
            Formatted string like "Flight Number: F123, from X to Y, Departure Time: ..., Arrival Time: ..."
        """
        return (
            f"Flight Number: {flight.flight_number}, from {flight.origin} to {flight.destination}, "
            f"Departure Time: {flight.dep_time}, Arrival Time: {flight.arr_time}"
        )

    @primitive(read_only=True)
    def format_driving(self, distance_info: DistanceInfo, origin: str, destination: str) -> str:
        """Format driving info into transport string.

        Args:
            distance_info: A DistanceInfo object.
            origin: Origin city name.
            destination: Destination city name.

        Returns:
            Formatted string like "Self-driving, from X to Y, Duration: ..., Distance: ..., Cost: ..."
        """
        mode_label = "Self-driving" if "self" in distance_info.mode.lower() else "Taxi"
        return (
            f"{mode_label}, from {origin} to {destination}, "
            f"Duration: {distance_info.duration}, Distance: {distance_info.distance}, Cost: {distance_info.cost}"
        )

    @primitive(read_only=True)
    def format_restaurant(self, restaurant: Restaurant, city: str) -> str:
        """Format a restaurant for plan output.

        Args:
            restaurant: A Restaurant object.
            city: City name.

        Returns:
            Formatted string like "Restaurant Name, City".
        """
        return f"{restaurant.name}, {city}"

    @primitive(read_only=True)
    def format_attractions(self, attractions: list[Attraction], city: str) -> str:
        """Format attractions for plan output (semicolon-separated).

        Args:
            attractions: List of Attraction objects.
            city: City name.

        Returns:
            Formatted string like "Attraction1, City;Attraction2, City" or "-".
        """
        if not attractions:
            return "-"
        return ";".join(f"{a.name}, {city}" for a in attractions)

    @primitive(read_only=True)
    def format_accommodation(self, accommodation: Accommodation, city: str) -> str:
        """Format an accommodation for plan output.

        Args:
            accommodation: An Accommodation object.
            city: City name.

        Returns:
            Formatted string like "Accommodation Name, City".
        """
        return f"{accommodation.name}, {city}"

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
            CURRENT_CITY: current_city,
            TRANSPORTATION: transportation,
            BREAKFAST: breakfast,
            ATTRACTION: attraction,
            LUNCH: lunch,
            DINNER: dinner,
            ACCOMMODATION: accommodation,
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
        flights_used = [f for f in [transport.outbound_flight, transport.return_flight] if f]
        cost = self.total_trip_cost(
            flights_used, [best_acc], [2], restaurants[:5], 1,
            driving_costs=transport.driving_costs,
        )
        ok = self.check_budget(cost, 1900.0)

        # Build days
        acc_str = self.format_accommodation(best_acc, "Chicago")
        day1 = self.build_day(
            1, "from Sarasota to Chicago", transport.outbound_str,
            meals[0][BREAKFAST],
            self.format_attractions(day_attractions[:2], "Chicago"),
            meals[0][LUNCH], meals[0][DINNER], acc_str,
        )
        day2 = self.build_day(
            2, "Chicago", "-",
            meals[1][BREAKFAST],
            self.format_attractions(day_attractions[2:4], "Chicago"),
            meals[1][LUNCH], meals[1][DINNER], acc_str,
        )
        day3 = self.build_day(
            3, "from Chicago to Sarasota", transport.return_str,
            meals[2][BREAKFAST], "-",
            meals[2][LUNCH], meals[2][DINNER], "-",
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
        flights_used = [f for f in [transport.outbound_flight, transport.return_flight] if f]
        cost = self.total_trip_cost(
            flights_used, [best_acc], [2],
            restaurants[:5], 2,
            driving_costs=transport.driving_costs,
        )
        ok = self.check_budget(cost, 1200.0)

        # Build days
        acc_str = self.format_accommodation(best_acc, "CityName")
        day1 = self.build_day(
            1, "from Origin to CityName", transport.outbound_str,
            meals[0][BREAKFAST],
            self.format_attractions(attrs[:2], "CityName"),
            meals[0][LUNCH], meals[0][DINNER], acc_str,
        )
        day2 = self.build_day(
            2, "CityName", "-",
            meals[1][BREAKFAST],
            self.format_attractions(attrs[2:4], "CityName"),
            meals[1][LUNCH], meals[1][DINNER], acc_str,
        )
        day3 = self.build_day(
            3, "from CityName to Origin", transport.return_str,
            meals[2][BREAKFAST], "-",
            meals[2][LUNCH], meals[2][DINNER], "-",
        )
        result = self.set_plan([day1, day2, day3])
        return result

    @decomposition(
        intent=(
            "Build a 3-day trip with tight budget $1100, "
            "no special room/cuisine constraints, 2 people, "
            "flights are expensive so self-driving is cheaper"
        ),
        expanded_intent=(
            "Tight budget: compare flight cost vs self-driving cost. "
            "Use self-driving if cheaper. Pick cheapest restaurants. "
            "NEVER raise on budget — always call set_plan(). "
            "Post-processing handles budget optimization (swapping flights "
            "to self-driving, trimming expensive meals)."
        ),
    )
    def _ex_tight_budget_3day(self) -> str:
        # Compare transport options: flights vs self-driving
        transport = self.plan_transport(
            outbound_flights, return_flights, distances, "",
            "Origin", "CityName",
        )

        # Accommodation: cheapest option
        best_acc = self.select_accommodation(accommodations, 2, 1100.0)

        # Meals: cheapest restaurants
        meals = self.prepare_meals(restaurants, 3, "CityName")

        # Attractions
        attrs = self.pick_diverse_attractions(attractions, 4)

        # Build plan — always submit, never raise on budget.
        # Post-processing will swap flights to self-driving and trim meals if needed.
        acc_str = self.format_accommodation(best_acc, "CityName")
        day1 = self.build_day(
            1, "from Origin to CityName", transport.outbound_str,
            meals[0][BREAKFAST],
            self.format_attractions(attrs[:2], "CityName"),
            meals[0][LUNCH], meals[0][DINNER], acc_str,
        )
        day2 = self.build_day(
            2, "CityName", "-",
            meals[1][BREAKFAST],
            self.format_attractions(attrs[2:4], "CityName"),
            meals[1][LUNCH], meals[1][DINNER], acc_str,
        )
        day3 = self.build_day(
            3, "from CityName to Origin", transport.return_str,
            meals[2][BREAKFAST], "-",
            meals[2][LUNCH], meals[2][DINNER], "-",
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
            "IMPORTANT: prepare_meals num_days = days_in_city + 1 (treat the "
            "inter-city transfer day as the departure day for that city). "
            "For a 5-day/2-city trip: city1 gets num_days=3 (arrival+stay+depart), "
            "city2 gets num_days=3 (arrival+stay+depart). "
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

        # --- City 1: San Antonio (2 nights, num_days=3: arrival+stay+transfer) ---
        acc1 = self.select_accommodation(san_antonio_accommodations, 2, 3100.0)
        meals1 = self.prepare_meals(san_antonio_restaurants, 3, "San Antonio")
        attrs1 = self.pick_diverse_attractions(san_antonio_attractions, 4)
        acc1_str = self.format_accommodation(acc1, "San Antonio")

        # --- City 2: Houston (2 nights, num_days=3: arrival+stay+depart) ---
        acc2 = self.select_accommodation(houston_accommodations, 2, 3100.0)
        meals2 = self.prepare_meals(houston_restaurants, 3, "Houston")
        attrs2 = self.pick_diverse_attractions(houston_attractions, 4)
        acc2_str = self.format_accommodation(acc2, "Houston")

        # --- Build all 5 days ---
        day1 = self.build_day(
            1, "from Orlando to San Antonio", out_str,
            meals1[0][BREAKFAST],
            self.format_attractions(attrs1[:2], "San Antonio"),
            meals1[0][LUNCH], meals1[0][DINNER], acc1_str,
        )
        day2 = self.build_day(
            2, "San Antonio", "-",
            meals1[1][BREAKFAST],
            self.format_attractions(attrs1[2:4], "San Antonio"),
            meals1[1][LUNCH], meals1[1][DINNER], acc1_str,
        )
        day3 = self.build_day(
            3, "from San Antonio to Houston", inter_str,
            meals2[0][BREAKFAST],
            self.format_attractions(attrs2[:2], "Houston"),
            meals2[0][LUNCH], meals2[0][DINNER], acc2_str,
        )
        day4 = self.build_day(
            4, "Houston", "-",
            meals2[1][BREAKFAST],
            self.format_attractions(attrs2[2:4], "Houston"),
            meals2[1][LUNCH], meals2[1][DINNER], acc2_str,
        )
        day5 = self.build_day(
            5, "from Houston to Orlando", ret_str,
            meals2[2][BREAKFAST], "-",
            meals2[2][LUNCH], meals2[2][DINNER], "-",
        )

        result = self.set_plan([day1, day2, day3, day4, day5])
        return result

    @decomposition(
        intent=(
            "Build a 7-day multi-city trip plan visiting 3 cities: "
            "Denver -> Pellston (2 days) -> Kalamazoo (2 days) -> Detroit (2 days) -> Denver, "
            "budget $5000, 2 people, room type 'private room'"
        ),
        expanded_intent=(
            "3-city multi-city: handle 4 transport legs. Use leg1_flights..leg4_flights "
            "or numbered flight variables. When no flights available for a leg, fall back "
            "to self-driving using distances. Each city gets arrival + stay + departure days. "
            "CRITICAL: never raise on budget — always call set_plan(). Post-processing "
            "handles budget optimization (swapping flights to self-driving, trimming meals)."
        ),
    )
    def _ex_multi_city_3cities_7day(self) -> str:
        # --- Transport legs (4 legs for 3 cities) ---
        # Leg 1: Denver -> Pellston
        leg1 = cheapest_flights(leg1_flights, n=1)
        leg1_f = leg1[0] if leg1 else None
        leg1_str = self.format_flight(leg1_f) if leg1_f else "-"

        # Leg 2: Pellston -> Kalamazoo
        leg2 = cheapest_flights(leg2_flights, n=1)
        leg2_f = leg2[0] if leg2 else None
        leg2_str = self.format_flight(leg2_f) if leg2_f else "-"

        # Leg 3: Kalamazoo -> Detroit
        leg3 = cheapest_flights(leg3_flights, n=1)
        leg3_f = leg3[0] if leg3 else None
        leg3_str = self.format_flight(leg3_f) if leg3_f else "-"

        # Leg 4: Detroit -> Denver (return)
        leg4 = cheapest_flights(leg4_flights, n=1)
        leg4_f = leg4[0] if leg4 else None
        leg4_str = self.format_flight(leg4_f) if leg4_f else "-"

        # --- City 1: Pellston (2 nights, num_days=3: arrival+stay+depart) ---
        acc1 = self.select_accommodation(
            pellston_accommodations, 2, 5000.0, room_type="private room"
        )
        meals1 = self.prepare_meals(pellston_restaurants, 3, "Pellston")
        attrs1 = self.pick_diverse_attractions(pellston_attractions, 4)
        acc1_str = self.format_accommodation(acc1, "Pellston")

        # --- City 2: Kalamazoo (2 nights, num_days=3: arrival+stay+depart) ---
        acc2 = self.select_accommodation(
            kalamazoo_accommodations, 2, 5000.0, room_type="private room"
        )
        meals2 = self.prepare_meals(kalamazoo_restaurants, 3, "Kalamazoo")
        attrs2 = self.pick_diverse_attractions(kalamazoo_attractions, 4)
        acc2_str = self.format_accommodation(acc2, "Kalamazoo")

        # --- City 3: Detroit (2 nights, num_days=3: arrival+stay+depart) ---
        acc3 = self.select_accommodation(
            detroit_accommodations, 2, 5000.0, room_type="private room"
        )
        meals3 = self.prepare_meals(detroit_restaurants, 3, "Detroit")
        attrs3 = self.pick_diverse_attractions(detroit_attractions, 4)
        acc3_str = self.format_accommodation(acc3, "Detroit")

        # --- Build all 7 days ---
        day1 = self.build_day(
            1, "from Denver to Pellston", leg1_str,
            meals1[0][BREAKFAST],
            self.format_attractions(attrs1[:2], "Pellston"),
            meals1[0][LUNCH], meals1[0][DINNER], acc1_str,
        )
        day2 = self.build_day(
            2, "Pellston", "-",
            meals1[1][BREAKFAST],
            self.format_attractions(attrs1[2:4], "Pellston"),
            meals1[1][LUNCH], meals1[1][DINNER], acc1_str,
        )
        day3 = self.build_day(
            3, "from Pellston to Kalamazoo", leg2_str,
            meals2[0][BREAKFAST],
            self.format_attractions(attrs2[:2], "Kalamazoo"),
            meals2[0][LUNCH], meals2[0][DINNER], acc2_str,
        )
        day4 = self.build_day(
            4, "Kalamazoo", "-",
            meals2[1][BREAKFAST],
            self.format_attractions(attrs2[2:4], "Kalamazoo"),
            meals2[1][LUNCH], meals2[1][DINNER], acc2_str,
        )
        day5 = self.build_day(
            5, "from Kalamazoo to Detroit", leg3_str,
            meals3[0][BREAKFAST],
            self.format_attractions(attrs3[:2], "Detroit"),
            meals3[0][LUNCH], meals3[0][DINNER], acc3_str,
        )
        day6 = self.build_day(
            6, "Detroit", "-",
            meals3[1][BREAKFAST],
            self.format_attractions(attrs3[2:4], "Detroit"),
            meals3[1][LUNCH], meals3[1][DINNER], acc3_str,
        )
        day7 = self.build_day(
            7, "from Detroit to Denver", leg4_str,
            meals3[2][BREAKFAST], "-",
            meals3[2][LUNCH], meals3[2][DINNER], "-",
        )

        # Budget check — never raise, always submit.
        # Post-processing handles budget optimization.
        result = self.set_plan([day1, day2, day3, day4, day5, day6, day7])
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
        self._run_result = result  # Expose for logging

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

        # --- Phase 1c: Fix return-to-origin on last day ---
        self._fix_return_to_origin(plan, gathered, task)

        # --- Phase 2: Clear last/return day ---
        self._clear_return_day(plan, task)

        # --- Phase 3: Fix meals, attractions, and accommodation referencing wrong city ---
        self._fix_wrong_city_meals(plan, gathered, task)
        self._fix_wrong_city_attractions(plan, gathered, task)
        self._fix_wrong_city_accommodation(plan, gathered, task)

        # --- Phase 4: Fill missing meals and accommodation ---
        required_cuisine_set: set[str] = set()
        if task.local_constraint.cuisine:
            required_cuisine_set = {
                c.strip().lower() for c in task.local_constraint.cuisine
            }

        used_restaurants: set[str] = set()
        for day in plan:
            for meal_key in MEAL_KEYS:
                name = day.get(meal_key, "-").strip()
                if name and name != "-":
                    used_restaurants.add(name.split(",")[0].strip().lower())

        # Determine which required cuisines are already covered by existing meals
        def _covered_cuisines(p: list[dict], g: GatheredData) -> set[str]:
            covered: set[str] = set()
            for d in p:
                for mk in MEAL_KEYS:
                    v = d.get(mk, "-").strip()
                    if not v or v == "-":
                        continue
                    base = v.rsplit(",", 1)[0].strip()
                    for city_rests in g.restaurants.values():
                        for r in city_rests:
                            if r.name.strip().lower() == base.lower():
                                covered |= r.cuisine_set()
            return covered

        covered = _covered_cuisines(plan, gathered)
        uncovered = required_cuisine_set - covered

        for day in plan:
            day_num = day.get("days", 0)
            city = self._infer_stay_city(day, task, gathered)
            if not city:
                continue

            city_restaurants = self._find_city_data(gathered.restaurants, city)
            city_accs = self._find_city_data(gathered.accommodations, city)

            # Fill missing meals — prefer restaurants covering uncovered cuisines
            # Skip Day 1 breakfast: arrival-day convention (prepare_meals leaves
            # it empty, agent's budget doesn't include it, and
            # check_complete_info only requires filled >= 2 on day 1).
            # Skip last day (return day): _clear_return_day already set meals to "-".
            if day_num != task.days:
                for meal_key in MEAL_KEYS:
                    if day_num == 1 and meal_key == BREAKFAST:
                        continue
                    val = day.get(meal_key, "-").strip()
                    if not val or val == "-":
                        rest = self._pick_unused_restaurant(
                            city_restaurants,
                            used_restaurants,
                            preferred_cuisines=uncovered if uncovered else None,
                        )
                        if rest:
                            day[meal_key] = f"{rest.name}, {city}"
                            used_restaurants.add(rest.name.lower())
                            # Update uncovered set
                            if uncovered:
                                uncovered -= rest.cuisine_set()

            # Fill missing accommodation (skip last day)
            if day_num != task.days:
                val = day.get(ACCOMMODATION, NO_DATA).strip()
                if (not val or val == "-") and city_accs:
                    acc = min(city_accs, key=lambda a: a.price)
                    day[ACCOMMODATION] = f"{acc.name}, {city}"

        # --- Phase 5: Cuisine coverage sweep ---
        # If required cuisines are still uncovered after filling, swap a
        # non-cuisine-critical restaurant with one that covers the gap.
        if required_cuisine_set:
            self._ensure_cuisine_coverage(plan, gathered, task, required_cuisine_set)

        # --- Phase 6: Deduplicate restaurants across cities ---
        self._deduplicate_restaurants(plan, gathered)

        # --- Phase 7: Deduplicate attractions ---
        self._deduplicate_attractions(plan, gathered)

        # --- Phase 8: Budget guard ---
        # If a budget is specified, verify total cost.  When over-budget,
        # remove the most-expensive *expendable* meals (those not solely
        # covering a required cuisine) until within budget.
        if task.budget:
            self._budget_guard(plan, gathered, task, required_cuisine_set)

    def _budget_guard(
        self,
        plan: list[dict[str, Any]],
        gathered: GatheredData,
        task: TravelPlannerTask,
        required_cuisine_set: set[str],
    ) -> None:
        """Remove expendable meals if total plan cost exceeds the budget."""
        import re as _re

        people = task.people_number or 1

        def _restaurant_cost(name: str) -> float:
            n = name.strip().lower()
            for rests in gathered.restaurants.values():
                for r in rests:
                    if r.name.strip().lower() == n:
                        return r.average_cost
            return 0.0

        def _flight_cost(flight_number: str) -> float:
            fn = flight_number.strip().lower()
            for flights in gathered.flights.values():
                for f in flights:
                    if f.flight_number.strip().lower() == fn:
                        return f.price
            return 0.0

        def _accommodation_cost(name: str) -> float:
            n = name.strip().lower()
            for accs in gathered.accommodations.values():
                for a in accs:
                    if a.name.strip().lower() == n:
                        return a.price
            return 0.0

        def _total_cost() -> float:
            total = 0.0
            for day in plan:
                # Flights
                transport = day.get(TRANSPORTATION, NO_DATA)
                if transport and transport.strip() != "-":
                    fm = _re.search(r"Flight\s+Number:\s*([A-Za-z0-9]+)", transport)
                    if fm:
                        total += _flight_cost(fm.group(1)) * people
                    else:
                        cm = _re.search(r"[Cc]ost:?\s*\$?([\d,.]+)", transport)
                        if cm:
                            total += parse_cost(cm.group(1))
                # Meals
                for mk in MEAL_KEYS:
                    v = day.get(mk, "-").strip()
                    if v and v != "-":
                        base = v.rsplit(",", 1)[0].strip()
                        total += _restaurant_cost(base) * people
                # Accommodation
                acc = day.get(ACCOMMODATION, NO_DATA).strip()
                if acc and acc != "-":
                    base = acc.rsplit(",", 1)[0].strip()
                    total += _accommodation_cost(base)
            return total

        cost = _total_cost()
        if cost <= task.budget:
            return

        log.warning(
            "BUDGET GUARD: plan cost $%.0f exceeds budget $%.0f — trimming meals",
            cost, task.budget,
        )

        # Build set of cuisine-critical restaurant names (sole providers)
        critical_names: set[str] = set()
        if required_cuisine_set:
            # cuisine → set of restaurant names covering it
            cuisine_providers: dict[str, set[str]] = {c: set() for c in required_cuisine_set}
            for day in plan:
                for mk in MEAL_KEYS:
                    v = day.get(mk, "-").strip()
                    if not v or v == "-":
                        continue
                    base = v.rsplit(",", 1)[0].strip().lower()
                    for rests in gathered.restaurants.values():
                        for r in rests:
                            if r.name.strip().lower() == base:
                                for c in r.cuisines.split(","):
                                    cl = c.strip().lower()
                                    if cl in cuisine_providers:
                                        cuisine_providers[cl].add(base)
            for _cuisine, providers in cuisine_providers.items():
                if len(providers) == 1:
                    critical_names |= providers

        # Collect removable meal slots: (cost_per_person, day_idx, meal_key)
        removable: list[tuple[float, int, str]] = []
        for i, day in enumerate(plan):
            for mk in MEAL_KEYS:
                v = day.get(mk, "-").strip()
                if not v or v == "-":
                    continue
                base = v.rsplit(",", 1)[0].strip()
                if base.lower() in critical_names:
                    continue
                mc = _restaurant_cost(base)
                removable.append((mc, i, mk))

        # Sort by cost descending — remove the most expensive first
        removable.sort(key=lambda x: x[0], reverse=True)

        for mc, i, mk in removable:
            plan[i][mk] = "-"
            cost -= mc * people
            log.info(
                "BUDGET GUARD: removed %s day %d (saved $%.0f) — new total $%.0f",
                mk, plan[i].get("days", i + 1), mc * people, cost,
            )
            if cost <= task.budget:
                break

        # If still over budget after removing all meals, try swapping
        # flights to self-driving (much cheaper for tight budgets).
        if cost > task.budget:
            self._budget_guard_swap_transport(plan, gathered, task, _re, _flight_cost, people)
            cost = _total_cost()
            if cost <= task.budget:
                log.info("BUDGET GUARD: transport swap resolved budget — $%.0f", cost)

        # Re-fill meals after transport swap freed up budget
        if cost <= task.budget:
            # Check if any non-return day has all meals empty — refill them
            has_empty = False
            for day in plan:
                dn = day.get("days", 0)
                if dn == task.days:
                    continue  # skip return day
                for mk in MEAL_KEYS:
                    if dn == 1 and mk == BREAKFAST:
                        continue
                    v = day.get(mk, "-").strip()
                    if not v or v == "-":
                        has_empty = True
                        break
            if has_empty:
                # Re-run Phase 4 style filling with remaining budget awareness
                used: set[str] = set()
                for day in plan:
                    for mk in MEAL_KEYS:
                        v = day.get(mk, "-").strip()
                        if v and v != "-":
                            used.add(v.rsplit(",", 1)[0].strip().lower())
                for day in plan:
                    dn = day.get("days", 0)
                    city = self._infer_stay_city(day, task, gathered)
                    if not city:
                        continue
                    city_rests = self._find_city_data(gathered.restaurants, city)
                    for mk in MEAL_KEYS:
                        if dn == 1 and mk == BREAKFAST:
                            continue
                        v = day.get(mk, "-").strip()
                        if v and v != "-":
                            continue
                        rest = self._pick_unused_restaurant(city_rests, used)
                        if rest:
                            if cost + rest.average_cost * people <= task.budget:
                                day[mk] = f"{rest.name}, {city}"
                                used.add(rest.name.lower())
                                cost += rest.average_cost * people

    @staticmethod
    def _budget_guard_swap_transport(
        plan: list[dict[str, Any]],
        gathered: GatheredData,
        task: TravelPlannerTask,
        _re: Any,
        _flight_cost: Any,
        people: int,
    ) -> None:
        """Swap flight-based transport to self-driving if cheaper."""
        for day in plan:
            transport = day.get(TRANSPORTATION, NO_DATA)
            if not transport or transport.strip() == "-":
                continue
            fm = _re.search(r"Flight\s+Number:\s*([A-Za-z0-9]+)", transport)
            if not fm:
                continue
            # This day uses a flight — check if self-driving is available
            origin_dest = _re.search(
                r"from\s+(.+?)\s+to\s+(.+?)(?:,|\s*$)",
                transport,
            )
            if not origin_dest:
                continue
            orig = origin_dest.group(1).strip()
            dest = origin_dest.group(2).strip()
            # Find self-driving distance data
            drive_data: DistanceInfo | None = None
            for _key, d in gathered.distances.items():
                if (d.origin.lower() == orig.lower()
                        and d.destination.lower() == dest.lower()
                        and d.mode == SELF_DRIVING):
                    drive_data = d
                    break
            if not drive_data:
                continue
            flight_cost_val = _flight_cost(fm.group(1)) * people
            if drive_data.cost < flight_cost_val:
                day[TRANSPORTATION] = (
                    f"Self-driving, from {orig} to {dest}, "
                    f"Duration: {drive_data.duration}, Distance: {drive_data.distance} km, "
                    f"Cost: {drive_data.cost}"
                )
                log.info(
                    "BUDGET GUARD: swapped flight $%.0f → self-driving $%.0f "
                    "for %s → %s",
                    flight_cost_val, drive_data.cost, orig, dest,
                )

    def _ensure_cuisine_coverage(
        self,
        plan: list[dict[str, Any]],
        gathered: GatheredData,
        task: TravelPlannerTask,
        required_cuisine_set: set[str],
    ) -> None:
        """Swap restaurants in the plan to guarantee all required cuisines appear.

        1. Determine which required cuisines are already covered.
        2. For each uncovered cuisine, find a candidate restaurant from any
           city's data that serves it.
        3. Swap it with the cheapest non-cuisine-critical meal slot (one whose
           restaurant doesn't uniquely cover any required cuisine).
        """
        # Build restaurant-name → cuisines lookup from all gathered data
        rest_cuisines: dict[str, set[str]] = {}  # lower(name) → set of lower cuisines
        rest_city: dict[str, str] = {}  # lower(name) → city
        rest_obj: dict[str, Restaurant] = {}  # lower(name) → Restaurant
        for city, rests in gathered.restaurants.items():
            for r in rests:
                if not r.name:
                    continue
                key = r.name.lower()
                rest_cuisines[key] = r.cuisine_set()
                rest_city[key] = city
                rest_obj[key] = r

        # Identify which required cuisines each plan-meal covers
        def _plan_covered() -> set[str]:
            covered: set[str] = set()
            for d in plan:
                for mk in MEAL_KEYS:
                    v = d.get(mk, "-").strip()
                    if not v or v == "-":
                        continue
                    base = v.rsplit(",", 1)[0].strip().lower()
                    covered |= rest_cuisines.get(base, set())
            return covered

        covered = _plan_covered()
        uncovered = required_cuisine_set - covered
        if not uncovered:
            return

        log.info("CUISINE_SWEEP: uncovered=%s, attempting swaps", uncovered)

        # Collect current plan meal entries as (day_idx, meal_key, base_name_lower)
        plan_meals: list[tuple[int, str, str]] = []
        used_names: set[str] = set()
        for i, d in enumerate(plan):
            for mk in MEAL_KEYS:
                v = d.get(mk, "-").strip()
                if not v or v == "-":
                    continue
                base = v.rsplit(",", 1)[0].strip().lower()
                plan_meals.append((i, mk, base))
                used_names.add(base)

        for missing_cuisine in list(uncovered):
            # Find a candidate restaurant serving this cuisine that isn't
            # already in the plan
            candidate: Restaurant | None = None
            candidate_city: str | None = None
            for city, rests in gathered.restaurants.items():
                sorted_rests = sorted(rests, key=lambda r: r.average_cost)
                for r in sorted_rests:
                    if not r.name or r.name.lower() in used_names:
                        continue
                    if missing_cuisine in r.cuisine_set():
                        candidate = r
                        candidate_city = city
                        break
                if candidate:
                    break

            if not candidate or not candidate_city:
                continue

            # Find the best slot to swap: prefer a meal whose restaurant
            # doesn't uniquely cover any required cuisine. Also prefer slots
            # in the same city as the candidate.
            best_slot = None
            best_score = -1
            for i, mk, base in plan_meals:
                r_cuisines = rest_cuisines.get(base, set())
                # Is this restaurant the sole provider of any required cuisine?
                sole_provider = False
                for req_c in required_cuisine_set:
                    if req_c in r_cuisines:
                        # Check if any OTHER plan restaurant also covers it
                        other_covers = False
                        for _, _, other_base in plan_meals:
                            if other_base == base:
                                continue
                            if req_c in rest_cuisines.get(other_base, set()):
                                other_covers = True
                                break
                        if not other_covers:
                            sole_provider = True
                            break
                if sole_provider:
                    continue  # Don't swap out sole providers

                # Score: prefer same city (2), otherwise any city (1)
                score = 1
                slot_city = rest_city.get(base, "")
                if slot_city.lower() == candidate_city.lower():
                    score = 2
                if score > best_score:
                    best_score = score
                    best_slot = (i, mk, base)

            if best_slot:
                day_idx, meal_key, old_base = best_slot
                plan[day_idx][meal_key] = f"{candidate.name}, {candidate_city}"
                used_names.discard(old_base)
                used_names.add(candidate.name.lower())
                # Update plan_meals list
                plan_meals = [
                    (ci, cmk, candidate.name.lower()) if (ci == day_idx and cmk == meal_key)
                    else (ci, cmk, cb)
                    for ci, cmk, cb in plan_meals
                ]
                log.info(
                    "CUISINE_SWEEP: swapped %r → %r to cover %r",
                    old_base, candidate.name, missing_cuisine,
                )

    def _fix_transport_conflicts(
        self, plan: list[dict[str, Any]], gathered: GatheredData
    ) -> None:
        """Ensure all transport in the plan uses a single mode.

        Respects the transport constraint when resolving conflicts:
        - "no self-driving": convert self-driving legs to taxi/flight
        - "no flight": convert flight legs to self-driving/taxi
        - Otherwise: convert flights to self-driving (cheapest ground option)
        """
        constraint = ""
        if hasattr(self, "_task") and self._task:
            constraint = (self._task.local_constraint.transportation or "").lower()

        has_flight = False
        has_driving = False  # self-driving or taxi
        for day in plan:
            trans = day.get(TRANSPORTATION, NO_DATA).strip()
            if not trans or trans == "-":
                continue
            t_lower = trans.lower()
            if FLIGHT in t_lower:
                has_flight = True
            if SELF_DRIVING in t_lower or "self driving" in t_lower or TAXI in t_lower:
                has_driving = True

        if not (has_flight and has_driving):
            return  # No conflict

        # Decide which mode to keep based on constraint
        if f"no {SELF_DRIVING}" in constraint:
            # Keep flights, convert self-driving to taxi
            convert_mode = SELF_DRIVING
            target_mode = TAXI
        elif f"no {FLIGHT}" in constraint:
            # Keep driving, convert flights to self-driving
            convert_mode = FLIGHT
            target_mode = SELF_DRIVING
        else:
            # Default: convert flights to self-driving
            convert_mode = FLIGHT
            target_mode = SELF_DRIVING

        for day in plan:
            trans = day.get(TRANSPORTATION, NO_DATA).strip()
            if not trans or trans == "-":
                continue
            t_lower = trans.lower()

            needs_convert = False
            if convert_mode == FLIGHT and FLIGHT in t_lower:
                needs_convert = True
            elif convert_mode == SELF_DRIVING and (
                SELF_DRIVING in t_lower or "self driving" in t_lower
            ):
                needs_convert = True

            if not needs_convert:
                continue

            current_city = day.get(CURRENT_CITY, "")
            match = re.match(r"from\s+(.+?)\s+to\s+(.+)", current_city, re.IGNORECASE)
            if not match:
                continue
            origin_part = match.group(1).strip()
            dest_part = match.group(2).strip()

            # Find a matching distance for the target mode
            replaced = False
            for route_key, dist in gathered.distances.items():
                rk_lower = route_key.lower()
                if (
                    origin_part.lower() in rk_lower
                    and dest_part.lower() in rk_lower
                ):
                    if target_mode in dist.mode.lower() or (
                        target_mode == SELF_DRIVING and "self" in dist.mode.lower()
                    ):
                        day[TRANSPORTATION] = self.format_driving(
                            dist, origin_part, dest_part
                        )
                        replaced = True
                        break

            if not replaced:
                # Try any available distance
                for route_key, dist in gathered.distances.items():
                    rk_lower = route_key.lower()
                    if (
                        origin_part.lower() in rk_lower
                        and dest_part.lower() in rk_lower
                    ):
                        day[TRANSPORTATION] = self.format_driving(
                            dist, origin_part, dest_part
                        )
                        replaced = True
                        break

            if not replaced:
                mode_label = "Taxi" if target_mode == TAXI else "Self-driving"
                day[TRANSPORTATION] = (
                    f"{mode_label}, from {origin_part} to {dest_part}"
                )

    def _fill_missing_transport(
        self, plan: list[dict[str, Any]], gathered: GatheredData
    ) -> None:
        """Fill missing transportation on transition days ('from X to Y').

        Respects the transport constraint: won't use flights when "no flight",
        won't use self-driving when "no self-driving".  Searches ALL distance
        modes (taxi **and** self-driving) so taxi-only routes are found.
        """
        constraint = ""
        if hasattr(self, "_task") and self._task:
            constraint = (self._task.local_constraint.transportation or "").lower()

        for day in plan:
            trans = day.get(TRANSPORTATION, NO_DATA).strip()
            if trans and trans != "-":
                continue
            current = day.get(CURRENT_CITY, "")
            match = re.match(r"from\s+(.+?)\s+to\s+(.+)", current, re.IGNORECASE)
            if not match:
                continue
            origin = match.group(1).strip()
            dest = match.group(2).strip()

            filled = False

            # Try flights first (unless "no flight" constraint)
            if "no flight" not in constraint:
                for fk, flights in gathered.flights.items():
                    if origin.lower() in fk.lower() and dest.lower() in fk.lower() and flights:
                        cheapest = min(flights, key=lambda f: f.price)
                        day[TRANSPORTATION] = self.format_flight(cheapest)
                        filled = True
                        break

            # Try distance data (taxi and self-driving)
            if not filled:
                for dk, dist in gathered.distances.items():
                    dk_lower = dk.lower()
                    if origin.lower() not in dk_lower or dest.lower() not in dk_lower:
                        continue
                    # Respect "no self-driving" constraint
                    if f"no {SELF_DRIVING}" in constraint and SELF_DRIVING in dist.mode.lower():
                        continue
                    day[TRANSPORTATION] = self.format_driving(dist, origin, dest)
                    filled = True
                    break

            # Fallback: use mode consistent with constraint
            if not filled:
                if "no self-driving" in constraint:
                    day[TRANSPORTATION] = f"Taxi, from {origin} to {dest}"
                else:
                    day[TRANSPORTATION] = f"Self-driving, from {origin} to {dest}"

    def _fix_return_to_origin(
        self,
        plan: list[dict[str, Any]],
        gathered: GatheredData,
        task: TravelPlannerTask,
    ) -> None:
        """Ensure the last day returns to the trip origin.

        The evaluation requires the last day to be 'from [last_city] to [origin]'.
        If the LLM routed to the destination state instead, fix it.
        Also normalizes the origin city name to match task.org exactly.
        """
        if not plan:
            return

        def _fuzzy_city_eq(a: str, b: str) -> bool:
            """Compare city names leniently (ignore periods, case, whitespace)."""
            a_clean = re.sub(r"[.\s]+", " ", a.lower()).strip()
            b_clean = re.sub(r"[.\s]+", " ", b.lower()).strip()
            return a_clean == b_clean

        # Fix first day's origin city name to match task.org exactly
        first_day = plan[0]
        first_cc = first_day.get(CURRENT_CITY, "")
        first_match = re.match(r"from\s+(.+?)\s+to\s+(.+)", first_cc, re.IGNORECASE)
        if first_match:
            first_origin = first_match.group(1).strip()
            first_dest = first_match.group(2).strip()
            origin = task.org.strip()
            if _fuzzy_city_eq(first_origin, origin) and first_origin != origin:
                first_day[CURRENT_CITY] = f"from {origin} to {first_dest}"
                log.info("FIX_RETURN: normalized first day origin %r → %r", first_origin, origin)

        last_day = plan[-1]
        current = last_day.get(CURRENT_CITY, "")
        match = re.match(r"from\s+(.+?)\s+to\s+(.+)", current, re.IGNORECASE)
        if not match:
            return

        dest = match.group(2).strip()
        from_city = match.group(1).strip()
        origin = task.org.strip()

        # Check if destination matches origin (exact or fuzzy)
        if _fuzzy_city_eq(dest, origin):
            # Destination matches but may have different formatting (e.g., "St Louis" vs "St. Louis")
            if dest != origin:
                last_day[CURRENT_CITY] = f"from {from_city} to {origin}"
                log.info("FIX_RETURN: normalized origin name %r → %r", dest, origin)
            return

        # Last day doesn't return to origin — fix it
        log.info(
            "FIX_RETURN: last day goes to %r instead of %r — fixing",
            dest, origin,
        )
        last_day[CURRENT_CITY] = f"from {from_city} to {origin}"

        # Fix transport string
        transport = last_day.get(TRANSPORTATION, NO_DATA).strip()
        if transport and transport != "-":
            # Replace destination in transport string
            transport_match = re.search(
                r"from\s+(.+?)\s+to\s+(.+?)(?:,|$)", transport
            )
            if transport_match:
                old_dest = transport_match.group(2).strip()
                last_day[TRANSPORTATION] = transport.replace(
                    f"to {old_dest}", f"to {origin}", 1
                )

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
        current = last_day.get(CURRENT_CITY, "")
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
        for key in (*MEAL_KEYS, ATTRACTION, ACCOMMODATION):
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
            for mk in MEAL_KEYS:
                v = day.get(mk, "-").strip()
                if v and v != "-":
                    used_restaurants.add(v.split(",")[0].strip().lower())

        for day in plan:
            current = day.get(CURRENT_CITY, "")
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

            for meal_key in MEAL_KEYS:
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
                    day[meal_key] = f"{rest.name}, {expected_city}"
                    used_restaurants.add(rest.name.lower())

    def _fix_wrong_city_attractions(
        self,
        plan: list[dict[str, Any]],
        gathered: GatheredData,
        task: TravelPlannerTask,
    ) -> None:
        """Fix attractions that reference the wrong city.

        On "from X to Y" days the evaluation expects attractions in city Y.
        Replace any attraction whose city suffix doesn't match with one from
        the expected city.
        """
        used_attrs: set[str] = set()
        for day in plan:
            raw = day.get(ATTRACTION, NO_DATA).strip()
            if not raw or raw == "-":
                continue
            for attr in raw.split(";"):
                attr = attr.strip()
                if attr and attr != "-" and "," in attr:
                    used_attrs.add(attr.rsplit(",", 1)[0].strip().lower())

        for day in plan:
            current = day.get(CURRENT_CITY, "")
            if not current or current.strip() == "-":
                continue
            match = re.match(r"from\s+(.+?)\s+to\s+(.+)", current, re.IGNORECASE)
            if not match:
                continue  # Only fix transition days
            expected_city_raw = match.group(2).strip()
            expected_city = self._match_gathered_city(expected_city_raw, gathered)
            city_attractions = self._find_city_data(gathered.attractions, expected_city)
            if not city_attractions:
                continue

            raw = day.get(ATTRACTION, NO_DATA).strip()
            if not raw or raw == "-":
                continue

            new_parts: list[str] = []
            changed = False
            for attr in raw.split(";"):
                attr = attr.strip()
                if not attr or attr == "-":
                    continue
                if "," not in attr:
                    new_parts.append(attr)
                    continue
                attr_city = attr.rsplit(",", 1)[1].strip()
                if attr_city.lower() == expected_city.lower():
                    new_parts.append(attr)
                    continue
                # Wrong city — pick an unused attraction from the expected city
                replacement = self._pick_unseen_attraction(
                    gathered, expected_city, used_attrs
                )
                if replacement:
                    used_attrs.add(replacement.lower())
                    new_parts.append(f"{replacement}, {expected_city}")
                    changed = True

            if changed:
                day[ATTRACTION] = ";".join(new_parts) if new_parts else "-"

    def _fix_wrong_city_accommodation(
        self,
        plan: list[dict[str, Any]],
        gathered: GatheredData,
        task: TravelPlannerTask,
    ) -> None:
        """Fix accommodation that references the wrong city on transit days.

        On "from X to Y" days, the evaluation expects accommodation in city Y.
        Replace any accommodation whose city suffix doesn't match.
        """
        for day in plan:
            current = day.get(CURRENT_CITY, "")
            if not current or current.strip() == "-":
                continue
            match = re.match(r"from\s+(.+?)\s+to\s+(.+)", current, re.IGNORECASE)
            if not match:
                continue  # Only fix transit days

            expected_city_raw = match.group(2).strip()
            expected_city = self._match_gathered_city(expected_city_raw, gathered)

            acc_val = day.get(ACCOMMODATION, NO_DATA).strip()
            if not acc_val or acc_val == "-":
                continue
            if "," not in acc_val:
                continue

            acc_city = acc_val.rsplit(",", 1)[1].strip()
            if acc_city.lower() == expected_city.lower():
                continue

            # Wrong city — find an accommodation in the expected city
            city_accs = self._find_city_data(gathered.accommodations, expected_city)
            if city_accs:
                best = min(city_accs, key=lambda a: a.price)
                day[ACCOMMODATION] = f"{best.name}, {expected_city}"
                log.info(
                    "FIX_ACC_CITY: replaced %r with %r on day %d",
                    acc_val, day[ACCOMMODATION], day.get("days", 0),
                )

    def _deduplicate_restaurants(
        self,
        plan: list[dict[str, Any]],
        gathered: GatheredData,
    ) -> None:
        """Ensure no restaurant base name appears more than once across all days.

        The evaluation strips city suffixes before comparing, so "Subway, Chicago"
        and "Subway, Rockford" are considered duplicates.  When a duplicate is
        found, replace it with an unused restaurant from the same city.
        """
        seen_base: set[str] = set()  # normalized base names already used
        used_full: set[str] = set()  # lowercase full names already used

        # First pass: collect all used base names
        for day in plan:
            for mk in MEAL_KEYS:
                val = day.get(mk, NO_DATA).strip()
                if not val or val == "-":
                    continue
                base = strip_city_suffix(val).strip()
                key = normalize_name(base)
                seen_base.add(key)
                used_full.add(val.rsplit(",", 1)[0].strip().lower())

        # Second pass: detect and replace duplicates (process in order)
        seen_base.clear()
        for day in plan:
            city = self._infer_stay_city(day, self._task, gathered) if hasattr(self, '_task') else None
            for mk in MEAL_KEYS:
                val = day.get(mk, NO_DATA).strip()
                if not val or val == "-":
                    continue
                base = strip_city_suffix(val).strip()
                key = normalize_name(base)
                if key in seen_base:
                    # Duplicate — find a replacement from the same city
                    if city:
                        city_restaurants = self._find_city_data(gathered.restaurants, city)
                        replacement = None
                        for r in city_restaurants:
                            r_key = normalize_name(r.name)
                            if r_key not in seen_base and r.name.lower() not in used_full:
                                replacement = r
                                break
                        if replacement:
                            day[mk] = f"{replacement.name}, {city}"
                            new_key = normalize_name(replacement.name)
                            seen_base.add(new_key)
                            used_full.add(replacement.name.lower())
                            log.info(
                                "DEDUP_RESTAURANT: replaced %r with %r on day %d",
                                val, day[mk], day.get("days", 0),
                            )
                            continue
                    # No replacement found — clear the slot
                    day[mk] = "-"
                    log.info(
                        "DEDUP_RESTAURANT: cleared duplicate %r on day %d (no replacement)",
                        val, day.get("days", 0),
                    )
                else:
                    seen_base.add(key)

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
            raw = day.get(ATTRACTION, NO_DATA).strip()
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
                day[ATTRACTION] = ";".join(new_parts)
            elif raw != "-":
                day[ATTRACTION] = "-"

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
                if attr.name and attr.name.lower() not in seen:
                    return attr.name
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
        current = day.get(CURRENT_CITY, "")
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
                key_clean = key.lower().replace("'", "").replace("\u2019", "")
                city_clean = city_lower.replace("'", "").replace("\u2019", "")
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
        city_clean = city_lower.replace("'", "").replace("\u2019", "")
        for k, v in data_dict.items():
            k_lower = k.lower()
            if k_lower == city_lower:
                return v
            k_clean = k_lower.replace("'", "").replace("\u2019", "")
            if k_clean == city_clean:
                return v
            if city_clean in k_clean or k_clean in city_clean:
                return v
        return []

    @staticmethod
    def _pick_unused_restaurant(
        restaurants: list[Restaurant],
        used: set[str],
        preferred_cuisines: set[str] | None = None,
    ) -> Restaurant | None:
        """Pick a restaurant not yet used in the plan.

        When *preferred_cuisines* is provided, prioritise restaurants that
        serve at least one of the preferred cuisines before falling back to
        any unused restaurant.
        """
        if preferred_cuisines:
            pref_lower = {c.lower().strip() for c in preferred_cuisines}
            for r in restaurants:
                if not r.name or r.name.lower() in used:
                    continue
                if r.cuisine_set() & pref_lower:
                    return r

        for r in restaurants:
            if r.name and r.name.lower() not in used:
                return r
        return restaurants[0] if restaurants else None

    def _build_task_string(
        self, gathered: GatheredData, task: TravelPlannerTask
    ) -> str:
        """Build structured task description for the LLM."""
        # Determine destination cities from gathered data
        dest_cities = list(gathered.restaurants.keys())
        cities_info = f" visiting {len(dest_cities)} cities: {', '.join(dest_cities)}" if len(dest_cities) > 1 else ""

        parts = [
            f"Build a {task.days}-day travel plan from {task.org} to {task.dest}{cities_info}.",
            f"People: {task.people_number}",
            f"Dates: {task.date}",
        ]

        if task.budget:
            parts.append(f"Budget: ${task.budget}")

        # Constraints
        constraints = []
        if task.local_constraint.room_type:
            constraints.append(f"Room type: {task.local_constraint.room_type}")
        if task.local_constraint.room_rule:
            constraints.append(f"Room rule: {task.local_constraint.room_rule}")
        if task.local_constraint.cuisine:
            constraints.append(f"Cuisine: {task.local_constraint.cuisine}")
        if task.local_constraint.transportation:
            constraints.append(
                f"Transportation: {task.local_constraint.transportation}"
            )
        if constraints:
            parts.append("Constraints: " + ", ".join(constraints))

        parts.append("")
        parts.append("AVAILABLE DATA (use these variable names in your plan):")
        parts.append("")

        # Flights
        for route_key, flights in gathered.flights.items():
            var_name = _safe_var_name(f"flights_{route_key}")
            prices = [f.price for f in flights]
            parts.append(
                f"  {var_name} = <{len(flights)} flights, "
                f"prices ${min(prices):.0f}-${max(prices):.0f}>"
            )

        # Restaurants per city
        for city, restaurants in gathered.restaurants.items():
            var_name = _safe_var_name(f"restaurants_{city}")
            costs = [r.average_cost for r in restaurants]
            cuisines_set: set[str] = set()
            for r in restaurants:
                for c in r.cuisines.split(","):
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
            prices = [a.price for a in accs]
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
                f"  {var_name} = <cost: {dist.cost}, "
                f"duration: {dist.duration}>"
            )

        parts.append("")
        parts.append(
            "IMPORTANT: The data variables above are pre-loaded in your "
            "execution environment. Use them directly in primitive calls."
        )

        return "\n".join(parts)

    def _build_execution_namespace(self) -> dict[str, Any]:
        """Inject gathered data as variables into the execution namespace.

        Entity models are injected directly — their pre-parsed cost attributes
        (price, average_cost, cost) are already floats, so the LLM can safely
        do arithmetic. The models also support dict-like access via __getitem__
        and get() for backward compatibility with raw field names.

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
        * ``distances`` – list of all DistanceInfo objects
        """
        ns: dict[str, Any] = {}
        if not hasattr(self, "_gathered"):
            return ns

        gathered = self._gathered

        # ------------------------------------------------------------------
        # Flights  (canonical + convenience aliases)
        # ------------------------------------------------------------------
        flight_lists: list[list[Flight]] = []
        for route_key, flights in gathered.flights.items():
            var_name = _safe_var_name(f"flights_{route_key}")
            ns[var_name] = flights
            flight_lists.append(flights)

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
        first_restaurants: list[Restaurant] | None = None
        for city, restaurants in gathered.restaurants.items():
            ns[_safe_var_name(f"restaurants_{city}")] = restaurants
            city_alias = _safe_var_name(city)
            ns[f"{city_alias}_restaurants"] = restaurants
            if first_restaurants is None:
                first_restaurants = restaurants
        ns["restaurants"] = first_restaurants or []

        # ------------------------------------------------------------------
        # Accommodations  (canonical + convenience aliases)
        # ------------------------------------------------------------------
        first_accommodations: list[Accommodation] | None = None
        for city, accs in gathered.accommodations.items():
            ns[_safe_var_name(f"accommodations_{city}")] = accs
            city_alias = _safe_var_name(city)
            ns[f"{city_alias}_accommodations"] = accs
            if first_accommodations is None:
                first_accommodations = accs
        ns["accommodations"] = first_accommodations or []

        # ------------------------------------------------------------------
        # Attractions  (canonical + convenience aliases)
        # ------------------------------------------------------------------
        first_attractions: list[Attraction] | None = None
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
        distance_list: list[DistanceInfo] = []
        for route_key, dist in gathered.distances.items():
            ns[_safe_var_name(f"distance_{route_key}")] = dist
            distance_list.append(dist)
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
