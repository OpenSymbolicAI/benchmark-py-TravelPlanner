"""Retrieval Agent: GoalSeeking agent for iterative travel data gathering.

Uses search primitives to collect flights, restaurants, accommodations,
attractions, distances, and cities from the reference database. Iterates
until all necessary data for the trip has been gathered.
"""

from __future__ import annotations

import logging
from typing import Any

from opensymbolicai.blueprints import GoalSeeking
from opensymbolicai.core import decomposition, evaluator, primitive
from opensymbolicai.llm import LLM, LLMConfig
from opensymbolicai.models import (
    ExecutionResult,
    GoalContext,
    GoalEvaluation,
    GoalSeekingConfig,
)

from travelplanner_bench.models import GatheredData, RetrievalContext, TravelPlannerTask
from travelplanner_bench.tools import (
    ReferenceDatabase,
    get_distance,
    search_accommodations,
    search_attractions,
    search_cities,
    search_flights,
    search_restaurants,
)

log = logging.getLogger(__name__)


class RetrievalAgent(GoalSeeking):
    """Gathers travel information by iteratively searching the reference database.

    Searches for flights, restaurants, accommodations, attractions, distances,
    and cities. Each iteration gathers data for one aspect of the trip.
    Continues until all necessary data for the trip has been collected:
    - Outbound and return flights (or distance info for driving)
    - Restaurants, accommodations, and attractions for each destination city
    - Inter-city distances for multi-city trips
    - City lists for state-level destinations

    RULES:
    1. Search flights for EACH leg: origin->destination on departure date,
       destination->origin on return date.
    2. Search restaurants, accommodations, and attractions for EACH destination city.
    3. For multi-city trips: search cities in the state first, then gather data
       per city, and get inter-city distances.
    4. Use get_distance for self-driving or taxi cost between cities.
    """

    def __init__(
        self,
        llm: LLMConfig | LLM,
        db: ReferenceDatabase,
        max_iterations: int = 5,
    ) -> None:
        super().__init__(
            llm=llm,
            name="RetrievalAgent",
            description=(
                "Gathers travel information by iteratively searching the reference "
                "database. Searches for flights, restaurants, accommodations, "
                "attractions, distances, and cities. Each iteration gathers data "
                "for one aspect of the trip. Continues until all necessary data "
                "has been collected.\n\n"
                "RULES:\n"
                "1. Search flights for EACH leg: origin->destination and back.\n"
                "2. Search restaurants, accommodations, and attractions for EACH "
                "destination city.\n"
                "3. For multi-city trips (visiting_city_number > 1): search cities "
                "in the state first, then gather data per city, and get inter-city "
                "distances.\n"
                "4. Use get_distance for driving/taxi cost between cities.\n"
                "5. Check the Accumulated Knowledge to see what has already been "
                "gathered and what is still missing."
            ),
            config=GoalSeekingConfig(max_iterations=max_iterations),
        )
        self._db = db

    # =========================================================================
    # SEARCH PRIMITIVES
    # =========================================================================

    @primitive(read_only=True)
    def search_flights(self, origin: str, destination: str, date: str) -> list[dict]:
        """Search for flights from origin to destination on a specific date.

        Args:
            origin: Origin city name (e.g., "Sarasota")
            destination: Destination city name (e.g., "Chicago")
            date: Flight date in YYYY-MM-DD format (e.g., "2022-03-22")

        Returns:
            List of flights with: Flight Number, Price, DepTime, ArrTime,
            ActualElapsedTime, FlightDate, OriginCityName, DestCityName, Distance.
            Empty list if no flights found.
        """
        return search_flights(self._db, origin, destination, date)

    @primitive(read_only=True)
    def search_accommodations(self, city: str) -> list[dict]:
        """Search for accommodations in a city.

        Args:
            city: City name (e.g., "Chicago")

        Returns:
            List of accommodations with: NAME, price, room type, house_rules,
            minimum nights, maximum occupancy, review rate number, city.
        """
        return search_accommodations(self._db, city)

    @primitive(read_only=True)
    def search_restaurants(self, city: str) -> list[dict]:
        """Search for restaurants in a city.

        Args:
            city: City name (e.g., "Chicago")

        Returns:
            List of restaurants with: Name, Average Cost, Cuisines,
            Aggregate Rating, City.
        """
        return search_restaurants(self._db, city)

    @primitive(read_only=True)
    def search_attractions(self, city: str) -> list[dict]:
        """Search for attractions in a city.

        Args:
            city: City name (e.g., "Chicago")

        Returns:
            List of attractions with: Name, Latitude, Longitude, Address,
            Phone, Website, City.
        """
        return search_attractions(self._db, city)

    @primitive(read_only=True)
    def get_distance(
        self, origin: str, destination: str, mode: str = "self-driving"
    ) -> dict | None:
        """Get distance, duration, and cost between two cities.

        Args:
            origin: Origin city (e.g., "Philadelphia")
            destination: Destination city (e.g., "Pittsburgh")
            mode: Travel mode: "self-driving" or "taxi"

        Returns:
            Dict with duration, distance, cost, or None if not found.
        """
        return get_distance(self._db, origin, destination, mode)

    @primitive(read_only=True)
    def search_cities(self, state: str) -> list[str]:
        """Get list of cities in a state.

        Args:
            state: State name (e.g., "Pennsylvania")

        Returns:
            List of city names in the state.
        """
        return search_cities(self._db, state)

    # =========================================================================
    # DECOMPOSITION EXAMPLES
    # =========================================================================

    @decomposition(
        intent=(
            "Gather all information for a single-city 3-day trip from "
            "Sarasota to Chicago, dates 2022-03-22 to 2022-03-24"
        ),
        expanded_intent=(
            "For a single-city trip: search flights out and back, "
            "then search restaurants, accommodations, and attractions "
            "in the destination city. All in one iteration."
        ),
    )
    def _ex_single_city_gather(self) -> str:
        flights_out = self.search_flights("Sarasota", "Chicago", "2022-03-22")
        flights_back = self.search_flights("Chicago", "Sarasota", "2022-03-24")
        restaurants = self.search_restaurants("Chicago")
        accommodations = self.search_accommodations("Chicago")
        attractions = self.search_attractions("Chicago")
        return "Single-city data gathered."

    @decomposition(
        intent=(
            "Gather information for a multi-city 5-day trip visiting "
            "2 cities in Pennsylvania from New York, dates 2022-03-22 "
            "to 2022-03-26"
        ),
        expanded_intent=(
            "For multi-city trips: first search cities in the state, "
            "then search flights for the first leg, search data per city "
            "(restaurants, accommodations, attractions), get inter-city "
            "distances, and search return flights."
        ),
    )
    def _ex_multi_city_gather(self) -> str:
        cities = self.search_cities("Pennsylvania")
        flights_out = self.search_flights("New York", "Philadelphia", "2022-03-22")
        rest_philly = self.search_restaurants("Philadelphia")
        acc_philly = self.search_accommodations("Philadelphia")
        attr_philly = self.search_attractions("Philadelphia")
        dist = self.get_distance("Philadelphia", "Pittsburgh", "self-driving")
        rest_pitt = self.search_restaurants("Pittsburgh")
        acc_pitt = self.search_accommodations("Pittsburgh")
        attr_pitt = self.search_attractions("Pittsburgh")
        flights_back = self.search_flights("Pittsburgh", "New York", "2022-03-26")
        return "Multi-city data gathered for Philadelphia and Pittsburgh."

    # =========================================================================
    # GOALSEEKING OVERRIDES
    # =========================================================================

    def update_context(
        self, context: GoalContext, execution_result: ExecutionResult
    ) -> None:
        assert isinstance(context, RetrievalContext)
        gathered = context.gathered

        for step in execution_result.trace.steps:
            if not step.success:
                continue
            prim = step.primitive_called
            result = step.result_value

            if prim == "search_flights" and isinstance(result, list) and result:
                origin = step.args.get("origin")
                dest = step.args.get("destination")
                date = step.args.get("date")
                if origin and dest and date:
                    key = f"{origin.resolved_value}->{dest.resolved_value} on {date.resolved_value}"
                    gathered.flights[key] = result
                    # Track direction
                    org_val = str(origin.resolved_value).lower()
                    dest_val = str(dest.resolved_value).lower()
                    ctx_org = context.org.lower()
                    if org_val == ctx_org or ctx_org in org_val:
                        context.has_outbound_flights = True
                    if dest_val == ctx_org or ctx_org in dest_val:
                        context.has_return_flights = True

            elif prim == "search_restaurants" and isinstance(result, list) and result:
                city = step.args.get("city")
                if city:
                    city_name = str(city.resolved_value)
                    gathered.restaurants[city_name] = result
                    context.restaurants_per_city[city_name] = len(result)
                    if city_name not in context.destination_cities:
                        context.destination_cities.append(city_name)

            elif prim == "search_accommodations" and isinstance(result, list) and result:
                city = step.args.get("city")
                if city:
                    city_name = str(city.resolved_value)
                    gathered.accommodations[city_name] = result
                    context.accommodations_per_city[city_name] = len(result)

            elif prim == "search_attractions" and isinstance(result, list) and result:
                city = step.args.get("city")
                if city:
                    city_name = str(city.resolved_value)
                    gathered.attractions[city_name] = result
                    context.attractions_per_city[city_name] = len(result)

            elif prim == "get_distance" and isinstance(result, dict):
                origin = step.args.get("origin")
                dest = step.args.get("destination")
                mode = step.args.get("mode")
                if origin and dest:
                    mode_val = str(mode.resolved_value) if mode else "self-driving"
                    key = f"{origin.resolved_value}->{dest.resolved_value} ({mode_val})"
                    gathered.distances[key] = result
                    context.has_distances = True

            elif prim == "search_cities" and isinstance(result, list) and result:
                state_arg = step.args.get("state")
                if state_arg:
                    gathered.cities[str(state_arg.resolved_value)] = result
                    context.has_cities_list = True

    @evaluator
    def check_data_complete(
        self, goal: str, context: GoalContext
    ) -> GoalEvaluation:
        """Data is complete when we have flights + per-city data."""
        assert isinstance(context, RetrievalContext)

        # Need at least some flight or distance data for travel
        has_travel = context.has_outbound_flights or context.has_distances

        # Need restaurants, accommodations, attractions for at least one city
        has_city_data = bool(
            context.restaurants_per_city
            and context.accommodations_per_city
            and context.attractions_per_city
        )

        return GoalEvaluation(goal_achieved=has_travel and has_city_data)

    def _extract_final_answer(self, context: GoalContext) -> Any:
        assert isinstance(context, RetrievalContext)
        return context.gathered

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def gather(self, task: TravelPlannerTask) -> GatheredData:
        """Gather all necessary travel data for a task.

        Returns:
            GatheredData with flights, restaurants, accommodations, etc.
        """
        # Set context metadata for the evaluator
        self._task = task

        # Pre-resolve city names so the LLM knows exactly which cities to search
        city_names = search_cities(self._db, task.dest)

        # Pre-resolve available flight routes so the LLM knows exact legs
        flight_routes: list[str] = []
        for (orig, dest, date) in self._db.flights:
            flight_routes.append(f"{orig.title()} -> {dest.title()} on {date}")

        goal_parts = [
            f"Gather all travel information needed for this trip:",
            f"Origin: {task.org}",
            f"Destination: {task.dest}",
            f"Dates: {task.date}",
            f"Days: {task.days}",
            f"Cities to visit: {task.visiting_city_number}",
        ]
        if city_names:
            goal_parts.append(
                f"Destination cities: {', '.join(c.title() for c in city_names)}"
            )
        if flight_routes:
            goal_parts.append(
                f"Available flight routes: {'; '.join(flight_routes)}"
            )
        # Pre-resolve available distance routes
        distance_routes: list[str] = []
        for (orig, dest, mode) in self._db.distances:
            distance_routes.append(f"{orig.title()} -> {dest.title()} ({mode})")
        if distance_routes:
            goal_parts.append(
                f"Available distance routes: {'; '.join(distance_routes)}"
            )
        if task.local_constraint.get("transportation"):
            goal_parts.append(
                f"Transportation constraint: {task.local_constraint['transportation']}"
            )
        goal = "\n".join(goal_parts)

        result = self.seek(goal)
        if isinstance(result.final_answer, GatheredData):
            return result.final_answer
        return GatheredData()

    def create_context(self, goal: str) -> RetrievalContext:
        ctx = RetrievalContext(goal=goal)
        if hasattr(self, "_task") and self._task:
            ctx.org = self._task.org
            ctx.dest = self._task.dest
            ctx.days = self._task.days
            ctx.date = self._task.date
            ctx.visiting_city_number = self._task.visiting_city_number
        return ctx
