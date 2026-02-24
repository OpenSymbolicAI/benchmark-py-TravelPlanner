"""Retrieval Agent: GoalSeeking agent for iterative travel data gathering.

Uses search primitives to collect flights, restaurants, accommodations,
attractions, distances, and cities from the reference database. Iterates
until all necessary data for the trip has been gathered.
"""

from __future__ import annotations

import logging

from opensymbolicai.blueprints import GoalSeeking
from opensymbolicai.core import decomposition, evaluator, primitive
from opensymbolicai.llm import LLM, LLMConfig
from opensymbolicai.models import (
    ExecutionResult,
    GoalContext,
    GoalEvaluation,
    GoalSeekingConfig,
)
from opensymbolicai.observability import ObservabilityConfig

from travelplanner_bench.constants import SELF_DRIVING
from travelplanner_bench.models import (
    Accommodation,
    Attraction,
    DistanceInfo,
    Flight,
    GatheredData,
    Restaurant,
    RetrievalContext,
    TravelPlannerTask,
)
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
    """Iterative travel data gatherer using reference database search primitives."""

    def __init__(
        self,
        llm: LLMConfig | LLM,
        db: ReferenceDatabase,
        max_iterations: int = 5,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            name="RetrievalAgent",
            description="Iterative travel data gatherer using reference database search primitives.",
            config=GoalSeekingConfig(
                max_iterations=max_iterations,
                observability=observability,
            ),
        )
        self._db = db

    # =========================================================================
    # SEARCH PRIMITIVES
    # =========================================================================

    @primitive(read_only=True)
    def search_flights(self, origin: str, destination: str, date: str) -> list[Flight]:
        """Search for flights from origin to destination on a specific date.

        Args:
            origin: Origin city name (e.g., "Sarasota")
            destination: Destination city name (e.g., "Chicago")
            date: Flight date in YYYY-MM-DD format (e.g., "2022-03-22")

        Returns:
            List of Flight objects with: flight_number, price, dep_time, arr_time,
            elapsed_time, date, origin, destination, distance.
            Empty list if no flights found.
        """
        return search_flights(self._db, origin, destination, date)

    @primitive(read_only=True)
    def search_accommodations(self, city: str) -> list[Accommodation]:
        """Search for accommodations in a city.

        Args:
            city: City name (e.g., "Chicago")

        Returns:
            List of Accommodation objects with: name, price, room_type, house_rules,
            min_nights, max_occupancy, review_rate, city.
        """
        return search_accommodations(self._db, city)

    @primitive(read_only=True)
    def search_restaurants(self, city: str) -> list[Restaurant]:
        """Search for restaurants in a city.

        Args:
            city: City name (e.g., "Chicago")

        Returns:
            List of Restaurant objects with: name, average_cost, cuisines, rating, city.
        """
        return search_restaurants(self._db, city)

    @primitive(read_only=True)
    def search_attractions(self, city: str) -> list[Attraction]:
        """Search for attractions in a city.

        Args:
            city: City name (e.g., "Chicago")

        Returns:
            List of Attraction objects with: name, latitude, longitude, address,
            phone, website, city.
        """
        return search_attractions(self._db, city)

    @primitive(read_only=True)
    def get_distance(
        self, origin: str, destination: str, mode: str = SELF_DRIVING
    ) -> DistanceInfo | None:
        """Get distance, duration, and cost between two cities.

        Args:
            origin: Origin city (e.g., "Philadelphia")
            destination: Destination city (e.g., "Pittsburgh")
            mode: Travel mode: SELF_DRIVING or "taxi"

        Returns:
            DistanceInfo with: duration, distance, cost, mode, origin, destination.
            Or None if not found.
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
        dist = self.get_distance("Philadelphia", "Pittsburgh", SELF_DRIVING)
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

            elif prim == "get_distance" and result is not None:
                origin = step.args.get("origin")
                dest = step.args.get("destination")
                mode = step.args.get("mode")
                if origin and dest:
                    mode_val = str(mode.resolved_value) if mode else SELF_DRIVING
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

    def _extract_final_answer(self, context: GoalContext) -> GatheredData:
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
        if task.local_constraint.transportation:
            goal_parts.append(
                f"Transportation constraint: {task.local_constraint.transportation}"
            )
        goal = "\n".join(goal_parts)

        result = self.seek(goal)
        self._seek_result = result  # Expose for logging
        gathered: GatheredData
        if isinstance(result.final_answer, GatheredData):
            gathered = result.final_answer
        else:
            gathered = GatheredData()

        # Safety net: ensure all discovered cities have restaurant/accommodation/
        # attraction data.  The LLM may exhaust iterations before searching every
        # city, especially on 3-city hard tasks.
        self._backfill_city_data(gathered, city_names)

        return gathered

    def _backfill_city_data(
        self, gathered: GatheredData, city_names: list[str]
    ) -> None:
        """Auto-fill missing per-city data for any discovered destination city."""
        # Collect all cities we should have data for
        all_cities: set[str] = set()
        for cities_list in gathered.cities.values():
            for c in cities_list:
                all_cities.add(c.title())
        for c in city_names:
            all_cities.add(c.title())
        # Also include cities already in gathered data
        for c in list(gathered.restaurants.keys()) + list(gathered.accommodations.keys()) + list(gathered.attractions.keys()):
            all_cities.add(c.title())

        for city in all_cities:
            if city not in gathered.restaurants:
                results = search_restaurants(self._db, city)
                if results:
                    gathered.restaurants[city] = results
                    log.info("BACKFILL: found %d restaurants for %s", len(results), city)

            if city not in gathered.accommodations:
                results = search_accommodations(self._db, city)
                if results:
                    gathered.accommodations[city] = results
                    log.info("BACKFILL: found %d accommodations for %s", len(results), city)

            if city not in gathered.attractions:
                results = search_attractions(self._db, city)
                if results:
                    gathered.attractions[city] = results
                    log.info("BACKFILL: found %d attractions for %s", len(results), city)

    def create_context(self, goal: str) -> RetrievalContext:
        ctx = RetrievalContext(goal=goal)
        if hasattr(self, "_task") and self._task:
            ctx.org = self._task.org
            ctx.dest = self._task.dest
            ctx.days = self._task.days
            ctx.date = self._task.date
            ctx.visiting_city_number = self._task.visiting_city_number
        return ctx
