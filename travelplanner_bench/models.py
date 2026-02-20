"""Data models for the TravelPlanner benchmark."""

from __future__ import annotations

from typing import Any

from opensymbolicai.models import GoalContext
from pydantic import BaseModel, Field


class TravelPlannerTask(BaseModel):
    """A single TravelPlanner benchmark task."""

    task_id: str = Field(..., description="Index-based task ID")
    query: str = Field(..., description="Natural language travel request")
    org: str = Field(..., description="Origin city")
    dest: str = Field(..., description="Destination city or state")
    days: int = Field(..., description="Trip duration (3, 5, or 7)")
    date: list[str] = Field(default_factory=list, description="Travel dates")
    level: str = Field(default="easy", description="Difficulty: easy, medium, hard")
    visiting_city_number: int = Field(default=1, description="Number of cities to visit")
    people_number: int = Field(default=1, description="Number of travelers")
    local_constraint: dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed constraints: cuisine, room_type, room_rule, transportation",
    )
    budget: int = Field(default=0, description="Total budget in dollars")
    reference_information: list[dict[str, str]] = Field(
        default_factory=list, description="Reference info entries with Description/Content",
    )
    annotated_plan: list[dict[str, Any]] | None = Field(
        default=None, description="Ground truth plan (None for test split)",
    )


class DayPlan(BaseModel):
    """A single day in the travel itinerary."""

    days: int = Field(..., description="Day number")
    current_city: str = Field(..., description="'from X to Y' or city name")
    transportation: str = Field(default="-", description="Flight info, taxi, self-driving, or '-'")
    breakfast: str = Field(default="-", description="Restaurant name or '-'")
    attraction: str = Field(default="-", description="Semicolon-separated attraction names or '-'")
    lunch: str = Field(default="-", description="Restaurant name or '-'")
    dinner: str = Field(default="-", description="Restaurant name or '-'")
    accommodation: str = Field(default="-", description="Accommodation name or '-'")


# =========================================================================
# Gathered Data (output of retrieval, input to plan assembler)
# =========================================================================


class GatheredData(BaseModel):
    """Structured container for all retrieved data from the reference database.

    Organized by city for easy access during plan assembly.
    """

    # Flights keyed by "(origin, dest, date)" as string
    flights: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    # Restaurants keyed by city name
    restaurants: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    # Accommodations keyed by city name
    accommodations: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    # Attractions keyed by city name
    attractions: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    # Distances keyed by "(origin, dest, mode)" as string
    distances: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Cities keyed by state name
    cities: dict[str, list[str]] = Field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary for prompt inclusion."""
        parts = []
        total_flights = sum(len(v) for v in self.flights.values())
        if total_flights:
            routes = ", ".join(self.flights.keys())
            parts.append(f"Flights: {total_flights} across routes [{routes}]")
        for city, rests in self.restaurants.items():
            parts.append(f"Restaurants in {city}: {len(rests)}")
        for city, accs in self.accommodations.items():
            parts.append(f"Accommodations in {city}: {len(accs)}")
        for city, attrs in self.attractions.items():
            parts.append(f"Attractions in {city}: {len(attrs)}")
        if self.distances:
            parts.append(f"Distances: {len(self.distances)} routes")
        for state, city_list in self.cities.items():
            parts.append(f"Cities in {state}: {city_list}")
        return "\n".join(parts) if parts else "No data gathered yet."


# =========================================================================
# Contexts (introspection boundaries)
# =========================================================================


class RetrievalContext(GoalContext):
    """Context for the RetrievalAgent (introspection boundary).

    Tracks what data has been gathered so the evaluator can determine
    when enough information is available.
    """

    # Task metadata needed for evaluator
    org: str = ""
    dest: str = ""
    days: int = 0
    date: list[str] = Field(default_factory=list)
    visiting_city_number: int = 1

    # Gathered data counts (for evaluator / prompt context)
    has_outbound_flights: bool = False
    has_return_flights: bool = False
    destination_cities: list[str] = Field(default_factory=list)
    restaurants_per_city: dict[str, int] = Field(default_factory=dict)
    accommodations_per_city: dict[str, int] = Field(default_factory=dict)
    attractions_per_city: dict[str, int] = Field(default_factory=dict)
    has_distances: bool = False
    has_cities_list: bool = False

    # The actual gathered data (accumulated across iterations)
    gathered: GatheredData = Field(default_factory=GatheredData)


class TravelPlanContext(GoalContext):
    """Context for the top-level TravelPlannerAgent (introspection boundary).

    Tracks orchestration state: have we gathered data? Have we built a plan?
    """

    # Task metadata
    query: str = ""
    org: str = ""
    dest: str = ""
    days: int = 0
    people_number: int = 1
    budget: int = 0
    local_constraint: dict[str, Any] = Field(default_factory=dict)

    # Phase tracking
    data_gathered: bool = False
    gathered_summary: str = ""
    plan_built: bool = False
    current_plan: list[dict[str, Any]] | None = None
    plan_complete: bool = False
    solver_error: str | None = None


class TravelPlannerResult(BaseModel):
    """Result of evaluating a single TravelPlanner task."""

    task_id: str = Field(..., description="Task identifier")
    query: str = ""
    level: str = "easy"
    days: int = 3

    # Agent output
    plan: list[dict[str, Any]] | None = None
    plan_delivered: bool = False

    # Commonsense constraint results (8 checks)
    within_sandbox: bool = False
    complete_info: bool = False
    within_current_city: bool = False
    reasonable_city_route: bool = False
    diverse_restaurants: bool = False
    diverse_attractions: bool = False
    non_conflicting_transport: bool = False
    valid_accommodation: bool = False

    # Hard constraint results (None = not applicable)
    budget_ok: bool | None = None
    room_rule_ok: bool | None = None
    room_type_ok: bool | None = None
    cuisine_ok: bool | None = None
    transportation_ok: bool | None = None

    # Aggregate scores
    commonsense_micro: float = 0.0
    commonsense_macro: bool = False
    hard_micro: float = 0.0
    hard_macro: bool = False
    final_pass: bool = False

    # Execution metadata
    model: str = ""
    framework: str = "opensymbolicai"
    iterations: int = 0
    wall_time_seconds: float = 0.0
    error: str | None = None
