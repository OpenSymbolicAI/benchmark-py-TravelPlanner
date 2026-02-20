"""Tests for TravelPlanner evaluation constraints."""

from travelplanner_bench.evaluation import (
    check_budget,
    check_complete_info,
    check_cuisine,
    check_diverse_attractions,
    check_diverse_restaurants,
    check_non_conflicting_transport,
    check_reasonable_city_route,
    check_room_rule,
    check_room_type,
    check_transportation_constraint,
    check_valid_accommodation,
    check_within_current_city,
    check_within_sandbox,
    evaluate_plan,
)
from travelplanner_bench.models import TravelPlannerTask
from travelplanner_bench.tools import ReferenceDatabase

# Reuse sample data from test_tools
from tests.test_tools import SAMPLE_REF_INFO


def _make_db() -> ReferenceDatabase:
    return ReferenceDatabase(SAMPLE_REF_INFO)


def _make_task(**overrides) -> TravelPlannerTask:
    defaults = {
        "task_id": "tp_test",
        "query": "Plan a 3-day trip from Sarasota to Chicago",
        "org": "Sarasota",
        "dest": "Chicago",
        "days": 3,
        "date": ["2022-03-22", "2022-03-23", "2022-03-24"],
        "level": "easy",
        "budget": 1900,
        "people_number": 1,
    }
    defaults.update(overrides)
    return TravelPlannerTask(**defaults)


def _make_good_plan() -> list[dict]:
    return [
        {
            "days": 1,
            "current_city": "from Sarasota to Chicago",
            "transportation": "Flight Number: F3600033, from Sarasota to Chicago",
            "breakfast": "-",
            "attraction": "Navy Pier",
            "lunch": "The Black Pearl, Chicago",
            "dinner": "Giordano's, Chicago",
            "accommodation": "Cozy Studio in Lincoln Park, Chicago",
        },
        {
            "days": 2,
            "current_city": "Chicago",
            "transportation": "-",
            "breakfast": "Portillo's, Chicago",
            "attraction": "Millennium Park",
            "lunch": "-",
            "dinner": "-",
            "accommodation": "Cozy Studio in Lincoln Park, Chicago",
        },
        {
            "days": 3,
            "current_city": "from Chicago to Sarasota",
            "transportation": "Flight Number: F3600078, from Chicago to Sarasota",
            "breakfast": "-",
            "attraction": "-",
            "lunch": "-",
            "dinner": "-",
            "accommodation": "-",
        },
    ]


# ===========================================================================
# Commonsense constraint tests
# ===========================================================================


class TestWithinSandbox:
    def test_good_plan(self):
        db = _make_db()
        task = _make_task()
        assert check_within_sandbox(_make_good_plan(), db, task) is True

    def test_hallucinated_restaurant(self):
        db = _make_db()
        task = _make_task()
        plan = _make_good_plan()
        plan[0]["lunch"] = "Fake Restaurant, Chicago"
        assert check_within_sandbox(plan, db, task) is False

    def test_hallucinated_flight(self):
        db = _make_db()
        task = _make_task()
        plan = _make_good_plan()
        plan[0]["transportation"] = "Flight Number: FAKE123"
        assert check_within_sandbox(plan, db, task) is False


class TestCompleteInfo:
    def test_good_plan(self):
        task = _make_task()
        assert check_complete_info(_make_good_plan(), task) is True

    def test_wrong_number_of_days(self):
        task = _make_task()
        plan = _make_good_plan()[:2]  # Only 2 days for a 3-day trip
        assert check_complete_info(plan, task) is False


class TestWithinCurrentCity:
    def test_good_plan(self):
        db = _make_db()
        task = _make_task()
        assert check_within_current_city(_make_good_plan(), db, task) is True


class TestReasonableCityRoute:
    def test_good_plan(self):
        task = _make_task()
        assert check_reasonable_city_route(_make_good_plan(), task) is True

    def test_wrong_origin(self):
        task = _make_task()
        plan = _make_good_plan()
        plan[0]["current_city"] = "from Boston to Chicago"
        assert check_reasonable_city_route(plan, task) is False


class TestDiverseRestaurants:
    def test_good_plan(self):
        assert check_diverse_restaurants(_make_good_plan()) is True

    def test_duplicate_restaurant(self):
        plan = _make_good_plan()
        plan[1]["breakfast"] = "The Black Pearl, Chicago"  # Already used in day 1 lunch
        assert check_diverse_restaurants(plan) is False


class TestDiverseAttractions:
    def test_good_plan(self):
        assert check_diverse_attractions(_make_good_plan()) is True

    def test_duplicate_attraction(self):
        plan = _make_good_plan()
        plan[1]["attraction"] = "Navy Pier"  # Already used in day 1
        assert check_diverse_attractions(plan) is False


class TestNonConflictingTransport:
    def test_flights_only(self):
        assert check_non_conflicting_transport(_make_good_plan()) is True

    def test_mixed_transport(self):
        plan = _make_good_plan()
        plan[1]["transportation"] = "Self-driving from Chicago to somewhere"
        assert check_non_conflicting_transport(plan) is False


class TestValidAccommodation:
    def test_good_plan(self):
        db = _make_db()
        assert check_valid_accommodation(_make_good_plan(), db) is True


# ===========================================================================
# Hard constraint tests
# ===========================================================================


class TestBudget:
    def test_no_budget(self):
        db = _make_db()
        task = _make_task(budget=0)
        assert check_budget(_make_good_plan(), db, task) is None

    def test_within_budget(self):
        db = _make_db()
        task = _make_task(budget=1900)
        result = check_budget(_make_good_plan(), db, task)
        assert result is True

    def test_over_budget(self):
        db = _make_db()
        task = _make_task(budget=100)  # Very low budget
        result = check_budget(_make_good_plan(), db, task)
        assert result is False


class TestRoomRule:
    def test_no_constraint(self):
        db = _make_db()
        task = _make_task()
        assert check_room_rule(_make_good_plan(), db, task) is None

    def test_matching_rule(self):
        db = _make_db()
        task = _make_task(local_constraint={"room_rule": "No smoking"})
        assert check_room_rule(_make_good_plan(), db, task) is True


class TestRoomType:
    def test_no_constraint(self):
        db = _make_db()
        task = _make_task()
        assert check_room_type(_make_good_plan(), db, task) is None

    def test_matching_type(self):
        db = _make_db()
        task = _make_task(local_constraint={"room_type": "entire home"})
        assert check_room_type(_make_good_plan(), db, task) is True


class TestCuisine:
    def test_no_constraint(self):
        db = _make_db()
        task = _make_task()
        assert check_cuisine(_make_good_plan(), db, task) is None

    def test_cuisine_found(self):
        db = _make_db()
        task = _make_task(local_constraint={"cuisine": ["Italian"]})
        assert check_cuisine(_make_good_plan(), db, task) is True

    def test_cuisine_missing(self):
        db = _make_db()
        task = _make_task(local_constraint={"cuisine": ["French"]})
        assert check_cuisine(_make_good_plan(), db, task) is False


class TestTransportationConstraint:
    def test_no_constraint(self):
        task = _make_task()
        assert check_transportation_constraint(_make_good_plan(), task) is None

    def test_no_flight_constraint_violated(self):
        task = _make_task(local_constraint={"transportation": "no flight"})
        assert check_transportation_constraint(_make_good_plan(), task) is False

    def test_no_self_driving_ok(self):
        task = _make_task(local_constraint={"transportation": "no self-driving"})
        assert check_transportation_constraint(_make_good_plan(), task) is True


# ===========================================================================
# Aggregate evaluation test
# ===========================================================================


class TestEvaluatePlan:
    def test_good_plan_passes(self):
        db = _make_db()
        task = _make_task()
        result = evaluate_plan(_make_good_plan(), task, db)
        assert result.plan_delivered is True
        assert result.commonsense_micro > 0.5

    def test_none_plan(self):
        db = _make_db()
        task = _make_task()
        result = evaluate_plan(None, task, db)
        assert result.plan_delivered is False
        assert result.final_pass is False

    def test_empty_plan(self):
        db = _make_db()
        task = _make_task()
        result = evaluate_plan([], task, db)
        assert result.plan_delivered is False
