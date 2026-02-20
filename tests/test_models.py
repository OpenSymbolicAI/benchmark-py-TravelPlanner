"""Tests for TravelPlanner data models."""

from travelplanner_bench.models import (
    DayPlan,
    TravelPlanContext,
    TravelPlannerResult,
    TravelPlannerTask,
)


def test_task_creation():
    task = TravelPlannerTask(
        task_id="tp_0001",
        query="Plan a 3-day trip from Sarasota to Chicago",
        org="Sarasota",
        dest="Chicago",
        days=3,
        date=["2022-03-22", "2022-03-23", "2022-03-24"],
        level="easy",
        budget=1900,
    )
    assert task.task_id == "tp_0001"
    assert task.days == 3
    assert task.level == "easy"
    assert task.budget == 1900
    assert len(task.date) == 3


def test_task_defaults():
    task = TravelPlannerTask(
        task_id="tp_0002",
        query="Test",
        org="A",
        dest="B",
        days=3,
    )
    assert task.level == "easy"
    assert task.people_number == 1
    assert task.budget == 0
    assert task.local_constraint == {}
    assert task.annotated_plan is None


def test_day_plan():
    day = DayPlan(
        days=1,
        current_city="from Sarasota to Chicago",
        transportation="Flight Number: F123",
        breakfast="-",
        attraction="Navy Pier;Millennium Park",
        lunch="The Black Pearl, Chicago",
        dinner="Giordano's, Chicago",
        accommodation="Cozy Studio, Chicago",
    )
    assert day.days == 1
    assert "Navy Pier" in day.attraction


def test_context_creation():
    ctx = TravelPlanContext(
        goal="Plan a trip",
        query="Plan a 3-day trip",
        org="Sarasota",
        dest="Chicago",
        days=3,
    )
    assert ctx.org == "Sarasota"
    assert ctx.plan_complete is False
    assert ctx.data_gathered is False
    assert ctx.iteration_count == 0


def test_result_defaults():
    result = TravelPlannerResult(task_id="tp_0001")
    assert result.plan_delivered is False
    assert result.final_pass is False
    assert result.commonsense_micro == 0.0
    assert result.hard_micro == 0.0


def test_result_serialization():
    result = TravelPlannerResult(
        task_id="tp_0001",
        plan_delivered=True,
        within_sandbox=True,
        complete_info=True,
        commonsense_micro=0.75,
        final_pass=False,
    )
    d = result.model_dump()
    assert d["task_id"] == "tp_0001"
    assert d["plan_delivered"] is True
    assert d["commonsense_micro"] == 0.75
