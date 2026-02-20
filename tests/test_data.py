"""Tests for data loading and parsing."""

from travelplanner_bench.data import (
    parse_annotated_plan,
    parse_date_field,
    parse_local_constraint,
    parse_reference_information,
    _extract_budget_from_query,
    _extract_people_from_query,
)


def test_parse_reference_information_list():
    data = [{"Description": "Flights", "Content": "col1\tcol2\nval1\tval2"}]
    result = parse_reference_information(data)
    assert result == data


def test_parse_reference_information_json_string():
    import json
    data = [{"Description": "Flights", "Content": "col1\tcol2"}]
    result = parse_reference_information(json.dumps(data))
    assert result == data


def test_parse_reference_information_empty():
    assert parse_reference_information("") == []
    assert parse_reference_information(None) == []
    assert parse_reference_information([]) == []


def test_parse_local_constraint_dict():
    data = {"cuisine": ["Chinese", "Italian"], "room_type": "entire room"}
    result = parse_local_constraint(data)
    assert result == data


def test_parse_local_constraint_json_string():
    import json
    data = {"cuisine": ["Chinese"], "transportation": "no flight"}
    result = parse_local_constraint(json.dumps(data))
    assert result == data


def test_parse_local_constraint_empty():
    assert parse_local_constraint("") == {}
    assert parse_local_constraint(None) == {}


def test_parse_annotated_plan():
    plan = [{"days": 1, "current_city": "Chicago"}]
    assert parse_annotated_plan(plan) == plan


def test_parse_annotated_plan_string():
    import json
    plan = [{"days": 1, "current_city": "Chicago"}]
    result = parse_annotated_plan(json.dumps(plan))
    assert result == plan


def test_parse_annotated_plan_empty():
    assert parse_annotated_plan("") is None
    assert parse_annotated_plan(None) is None


def test_parse_date_field_list():
    dates = ["2022-03-22", "2022-03-23"]
    assert parse_date_field(dates) == dates


def test_parse_date_field_string():
    import json
    dates = ["2022-03-22", "2022-03-23"]
    result = parse_date_field(json.dumps(dates))
    assert result == dates


def test_parse_date_field_python_repr():
    result = parse_date_field("['2022-03-22', '2022-03-23']")
    assert result == ["2022-03-22", "2022-03-23"]


def test_extract_budget_from_query():
    assert _extract_budget_from_query("budget of $1,900") == 1900
    assert _extract_budget_from_query("budget is set at $2000") == 2000
    assert _extract_budget_from_query("no budget mentioned") == 0


def test_extract_people_from_query():
    assert _extract_people_from_query("for 2 people") == 2
    assert _extract_people_from_query("for 1 person") == 1
    assert _extract_people_from_query("solo trip") == 1
