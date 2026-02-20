"""Tests for ReferenceDatabase and search tools."""

from travelplanner_bench.tools import (
    ReferenceDatabase,
    get_distance,
    search_accommodations,
    search_attractions,
    search_cities,
    search_flights,
    search_restaurants,
)

SAMPLE_REF_INFO = [
    {
        "Description": "Flight from Sarasota to Chicago on 2022-03-22",
        "Content": (
            "Flight Number\tPrice\tDepTime\tArrTime\tActualElapsedTime\t"
            "FlightDate\tOriginCityName\tDestCityName\tDistance\n"
            "F3600033\t234\t15:30\t17:30\t120\t2022-03-22\tSarasota\tChicago\t1000"
        ),
    },
    {
        "Description": "Flight from Chicago to Sarasota on 2022-03-24",
        "Content": (
            "Flight Number\tPrice\tDepTime\tArrTime\tActualElapsedTime\t"
            "FlightDate\tOriginCityName\tDestCityName\tDistance\n"
            "F3600078\t198\t10:00\t14:00\t240\t2022-03-24\tChicago\tSarasota\t1000"
        ),
    },
    {
        "Description": "Restaurants in Chicago",
        "Content": (
            "Name\tAverage Cost\tCuisines\tAggregate Rating\tCity\n"
            "The Black Pearl\t63\tCafe, Bakery\t4.1\tChicago\n"
            "Giordano's\t40\tItalian, Pizza\t4.5\tChicago\n"
            "Portillo's\t15\tAmerican, Fast Food\t4.3\tChicago"
        ),
    },
    {
        "Description": "Accommodations in Chicago",
        "Content": (
            "NAME\tprice\troom type\thouse_rules\tminimum nights\t"
            "maximum occupancy\treview rate number\tcity\n"
            "Cozy Studio in Lincoln Park\t150\tEntire home/apt\t"
            "No smoking\t2\t4\t4.5\tChicago"
        ),
    },
    {
        "Description": "Attractions in Chicago",
        "Content": (
            "Name\tLatitude\tLongitude\tAddress\tPhone\tWebsite\tCity\n"
            "Navy Pier\t41.89\t-87.60\t600 E Grand Ave\t555-1234\twww.navypier.org\tChicago\n"
            "Millennium Park\t41.88\t-87.62\t201 E Randolph St\t555-5678\twww.mp.org\tChicago"
        ),
    },
    {
        "Description": "Self-driving from Chicago to Milwaukee",
        "Content": (
            "duration\tdistance\tcost\n"
            "1 hour 30 mins\t150 km\t7.50"
        ),
    },
    {
        "Description": "Cities in Illinois",
        "Content": "Chicago\nSpringfield\nPeoria",
    },
]


def _make_db() -> ReferenceDatabase:
    return ReferenceDatabase(SAMPLE_REF_INFO)


def test_parse_flights():
    db = _make_db()
    assert ("sarasota", "chicago", "2022-03-22") in db.flights
    flights = db.flights[("sarasota", "chicago", "2022-03-22")]
    assert len(flights) == 1
    assert flights[0]["Flight Number"] == "F3600033"
    assert flights[0]["Price"] == "234"


def test_parse_restaurants():
    db = _make_db()
    assert "chicago" in db.restaurants
    restaurants = db.restaurants["chicago"]
    assert len(restaurants) == 3
    names = {r["Name"] for r in restaurants}
    assert "The Black Pearl" in names
    assert "Giordano's" in names


def test_parse_accommodations():
    db = _make_db()
    assert "chicago" in db.accommodations
    accs = db.accommodations["chicago"]
    assert len(accs) == 1
    assert accs[0]["NAME"] == "Cozy Studio in Lincoln Park"


def test_parse_attractions():
    db = _make_db()
    assert "chicago" in db.attractions
    attrs = db.attractions["chicago"]
    assert len(attrs) == 2
    names = {a["Name"] for a in attrs}
    assert "Navy Pier" in names


def test_parse_distances():
    db = _make_db()
    assert ("chicago", "milwaukee", "self-driving") in db.distances


def test_parse_cities():
    db = _make_db()
    assert "illinois" in db.cities
    assert "Chicago" in db.cities["illinois"]


def test_sandbox_sets():
    db = _make_db()
    assert "F3600033" in db.all_flight_numbers
    assert "F3600078" in db.all_flight_numbers
    assert "The Black Pearl" in db.all_restaurant_names
    assert "Cozy Studio in Lincoln Park" in db.all_accommodation_names
    assert "Navy Pier" in db.all_attraction_names


def test_search_flights():
    db = _make_db()
    results = search_flights(db, "Sarasota", "Chicago", "2022-03-22")
    assert len(results) == 1
    assert results[0]["Flight Number"] == "F3600033"


def test_search_flights_not_found():
    db = _make_db()
    results = search_flights(db, "Boston", "Chicago", "2022-03-22")
    assert results == []


def test_search_restaurants():
    db = _make_db()
    results = search_restaurants(db, "Chicago")
    assert len(results) == 3


def test_search_restaurants_case_insensitive():
    db = _make_db()
    results = search_restaurants(db, "chicago")
    assert len(results) == 3


def test_search_accommodations():
    db = _make_db()
    results = search_accommodations(db, "Chicago")
    assert len(results) == 1
    assert results[0]["NAME"] == "Cozy Studio in Lincoln Park"


def test_search_attractions():
    db = _make_db()
    results = search_attractions(db, "Chicago")
    assert len(results) == 2


def test_get_distance():
    db = _make_db()
    result = get_distance(db, "Chicago", "Milwaukee", "self-driving")
    assert result is not None
    assert result["distance"] == "150 km"


def test_get_distance_not_found():
    db = _make_db()
    result = get_distance(db, "Boston", "Chicago", "self-driving")
    assert result is None


def test_search_cities():
    db = _make_db()
    cities = search_cities(db, "Illinois")
    assert "Chicago" in cities
    assert "Springfield" in cities
