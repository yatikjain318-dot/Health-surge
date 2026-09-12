"""
Unit and Integration Tests for Emergency Bed Finder & Routing Module.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.emergency_service import (
    get_emergency_service,
    haversine_distance,
    calculate_travel_eta
)

client = TestClient(app)


def test_haversine_distance_accuracy():
    # Known distance: Jaipur (26.9124, 75.7873) to Jodhpur (26.2389, 73.0243) is ~285-290 km
    dist = haversine_distance(26.9124, 75.7873, 26.2389, 73.0243)
    assert 280.0 <= dist <= 295.0

    # Same location distance is 0.0
    zero_dist = haversine_distance(26.9124, 75.7873, 26.9124, 75.7873)
    assert zero_dist == 0.0


def test_calculate_travel_eta():
    eta_short = calculate_travel_eta(3.0)
    assert 4 <= eta_short <= 12

    eta_long = calculate_travel_eta(100.0)
    assert eta_long >= 60


def test_emergency_search_icu_filtering():
    """
    Critical Requirement: A closer hospital with 0 available ICU beds
    must NOT be recommended when an ICU bed is requested!
    """
    # Search from Bani Park area in Jaipur near Hospital H-JAI-003 (which has 0 ICU beds)
    res = client.post("/emergency/search", json={
        "latitude": 26.9300,
        "longitude": 75.7920,
        "bed_type": "ICU",
        "quantity": 1,
        "district": "Jaipur"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["recommended"] is not None

    # The recommended hospital MUST actually have >= 1 ICU bed!
    rec = data["recommended"]
    assert rec["available_beds"] >= 1
    assert rec["hospital_id"] != "H-JAI-003"  # H-JAI-003 has 0 ICU beds and must NOT be recommended!

    # Check that H-JAI-003 was partitioned into unavailable_nearby with proper reason
    unavail_ids = [u["hospital_id"] for u in data["unavailable_nearby"]]
    assert "H-JAI-003" in unavail_ids
    bani_park = next(u for u in data["unavailable_nearby"] if u["hospital_id"] == "H-JAI-003")
    assert "0 ICU" in bani_park["rejection_reason"]


def test_emergency_search_radius_expansion():
    """
    If no hospital has available beds within initial search radius (e.g. 5 km in a rural coordinate),
    the system must automatically expand search radius until a suitable hospital is found.
    """
    # Coordinates in rural Sambhar Lake region (far from major hospitals)
    res = client.post("/emergency/search", json={
        "latitude": 26.9000,
        "longitude": 75.2000,
        "bed_type": "NICU",
        "quantity": 2,
        "max_radius_km": 10.0
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["expanded_search"] is True
    assert data["search_radius_km"] > 10.0
    assert "NO IMMEDIATE" in data["expansion_message"]
    assert data["recommended"]["available_beds"] >= 2


def test_emergency_route_generation():
    """Verify route generation from patient to hospital."""
    res = client.post("/emergency/route", json={
        "origin": {"latitude": 26.9124, "longitude": 75.7873},
        "destination": {"latitude": 26.8978, "longitude": 75.8167}
    })
    assert res.status_code == 200
    data = res.json()
    assert "distance_km" in data
    assert data["distance_km"] > 0.0
    assert "eta_minutes" in data
    assert data["eta_minutes"] >= 2
    assert "route" in data
    assert len(data["route"]) >= 5
    assert "turn_by_turn" in data


def test_live_beds_catalog_and_disclaimer():
    """Verify /beds returns hospital availability and clear simulated data labeling."""
    res = client.get("/beds")
    assert res.status_code == 200
    data = res.json()
    assert data["is_simulated"] is True
    assert "DEMO DATA" in data["disclaimer"]
    assert "summary" in data
    assert data["summary"]["vacant_icu_beds"] > 0
    assert len(data["hospitals"]) >= 20


def test_forecast_integration_warning():
    """Verify district AI forecast shortage attaches a future_capacity_risk warning."""
    res = client.post("/emergency/search", json={
        "latitude": 26.9124,
        "longitude": 75.7873,
        "bed_type": "ICU",
        "quantity": 1,
        "district": "Jaipur"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["future_capacity_risk"] is not None
    fc = data["future_capacity_risk"]
    assert fc["district"] == "Jaipur"
    assert "surge" in fc["projected_deficit"].lower()


def test_district_emergency_view():
    """Verify district-level unified emergency view."""
    res = client.get("/api/emergency/district/Jaipur?bed_type=ICU")
    assert res.status_code == 200
    data = res.json()
    assert data["district"] == "Jaipur"
    assert "current_status" in data
    assert "forecast_projection" in data
    assert data["current_status"]["vacant_beds"] > 0
    assert data["nearest_recommended_hospital"] is not None
