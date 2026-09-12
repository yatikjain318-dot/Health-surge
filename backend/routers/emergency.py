"""
Emergency Bed Finder & Routing API Router.
Handles emergency bed queries, hospital suitability scoring, road navigation routes, and live bed feeds.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

from ..services.emergency_service import get_emergency_service

router = APIRouter(tags=["Emergency Bed Finder & Routing"])


class LocationPoint(BaseModel):
    latitude: float
    longitude: float


class EmergencySearchRequest(BaseModel):
    latitude: Optional[float] = Field(default=None, description="Patient current latitude (e.g. 26.9124)")
    longitude: Optional[float] = Field(default=None, description="Patient current longitude (e.g. 75.7873)")
    patient_lat: Optional[float] = Field(default=None, description="Patient latitude alias")
    patient_lon: Optional[float] = Field(default=None, description="Patient longitude alias")
    bed_type: str = Field(default="ICU", description="Requested bed type: General, ICU, HDU, NICU, Emergency, Ventilator")
    quantity: int = Field(default=1, ge=1, le=20, description="Number of beds required")
    district: Optional[str] = Field(default=None, description="Optional target district")
    max_radius_km: Optional[float] = Field(default=25.0, description="Initial search radius in km")


class EmergencyRouteRequest(BaseModel):
    origin: Optional[LocationPoint] = Field(default=None, description="Patient origin coordinates")
    destination: Optional[LocationPoint] = Field(default=None, description="Hospital destination coordinates")
    origin_lat: Optional[float] = Field(default=None, description="Origin latitude alias")
    origin_lon: Optional[float] = Field(default=None, description="Origin longitude alias")
    dest_lat: Optional[float] = Field(default=None, description="Destination latitude alias")
    dest_lon: Optional[float] = Field(default=None, description="Destination longitude alias")
    bed_type: Optional[str] = Field(default="ICU", description="Requested bed type")


@router.post("/emergency/search")
@router.post("/api/emergency/search")
def search_emergency_beds(req: EmergencySearchRequest):
    """
    Finds nearest suitable hospital with vacant beds matching requested bed type and quantity.
    Filters out hospitals with zero available beds of the requested type.
    Expands search radius automatically if no immediate capacity is found.
    """
    lat = req.latitude if req.latitude is not None else req.patient_lat
    lon = req.longitude if req.longitude is not None else req.patient_lon
    if lat is None or lon is None:
        lat = 26.9124
        lon = 75.7873

    svc = get_emergency_service()
    result = svc.search_nearest_suitable_hospitals(
        latitude=lat,
        longitude=lon,
        bed_type=req.bed_type,
        quantity=req.quantity,
        district=req.district,
        initial_radius_km=req.max_radius_km or 25.0
    )
    return result


@router.post("/emergency/route")
@router.post("/api/emergency/route")
def get_emergency_route(req: EmergencyRouteRequest):
    """
    Generates fastest emergency transit route from patient location to hospital destination
    with distance, travel ETA in minutes, and road waypoints.
    """
    orig_lat = req.origin.latitude if req.origin else req.origin_lat
    orig_lon = req.origin.longitude if req.origin else req.origin_lon
    d_lat = req.destination.latitude if req.destination else req.dest_lat
    d_lon = req.destination.longitude if req.destination else req.dest_lon

    if orig_lat is None or orig_lon is None or d_lat is None or d_lon is None:
        raise HTTPException(status_code=400, detail="Missing origin or destination coordinates")

    svc = get_emergency_service()
    route_info = svc.generate_emergency_route(
        origin_lat=orig_lat,
        origin_lon=orig_lon,
        dest_lat=d_lat,
        dest_lon=d_lon
    )
    return route_info


@router.get("/beds")
@router.get("/api/beds")
def get_live_hospital_beds():
    """
    Returns current hospital bed availability across all facilities in the network.
    Clearly labeled as simulated demo data.
    """
    svc = get_emergency_service()
    return svc.get_live_beds_catalog()


@router.get("/api/emergency/district/{district_name}")
def get_district_emergency_status(district_name: str, bed_type: str = Query(default="ICU")):
    """
    Returns unified district-level emergency view combining instantaneous hospital bed
    availability with AI-projected 30-day demand surges and shortages.
    """
    svc = get_emergency_service()
    d_clean = district_name.strip()
    
    # Filter hospitals in this district
    dist_hospitals = [h for h in svc._hospitals if h["district"].lower() == d_clean.lower()]
    if not dist_hospitals:
        # Fallback to closest matching
        dist_hospitals = [h for h in svc._hospitals if d_clean.lower() in h["district"].lower()]

    bed_field = svc._get_bed_field(bed_type)
    current_vacant_beds = sum(h.get(bed_field, 0) for h in dist_hospitals)
    total_beds_in_district = sum(h.get("total_beds", 0) for h in dist_hospitals)

    # Forecast context
    fc_info = svc._hmis_forecasts.get(d_clean.lower(), {
        "district": d_clean,
        "predicted_demand": 0,
        "base_capacity": total_beds_in_district,
        "bed_shortage": 0,
        "surge_probability": 0.20,
        "risk_level": "NORMAL"
    })

    # Nearest available hospital within district
    available_in_district = [h for h in dist_hospitals if h.get(bed_field, 0) > 0]
    available_in_district.sort(key=lambda h: h.get(bed_field, 0), reverse=True)
    nearest_hospital = available_in_district[0] if available_in_district else (dist_hospitals[0] if dist_hospitals else None)

    return {
        "district": d_clean,
        "bed_type": bed_type.upper(),
        "current_status": {
            "vacant_beds": current_vacant_beds,
            "total_district_beds": total_beds_in_district,
            "operational_facilities": len(dist_hospitals),
            "critical_facilities": [
                {
                    "hospital_id": h["hospital_id"],
                    "hospital_name": h["hospital_name"],
                    "available_beds": h.get(bed_field, 0),
                    "emergency_status": h.get("emergency_status", "ACCEPTING_EMERGENCIES")
                }
                for h in dist_hospitals[:4]
            ]
        },
        "forecast_projection": {
            "predicted_monthly_demand": fc_info.get("predicted_demand", 0),
            "nominal_base_capacity": fc_info.get("base_capacity", total_beds_in_district),
            "expected_shortage": fc_info.get("bed_shortage", 0),
            "surge_probability": fc_info.get("surge_probability", 0.20),
            "risk_level": fc_info.get("risk_level", "NORMAL")
        },
        "nearest_recommended_hospital": nearest_hospital,
        "data_mode": "SIMULATED_DEMO_DATA",
        "disclaimer": "DEMO DATA — Not real-time hospital availability"
    }
