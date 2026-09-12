"""Main Application Entrypoint for SwasthyaSurge AI Emergency Operations Command Center."""

import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from .config import BASE_DIR
from .database import init_db
from .routers import (
    districts,
    forecast,
    risks,
    alerts,
    allocation,
    simulate,
    evaluation,
    data_sources,
    system_health,
    emergency,
)

# Initialize database
init_db()

app = FastAPI(
    title="SwasthyaSurge AI",
    description="AI-Powered District Health Surge Forecast & Resource Allocation Command Center",
    version="2.4.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register All Subsystem Routers
app.include_router(districts.router)
app.include_router(forecast.router)
app.include_router(risks.router)
app.include_router(alerts.router)
app.include_router(allocation.router)
app.include_router(simulate.router)
app.include_router(evaluation.router)
app.include_router(data_sources.router)
app.include_router(system_health.router)
app.include_router(emergency.router)

# Mount static frontend build if present
frontend_dist = BASE_DIR / "backend" / "static"
if not (frontend_dist / "index.html").exists():
    frontend_dist = BASE_DIR / "frontend" / "out"
if not (frontend_dist / "index.html").exists():
    frontend_dist = BASE_DIR / "frontend" / "build"
if not (frontend_dist / "index.html").exists():
    frontend_dist = BASE_DIR / "frontend" / "dist"

if frontend_dist.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dist)), name="static")

@app.get("/")
def root():
    """Root platform status and API route overview."""
    if frontend_dist.exists() and (frontend_dist / "index.html").exists():
        return FileResponse(str(frontend_dist / "index.html"))

    return {
        "platform": "SwasthyaSurge AI",
        "tagline": "AI-Powered District Health Surge Forecast & Resource Allocation Command Center",
        "system_status": "OPERATIONAL",
        "version": "v2.4.0",
        "api_documentation": "/docs",
        "active_endpoints": [
            "/api/districts",
            "/api/district/{id}",
            "/api/forecast/{district_id}",
            "/api/risks",
            "/api/alerts",
            "/api/allocation",
            "/api/allocation/recalculate",
            "/api/simulate",
            "/api/evaluation",
            "/api/data-sources",
            "/api/system-health"
        ]
    }
