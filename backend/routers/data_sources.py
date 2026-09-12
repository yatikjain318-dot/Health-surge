"""Data Sources and Catalog Transparency Router."""

from fastapi import APIRouter
from typing import Dict, List
from ..services.data_service import get_data_service

router = APIRouter(prefix="/api", tags=["Data Catalog Transparency"])


@router.get("/data-sources")
def get_data_sources():
    """
    Returns complete transparent catalog of all public datasets ingested and simulation benchmarks.
    Zero synthetic live claims; full attribution to MoHFW, NHA ABDM, and NCDC IDSP.
    """
    data_svc = get_data_service()
    catalog = data_svc.get_catalog()

    return {
        "transparency_policy": "Strict Open Government Data Attribution (Zero Invented APIs)",
        "national_jurisdiction": "Republic of India",
        "total_datasets": len(catalog),
        "datasets": catalog,
        "governance_note": (
            "All operational health indicators are derived from public gazettes published under "
            "the National Data Sharing and Accessibility Policy (NDSAP). Synthetic evaluations "
            "are explicitly demarcated as HC-05 challenge benchmarks."
        )
    }
