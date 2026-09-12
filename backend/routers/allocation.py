"""Resource Allocation Router: 60-unit marginal allocation and inter-district transfers."""

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import numpy as np

from ..services.data_service import get_data_service
from ..services.hc05_engine import get_hc05_env
from ..services.forecasting_service import EnsembleForecastingEngine
from ..services.uncertainty_service import UncertaintyEngine
from ..services.surge_service import SurgeService
from ..services.allocation_service import AllocationService
from ..services.transfer_service import ResourceTransferOptimizer

router = APIRouter(prefix="/api", tags=["Allocation & Transfers"])


class RecalculateRequest(BaseModel):
    reserve_budget: int = Field(default=60, ge=0, le=250)
    w_surge: float = Field(default=0.20, ge=0.0, le=1.0)
    w_fair: float = Field(default=0.50, ge=0.0, le=1.0)
    w_unc: float = Field(default=0.05, ge=0.0, le=1.0)
    mode: str = Field(default="REAL_HEALTH_DATA")


@router.get("/allocation")
def get_allocation(mode: str = Query(default="REAL_HEALTH_DATA")):
    """
    Returns the optimal reserve allocation across districts and inter-district transfer recommendations.
    """
    return run_allocation_pipeline(mode=mode, budget=60, w_surge=0.20, w_fair=0.50, w_unc=0.05)


@router.post("/allocation/recalculate")
def recalculate_allocation(payload: RecalculateRequest):
    """
    Allows duty commander to adjust reserve capacity or fairness weights and observe instant reallocation.
    """
    return run_allocation_pipeline(
        mode=payload.mode,
        budget=payload.reserve_budget,
        w_surge=payload.w_surge,
        w_fair=payload.w_fair,
        w_unc=payload.w_unc
    )


def run_allocation_pipeline(mode: str, budget: int, w_surge: float, w_fair: float, w_unc: float):
    data_svc = get_data_service()
    alloc_svc = AllocationService(w_unc=w_unc, w_surge=w_surge, w_fair=w_fair)
    transfer_opt = ResourceTransferOptimizer()

    if mode == "REAL_HEALTH_DATA":
        districts = data_svc.get_hmis_districts()
        n = len(districts)

        forecasts = np.array([d.get("predicted_demand", 0) for d in districts], dtype=float)
        capacities = np.array([d.get("base_capacity", 0) for d in districts], dtype=float)
        stds = np.array([max(8.0, d.get("predicted_demand", 0) * 0.08) for d in districts], dtype=float)
        surge_risks = np.array([d.get("surge_probability", 0.0) for d in districts], dtype=float)
        cumulative_service = np.array([0.90 for _ in range(n)], dtype=float)

        alloc_vector, details = alloc_svc.optimize_allocation(
            forecasts=forecasts,
            stds=stds,
            capacities=capacities,
            surge_risks=surge_risks,
            cumulative_service=cumulative_service,
            total_budget=budget
        )

        # Build district allocations list
        dist_allocs = []
        for i, d in enumerate(districts):
            det = details[i]
            dist_allocs.append({
                "district_id": d.get("id"),
                "district_name": d.get("name"),
                "base_capacity": d.get("base_capacity"),
                "predicted_demand": d.get("predicted_demand"),
                "allocated_units": int(alloc_vector[i]),
                "effective_capacity": int(d.get("base_capacity") + alloc_vector[i]),
                "shortage_before": int(d.get("bed_shortage", 0)),
                "shortage_after": max(0, int(d.get("bed_shortage", 0) - alloc_vector[i])),
                "expected_shortage_before": det["expected_shortage_before"],
                "expected_shortage_after": det["expected_shortage_after"],
                "shortage_reduction_pct": det["reduction_pct"],
                "surge_risk": d.get("surge_probability", 0.0),
                "rationale": det["rationale"]
            })

        # Inter-district transfer recommendations
        transfer_recs = transfer_opt.generate_recommendations(dist_allocs)

        total_shortage_before = sum(d["shortage_before"] for d in dist_allocs)
        total_shortage_after = sum(d["shortage_after"] for d in dist_allocs)
        overall_reduction = round(
            ((total_shortage_before - total_shortage_after) / max(total_shortage_before, 1)) * 100, 1
        )

        return {
            "mode": mode,
            "reserve_budget": budget,
            "total_allocated": int(alloc_vector.sum()),
            "total_shortage_before": total_shortage_before,
            "total_shortage_after": total_shortage_after,
            "shortage_reduction_pct": overall_reduction,
            "district_allocations": dist_allocs,
            "inter_district_transfers": transfer_recs,
            "weights_used": {"w_surge": w_surge, "w_fair": w_fair, "w_unc": w_unc}
        }

    else:
        env = get_hc05_env()
        history = env.get_history(up_to_month=36)
        n = env.NUM_DISTRICTS

        engine = EnsembleForecastingEngine()
        engine.optimize_weights_backtest(history, start_month=24)
        forecasts, _ = engine.predict(history, t=36)

        unc_engine = UncertaintyEngine()
        stds = unc_engine.estimate_uncertainty(history, t=36)

        surge_svc = SurgeService()
        surge_items = surge_svc.compute_surge_risk_and_xai(
            history, forecasts, env.nominal_capacity, [[] for _ in range(n)], stds
        )
        surge_risks = np.array([item["surge_probability"] for item in surge_items])
        cumulative_service = np.array([1.0 for _ in range(n)], dtype=float)

        alloc_vector, details = alloc_svc.optimize_allocation(
            forecasts=forecasts.astype(float),
            stds=stds,
            capacities=env.nominal_capacity.astype(float),
            surge_risks=surge_risks,
            cumulative_service=cumulative_service,
            total_budget=budget
        )

        dist_allocs = []
        for d in range(n):
            det = details[d]
            cap = int(env.nominal_capacity[d])
            pred = int(forecasts[d])
            alloc = int(alloc_vector[d])
            shortage_before = max(0, pred - cap)
            shortage_after = max(0, pred - (cap + alloc))

            dist_allocs.append({
                "district_id": d,
                "district_name": f"District {d}",
                "base_capacity": cap,
                "predicted_demand": pred,
                "allocated_units": alloc,
                "effective_capacity": cap + alloc,
                "shortage_before": shortage_before,
                "shortage_after": shortage_after,
                "expected_shortage_before": det["expected_shortage_before"],
                "expected_shortage_after": det["expected_shortage_after"],
                "shortage_reduction_pct": det["reduction_pct"],
                "surge_risk": round(float(surge_risks[d]), 3),
                "rationale": det["rationale"]
            })

        transfer_recs = transfer_opt.generate_recommendations(dist_allocs)

        total_shortage_before = sum(d["shortage_before"] for d in dist_allocs)
        total_shortage_after = sum(d["shortage_after"] for d in dist_allocs)
        overall_reduction = round(
            ((total_shortage_before - total_shortage_after) / max(total_shortage_before, 1)) * 100, 1
        )

        return {
            "mode": mode,
            "reserve_budget": budget,
            "total_allocated": int(alloc_vector.sum()),
            "total_shortage_before": total_shortage_before,
            "total_shortage_after": total_shortage_after,
            "shortage_reduction_pct": overall_reduction,
            "district_allocations": dist_allocs,
            "inter_district_transfers": transfer_recs,
            "weights_used": {"w_surge": w_surge, "w_fair": w_fair, "w_unc": w_unc}
        }
