"""What-If Crisis Simulator Router: Dynamic stress-testing and catastrophe modeling."""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import numpy as np

from ..services.data_service import get_data_service
from ..services.hc05_engine import get_hc05_env
from ..services.allocation_service import AllocationService
from ..services.transfer_service import ResourceTransferOptimizer

router = APIRouter(prefix="/api", tags=["Crisis Simulation"])


class SimulationScenario(BaseModel):
    district_id: int = Field(default=0, description="Target district ID or -1 for system-wide")
    surge_multiplier: float = Field(default=1.30, ge=0.5, le=3.0, description="Demand multiplier, e.g. 1.30 = +30% surge")
    capacity_loss_pct: float = Field(default=0.10, ge=0.0, le=0.7, description="Operational capacity loss (e.g. strike, damage)")
    reserve_budget: int = Field(default=60, ge=0, le=250, description="Available reserve budget")
    mode: str = Field(default="REAL_HEALTH_DATA")
    scenario_name: str = Field(default="Acute Viral Outbreak Stress Test")


@router.post("/simulate")
def run_simulation(scenario: SimulationScenario):
    """
    Simulates a crisis scenario, re-running marginal allocation and inter-district balancing
    to provide instant before-and-after operational impacts.
    """
    data_svc = get_data_service()
    alloc_svc = AllocationService()
    transfer_opt = ResourceTransferOptimizer()

    if scenario.mode == "REAL_HEALTH_DATA":
        districts = data_svc.get_hmis_districts()
        n = len(districts)

        # Baseline vectors
        base_demand = np.array([d.get("predicted_demand", 0) for d in districts], dtype=float)
        base_cap = np.array([d.get("base_capacity", 0) for d in districts], dtype=float)
        names = [d.get("name") for d in districts]
    else:
        env = get_hc05_env()
        history = env.get_history(up_to_month=36)
        n = env.NUM_DISTRICTS
        base_demand = np.array([int(history[d, -1]) for d in range(n)], dtype=float)
        base_cap = env.nominal_capacity.astype(float)
        names = [f"District {d}" for d in range(n)]

    # Apply scenario perturbations
    stressed_demand = base_demand.copy()
    stressed_cap = base_cap.copy()

    for d in range(n):
        is_target = (scenario.district_id == -1) or (scenario.district_id == d)
        if is_target:
            stressed_demand[d] = round(base_demand[d] * scenario.surge_multiplier)
            stressed_cap[d] = round(base_cap[d] * (1.0 - scenario.capacity_loss_pct))

    stds = np.maximum(8.0, stressed_demand * 0.08)
    surge_risks = np.clip((stressed_demand - stressed_cap) / np.maximum(1.0, stressed_cap), 0.1, 0.99)
    cumulative_service = np.full(n, 0.90)

    # 1. Baseline unmanaged shortage under stress
    raw_stress_shortage = np.maximum(0, stressed_demand - stressed_cap).astype(int)

    # 2. Optimal reserve allocation under stress
    alloc_vector, details = alloc_svc.optimize_allocation(
        forecasts=stressed_demand,
        stds=stds,
        capacities=stressed_cap,
        surge_risks=surge_risks,
        cumulative_service=cumulative_service,
        total_budget=scenario.reserve_budget
    )

    effective_cap = stressed_cap + alloc_vector
    managed_shortage = np.maximum(0, stressed_demand - effective_cap).astype(int)

    # District level comparison
    comparison_table = []
    for d in range(n):
        comparison_table.append({
            "district_id": d,
            "district_name": names[d],
            "baseline_demand": int(base_demand[d]),
            "stressed_demand": int(stressed_demand[d]),
            "demand_delta": int(stressed_demand[d] - base_demand[d]),
            "stressed_capacity": int(stressed_cap[d]),
            "allocated_reserve": int(alloc_vector[d]),
            "unmanaged_shortage": int(raw_stress_shortage[d]),
            "managed_shortage": int(managed_shortage[d]),
            "shortage_prevented": int(raw_stress_shortage[d] - managed_shortage[d]),
            "surge_risk": round(float(surge_risks[d]), 2)
        })

    # Transfers under stress
    transfers = transfer_opt.generate_recommendations([
        {
            "id": d,
            "name": names[d],
            "base_capacity": int(effective_cap[d]),
            "predicted_demand": int(stressed_demand[d]),
            "bed_shortage": int(managed_shortage[d]),
            "risk_level": "CRITICAL" if managed_shortage[d] > 50 else "NORMAL"
        }
        for d in range(n)
    ])

    total_unmanaged = int(raw_stress_shortage.sum())
    total_managed = int(managed_shortage.sum())
    lives_protected = total_unmanaged - total_managed
    mitigation_rate = round((lives_protected / max(1, total_unmanaged)) * 100, 1)

    return {
        "scenario_name": scenario.scenario_name,
        "mode": scenario.mode,
        "parameters": {
            "target_district": scenario.district_id,
            "surge_multiplier": scenario.surge_multiplier,
            "capacity_loss_pct": scenario.capacity_loss_pct,
            "reserve_budget": scenario.reserve_budget,
        },
        "impact_summary": {
            "total_unmanaged_shortage": total_unmanaged,
            "total_managed_shortage": total_managed,
            "unmet_demand_mitigated": lives_protected,
            "mitigation_rate_pct": mitigation_rate,
            "critical_districts_unmanaged": int(np.sum(raw_stress_shortage > 50)),
            "critical_districts_managed": int(np.sum(managed_shortage > 50)),
        },
        "district_comparison": comparison_table,
        "emergency_transfers": transfers,
        "commander_brief": (
            f"Under '{scenario.scenario_name}', unmanaged emergency deficit surges to {total_unmanaged} beds. "
            f"Deploying {scenario.reserve_budget} reserve units absorbs {lives_protected} deficit units ({mitigation_rate}% mitigation). "
            f"Inter-district transfers further stabilize adjacent zones."
        )
    }
