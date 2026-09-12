"""Smart Alerts Router: Prioritized emergency alerts and automated clinical triggers."""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np

from ..services.data_service import get_data_service
from ..services.hc05_engine import get_hc05_env
from ..services.forecasting_service import EnsembleForecastingEngine
from ..services.uncertainty_service import UncertaintyEngine
from ..services.surge_service import SurgeService

router = APIRouter(prefix="/api", tags=["Smart Alerts"])

# In-memory alert acknowledgment tracker
ACKNOWLEDGED_ALERTS = set()


class AckRequest(BaseModel):
    acknowledged: bool = True
    user_notes: Optional[str] = None


@router.get("/alerts")
def get_alerts(mode: str = Query(default="REAL_HEALTH_DATA")):
    """
    Returns prioritized emergency alerts ranked by severity, predicted shortage,
    and surge probability, complete with actionable operational buttons.
    """
    data_svc = get_data_service()
    alerts = []

    if mode == "REAL_HEALTH_DATA":
        districts = data_svc.get_hmis_districts()
        idsp_signals = {s.get("district", ""): s for s in data_svc.get_idsp_signals()}

        for d in districts:
            shortage = d.get("bed_shortage", 0)
            prob = d.get("surge_probability", 0.0)
            risk = d.get("risk_level", "NORMAL")
            cap = d.get("base_capacity", 1)

            if risk in ("CRITICAL", "HIGH RISK") or shortage > 0:
                shortage_ratio = min(1.0, shortage / max(cap, 1))
                severity_weight = 1.0 if risk == "CRITICAL" else 0.75
                # Priority Score between 0 and 100
                priority_score = round(
                    (0.40 * prob + 0.35 * shortage_ratio + 0.25 * severity_weight) * 100, 1
                )

                sig = idsp_signals.get(d.get("name", ""), {})
                pathogen = "Acute Outbreak Cluster" if sig.get("cluster_detected") else "Inpatient Surge"

                alert_id = f"alert-real-{d['id']}"
                alerts.append({
                    "id": alert_id,
                    "district_id": d["id"],
                    "district_name": d["name"],
                    "state": d["state"],
                    "severity": risk,
                    "priority_score": priority_score,
                    "predicted_shortage": shortage,
                    "surge_probability": prob,
                    "title": f"{d['name']}: Acute Surge Alert ({pathogen})",
                    "reason": f"Predicted demand {d.get('predicted_demand')} exceeds bed baseline of {cap}. Pathogen signal: {pathogen}.",
                    "recommended_action": f"Dispatch +{min(60, shortage)} emergency buffer beds from State Reserve. Initiate triage protocol.",
                    "actions": [
                        {"type": "VIEW_DISTRICT", "label": "Inspect District", "target": f"/district/{d['id']}"},
                        {"type": "ALLOCATE", "label": "Deploy Reserve", "target": "/allocation"},
                        {"type": "SIMULATE", "label": "Run What-If", "target": f"/simulate?district={d['id']}"}
                    ],
                    "is_acknowledged": alert_id in ACKNOWLEDGED_ALERTS,
                    "timestamp": "2026-09-11 20:30 IST"
                })

    else:
        env = get_hc05_env()
        history = env.get_history(up_to_month=36)
        engine = EnsembleForecastingEngine()
        engine.optimize_weights_backtest(history, start_month=24)
        forecasts, _ = engine.predict(history, t=36)

        unc_engine = UncertaintyEngine()
        stds = unc_engine.estimate_uncertainty(history, t=36)

        surge_svc = SurgeService()
        surge_results = surge_svc.compute_surge_risk_and_xai(
            history, forecasts, env.nominal_capacity, [[] for _ in range(12)], stds
        )

        for item in surge_results:
            d = item["district_id"]
            lvl = item["risk_level"]
            prob = item["surge_probability"]
            cap = int(env.nominal_capacity[d])
            pred = int(forecasts[d])
            shortage = max(0, pred - cap)

            if lvl in ("CRITICAL", "HIGH RISK") or shortage > 0:
                shortage_ratio = min(1.0, shortage / max(cap, 1))
                severity_weight = 1.0 if lvl == "CRITICAL" else 0.70
                priority_score = round(
                    (0.40 * prob + 0.35 * shortage_ratio + 0.25 * severity_weight) * 100, 1
                )

                alert_id = f"alert-hc05-{d}"
                alerts.append({
                    "id": alert_id,
                    "district_id": d,
                    "district_name": f"District {d}",
                    "state": "HC-05 Sector Alpha",
                    "severity": lvl,
                    "priority_score": priority_score,
                    "predicted_shortage": shortage,
                    "surge_probability": prob,
                    "title": f"District {d}: HC-05 Surge Warning",
                    "reason": f"Forecasted demand {pred} exceeds capacity {cap} by {shortage} units.",
                    "recommended_action": f"Allocate emergency units sequentially under 60-unit budget limit.",
                    "actions": [
                        {"type": "VIEW_DISTRICT", "label": "Inspect District", "target": f"/district/{d}"},
                        {"type": "ALLOCATE", "label": "Apply Optimal Allocation", "target": "/allocation"},
                        {"type": "SIMULATE", "label": "Run Stress Test", "target": f"/simulate?district={d}"}
                    ],
                    "is_acknowledged": alert_id in ACKNOWLEDGED_ALERTS,
                    "timestamp": "2026-09-11 20:30 IST"
                })

    # Sort descending by priority score
    alerts.sort(key=lambda x: x["priority_score"], reverse=True)

    return {
        "mode": mode,
        "total_active_alerts": len(alerts),
        "critical_alerts_count": sum(1 for a in alerts if a["severity"] == "CRITICAL"),
        "alerts": alerts
    }


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, payload: AckRequest):
    """Marks an alert as acknowledged by duty officer."""
    if payload.acknowledged:
        ACKNOWLEDGED_ALERTS.add(alert_id)
    else:
        ACKNOWLEDGED_ALERTS.discard(alert_id)
    return {"status": "success", "alert_id": alert_id, "acknowledged": payload.acknowledged}
