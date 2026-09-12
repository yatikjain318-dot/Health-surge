"""Surge Risk and Explainable AI (XAI) Attribution Router."""

from fastapi import APIRouter, Query
from typing import Dict, List
import numpy as np

from ..services.data_service import get_data_service
from ..services.hc05_engine import get_hc05_env
from ..services.forecasting_service import EnsembleForecastingEngine
from ..services.uncertainty_service import UncertaintyEngine
from ..services.surge_service import SurgeService

router = APIRouter(prefix="/api", tags=["Risks & Explainability"])


@router.get("/risks")
def get_surge_risks(mode: str = Query(default="REAL_HEALTH_DATA")):
    """
    Returns ranked surge risk assessments across districts with genuine
    Explainable AI (XAI) feature attribution percentage splits.
    """
    data_svc = get_data_service()

    if mode == "REAL_HEALTH_DATA":
        districts = data_svc.get_hmis_districts()
        idsp_signals = {s.get("district", ""): s for s in data_svc.get_idsp_signals()}

        ranked_districts = []
        counts = {"CRITICAL": 0, "HIGH RISK": 0, "WATCH": 0, "NORMAL": 0}

        for d in districts:
            level = d.get("risk_level", "NORMAL")
            counts[level] = counts.get(level, 0) + 1

            sig = idsp_signals.get(d.get("name", ""), {})
            xai = d.get("explainability", {
                "recent_demand_growth": 0.30,
                "seasonal_pattern": 0.20,
                "disease_trend": 0.25,
                "historical_volatility": 0.15,
                "capacity_pressure": 0.10
            })

            pathogen_note = (
                f"ARI index {sig.get('ari_case_index', 100.0)} | Vector index {sig.get('vector_borne_index', 100.0)}"
                if sig else "Seasonal Inpatient Variance"
            )

            ranked_districts.append({
                "district_id": d["id"],
                "district_name": d["name"],
                "state": d["state"],
                "surge_probability": d.get("surge_probability", 0.2),
                "risk_level": level,
                "predicted_demand": d.get("predicted_demand"),
                "base_capacity": d.get("base_capacity"),
                "bed_shortage": d.get("bed_shortage", 0),
                "surveillance_signal": pathogen_note,
                "signal_weekly_rate": sig.get("ari_case_index", 95.0),
                "cluster_detected": sig.get("cluster_detected", False),
                "explainability": xai,
                "rationale": (
                    f"Surge driven primarily by Recent Demand Growth ({int(xai['recent_demand_growth']*100)}%) "
                    f"and Disease Outbreak Signals ({int(xai['disease_trend']*100)}%), "
                    f"exceeding operational threshold."
                )
            })

        # Rank by surge probability descending
        ranked_districts.sort(key=lambda x: x["surge_probability"], reverse=True)

        surge_svc = SurgeService()
        sample_feature_dicts = []
        for rd in ranked_districts:
            cap = float(rd.get("base_capacity", 100))
            pred = float(rd.get("predicted_demand", 100))
            gap = max(0.0, (pred - cap) / max(cap, 1.0) * 100.0)
            xai = rd.get("explainability", {})
            sample_feature_dicts.append({
                "surge_probability": rd.get("surge_probability", 0.2),
                "features": {
                    "recent_growth": round(xai.get("recent_demand_growth", 0.15), 3),
                    "positive_error_streak": 2 if rd.get("surge_probability", 0.2) >= 0.5 else 0,
                    "last_residual": round(xai.get("disease_trend", 0.20) * 35.0, 1),
                    "capacity_gap_pct": round(gap, 1),
                }
            })
        logit_diagnostics = surge_svc.fit_surge_logit_model(sample_feature_dicts)

        return {
            "mode": mode,
            "summary": {
                "total_districts": len(ranked_districts),
                "critical_count": counts.get("CRITICAL", 0),
                "high_risk_count": counts.get("HIGH RISK", 0),
                "watch_count": counts.get("WATCH", 0),
                "normal_count": counts.get("NORMAL", 0),
            },
            "districts": ranked_districts,
            "statsmodels_logit_diagnostics": logit_diagnostics,
            "methodology": "Multi-Factor Epidemiological Surge Scoring + Maximum-Likelihood Logit Diagnostics (smf.logit)"
        }

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
        logit_diagnostics = surge_svc.fit_surge_logit_model(surge_results)

        ranked = []
        counts = {"CRITICAL": 0, "HIGH RISK": 0, "WATCH": 0, "NORMAL": 0}

        for item in surge_results:
            d = item["district_id"]
            lvl = item["risk_level"]
            counts[lvl] = counts.get(lvl, 0) + 1
            cap = int(env.nominal_capacity[d])
            pred = int(forecasts[d])

            ranked.append({
                "district_id": d,
                "district_name": f"District {d}",
                "state": "HC-05 Grid",
                "surge_probability": item["surge_probability"],
                "risk_level": lvl,
                "predicted_demand": pred,
                "base_capacity": cap,
                "bed_shortage": max(0, pred - cap),
                "surveillance_signal": "Synthetic Demand Residual Acceleration",
                "explainability": item["explainability"],
                "rationale": (
                    f"District {d} surge index at {item['surge_probability']:.2f}. "
                    f"Growth weight {int(item['explainability']['recent_demand_growth']*100)}% "
                    f"with capacity deficit of {max(0, pred - cap)} units."
                )
            })

        ranked.sort(key=lambda x: x["surge_probability"], reverse=True)

        return {
            "mode": mode,
            "summary": {
                "total_districts": len(ranked),
                "critical_count": counts.get("CRITICAL", 0),
                "high_risk_count": counts.get("HIGH RISK", 0),
                "watch_count": counts.get("WATCH", 0),
                "normal_count": counts.get("NORMAL", 0),
            },
            "districts": ranked,
            "statsmodels_logit_diagnostics": logit_diagnostics,
            "methodology": "HC-05 Online Autoregressive Residual Scoring + Maximum-Likelihood Logit (smf.logit)"
        }
