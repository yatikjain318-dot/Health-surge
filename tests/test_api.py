"""Integration tests for all SwasthyaSurge AI API routes."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "SwasthyaSurge" in response.text
    assert "Command Center" in response.text


def test_districts_real_and_hc05():
    # Real health data
    res_real = client.get("/api/districts?mode=REAL_HEALTH_DATA")
    assert res_real.status_code == 200
    data_real = res_real.json()
    assert data_real["mode"] == "REAL_HEALTH_DATA"
    assert len(data_real["districts"]) == 12
    assert data_real["districts"][0]["name"] == "Jaipur"

    # HC-05 challenge mode
    res_hc05 = client.get("/api/districts?mode=HC05_EVALUATION")
    assert res_hc05.status_code == 200
    data_hc05 = res_hc05.json()
    assert data_hc05["mode"] == "HC05_EVALUATION"
    assert len(data_hc05["districts"]) == 12
    assert "District 0" in data_hc05["districts"][0]["name"]


def test_district_detail():
    res = client.get("/api/district/0?mode=REAL_HEALTH_DATA")
    assert res.status_code == 200
    data = res.json()
    assert data["district"]["name"] == "Jaipur"
    assert "multi_horizon_forecast" in data
    assert "bed_analytics" in data


def test_forecast_multi_horizon():
    res = client.get("/api/forecast/0?mode=REAL_HEALTH_DATA")
    assert res.status_code == 200
    data = res.json()
    assert "horizon_1_month" in data
    assert "horizon_3_month" in data
    assert "horizon_6_month" in data
    assert "confidence_interval_95" in data
    assert data["confidence_interval_95"]["lower"] <= data["confidence_interval_95"]["upper"]
    assert "ensemble_weights" in data
    assert "statsmodels_diagnostics" in data
    assert "rsquared" in data["statsmodels_diagnostics"]
    assert data["statsmodels_diagnostics"]["rsquared"] > 0.0


def test_forecast_statsmodels_decomposition():
    res = client.get("/api/forecast/0/decomposition?mode=REAL_HEALTH_DATA")
    assert res.status_code == 200
    data = res.json()
    assert "decomposition" in data
    decomp = data["decomposition"]
    assert "trend" in decomp
    assert "seasonal" in decomp
    assert "residual" in decomp
    assert len(decomp["trend"]) == len(decomp["observed"])
    assert "statsmodels.tsa" in decomp["method"]


def test_surge_risks_and_xai():
    res = client.get("/api/risks?mode=REAL_HEALTH_DATA")
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert len(data["districts"]) == 12
    first = data["districts"][0]
    assert "surge_probability" in first
    assert "explainability" in first
    # XAI splits sum to ~1.0
    xai = first["explainability"]
    total_xai = sum(xai.values())
    assert abs(total_xai - 1.0) < 0.05


def test_smart_alerts():
    res = client.get("/api/alerts?mode=REAL_HEALTH_DATA")
    assert res.status_code == 200
    data = res.json()
    assert "alerts" in data
    if len(data["alerts"]) > 0:
        first = data["alerts"][0]
        assert "priority_score" in first
        assert "recommended_action" in first
        assert "actions" in first


def test_allocation_and_recalculate():
    # Default allocation
    res = client.get("/api/allocation?mode=REAL_HEALTH_DATA")
    assert res.status_code == 200
    data = res.json()
    assert data["total_allocated"] <= 60
    assert len(data["district_allocations"]) == 12

    # Recalculate allocation POST
    post_res = client.post("/api/allocation/recalculate", json={
        "reserve_budget": 80,
        "w_surge": 0.30,
        "w_fair": 0.40,
        "w_unc": 0.10,
        "mode": "REAL_HEALTH_DATA"
    })
    assert post_res.status_code == 200
    post_data = post_res.json()
    assert post_data["reserve_budget"] == 80
    assert post_data["total_allocated"] <= 80


def test_crisis_simulation():
    payload = {
        "district_id": 0,
        "surge_multiplier": 1.30,
        "capacity_loss_pct": 0.10,
        "reserve_budget": 60,
        "mode": "REAL_HEALTH_DATA",
        "scenario_name": "Test Viral Outbreak"
    }
    res = client.post("/api/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "impact_summary" in data
    assert data["impact_summary"]["unmet_demand_mitigated"] >= 0
    assert "commander_brief" in data


def test_hc05_evaluation_benchmark():
    res = client.get("/api/evaluation")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OFFICIALLY_VERIFIED"
    assert data["seed"] == 20260911
    assert "policy_comparison" in data
    assert len(data["policy_comparison"]) == 4


def test_data_sources_catalog():
    res = client.get("/api/data-sources")
    assert res.status_code == 200
    data = res.json()
    assert len(data["datasets"]) >= 4


def test_system_health():
    res = client.get("/api/system-health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OPERATIONAL"
    assert len(data["active_models"]) == 5


def test_statsmodels_diagnostic_suite():
    res = client.get("/api/forecast/0/diagnostics?mode=REAL_HEALTH_DATA")
    assert res.status_code == 200
    data = res.json()
    assert "diagnostics" in data
    diag = data["diagnostics"]
    assert "stationarity" in diag
    assert "adfuller" in diag["stationarity"]
    assert "kpss" in diag["stationarity"]
    assert "quantile_surge_envelope" in diag
    assert diag["quantile_surge_envelope"]["tau_90_surge_ceiling"] >= diag["quantile_surge_envelope"]["tau_10_floor"]
    assert "autocorrelation" in diag
    assert len(diag["autocorrelation"]["acf_lags"]) > 0
    assert "robust_and_autoregressive" in diag
    assert "sm" in data["api_stack"]
    assert "tsa" in data["api_stack"]
    assert "smf" in data["api_stack"]


def test_statsmodels_surge_logit_diagnostics():
    res = client.get("/api/risks?mode=REAL_HEALTH_DATA")
    assert res.status_code == 200
    data = res.json()
    assert "statsmodels_logit_diagnostics" in data
    logit_diag = data["statsmodels_logit_diagnostics"]
    assert "engine" in logit_diag
    assert "statsmodels" in logit_diag["engine"]
    assert "pseudo_rsquared" in logit_diag
    assert logit_diag["pseudo_rsquared"] >= 0.0
    assert "odds_ratios" in logit_diag


def test_statsmodels_direct_classes():
    import numpy as np
    from backend.services.forecasting_service import (
        AutoRegModel,
        QuantileRegressionForecaster,
        RobustHarmonicRegressionModel,
        WeightedHarmonicRegressionModel
    )
    s = np.array([100, 110, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180], dtype=float)
    
    ar = AutoRegModel(lags=2)
    pred_ar = ar.predict_next(s, 12)
    assert pred_ar > 0.0

    qr = QuantileRegressionForecaster(quantiles=[0.10, 0.50, 0.90])
    q_dict = qr.fit_predict_quantiles(s, 12)
    assert q_dict[0.90] >= q_dict[0.10]

    rlm = RobustHarmonicRegressionModel()
    pred_rlm = rlm.predict_next(s, 12)
    assert pred_rlm > 0.0

    wls = WeightedHarmonicRegressionModel()
    pred_wls = wls.predict_next(s, 12)
    assert pred_wls > 0.0

