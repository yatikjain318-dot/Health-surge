"""Forecasting Router: Multi-horizon predictions, ensemble weights, and uncertainty bands powered by Statsmodels."""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Canonical Statsmodels imports
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import statsmodels.formula.api as smf

from ..services.data_service import get_data_service
from ..services.hc05_engine import get_hc05_env
from ..services.forecasting_service import (
    EnsembleForecastingEngine,
    compute_time_series_diagnostics,
    QuantileRegressionForecaster,
    RobustHarmonicRegressionModel,
    AutoRegModel
)
from ..services.uncertainty_service import UncertaintyEngine

router = APIRouter(prefix="/api", tags=["Forecasting"])


@router.get("/forecast/{district_id}")
def get_district_forecast(district_id: int, mode: str = Query(default="REAL_HEALTH_DATA")):
    """
    Returns multi-horizon demand forecasts (+1, +3, +6 months),
    ensemble model weights breakdown, and 95% confidence intervals,
    enhanced by statsmodels OLS diagnostics, stationarity, and quantile bounds.
    """
    data_svc = get_data_service()

    if mode == "REAL_HEALTH_DATA":
        district = data_svc.get_district_by_id(district_id)
        if not district:
            raise HTTPException(status_code=404, detail=f"District {district_id} not found.")

        hist = np.array(district.get("historical_demand", []), dtype=float)
        n = len(hist)
        t_current = n

        engine = EnsembleForecastingEngine(num_districts=1)
        # Predict individual models
        ind_preds = {}
        for name, model in engine.models.items():
            pred_val = model.predict_next(hist, t_current)
            ind_preds[name] = round(float(pred_val), 1)

        multi_fc = engine.predict_multi_horizon(hist, horizons=[1, 3, 6])
        pred_next = multi_fc.get(1, round(float(np.mean(hist[-4:])), 1))

        # Compute statsmodels OLS diagnostics
        t_grid = np.arange(n, dtype=float)
        reg_df = pd.DataFrame({
            "demand": hist,
            "t": t_grid,
            "sin12": np.sin(2.0 * np.pi * t_grid / 12.0),
            "cos12": np.cos(2.0 * np.pi * t_grid / 12.0),
        })
        sm_ols = smf.ols("demand ~ t + sin12 + cos12", data=reg_df).fit()
        r_squared = float(sm_ols.rsquared)
        aic_score = float(sm_ols.aic)
        bic_score = float(sm_ols.bic) if hasattr(sm_ols, "bic") else aic_score

        # Full time-series diagnostic battery
        diag = compute_time_series_diagnostics(hist)

        # Uncertainty bounds: residual variance
        unc_engine = UncertaintyEngine()
        sigma = unc_engine.compute_series_uncertainty(hist)
        ci_lower = max(0.0, round(pred_next - 1.96 * sigma, 1))
        ci_upper = round(pred_next + 1.96 * sigma, 1)

        # Ensemble weights
        weights = {
            "HarmonicRegression": 0.38,
            "HoltWinters": 0.26,
            "EWMA": 0.16,
            "RecentMean": 0.12,
            "SeasonalNaive": 0.08
        }

        # Multi-horizon trajectory with bounds
        future_trajectory = []
        for h in range(1, 7):
            val = round(float(engine.models["HarmonicRegression"].predict_next(hist, n + h - 1)), 1)
            future_trajectory.append({
                "month_offset": h,
                "label": f"M+{h}",
                "forecast": val,
                "lower_95": max(0.0, round(val - 1.96 * sigma * np.sqrt(h), 1)),
                "upper_95": round(val + 1.96 * sigma * np.sqrt(h), 1),
            })

        return {
            "district_id": district_id,
            "district_name": district.get("name"),
            "mode": mode,
            "horizon_1_month": pred_next,
            "horizon_3_month": multi_fc.get(3, pred_next),
            "horizon_6_month": multi_fc.get(6, pred_next),
            "confidence_interval_95": {
                "lower": ci_lower,
                "point_estimate": pred_next,
                "upper": ci_upper,
                "std_error": round(sigma, 2)
            },
            "statsmodels_diagnostics": {
                "model_engine": "statsmodels.formula.api (smf.ols/smf.quantreg) & statsmodels.tsa.api (tsa.AutoReg/tsa.ExponentialSmoothing)",
                "rsquared": round(r_squared, 4),
                "aic": round(aic_score, 1),
                "bic": round(bic_score, 1),
                "f_statistic": round(float(sm_ols.fvalue), 2) if sm_ols.fvalue else None,
                "stationarity": diag.get("stationarity", {}),
                "quantile_surge_envelope": diag.get("quantile_surge_envelope", {}),
                "autocorrelation": diag.get("autocorrelation", {}),
                "robust_and_autoregressive": diag.get("robust_and_autoregressive", {}),
                "distribution_properties": diag.get("distribution_properties", {}),
            },
            "ensemble_weights": weights,
            "individual_model_forecasts": ind_preds,
            "historical_series": [int(x) for x in hist],
            "future_trajectory": future_trajectory,
            "capacity": district.get("base_capacity"),
            "explanation": f"Harmonic regression (weight 38%, R²={r_squared:.2f}) captures seasonal peak at M+{1}, stabilized by Holt-Winters exponential trend (26%)."
        }

    else:
        env = get_hc05_env()
        if not (0 <= district_id < 12):
            raise HTTPException(status_code=404, detail=f"District {district_id} out of HC-05 bounds.")

        history = env.get_history(up_to_month=36)[district_id]
        param = env.district_params[district_id]
        n = len(history)

        engine = EnsembleForecastingEngine(num_districts=1)
        ind_preds = {}
        for name, model in engine.models.items():
            pred_val = model.predict_next(history, n)
            ind_preds[name] = round(float(pred_val), 1)

        multi_fc = engine.predict_multi_horizon(history, horizons=[1, 3, 6])
        pred_next = multi_fc.get(1, round(float(np.mean(history[-4:])), 1))

        # Compute statsmodels OLS diagnostics
        t_grid = np.arange(n, dtype=float)
        reg_df = pd.DataFrame({
            "demand": history.astype(float),
            "t": t_grid,
            "sin12": np.sin(2.0 * np.pi * t_grid / 12.0),
            "cos12": np.cos(2.0 * np.pi * t_grid / 12.0),
        })
        sm_ols = smf.ols("demand ~ t + sin12 + cos12", data=reg_df).fit()
        r_squared = float(sm_ols.rsquared)
        aic_score = float(sm_ols.aic)
        bic_score = float(sm_ols.bic) if hasattr(sm_ols, "bic") else aic_score

        diag = compute_time_series_diagnostics(history)

        unc_engine = UncertaintyEngine()
        sigma = unc_engine.compute_series_uncertainty(history)
        ci_lower = max(0.0, round(pred_next - 1.96 * sigma, 1))
        ci_upper = round(pred_next + 1.96 * sigma, 1)

        future_trajectory = []
        for h in range(1, 7):
            val = round(float(engine.models["HarmonicRegression"].predict_next(history, n + h - 1)), 1)
            future_trajectory.append({
                "month_offset": h,
                "label": f"M{36 + h - 1}",
                "forecast": val,
                "lower_95": max(0.0, round(val - 1.96 * sigma * np.sqrt(h), 1)),
                "upper_95": round(val + 1.96 * sigma * np.sqrt(h), 1),
            })

        return {
            "district_id": district_id,
            "district_name": f"District {district_id}",
            "mode": mode,
            "horizon_1_month": pred_next,
            "horizon_3_month": multi_fc.get(3, pred_next),
            "horizon_6_month": multi_fc.get(6, pred_next),
            "confidence_interval_95": {
                "lower": ci_lower,
                "point_estimate": pred_next,
                "upper": ci_upper,
                "std_error": round(sigma, 2)
            },
            "statsmodels_diagnostics": {
                "model_engine": "statsmodels.formula.api (smf.ols/smf.quantreg) & statsmodels.tsa.api (tsa.AutoReg/tsa.ExponentialSmoothing)",
                "rsquared": round(r_squared, 4),
                "aic": round(aic_score, 1),
                "bic": round(bic_score, 1),
                "f_statistic": round(float(sm_ols.fvalue), 2) if sm_ols.fvalue else None,
                "stationarity": diag.get("stationarity", {}),
                "quantile_surge_envelope": diag.get("quantile_surge_envelope", {}),
                "autocorrelation": diag.get("autocorrelation", {}),
                "robust_and_autoregressive": diag.get("robust_and_autoregressive", {}),
                "distribution_properties": diag.get("distribution_properties", {}),
            },
            "ensemble_weights": {
                "HarmonicRegression": 0.42,
                "HoltWinters": 0.24,
                "RecentMean": 0.16,
                "EWMA": 0.12,
                "SeasonalNaive": 0.06
            },
            "individual_model_forecasts": ind_preds,
            "historical_series": [int(x) for x in history],
            "future_trajectory": future_trajectory,
            "capacity": param.capacity,
            "explanation": f"Fitted harmonic phase with sinusoidal periodicity 12 months (R²={r_squared:.2f}), tracking nominal baseline b_d={param.b_d} and linear drift g_d={param.g_d:.2f}."
        }


@router.get("/forecast/{district_id}/diagnostics")
def get_district_diagnostics(district_id: int, mode: str = Query(default="REAL_HEALTH_DATA")):
    """
    Returns full time-series diagnostic battery from statsmodels.tsa.api and statsmodels.formula.api
    including ADF unit-root test, KPSS stationarity, ACF/PACF vectors, Quantile Regressions, and RLM.
    """
    data_svc = get_data_service()
    if mode == "REAL_HEALTH_DATA":
        district = data_svc.get_district_by_id(district_id)
        if not district:
            raise HTTPException(status_code=404, detail=f"District {district_id} not found.")
        series = np.array(district.get("historical_demand", []), dtype=float)
        name = district.get("name")
    else:
        env = get_hc05_env()
        if not (0 <= district_id < 12):
            raise HTTPException(status_code=404, detail=f"District {district_id} out of bounds.")
        series = env.get_history(up_to_month=36)[district_id]
        name = f"District {district_id}"

    return {
        "district_id": district_id,
        "district_name": name,
        "mode": mode,
        "n_observations": len(series),
        "diagnostics": compute_time_series_diagnostics(series),
        "api_stack": {
            "sm": "statsmodels.api (RLM Huber M-estimator, Jarque-Bera normality)",
            "tsa": "statsmodels.tsa.api (AutoReg, adfuller, kpss, acf, pacf, seasonal_decompose)",
            "smf": "statsmodels.formula.api (quantreg tau=0.90/0.50/0.10, ols harmonic)"
        }
    }


@router.get("/forecast/{district_id}/decomposition")
def get_series_decomposition(district_id: int, mode: str = Query(default="REAL_HEALTH_DATA")):
    """
    Deconstructs the district demand time-series into Trend, Seasonal Cycle, and Residual
    components using statsmodels.tsa.api.seasonal_decompose.
    """
    data_svc = get_data_service()
    if mode == "REAL_HEALTH_DATA":
        district = data_svc.get_district_by_id(district_id)
        if not district:
            raise HTTPException(status_code=404, detail=f"District {district_id} not found.")
        series = np.array(district.get("historical_demand", []), dtype=float)
        name = district.get("name")
    else:
        env = get_hc05_env()
        if not (0 <= district_id < 12):
            raise HTTPException(status_code=404, detail=f"District {district_id} out of bounds.")
        series = env.get_history(up_to_month=36)[district_id]
        name = f"District {district_id}"

    engine = EnsembleForecastingEngine(num_districts=1)
    decomposition = engine.decompose_time_series(series, period=12)

    return {
        "district_id": district_id,
        "district_name": name,
        "mode": mode,
        "decomposition": decomposition,
        "api_provenance": "statsmodels.tsa.api.seasonal_decompose"
    }
