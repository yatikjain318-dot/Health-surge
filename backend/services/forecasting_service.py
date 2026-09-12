"""Forecasting Models, Multi-Horizon Projections, and Ensemble Engine powered by Statsmodels."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Canonical Statsmodels APIs as requested
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import statsmodels.formula.api as smf


class BaseForecaster(ABC):
    @abstractmethod
    def predict_next(self, series: np.ndarray, t: int) -> float:
        pass


class SeasonalNaiveModel(BaseForecaster):
    def predict_next(self, series: np.ndarray, t: int) -> float:
        if len(series) >= 12:
            return float(series[-12])
        return float(np.mean(series)) if len(series) > 0 else 100.0


class RecentMeanModel(BaseForecaster):
    def __init__(self, window: int = 4):
        self.window = window

    def predict_next(self, series: np.ndarray, t: int) -> float:
        n = len(series)
        if n == 0:
            return 100.0
        k = min(self.window, n)
        recent = series[-k:].astype(float)
        weights = np.arange(1, k + 1, dtype=float)
        weights /= weights.sum()
        return float(np.dot(recent, weights))


class EWMAModel(BaseForecaster):
    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha

    def predict_next(self, series: np.ndarray, t: int) -> float:
        if len(series) == 0:
            return 100.0
        try:
            # Using statsmodels.tsa.api as tsa: Simple Exponential Smoothing
            ses_fit = tsa.SimpleExpSmoothing(series.astype(float)).fit(
                smoothing_level=self.alpha, optimized=False
            )
            forecast = ses_fit.forecast(1)
            return max(0.0, float(forecast.iloc[0] if hasattr(forecast, "iloc") else forecast[0]))
        except Exception:
            s = float(series[0])
            for val in series[1:]:
                s = self.alpha * float(val) + (1.0 - self.alpha) * s
            return max(0.0, s)


class HarmonicRegressionModel(BaseForecaster):
    """Trigonometric Fourier Seasonal Regression using statsmodels.formula.api as smf."""

    def predict_next(self, series: np.ndarray, t: int) -> float:
        n = len(series)
        if n < 6:
            return float(np.mean(series)) if n > 0 else 100.0

        t_hist = np.arange(n, dtype=float)
        df = pd.DataFrame({
            "demand": series.astype(float),
            "t": t_hist,
            "sin12": np.sin(2.0 * np.pi * t_hist / 12.0),
            "cos12": np.cos(2.0 * np.pi * t_hist / 12.0),
        })

        try:
            # Canonical statsmodels.formula.api as smf: OLS formula estimation
            model = smf.ols("demand ~ t + sin12 + cos12", data=df).fit()
            pred_df = pd.DataFrame({
                "t": [float(t)],
                "sin12": [np.sin(2.0 * np.pi * float(t) / 12.0)],
                "cos12": [np.cos(2.0 * np.pi * float(t) / 12.0)],
            })
            pred_val = float(model.predict(pred_df).iloc[0])
            return max(0.0, pred_val)
        except Exception:
            # Fallback to statsmodels.api as sm: cross-sectional OLS with constant
            X = sm.add_constant(np.column_stack([
                t_hist,
                np.sin(2.0 * np.pi * t_hist / 12.0),
                np.cos(2.0 * np.pi * t_hist / 12.0),
            ]))
            model_sm = sm.OLS(series.astype(float), X).fit()
            x_next = np.array([1.0, float(t), np.sin(2.0 * np.pi * float(t) / 12.0), np.cos(2.0 * np.pi * float(t) / 12.0)])
            return max(0.0, float(model_sm.predict(x_next)[0]))


class HoltWintersModel(BaseForecaster):
    """Triple Exponential Smoothing with additive trend and seasonality using statsmodels.tsa.api as tsa."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.3, season_period: int = 12):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.period = season_period

    def predict_next(self, series: np.ndarray, t: int) -> float:
        n = len(series)
        if n < self.period * 2:
            return HarmonicRegressionModel().predict_next(series, t)

        try:
            # Canonical statsmodels.tsa.api as tsa: ExponentialSmoothing
            es_model = tsa.ExponentialSmoothing(
                series.astype(float),
                seasonal_periods=self.period,
                trend="add",
                seasonal="add",
                initialization_method="estimated"
            ).fit(
                smoothing_level=self.alpha,
                smoothing_trend=self.beta,
                smoothing_seasonal=self.gamma,
                optimized=False
            )
            steps = max(1, t - n + 1)
            forecast = es_model.forecast(steps)
            val = float(forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1])
            return max(0.0, val)
        except Exception:
            # Robust manual recurrence fallback
            p = self.period
            level = float(np.mean(series[:p]))
            trend = float(np.mean(series[p:2*p]) - np.mean(series[:p])) / float(p)
            seasonals = [float(series[i] - level) for i in range(p)]
            mean_season = float(np.mean(seasonals))
            seasonals = [s - mean_season for s in seasonals]

            for i in range(n):
                val = float(series[i])
                s_idx = i % p
                prev_level = level
                level = self.alpha * (val - seasonals[s_idx]) + (1.0 - self.alpha) * (level + trend)
                trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend
                seasonals[s_idx] = self.gamma * (val - level) + (1.0 - self.gamma) * seasonals[s_idx]

            next_s_idx = n % p
            forecast = level + trend + seasonals[next_s_idx]
            return max(0.0, float(forecast))


class AutoRegModel(BaseForecaster):
    """Autoregressive Time Series Model using statsmodels.tsa.api as tsa."""

    def __init__(self, lags: int = 2):
        self.lags = lags

    def predict_next(self, series: np.ndarray, t: int) -> float:
        n = len(series)
        if n < self.lags + 2:
            return float(np.mean(series)) if n > 0 else 100.0
        try:
            p = min(self.lags, max(1, n // 4))
            ar_model = tsa.AutoReg(series.astype(float), lags=p, trend="c").fit()
            pred = ar_model.forecast(1)
            val = float(pred.iloc[0] if hasattr(pred, "iloc") else pred[0])
            return max(0.0, val)
        except Exception:
            return float(np.mean(series[-3:]))


class RobustHarmonicRegressionModel(BaseForecaster):
    """
    Outlier-resistant Harmonic Regression using statsmodels.api as sm (sm.RLM with Huber M-estimator).
    Protects trend estimation against anomalous epidemiological outbreak spikes.
    """

    def predict_next(self, series: np.ndarray, t: int) -> float:
        n = len(series)
        if n < 6:
            return float(np.mean(series)) if n > 0 else 100.0

        t_hist = np.arange(n, dtype=float)
        X = sm.add_constant(np.column_stack([
            t_hist,
            np.sin(2.0 * np.pi * t_hist / 12.0),
            np.cos(2.0 * np.pi * t_hist / 12.0),
        ]))
        try:
            rlm_model = sm.RLM(series.astype(float), X, M=sm.robust.norms.HuberT()).fit()
            x_next = np.array([1.0, float(t), np.sin(2.0 * np.pi * float(t) / 12.0), np.cos(2.0 * np.pi * float(t) / 12.0)])
            return max(0.0, float(rlm_model.predict(x_next)[0]))
        except Exception:
            return float(np.mean(series))


class WeightedHarmonicRegressionModel(BaseForecaster):
    """
    Weighted Least Squares (sm.WLS) with exponential recency decay weights.
    Prioritizes recent structural changes over distant historical patterns.
    """

    def __init__(self, decay: float = 0.95):
        self.decay = decay

    def predict_next(self, series: np.ndarray, t: int) -> float:
        n = len(series)
        if n < 6:
            return float(np.mean(series)) if n > 0 else 100.0

        t_hist = np.arange(n, dtype=float)
        weights = self.decay ** (n - 1 - t_hist)
        X = sm.add_constant(np.column_stack([
            t_hist,
            np.sin(2.0 * np.pi * t_hist / 12.0),
            np.cos(2.0 * np.pi * t_hist / 12.0),
        ]))
        try:
            wls_model = sm.WLS(series.astype(float), X, weights=weights).fit()
            x_next = np.array([1.0, float(t), np.sin(2.0 * np.pi * float(t) / 12.0), np.cos(2.0 * np.pi * float(t) / 12.0)])
            return max(0.0, float(wls_model.predict(x_next)[0]))
        except Exception:
            return float(np.mean(series))


class QuantileRegressionForecaster:
    """
    Tail-Risk Surge Forecasting using statsmodels.formula.api as smf (QuantReg).
    Estimates upper surge quantiles (tau=0.90) vs median (tau=0.50).
    """

    def __init__(self, quantiles: Optional[List[float]] = None):
        self.quantiles = quantiles or [0.10, 0.50, 0.90]

    def fit_predict_quantiles(self, series: np.ndarray, t_target: int) -> Dict[float, float]:
        n = len(series)
        if n < 6:
            mean_val = float(np.mean(series)) if n > 0 else 100.0
            return {q: mean_val for q in self.quantiles}

        t_hist = np.arange(n, dtype=float)
        df = pd.DataFrame({
            "demand": series.astype(float),
            "t": t_hist,
            "sin12": np.sin(2.0 * np.pi * t_hist / 12.0),
            "cos12": np.cos(2.0 * np.pi * t_hist / 12.0),
        })

        pred_row = pd.DataFrame({
            "t": [float(t_target)],
            "sin12": [np.sin(2.0 * np.pi * float(t_target) / 12.0)],
            "cos12": [np.cos(2.0 * np.pi * float(t_target) / 12.0)],
        })

        results = {}
        for q in self.quantiles:
            try:
                mod = smf.quantreg("demand ~ t + sin12 + cos12", data=df).fit(q=q, max_iter=2000)
                pred_val = float(mod.predict(pred_row).iloc[0])
                results[q] = max(0.0, round(pred_val, 1))
            except Exception:
                base = float(np.mean(series[-4:]))
                std_est = float(np.std(series, ddof=1)) if len(series) > 1 else 10.0
                z = -1.28 if q == 0.10 else (0.0 if q == 0.50 else 1.28)
                results[q] = max(0.0, round(base + z * std_est, 1))
        return results


def compute_time_series_diagnostics(series: np.ndarray) -> Dict:
    """
    Comprehensive battery of statistical time-series tests using:
    - statsmodels.tsa.api as tsa (adfuller, kpss, acf, pacf, AutoReg)
    - statsmodels.formula.api as smf (quantreg, ols)
    - statsmodels.api as sm (RLM, OLS, jarque_bera)
    """
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n < 6:
        return {
            "status": "insufficient_data",
            "message": "At least 6 time periods required for statistical diagnostics."
        }

    # 1. ADF Unit Root Test (tsa.stattools.adfuller)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_res = tsa.stattools.adfuller(s, autolag="AIC")
        adf_stat = float(adf_res[0])
        adf_pvalue = float(adf_res[1])
        adf_usedlag = int(adf_res[2])
        is_stationary = bool(adf_pvalue < 0.05)
    except Exception:
        adf_stat, adf_pvalue, adf_usedlag, is_stationary = 0.0, 0.5, 0, False

    # 2. KPSS Stationarity Test (tsa.stattools.kpss)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_res = tsa.stattools.kpss(s, regression="c", nlags="auto")
        kpss_stat = float(kpss_res[0])
        kpss_pvalue = float(kpss_res[1])
    except Exception:
        kpss_stat, kpss_pvalue = 0.0, 0.10

    # 3. Autocorrelation (tsa.stattools.acf and pacf)
    try:
        max_nlags = min(12, n // 2)
        acf_vals = tsa.stattools.acf(s, nlags=max_nlags).tolist()
        pacf_vals = tsa.stattools.pacf(s, nlags=min(4, max_nlags)).tolist()
    except Exception:
        acf_vals = [1.0]
        pacf_vals = [1.0]

    # 4. Quantile Regression (smf.quantreg)
    q_forecaster = QuantileRegressionForecaster(quantiles=[0.10, 0.50, 0.90])
    q_preds = q_forecaster.fit_predict_quantiles(s, n)

    # 5. Robust M-Estimation (sm.RLM)
    rlm_forecaster = RobustHarmonicRegressionModel()
    rlm_pred = rlm_forecaster.predict_next(s, n)

    # 6. Autoregressive AR(p) (tsa.AutoReg)
    ar_forecaster = AutoRegModel(lags=min(2, max(1, n // 6)))
    ar_pred = ar_forecaster.predict_next(s, n)

    # 7. Descriptive & Normality (sm.stats.stattools.jarque_bera)
    try:
        jb_stat, jb_pvalue, skew, kurtosis = sm.stats.stattools.jarque_bera(s)
    except Exception:
        jb_stat, jb_pvalue, skew, kurtosis = 0.0, 1.0, 0.0, 3.0

    return {
        "stationarity": {
            "adfuller": {
                "test_statistic": round(adf_stat, 3),
                "p_value": round(adf_pvalue, 4),
                "lags_used": adf_usedlag,
                "is_stationary": is_stationary,
                "null_hypothesis": "Unit root present (non-stationary)"
            },
            "kpss": {
                "test_statistic": round(kpss_stat, 3),
                "p_value": round(kpss_pvalue, 4),
                "null_hypothesis": "Data is trend/level stationary"
            }
        },
        "autocorrelation": {
            "acf_lags": [round(float(x), 3) for x in acf_vals],
            "pacf_lags": [round(float(x), 3) for x in pacf_vals],
            "lag_1_persistence": round(float(acf_vals[1]), 3) if len(acf_vals) > 1 else 0.0,
            "lag_12_seasonality": round(float(acf_vals[12]), 3) if len(acf_vals) > 12 else 0.0
        },
        "quantile_surge_envelope": {
            "tau_10_floor": q_preds.get(0.10, 0.0),
            "tau_50_median": q_preds.get(0.50, 0.0),
            "tau_90_surge_ceiling": q_preds.get(0.90, 0.0),
            "surge_headroom_margin": round(q_preds.get(0.90, 0.0) - q_preds.get(0.50, 0.0), 1),
            "method": "statsmodels.formula.api.quantreg"
        },
        "robust_and_autoregressive": {
            "rlm_huber_forecast": round(float(rlm_pred), 1),
            "autoreg_forecast": round(float(ar_pred), 1),
            "methods": ["statsmodels.api.RLM", "statsmodels.tsa.api.AutoReg"]
        },
        "distribution_properties": {
            "skewness": round(float(skew), 3),
            "kurtosis": round(float(kurtosis), 3),
            "jarque_bera_stat": round(float(jb_stat), 2),
            "jarque_bera_pvalue": round(float(jb_pvalue), 4),
            "is_normal": bool(jb_pvalue > 0.05)
        }
    }


def fit_cross_sectional_ols(y: np.ndarray, X: np.ndarray) -> Dict:
    """
    Uses statsmodels.api as sm for cross-sectional regression models.
    Returns coefficients, robust standard errors (HC1), p-values, and R-squared.
    """
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit(cov_type="HC1")
    return {
        "params": [float(c) for c in model.params],
        "bse": [float(se) for se in model.bse],
        "pvalues": [float(p) for p in model.pvalues],
        "rsquared": float(model.rsquared),
        "fvalue": float(model.fvalue) if model.fvalue is not None else 0.0,
    }


class EnsembleForecastingEngine:
    """Ensemble of 5 models with rolling historical backtesting, seasonal decomposition, and bias correction."""

    MODEL_NAMES = ["SeasonalNaive", "RecentMean", "EWMA", "HarmonicRegression", "HoltWinters"]

    def __init__(self, num_districts: int = 12, bias_decay: float = 0.70):
        self.num_districts = num_districts
        self.bias_decay = bias_decay
        self.models: Dict[str, BaseForecaster] = {
            "SeasonalNaive": SeasonalNaiveModel(),
            "RecentMean": RecentMeanModel(),
            "EWMA": EWMAModel(),
            "HarmonicRegression": HarmonicRegressionModel(),
            "HoltWinters": HoltWintersModel(),
        }
        self.weights = np.ones((num_districts, len(self.MODEL_NAMES)), dtype=float) / len(self.MODEL_NAMES)
        self.bias = np.zeros(num_districts, dtype=float)

    def decompose_time_series(self, series: np.ndarray, period: int = 12) -> Dict:
        """
        Uses statsmodels.tsa.api as tsa for additive seasonal decomposition.
        Deconstructs series into trend, seasonal cycle, and residual components.
        """
        n = len(series)
        if n < period * 2:
            return {
                "observed": [float(x) for x in series],
                "trend": [float(x) for x in series],
                "seasonal": [0.0] * n,
                "residual": [0.0] * n,
                "method": "Insufficient history for full STL decomposition (requires >= 24 months)"
            }
        decomp = tsa.seasonal_decompose(pd.Series(series.astype(float)), model="additive", period=period)
        trend = decomp.trend.bfill().ffill().values
        seasonal = decomp.seasonal.values
        residual = decomp.resid.fillna(0.0).values
        return {
            "observed": [round(float(x), 1) for x in series],
            "trend": [round(float(x), 1) for x in trend],
            "seasonal": [round(float(x), 1) for x in seasonal],
            "residual": [round(float(x), 1) for x in residual],
            "method": "statsmodels.tsa.api.seasonal_decompose (Additive, Period=12)"
        }

    def optimize_weights_backtest(self, history: np.ndarray, start_month: int = 24, power: float = 2.0):
        total_months = history.shape[1]
        abs_errors = {d: [[] for _ in range(len(self.MODEL_NAMES))] for d in range(self.num_districts)}

        for k in range(start_month, total_months):
            for d in range(self.num_districts):
                d_series_k = history[d, :k]
                actual_val = float(history[d, k])
                for m_idx, name in enumerate(self.MODEL_NAMES):
                    pred = self.models[name].predict_next(d_series_k, k)
                    abs_errors[d][m_idx].append(abs(actual_val - pred))

        for d in range(self.num_districts):
            maes = np.array([float(np.mean(abs_errors[d][m])) for m in range(len(self.MODEL_NAMES))])
            inv_mae = 1.0 / (maes + 1e-4) ** power
            self.weights[d, :] = inv_mae / inv_mae.sum()

    def predict(self, history: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.zeros((self.num_districts, len(self.MODEL_NAMES)), dtype=float)
        for d in range(self.num_districts):
            d_series = history[d, :]
            for m_idx, name in enumerate(self.MODEL_NAMES):
                preds[d, m_idx] = self.models[name].predict_next(d_series, t)

        raw_ensemble = np.sum(preds * self.weights, axis=1)
        corrected = raw_ensemble + self.bias
        final_forecast = np.maximum(0, np.round(corrected)).astype(int)
        return final_forecast, raw_ensemble

    def update_bias(self, actual: np.ndarray, predicted: np.ndarray):
        err = actual.astype(float) - predicted.astype(float)
        self.bias = self.bias_decay * self.bias + (1.0 - self.bias_decay) * err

    def predict_multi_horizon(self, series: np.ndarray, horizons: List[int] = [1, 3, 6]) -> Dict[int, float]:
        """Generate forecasts for +1, +3, +6 months ahead using statsmodels harmonic regression."""
        results = {}
        curr_series = list(series.astype(float))
        n = len(curr_series)

        for h in range(1, max(horizons) + 1):
            pred = self.models["HarmonicRegression"].predict_next(np.array(curr_series), n + h - 1)
            if h in horizons:
                results[h] = round(float(pred), 1)
            curr_series.append(pred)

        return results
