"""Forecast Uncertainty Engine (Residual Standard Deviation & Confidence Intervals)."""

from typing import Dict, List, Tuple, Union
import numpy as np


class UncertaintyService:
    def __init__(self, num_districts: int = 12, recent_window: int = 6, z_critical: float = 1.96):
        self.num_districts = num_districts
        self.recent_window = recent_window
        self.z_critical = z_critical
        self.residuals: List[List[float]] = [[] for _ in range(num_districts)]

    def initialize_residuals(self, history: np.ndarray, predictions: np.ndarray):
        num_districts, t_len = history.shape
        for d in range(num_districts):
            for t in range(t_len):
                res = float(history[d, t]) - float(predictions[d, t])
                self.residuals[d].append(res)

    def compute_series_uncertainty(self, series: np.ndarray) -> float:
        """Estimates residual standard deviation from a 1D demand series."""
        n = len(series)
        if n < 4:
            return 8.0
        diffs = np.diff(series.astype(float))
        sigma = float(np.std(diffs, ddof=1)) / np.sqrt(2.0)
        return float(np.clip(sigma, 6.0, 25.0))

    def estimate_uncertainty(self, *args, **kwargs) -> Union[Dict, np.ndarray]:
        """
        Supports:
        1. estimate_uncertainty(history: np.ndarray, t: int = ...) -> np.ndarray of stds for each district
        2. estimate_uncertainty(district_id: int, forecast_val: float) -> Dict of bounds
        """
        # Check if first positional arg is a 2D numpy array (history)
        if len(args) > 0 and isinstance(args[0], np.ndarray) and args[0].ndim == 2:
            history = args[0]
            num_dist = history.shape[0]
            stds = np.zeros(num_dist, dtype=float)
            for d in range(num_dist):
                stds[d] = self.compute_series_uncertainty(history[d, :])
            return stds

        if "history" in kwargs and isinstance(kwargs["history"], np.ndarray) and kwargs["history"].ndim == 2:
            history = kwargs["history"]
            num_dist = history.shape[0]
            stds = np.zeros(num_dist, dtype=float)
            for d in range(num_dist):
                stds[d] = self.compute_series_uncertainty(history[d, :])
            return stds

        district_id = int(kwargs.get("district_id", args[0] if len(args) > 0 else 0))
        forecast_val = float(kwargs.get("forecast_val", args[1] if len(args) > 1 else 100.0))

        res_list = self.residuals[district_id] if 0 <= district_id < len(self.residuals) else []
        if len(res_list) >= 3:
            res_arr = np.array(res_list, dtype=float)
            res_std = float(np.std(res_arr, ddof=1))
            k = min(self.recent_window, len(res_arr))
            recent_std = float(np.std(res_arr[-k:], ddof=1)) if k > 1 else res_std
            eff_std = float(np.sqrt(0.5 * (res_std ** 2) + 0.5 * (recent_std ** 2) + 4.0))
        else:
            eff_std = 8.0

        lower_bound = max(0.0, forecast_val - self.z_critical * eff_std)
        upper_bound = forecast_val + self.z_critical * eff_std

        uncertainty_level = "LOW"
        if eff_std > 15.0:
            uncertainty_level = "HIGH"
        elif eff_std > 10.0:
            uncertainty_level = "MEDIUM"

        return {
            "forecast_std": round(eff_std, 2),
            "lower_bound": int(np.round(lower_bound)),
            "upper_bound": int(np.round(upper_bound)),
            "confidence_interval_pct": 95,
            "uncertainty_level": uncertainty_level
        }

    def update(self, actual: np.ndarray, predicted: np.ndarray):
        for d in range(min(self.num_districts, len(actual))):
            res = float(actual[d]) - float(predicted[d])
            self.residuals[d].append(res)


UncertaintyEngine = UncertaintyService
