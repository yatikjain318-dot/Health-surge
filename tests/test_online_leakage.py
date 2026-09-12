"""Tests ensuring strict online information rule (zero future leakage)."""

import numpy as np
from backend.services.hc05_engine import HC05Environment
from backend.services.forecasting_service import EnsembleForecastingEngine


def test_strict_online_data_availability():
    """Verify get_history(t) returns exactly t columns."""
    env = HC05Environment(seed=20260911)
    for t in range(36, 42):
        hist_t = env.get_history(up_to_month=t)
        assert hist_t.shape == (12, t)


def test_zero_future_leakage():
    """
    Verify that altering future months (t+1..41) does not change
    the model's prediction at month t.
    """
    env1 = HC05Environment(seed=20260911, add_surge=True)
    env2 = HC05Environment(seed=20260911, add_surge=False)

    # In env1, surge is added at D2 and D9 during months 38, 39, 40.
    # At month 36 (eval start), both environments MUST yield identical history up to month 36
    # because the surge hasn't happened yet!
    hist1_36 = env1.get_history(36)
    hist2_36 = env2.get_history(36)
    np.testing.assert_array_equal(hist1_36, hist2_36)

    # Consequently, forecaster initialized on history up to month 36 produces identical predictions
    f1 = EnsembleForecastingEngine(num_districts=12)
    f2 = EnsembleForecastingEngine(num_districts=12)

    pred1, _ = f1.predict(hist1_36, t=36)
    pred2, _ = f2.predict(hist2_36, t=36)
    np.testing.assert_array_equal(pred1, pred2)
