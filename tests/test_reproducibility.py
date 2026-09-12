"""Tests for deterministic reproducibility with NumPy PCG64 seed 20260911."""

import numpy as np
from backend.services.hc05_engine import HC05Environment
from fastapi.testclient import TestClient
from backend.main import app


def test_environment_reproducibility():
    """Verify PCG64 seed 20260911 is completely deterministic."""
    env1 = HC05Environment(seed=20260911, add_surge=True)
    env2 = HC05Environment(seed=20260911, add_surge=True)

    np.testing.assert_array_equal(env1.nominal_capacity, env2.nominal_capacity)
    np.testing.assert_array_equal(env1.demand, env2.demand)
    np.testing.assert_array_almost_equal(env1.mu_demand, env2.mu_demand)
    np.testing.assert_array_almost_equal(env1.noise, env2.noise)


def test_evaluation_reproducibility():
    """Verify evaluation API produces exact deterministic numbers."""
    client = TestClient(app)
    res1 = client.get("/api/evaluation").json()
    res2 = client.get("/api/evaluation").json()

    assert res1["metrics_summary"]["u_service"] == res2["metrics_summary"]["u_service"]
    assert res1["metrics_summary"]["u_worst"] == res2["metrics_summary"]["u_worst"]
    assert res1["metrics_summary"]["total_unmet_demand"] == res2["metrics_summary"]["total_unmet_demand"]
    assert res1["metrics_summary"]["total_unmet_demand"] <= 2160
    assert res1["metrics_summary"]["u_worst"] >= 0.6300
