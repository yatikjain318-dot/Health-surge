"""Tests for HC-05 challenge hard constraints compliance."""

import numpy as np
from backend.services.hc05_engine import get_hc05_env
from backend.services.allocation_service import AllocationService


def test_capacity_formula():
    """Verify nominal capacity is strictly round(0.90 * b_d)."""
    env = get_hc05_env()
    for d in range(env.NUM_DISTRICTS):
        param = env.district_params[d]
        expected_cap = int(np.round(0.90 * param.b_d))
        assert param.capacity == expected_cap
        assert env.nominal_capacity[d] == expected_cap


def test_allocation_hard_constraints():
    """Verify sum(a_d) <= 60, a_d >= 0, a_d is integer for arbitrary inputs."""
    alloc_svc = AllocationService()
    rng = np.random.default_rng(42)

    for trial in range(25):
        forecasts = rng.integers(50, 200, size=12).astype(float)
        capacities = rng.integers(40, 150, size=12).astype(float)
        stds = rng.uniform(5.0, 15.0, size=12)
        surge_risks = rng.uniform(0.0, 1.0, size=12)
        cum_serv = rng.uniform(0.5, 1.0, size=12)
        budget = 60

        alloc, details = alloc_svc.optimize_allocation(
            forecasts=forecasts,
            stds=stds,
            capacities=capacities,
            surge_risks=surge_risks,
            cumulative_service=cum_serv,
            total_budget=budget
        )

        # 1. Budget sum <= 60
        assert int(alloc.sum()) <= budget, f"Exceeded budget: {alloc.sum()}"

        # 2. Non-negative
        assert np.all(alloc >= 0), "Negative allocation detected"

        # 3. Strictly integers
        assert np.issubdtype(alloc.dtype, np.integer), "Non-integer allocation detected"

        # 4. Correct length
        assert len(alloc) == 12


def test_custom_budgets():
    """Verify allocator satisfies arbitrary budgets from 0 to 120."""
    alloc_svc = AllocationService()
    forecasts = np.array([120.0] * 12)
    capacities = np.array([100.0] * 12)
    stds = np.array([8.0] * 12)
    surge = np.array([0.5] * 12)
    cum_serv = np.array([0.9] * 12)

    for b in [0, 10, 45, 60, 100, 120]:
        alloc, _ = alloc_svc.optimize_allocation(forecasts, stds, capacities, surge, cum_serv, total_budget=b)
        assert alloc.sum() <= b
        assert np.all(alloc >= 0)
