"""Resource Allocation Engine (Gaussian Expected Shortage, Marginal Allocator & Explanations)."""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np


def normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def expected_shortage_gaussian(mu: float, sigma: float, capacity: float) -> float:
    if sigma <= 1e-6:
        return max(0.0, mu - capacity)
    z = (capacity - mu) / sigma
    phi_z = normal_pdf(z)
    Phi_z = normal_cdf(z)
    return max(0.0, float(sigma * phi_z + (mu - capacity) * (1.0 - Phi_z)))


class AllocationService:
    def __init__(
        self,
        w_unc: float = 0.05,
        w_surge: float = 0.20,
        w_fair: float = 0.50,
        fairness_target: float = 0.95,
    ):
        self.w_unc = w_unc
        self.w_surge = w_surge
        self.w_fair = w_fair
        self.fairness_target = fairness_target

    def optimize_allocation(
        self,
        forecasts: np.ndarray,
        stds: np.ndarray,
        capacities: np.ndarray,
        surge_risks: np.ndarray,
        cumulative_service: np.ndarray,
        total_budget: int = 60,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Greedy integer marginal allocation of total_budget reserve units.
        Guarantees: non-negative, integer, sum <= total_budget.
        Returns:
            allocation_vector: shape (n_districts,)
            district_allocation_details: explanation list
        """
        n = len(forecasts)
        allocation = np.zeros(n, dtype=int)
        mean_service = float(np.mean(cumulative_service)) if len(cumulative_service) > 0 else 1.0
        target = max(self.fairness_target, mean_service)

        # Unit-by-unit greedy allocation
        for _ in range(total_budget):
            best_district = -1
            best_benefit = -float("inf")

            for d in range(n):
                current_cap = float(capacities[d] + allocation[d])
                mu = float(forecasts[d])
                sigma = float(stds[d])

                shortage_now = expected_shortage_gaussian(mu, sigma, current_cap)
                shortage_next = expected_shortage_gaussian(mu, sigma, current_cap + 1.0)
                delta_s = max(0.0, shortage_now - shortage_next)

                unc_value = (sigma / 10.0) * delta_s
                surge_value = float(surge_risks[d]) * delta_s
                service_d = float(cumulative_service[d])
                fairness_boost = max(0.0, target - service_d) * delta_s

                total_benefit = (
                    delta_s
                    + self.w_unc * unc_value
                    + self.w_surge * surge_value
                    + self.w_fair * fairness_boost
                )

                if total_benefit > best_benefit:
                    best_benefit = total_benefit
                    best_district = d

            allocation[best_district] += 1

        # Build Explainable Allocation Details
        details = []
        for d in range(n):
            mu = float(forecasts[d])
            sigma = float(stds[d])
            base_cap = float(capacities[d])
            alloc = int(allocation[d])

            shortage_without = expected_shortage_gaussian(mu, sigma, base_cap)
            shortage_with = expected_shortage_gaussian(mu, sigma, base_cap + alloc)
            improvement_pct = (
                ((shortage_without - shortage_with) / max(shortage_without, 1e-6)) * 100.0
                if shortage_without > 0
                else 0.0
            )

            details.append({
                "district_id": d,
                "forecast": int(forecasts[d]),
                "base_capacity": int(capacities[d]),
                "allocated_reserve": alloc,
                "effective_capacity": int(capacities[d] + alloc),
                "expected_shortage_without": round(shortage_without, 1),
                "expected_shortage_with": round(shortage_with, 1),
                "expected_shortage_before": round(shortage_without, 1),
                "expected_shortage_after": round(shortage_with, 1),
                "shortage_reduction_pct": round(improvement_pct, 1),
                "reduction_pct": round(improvement_pct, 1),
                "surge_risk": round(float(surge_risks[d]), 3),
                "worst_district_protected": bool(cumulative_service[d] < mean_service and alloc > 0),
                "rationale": (
                    f"Without reserve: Expected unmet demand = {shortage_without:.1f}. "
                    f"With +{alloc} reserve units: Expected unmet demand = {shortage_with:.1f}. "
                    f"Shortage mitigated by {improvement_pct:.1f}%."
                ),
                "explanation": (
                    f"Without allocation: Expected unmet demand = {shortage_without:.1f}. "
                    f"With allocation: Expected unmet demand = {shortage_with:.1f}. "
                    f"Shortage reduced by {improvement_pct:.1f}%."
                ),
            })

        return allocation, details
