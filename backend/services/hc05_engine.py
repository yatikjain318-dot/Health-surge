"""Official HC-05 Challenge Generative Environment and Simulation Core."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DistrictParameters:
    district_id: int
    b_d: int
    a_d: float
    g_d: float
    capacity: int


class HC05Environment:
    """Official 12-District Healthcare Time-Series Environment for HC-05."""

    NUM_DISTRICTS: int = 12
    TOTAL_MONTHS: int = 42
    HISTORICAL_MONTHS: int = 36
    EVAL_START_MONTH: int = 36
    EVAL_MONTHS: int = 6
    RESERVE_BUDGET: int = 60
    NOISE_STD: float = 8.0

    SURGE_DISTRICTS: Tuple[int, ...] = (2, 9)
    SURGE_MONTHS: Tuple[int, ...] = (38, 39, 40)
    SURGE_MAGNITUDE: int = 35

    def __init__(self, seed: int = 20260911, add_surge: bool = True):
        self.seed = seed
        self.add_surge = add_surge
        self.rng = np.random.Generator(np.random.PCG64(seed))

        self.district_params: List[DistrictParameters] = []
        self.nominal_capacity: np.ndarray = np.zeros(self.NUM_DISTRICTS, dtype=int)
        self.mu_demand: np.ndarray = np.zeros((self.NUM_DISTRICTS, self.TOTAL_MONTHS), dtype=float)
        self.noise: np.ndarray = np.zeros((self.NUM_DISTRICTS, self.TOTAL_MONTHS), dtype=float)
        self.demand: np.ndarray = np.zeros((self.NUM_DISTRICTS, self.TOTAL_MONTHS), dtype=int)

        self._initialize()

    def _initialize(self):
        b_samples = self.rng.integers(60, 161, size=self.NUM_DISTRICTS)
        a_samples = self.rng.uniform(10.0, 40.0, size=self.NUM_DISTRICTS)
        g_samples = self.rng.uniform(-0.5, 1.5, size=self.NUM_DISTRICTS)

        for d in range(self.NUM_DISTRICTS):
            b_d = int(b_samples[d])
            a_d = float(a_samples[d])
            g_d = float(g_samples[d])
            c_d = int(np.round(0.90 * b_d))

            self.district_params.append(
                DistrictParameters(
                    district_id=d,
                    b_d=b_d,
                    a_d=a_d,
                    g_d=g_d,
                    capacity=c_d,
                )
            )
            self.nominal_capacity[d] = c_d

        t_grid = np.arange(self.TOTAL_MONTHS)
        for d in range(self.NUM_DISTRICTS):
            p = self.district_params[d]
            # mu(d,t) = bd + ad * sin(2*pi*t/12 + d*pi/6) + gd*t
            phase = (2.0 * np.pi * t_grid / 12.0) + (d * np.pi / 6.0)
            self.mu_demand[d, :] = p.b_d + p.a_d * np.sin(phase) + p.g_d * t_grid

        # eps(d,t) ~ Normal(0, 8^2)
        self.noise = self.rng.normal(0.0, self.NOISE_STD, size=(self.NUM_DISTRICTS, self.TOTAL_MONTHS))

        # y(d,t) = max(0, round(mu + eps))
        raw_demand = np.maximum(0, np.round(self.mu_demand + self.noise)).astype(int)

        if self.add_surge:
            for sd in self.SURGE_DISTRICTS:
                for sm in self.SURGE_MONTHS:
                    raw_demand[sd, sm] += self.SURGE_MAGNITUDE

        self.demand = raw_demand

    def get_history(self, up_to_month: int) -> np.ndarray:
        """Strict online rule: return demand strictly up to month t-1."""
        assert 0 <= up_to_month <= self.TOTAL_MONTHS
        return self.demand[:, :up_to_month].copy()

    def get_ground_truth_for_month(self, month: int) -> np.ndarray:
        assert 0 <= month < self.TOTAL_MONTHS
        return self.demand[:, month].copy()


_env_instance = None

def get_hc05_env() -> HC05Environment:
    global _env_instance
    if _env_instance is None:
        _env_instance = HC05Environment(seed=20260911, add_surge=True)
    return _env_instance
