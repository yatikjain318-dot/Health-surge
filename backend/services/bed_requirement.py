"""Bed Requirement and Healthcare Resource Estimation Engine."""

from typing import Dict, List, Optional
import numpy as np


class BedRequirementEngine:
    """Estimates required beds, bed shortage gaps, and specialized ICU/Oxygen bed requirements."""

    def __init__(
        self,
        avg_length_of_stay_days: float = 4.5,
        target_occupancy_rate: float = 0.85,
        icu_fraction: float = 0.10,
        oxygen_fraction: float = 0.25,
        general_fraction: float = 0.65,
    ):
        self.alos = avg_length_of_stay_days
        self.occupancy_rate = target_occupancy_rate
        self.icu_fraction = icu_fraction
        self.oxygen_fraction = oxygen_fraction
        self.general_fraction = general_fraction

    def calculate_district_beds(
        self,
        district_name: str,
        predicted_demand: int,
        available_beds: int,
        risk_level: str = "NORMAL",
    ) -> Dict:
        """
        Computes total beds required, shortage gap, and clinical splits.
        """
        # Adjusted demand considering monthly turnover:
        # In emergency operations, monthly admissions need bed-days = demand * ALOS / 30 / target_occupancy
        turnover_factor = (self.alos / 30.0) / self.occupancy_rate
        # For acute surge, beds required tracks peak predicted demand directly:
        required_beds = int(np.ceil(predicted_demand * max(0.95, turnover_factor * 5.5)))

        # Direct gap
        bed_gap = max(0, required_beds - available_beds)

        # Specialized bed requirements
        icu_required = int(np.ceil(required_beds * self.icu_fraction))
        oxygen_required = int(np.ceil(required_beds * self.oxygen_fraction))
        general_required = max(0, required_beds - icu_required - oxygen_required)

        # Recommended emergency reserve allocation
        recommended_reserve = bed_gap

        return {
            "district_name": district_name,
            "predicted_demand": predicted_demand,
            "available_beds": available_beds,
            "required_beds": required_beds,
            "bed_gap": bed_gap,
            "icu_required": icu_required,
            "oxygen_required": oxygen_required,
            "general_required": general_required,
            "recommended_reserve": recommended_reserve,
            "assumptions": {
                "avg_length_of_stay_days": self.alos,
                "target_occupancy_pct": int(self.occupancy_rate * 100),
                "icu_split_pct": int(self.icu_fraction * 100),
                "oxygen_split_pct": int(self.oxygen_fraction * 100),
            }
        }
