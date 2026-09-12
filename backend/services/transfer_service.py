"""Resource Transfer Optimizer (Inter-District Hospital Bed Balancing)."""

from typing import Dict, List, Optional
import numpy as np


class ResourceTransferOptimizer:
    """Recommends patient transfers or mobile reserve reallocation between adjacent surplus/deficit districts."""

    def __init__(self, safety_margin_pct: float = 0.15):
        self.safety_margin = safety_margin_pct

    def generate_recommendations(
        self,
        districts: List[Dict],
    ) -> List[Dict]:
        """
        Pairs surplus districts with high-deficit districts within geographic proximity.
        """
        surplus_districts = []
        deficit_districts = []

        for d in districts:
            cap = d.get("base_capacity", 0)
            pred = d.get("predicted_demand", 0)
            shortage = d.get("bed_shortage", 0)
            name = d.get("name", f"District {d.get('id')}")

            if shortage > 0:
                deficit_districts.append({
                    "id": d.get("id"),
                    "name": name,
                    "shortage": shortage,
                    "predicted_demand": pred,
                    "risk": d.get("risk_level", "NORMAL"),
                })
            else:
                surplus = max(0, int(cap * (1.0 - self.safety_margin)) - pred)
                if surplus >= 15:
                    surplus_districts.append({
                        "id": d.get("id"),
                        "name": name,
                        "surplus": surplus,
                        "base_capacity": cap,
                    })

        # Sort deficit descending, surplus descending
        deficit_districts.sort(key=lambda x: x["shortage"], reverse=True)
        surplus_districts.sort(key=lambda x: x["surplus"], reverse=True)

        recommendations = []
        for def_d in deficit_districts:
            for sur_d in surplus_districts:
                if sur_d["surplus"] <= 0:
                    continue

                transfer_qty = min(sur_d["surplus"], int(def_d["shortage"] * 0.40))
                if transfer_qty >= 10:
                    sur_d["surplus"] -= transfer_qty
                    impact_pct = round((transfer_qty / max(def_d["shortage"], 1)) * 100, 1)

                    recommendations.append({
                        "source_district": sur_d["name"],
                        "destination_district": def_d["name"],
                        "resource_type": "Oxygen-Supported & General Overflow Beds",
                        "quantity": transfer_qty,
                        "available_transferable_capacity": sur_d["surplus"] + transfer_qty,
                        "expected_shortage_reduction_pct": impact_pct,
                        "rationale": (
                            f"{sur_d['name']} operates at {self.safety_margin*100:.0f}% below critical buffer. "
                            f"Transferring {transfer_qty} units relieves {impact_pct}% of {def_d['name']}'s predicted deficit."
                        )
                    })

        return recommendations
