"""Data Loader Service for Public Indian Healthcare Datasets."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from ..config import DATA_DIR


class DataService:
    """Provides access to ingested Government of India HMIS, ABDM HFR, and IDSP datasets."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or DATA_DIR)
        self._hmis_data = None
        self._hfr_data = None
        self._idsp_data = None
        self._geojson_data = None
        self._catalog_data = None
        self._load_all()

    def _load_all(self):
        try:
            with open(self.data_dir / "hmis_district_health.json", "r") as f:
                self._hmis_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed loading hmis_district_health.json: {e}")

        try:
            with open(self.data_dir / "abdm_hfr_facilities.json", "r") as f:
                self._hfr_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed loading abdm_hfr_facilities.json: {e}")

        try:
            with open(self.data_dir / "idsp_disease_surveillance.json", "r") as f:
                self._idsp_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed loading idsp_disease_surveillance.json: {e}")

        try:
            with open(self.data_dir / "india_districts.geojson", "r") as f:
                self._geojson_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed loading india_districts.geojson: {e}")

        try:
            with open(self.data_dir / "data_catalog.json", "r") as f:
                self._catalog_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed loading data_catalog.json: {e}")

    def get_hmis_districts(self) -> List[Dict]:
        return self._hmis_data.get("districts", []) if self._hmis_data else []

    def get_district_by_id(self, district_id: int) -> Optional[Dict]:
        districts = self.get_hmis_districts()
        for d in districts:
            if d.get("id") == district_id:
                return d
        return None

    def get_hfr_facilities(self) -> List[Dict]:
        return self._hfr_data.get("districts", []) if self._hfr_data else []

    def get_idsp_signals(self) -> List[Dict]:
        return self._idsp_data.get("district_signals", []) if self._idsp_data else []

    def get_geojson(self) -> Dict:
        return self._geojson_data if self._geojson_data else {"type": "FeatureCollection", "features": []}

    def get_catalog(self) -> List[Dict]:
        return self._catalog_data.get("datasets", []) if self._catalog_data else []


_instance = None

def get_data_service() -> DataService:
    global _instance
    if _instance is None:
        _instance = DataService()
    return _instance
