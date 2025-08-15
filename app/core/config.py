from pydantic import BaseModel

EARTH_RADIUS_KM: float = 6371.0088
MUNICIPAL_COUNCIL_LATITUDE: float = 6.915788733342365
MUNICIPAL_COUNCIL_LONGITUDE: float = 79.86372182720865

class Settings(BaseModel):
    default_dbscan_eps_km: float = 0.5
    default_dbscan_min_samples: int = 3
    max_clusters_default: int = 10

def get_settings() -> Settings:
    return Settings()