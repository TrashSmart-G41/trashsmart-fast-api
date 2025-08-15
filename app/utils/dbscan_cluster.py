import numpy as np
from sklearn.cluster import DBSCAN
from .geo import latlon_to_radians_array
from ..core.config import EARTH_RADIUS_KM

def run_dbscan(coords: np.ndarray, eps_km: float, min_samples: int) -> np.ndarray:
    X_rad = latlon_to_radians_array(coords)
    eps_rad = eps_km / EARTH_RADIUS_KM
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(X_rad)
    return labels