from math import radians, cos, sin, sqrt, atan2
from ..core.config import EARTH_RADIUS_KM
import numpy as np

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * atan2(sqrt(a), sqrt(1 - a))

def latlon_to_radians_array(points: np.ndarray) -> np.ndarray:
    return np.radians(points)