import numpy as np

def calculate_centroid(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.array([0.0, 0.0], dtype=float)
    return points.mean(axis=0)

def calculate_centroid_with_municipal(points: np.ndarray, municipal_lat: float, municipal_lon: float) -> np.ndarray:
    if points.size == 0:
        return np.array([municipal_lat, municipal_lon], dtype=float)
    stacked = np.vstack([points, np.array([[municipal_lat, municipal_lon]], dtype=float)])
    return stacked.mean(axis=0)