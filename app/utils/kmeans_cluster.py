import numpy as np
from sklearn.cluster import KMeans

def run_kmeans(coords: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(coords)
    return labels