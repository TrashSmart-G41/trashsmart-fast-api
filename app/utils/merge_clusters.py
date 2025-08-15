import numpy as np
from sklearn.cluster import AgglomerativeClustering

def merge_clusters_by_centroids(coords: np.ndarray, labels: np.ndarray, target_k: int) -> np.ndarray:
    """Merge existing cluster labels down to target_k by agglomerative clustering on centroids."""
    labels = labels.copy()
    uniq = sorted(set(labels))
    k_current = len(uniq)
    if k_current <= target_k:
        return _normalize_labels(labels)

    # compute centroids for current clusters
    centroids = []
    id_to_idx = {}
    for i, c in enumerate(uniq):
        pts = coords[labels == c]
        centroid = pts.mean(axis=0) if pts.size else np.array([0.0, 0.0])
        centroids.append(centroid)
        id_to_idx[c] = i
    cents = np.array(centroids)

    # cluster centroids to target_k
    agg = AgglomerativeClustering(n_clusters=target_k)
    agg_labels = agg.fit_predict(np.radians(cents))

    # map original label -> merged label
    map_to_merged = {orig: int(agg_labels[idx]) for orig, idx in id_to_idx.items()}
    merged = np.array([map_to_merged[int(l)] for l in labels])
    return _normalize_labels(merged)

def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    uniq = sorted(set(labels))
    remap = {old: new for new, old in enumerate(uniq)}
    return np.array([remap[l] for l in labels])