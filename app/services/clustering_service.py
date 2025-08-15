from typing import List, Dict, Any
import numpy as np
from fastapi import HTTPException
from ..models.clustering_models import (
    WasteCollectionRequestModel,
    ClusterOptions,
    ClusterResult,
    ClusterResponse,
)
from ..core.config import get_settings
from ..utils.dbscan_cluster import run_dbscan
from ..utils.kmeans_cluster import run_kmeans
from ..utils.merge_clusters import merge_clusters_by_centroids
from ..utils.centroid import (
    calculate_centroid,
    calculate_centroid_with_municipal,
)
from .capacity_utils import suggest_k_by_capacity, apply_k_constraints

def validate_resources(trucks: int, drivers: int) -> None:
    if trucks <= 0 or drivers <= 0:
        raise HTTPException(status_code=400, detail="available_trucks and available_drivers must be > 0")

def cluster_requests(requests: List[WasteCollectionRequestModel], opts: ClusterOptions) -> ClusterResponse:
    settings = get_settings()
    coords = np.array([[r['latitude'], r['longitude']] for r in requests], dtype=float)
    volumes = np.array([float(r['volume_liters'] or 0.0) for r in requests], dtype=float)
    ids = np.array([int(r['id']) for r in requests], dtype=int)

    # Determine constraints & suggested k
    total_volume = float(volumes.sum())
    suggested_k_by_cap = suggest_k_by_capacity(total_volume, opts['avg_truck_capacity_liters'])
    available_resources = min(opts['available_trucks'], opts['available_drivers'])
    max_clusters = opts['max_clusters'] if (opts['max_clusters'] and opts['max_clusters'] > 0) else settings.max_clusters_default

    # 1) Try DBSCAN with haversine metric
    eps_km = opts['dbscan_eps_km'] if (opts['dbscan_eps_km'] is not None) else settings.default_dbscan_eps_km
    min_samples = opts['dbscan_min_samples'] if (opts['dbscan_min_samples'] is not None) else settings.default_dbscan_min_samples
    labels = run_dbscan(coords, eps_km=eps_km, min_samples=min_samples)

    # If DBSCAN failed (all noise) → KMeans
    if (set(labels) == {-1}) or (len(set([l for l in labels if l != -1])) == 0):
        # pick k using capacity first, then resource caps
        k_guess = suggested_k_by_cap if suggested_k_by_cap is not None else available_resources
        k = apply_k_constraints(k_guess, available_resources, max_clusters)
        labels = run_kmeans(coords, k)
    else:
        # Remap noise points to nearest cluster by centroid
        labels = _attach_noise_to_nearest_centroid(coords, labels)

    # 2) Enforce resource-aware upper bound
    n_clusters_found = len(set(labels))
    desired_max = min(available_resources, max_clusters)
    if n_clusters_found > desired_max:
        labels = merge_clusters_by_centroids(coords, labels, target_k=desired_max)

    # 3) Build response clusters
    clusters_out: List[ClusterResult] = []
    final_labels = np.array(labels)
    for c in range(final_labels.max() + 1):
        idxs = np.where(final_labels == c)[0]
        req_ids = ids[idxs].tolist()
        total_vol = float(volumes[idxs].sum()) if idxs.size > 0 else 0.0

        if opts['include_municipal_in_centroid'] and opts['municipal_latitude'] is not None and opts['municipal_longitude'] is not None:
            centroid = calculate_centroid_with_municipal(
                coords[idxs],
                municipal_lat=opts['municipal_latitude'],
                municipal_lon=opts['municipal_longitude'],
            )
        else:
            centroid = calculate_centroid(coords[idxs]) if idxs.size > 0 else np.array([0.0, 0.0])

        clusters_out.append(
            ClusterResult(
                cluster_id=int(c),
                centroid_latitude=float(centroid[0]),
                centroid_longitude=float(centroid[1]),
                request_ids=req_ids,
                total_volume_liters=total_vol,
            )
        )

    meta: Dict[str, Any] = {
        "requests_count": len(requests),
        "clusters_found": len(clusters_out),
        "available_resources": available_resources,
        "suggested_k_by_capacity": suggested_k_by_cap,
        "dbscan_eps_km_used": eps_km,
        "dbscan_min_samples_used": min_samples,
        "max_clusters": max_clusters,
    }
    return ClusterResponse(clusters=clusters_out, meta=meta)

def _attach_noise_to_nearest_centroid(coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Assign DBSCAN noise (-1) points to nearest existing cluster centroid (by haversine)."""
    from ..utils.geo import haversine_distance_km
    labels = labels.copy()
    cluster_ids = sorted([l for l in set(labels) if l != -1])
    if not cluster_ids:
        return labels

    centroids = []
    for c in cluster_ids:
        pts = coords[labels == c]
        centroids.append(pts.mean(axis=0) if pts.size else np.array([0.0, 0.0]))
    centroids = np.array(centroids)

    noise_idx = np.where(labels == -1)[0]
    for idx in noise_idx:
        lat, lon = coords[idx]
        dists = [haversine_distance_km(lat, lon, cy, cx) for cy, cx in centroids]
        nearest = int(np.argmin(dists))
        labels[idx] = cluster_ids[nearest]

    # normalize labels to 0..k-1
    uniq = sorted(set(labels))
    remap = {old: new for new, old in enumerate(uniq)}
    return np.array([remap[l] for l in labels])