from typing import Optional

def suggest_k_by_capacity(total_volume_liters: float, avg_truck_capacity_liters: Optional[float]) -> Optional[int]:
    if not avg_truck_capacity_liters or avg_truck_capacity_liters <= 0:
        return None
    import math
    return max(1, int(math.ceil(total_volume_liters / avg_truck_capacity_liters)))

def apply_k_constraints(k: int, available_resources: int, max_clusters: int) -> int:
    k = max(1, k)
    k = min(k, available_resources)
    k = min(k, max_clusters)
    return k