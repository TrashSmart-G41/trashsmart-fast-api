from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from ..core.config import MUNICIPAL_COUNCIL_LATITUDE, MUNICIPAL_COUNCIL_LONGITUDE

class WasteCollectionRequestModel(BaseModel):
    id: int
    latitude: float
    longitude: float
    volume_liters: Optional[float] = Field(default=0.0, ge=0.0)

class ClusterOptions(BaseModel):
    available_trucks: int
    available_drivers: int
    max_clusters: Optional[int] = 10
    avg_truck_capacity_liters: Optional[float] = None
    dbscan_eps_km: Optional[float] = None
    dbscan_min_samples: Optional[int] = None
    municipal_latitude: Optional[float] = MUNICIPAL_COUNCIL_LATITUDE
    municipal_longitude: Optional[float] = MUNICIPAL_COUNCIL_LONGITUDE
    include_municipal_in_centroid: Optional[bool] = True

class ClusterResult(BaseModel):
    cluster_id: int
    centroid_latitude: float
    centroid_longitude: float
    request_ids: List[int]
    total_volume_liters: float

class ClusterResponse(BaseModel):
    clusters: List[ClusterResult]
    meta: Dict[str, Any]