from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel
from ..services.clustering_service import cluster_requests, validate_resources

router = APIRouter(tags=["clustering"])

# Combine both request list and options into a single model
class ClusterRequestPayload(BaseModel):
    requests: List
    opts: dict

@router.post("/cluster")
def cluster_requests_endpoint(payload: ClusterRequestPayload):
    requests = payload.requests
    opts_data = payload.opts
    available_trucks = opts_data.get("available_trucks", 0)
    available_drivers = opts_data.get("available_drivers", 0)

    validate_resources(available_trucks, available_drivers)
    if not requests:
        raise HTTPException(status_code=400, detail="No requests provided")

    return cluster_requests(requests, opts_data)