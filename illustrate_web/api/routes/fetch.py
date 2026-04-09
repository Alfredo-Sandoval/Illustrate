"""Fetch PDB by ID from RCSB."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from illustrate.fetch import fetch_pdb

from illustrate_web.api.deps import RateLimitExceeded, enforce_rate_limit, rate_limit_client_host, register_upload
from illustrate_web.api.models import UploadResponse

router = APIRouter(prefix="/api")


class FetchRequest(BaseModel):
    pdb_id: str


@router.post("/fetch-pdb", response_model=UploadResponse)
async def fetch_pdb_route(body: FetchRequest, request: Request) -> UploadResponse:
    try:
        enforce_rate_limit("fetch", client_host=rate_limit_client_host(request))
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc

    try:
        path = fetch_pdb(body.pdb_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    data = path.read_bytes()
    upload_id = register_upload(data, ".pdb")
    return UploadResponse(pdb_id=upload_id)
