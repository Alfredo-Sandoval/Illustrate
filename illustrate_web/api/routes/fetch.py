"""Fetch PDB by ID from RCSB."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from illustrate.fetch import fetch_pdb

from illustrate_web.api.deps import register_upload
from illustrate_web.api.models import UploadResponse

router = APIRouter(prefix="/api")


class FetchRequest(BaseModel):
    pdb_id: str


@router.post("/fetch-pdb", response_model=UploadResponse)
async def fetch_pdb_route(body: FetchRequest) -> UploadResponse:
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
