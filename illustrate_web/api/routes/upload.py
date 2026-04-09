"""PDB upload route."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pathlib import Path

from illustrate_web.api.deps import RateLimitExceeded, enforce_rate_limit, store_upload_file
from illustrate_web.api.models import UploadResponse

router = APIRouter(prefix="/api")


@router.post("/upload-pdb", response_model=UploadResponse)
async def upload_pdb(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    try:
        enforce_rate_limit("upload", client_host=request.client.host if request.client is not None else None)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc

    suffix = Path(file.filename or "").suffix or ".pdb"
    try:
        pdb_id = await store_upload_file(file, suffix)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    return UploadResponse(pdb_id=pdb_id)
