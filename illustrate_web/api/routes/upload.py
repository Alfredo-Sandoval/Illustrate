"""PDB upload route."""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from pathlib import Path

from illustrate_web.api.deps import register_upload
from illustrate_web.api.models import UploadResponse

router = APIRouter(prefix="/api")


@router.post("/upload-pdb", response_model=UploadResponse)
async def upload_pdb(file: UploadFile = File(...)) -> UploadResponse:
    data = await file.read()
    suffix = Path(file.filename or "").suffix or ".pdb"
    pdb_id = register_upload(data, suffix)
    return UploadResponse(pdb_id=pdb_id)
