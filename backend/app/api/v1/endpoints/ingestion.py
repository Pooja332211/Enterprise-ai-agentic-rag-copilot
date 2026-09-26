from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.database.models import IngestionJob
from backend.app.database.session import get_db
from backend.app.models.schemas import ErrorResponse, IngestionJobResponse

router = APIRouter()


@router.get(
    "/{job_id}",
    summary="Get Ingestion Job Status",
    description="Returns real-time lifecycle progress, stage, and error details for an ingestion job.",
    response_model=IngestionJobResponse,
    responses={404: {"model": ErrorResponse, "description": "Ingestion job not found"}},
)
async def get_ingestion_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> IngestionJobResponse:
    stmt = select(IngestionJob).where(IngestionJob.id == job_id)
    res = await db.execute(stmt)
    job = res.scalars().first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job with ID '{job_id}' not found.",
        )

    return IngestionJobResponse.model_validate(job)
