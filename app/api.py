from __future__ import annotations

import os

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field

from .ingestion import DataIngestionService
from .learning import PatternLearningService
from .scanning import ScanningPredictionService


class IngestionRequest(BaseModel):
    entrepreneur_name: str = Field(..., description="Founder full name.")
    first_institutional_investment_date: str = Field(
        ...,
        description="DATE CUTOFF anchor in YYYY-MM-DD format (first seed/Series A institutional investment).",
    )
    source_urls: list[str] = Field(..., description="Public source URLs to ingest.")
    persist_to_db: bool = Field(
        default=False,
        description="Persist extracted attributes/timelines to PostgreSQL if db_dsn is configured.",
    )


class LearningRequest(BaseModel):
    n_clusters: int = Field(default=3, ge=2, le=12)


class ScanningRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=50)


class PredictRequest(BaseModel):
    profile: dict


router = APIRouter(tags=["seed-founder-intelligence"])
ingestion_service = DataIngestionService()
_db_dsn = os.getenv("DATABASE_URL", "")
learning_service = PatternLearningService(db_dsn=_db_dsn) if _db_dsn else None
scanning_service = ScanningPredictionService()


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "message": "Seed Founder Intelligence API is running."}


@router.post("/ingest")
@router.post("/api/ingest")
def ingest(payload: IngestionRequest) -> dict:
    return ingestion_service.process_sources(
        entrepreneur_name=payload.entrepreneur_name,
        first_institutional_investment_date=payload.first_institutional_investment_date,
        source_urls=payload.source_urls,
        persist_to_db=payload.persist_to_db,
    )


@router.post("/train")
@router.post("/api/train")
def train(payload: LearningRequest) -> dict:
    if learning_service is None:
        return {
            "rows_used": 0,
            "feature_count": 0,
            "model_name": "xgboost_classifier",
            "status": "skipped_missing_DATABASE_URL",
            "n_clusters": 0,
            "model_path": "",
            "cluster_path": "",
        }
    summary = learning_service.run_full_training(n_clusters=payload.n_clusters)
    return {
        "rows_used": summary.rows_used,
        "feature_count": summary.feature_count,
        "model_name": summary.model_name,
        "status": summary.status,
        "n_clusters": summary.n_clusters,
        "model_path": summary.model_path,
        "cluster_path": summary.cluster_path,
    }


@router.post("/scan")
@router.post("/api/scan")
def scan(payload: ScanningRequest) -> dict:
    matches = scanning_service.search_candidates(payload.query, top_k=payload.top_k)
    return {"query": payload.query, "count": len(matches), "matches": matches}


@router.post("/predict")
@router.post("/api/predict")
def predict(payload: PredictRequest) -> dict:
    result = scanning_service.predict_profile(payload.profile)
    return {"profile": payload.profile, "result": result}


def create_app() -> FastAPI:
    app = FastAPI(
        title="Seed Founder Intelligence",
        description=(
            "AI platform to identify high-potential seed-stage entrepreneurs "
            "using historical patterns, temporal filtering, and model-based scoring."
        ),
        version="0.2.0",
    )
    app.include_router(router)
    return app


app = create_app()

