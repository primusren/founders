# Seed Founder Intelligence

This project is an AI system for identifying high-potential, seed-stage entrepreneurs using historical founder patterns and pre-success signals.

## High-Level Architecture

```mermaid
flowchart LR
  subgraph ingestLayer [DataIngestionLayer]
    scraper[WebScrapers]
    extractor[LLMExtractionTransformers]
    temporal[TemporalFilterDateCutoff]
    scraper --> extractor --> temporal
  end

  subgraph learningLayer [PatternLearningLayer]
    features[FeatureEngineering]
    trainer[MLTrainingScikitLearn]
    predictor[PredictionService]
    temporal --> features --> trainer --> predictor
  end

  subgraph scanningLayer [ScanningPredictionLayer]
    scan[RealtimeCandidateScanner]
    vector[PineconeVectorSearch]
    scoring[FounderScoring]
    scan --> vector --> scoring
    predictor --> scoring
  end

  subgraph backendLayer [BackendAndStorage]
    api[FastAPIBackend]
    pg[(PostgreSQL)]
    pine[(Pinecone)]
    api --> pg
    api --> pine
  end

  scoring --> api
  temporal --> pg
```

## Project Structure

- `app/`: core application code and API modules
- `app/ingestion.py`: ingestion layer skeleton (scraping, extraction, temporal filtering)
- `app/learning.py`: pattern learning layer skeleton (feature engineering, training, prediction)
- `app/scanning.py`: scanning layer skeleton (candidate collection, vector lookup, scoring)
- `app/api.py`: FastAPI backend routes for ingestion, learning, and scanning
- `data/`: raw and processed data storage
- `models/`: machine learning components and training logic
- `scrapers/`: data ingestion and web scraping scripts
- `main.py`: FastAPI entrypoint

## Quick Start

1. Create and activate virtual environment:
   - PowerShell:
     - `py -m venv .venv`
     - `.\\.venv\\Scripts\\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Configure environment:
   - Copy `.env.template` to `.env`
   - Set `DATABASE_URL` in `.env`, for example:
     - `postgresql://user:pass@localhost:5432/dbname`
4. Run API:
   - `uvicorn main:app --reload`

### Runtime Warning Notes (Windows)

- `app/ingestion.py` suppresses two common non-fatal warnings:
  - spaCy/Confection `Pydantic V1` compatibility warning on Python 3.14+,
  - Hugging Face Hub symlink warning on Windows.
- If you still want the default warnings back, remove the related `warnings.filterwarnings(...)`
  and `HF_HUB_DISABLE_SYMLINKS_WARNING` lines in `app/ingestion.py`.

## Initial API Routes

- `GET /api/health`
- `POST /api/ingestion/run`
- `POST /api/learning/train`
- `POST /api/scanning/run`

### Ingestion Payload (Step 3)

`POST /api/ingestion/run` requires the DATE CUTOFF first:

```json
{
  "entrepreneur_name": "Elon Musk",
  "first_institutional_investment_date": "1996-01-01",
  "source_urls": [
    "https://en.wikipedia.org/wiki/Elon_Musk",
    "https://en.wikipedia.org/wiki/Zip2"
  ],
  "persist_to_db": false
}
```

### DATE CUTOFF Validation Test (Step 4)

Run sample Musk pre-1995 validation:

```powershell
.\.venv\Scripts\python scrapers/test_musk_cutoff_validation.py
```

Outputs:
- `data/musk_pre_1995_validation_output.json`
- anomaly log: `data/ingestion_anomalies.log`

## Pattern Learning Layer (Step 5)

`app/learning.py` implements:
- historical data loading from PostgreSQL,
- sentence-transformer embeddings,
- unsupervised K-means clustering for founder pattern groups,
- supervised `XGBoost` success classifier,
- artifact saving via `joblib`.

Train with 5-10 founder sample seed data:

```powershell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/seed_founder_intel"
.\.venv\Scripts\python scripts/train_pattern_learning.py --seed-sample --clusters 3
```

Artifacts:
- `models/xgboost_founder_success.joblib`
- `models/founder_clusters.csv`
- `models/pattern_training_summary.json`

## Scanning Layer (Step 6)

`app/scanning.py` implements:
- real-time candidate collection from LinkedIn/Crunchbase/X (API when available, simulated fallback otherwise),
- GitHub AI-era signal enrichment (ML repo count + stars),
- sentence-transformer profile embeddings,
- historical pattern matching via Pinecone cosine similarity (local cosine fallback if Pinecone unavailable),
- ranked output with explanations.

API call example:

```json
POST /api/scanning/run
{
  "query": "AI startups seed stage",
  "top_k": 10
}
```

## API and Dashboard (Step 7)

### FastAPI Endpoints

- `POST /ingest` (alias: `/api/ingest`) for historical data ingestion
- `POST /train` (alias: `/api/train`) for model training
- `POST /scan` (alias: `/api/scan`) for current entrepreneur search
- `POST /predict` (alias: `/api/predict`) for scoring a specific profile

### Run Backend (Uvicorn)

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload
```

### Streamlit Dashboard

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run dashboard.py
```

The dashboard visualizes:
- ranked entrepreneur matches from `/scan`,
- pattern similarity explanations,
- a Musk-style signal tag based on top historical pattern alignment.

### Ingestion Quality Dashboard

Run:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run quality_check_dashboard.py
```

Behavior:
- uses PostgreSQL when `DATABASE_URL` is configured and reachable,
- falls back to `data/elon_ingestion_detailed_log.json` when DB is not configured/available,
- shows accepted facts grouped by category with linked sources,
- shows excluded facts with discard reasons for audit.

If DB is empty, run the Elon sample ingestion first to generate fallback sample data:

```powershell
.\.venv\Scripts\python app\ingestion.py
```

## Testing (Step 8)

Pytest suite location:
- `tests/test_ingestion_elon.py`

Run tests:

```powershell
.\.venv\Scripts\python -m pytest -q
```

Elon ingestion tests verify:
- pre-1995 extraction acceptance for corroborated facts (e.g., move to Canada, UPenn transfer),
- strict DATE CUTOFF filtering of post-1995 facts,
- discard of unanchored facts without parseable dates,
- anomaly logging behavior.
