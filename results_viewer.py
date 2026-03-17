from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from typing import Any, Optional
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import psycopg2
import requests
import streamlit as st
from psycopg2.extras import RealDictCursor
from app.learning import PatternLearningService


st.set_page_config(
    page_title="AI Founder Model",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Polished UI/UX for investor demo (responsive on desktop/mobile).
st.markdown(
    """
    <style>
    :root {
        --bg: #eef3f9;
        --panel: #ffffff;
        --ink: #1b2430;
        --muted: #5a6b82;
        --brand: #1456d8;
        --brand-2: #5c87ff;
        --line: #d8e1ee;
        --success: #107a41;
        --warn: #995f00;
    }

    /* Global foundation */
    .stApp {
        background: radial-gradient(1200px 520px at 0% -10%, #dfe9ff 0%, #eef3f9 50%, #eef3f9 100%);
        color: var(--ink);
        font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.01em;
        font-weight: 700;
    }
    h4 {
        color: #224273;
        font-weight: 650;
    }

    /* Hero */
    .hero-wrap {
        background: linear-gradient(135deg, #103c99 0%, #1456d8 48%, #5c87ff 100%);
        color: #ffffff;
        border-radius: 16px;
        padding: 18px 18px 16px 18px;
        box-shadow: 0 10px 28px rgba(20, 86, 216, 0.22);
        margin-top: 2px;
        margin-bottom: 14px;
    }
    .hero-title {
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .hero-sub {
        opacity: 0.96;
        line-height: 1.55;
        font-size: 0.95rem;
    }

    /* Core widgets */
    .stButton > button {
        background: linear-gradient(135deg, var(--brand), var(--brand-2));
        color: #ffffff;
        border-radius: 10px;
        border: 0;
        padding: 10px 14px;
        transition: transform 0.16s ease, box-shadow 0.16s ease;
        box-shadow: 0 4px 14px rgba(20, 86, 216, 0.2);
        font-weight: 600;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(20, 86, 216, 0.28);
    }
    .stSelectbox > div {
        background-color: var(--panel);
        border-radius: 10px;
        border: 1px solid var(--line);
    }
    .stExpander {
        border: 1px solid var(--line);
        border-radius: 12px;
        background-color: var(--panel);
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 10px 12px;
        box-shadow: 0 2px 10px rgba(15, 35, 70, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        color: #36507d;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #0f2f66;
        font-weight: 720;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        overflow-x: auto;
        scrollbar-width: thin;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: nowrap;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: #f8fbff;
        color: #24406e;
        font-size: 0.88rem;
        padding: 0 14px;
    }
    .stTabs [aria-selected="true"] {
        background: #eaf2ff;
        border-color: #9cbcf3;
        color: #0f3476;
        font-weight: 700;
    }

    .meta-line {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-left: 8px;
        vertical-align: middle;
    }
    .badge-explicit {
        background-color: #e8f7ef;
        color: var(--success);
    }
    .badge-lifecycle {
        background-color: #e8f0fe;
        color: var(--brand);
    }
    .badge-context {
        background-color: #fff4df;
        color: var(--warn);
    }
    .fact-text {
        line-height: 1.5;
        white-space: pre-wrap;
    }
    .section-intro {
        color: var(--muted);
        line-height: 1.65;
        margin-top: 2px;
        margin-bottom: 16px;
        font-size: 1rem;
    }
    .category-header {
        margin-top: 24px;
        margin-bottom: 10px;
        color: #173d80;
        font-weight: 700;
    }
    .excluded-box {
        background-color: #fff9f8;
        border: 1px solid #ffd7d2;
        border-radius: 10px;
        padding: 12px;
        margin-top: 8px;
        margin-bottom: 14px;
    }
    @media (max-width: 768px) {
        .hero-wrap {
            padding: 14px 12px 12px 12px;
            border-radius: 14px;
        }
        h1 { font-size: 1.55rem; }
        h2, h3 { font-size: 1.15rem; }
        .section-intro { font-size: 0.95rem; }
        .stButton > button { width: 100%; }
    }
    </style>
""",
    unsafe_allow_html=True,
)


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = str(BASE_DIR / "data")
MODELS_DIR = str(BASE_DIR / "models")
DEEPSEEK_NARRATIVE_CACHE_PATH = os.path.join(LOG_DIR, "deepseek_narrative_cache.json")
UI_TRANSLATION_CACHE_PATH = os.path.join(LOG_DIR, "ui_translation_cache.json")
DEFAULT_FOUNDER_COMPANY_VALUES_USD: dict[str, list[dict[str, Any]]] = {
    "Elon Musk": [
        {
            "company": "Tesla",
            "value_usd": 1_240_000_000_000,
            "type": "public_market_cap",
            "as_of": "2026-03-17",
            "source_url": "https://www.google.com/finance/quote/TSLA",
            "confidence": "high",
        },
        {
            "company": "SpaceX-xAI (merged entity)",
            "value_usd": 1_100_000_000_000,
            "type": "private_valuation",
            "as_of": "2026-02-03",
            "source_url": "https://www.reuters.com/business/",
            "confidence": "medium",
        },
        {
            "company": "The Boring Company",
            "value_usd": 5_700_000_000,
            "type": "private_valuation",
            "as_of": "2025-01-01",
            "source_url": "https://www.reuters.com/business/autos-transportation/musks-boring-company-valued-56-bln-after-latest-funding-round-2022-04-21/",
            "confidence": "low",
        },
    ],
    "Jeff Bezos": [
        {
            "company": "Amazon",
            "value_usd": 2_290_000_000_000,
            "type": "public_market_cap",
            "as_of": "2026-03-17",
            "source_url": "https://www.google.com/finance/quote/AMZN:NASDAQ",
            "confidence": "high",
        },
        {
            "company": "Blue Origin",
            "value_usd": None,
            "type": "private_valuation_undisclosed",
            "as_of": "2026-03-17",
            "source_url": "https://sacra.com/c/blue-origin/",
            "confidence": "low",
            "notes": "No reliable recent public valuation disclosure.",
        },
    ],
    "Mark Zuckerberg": [
        {
            "company": "Meta",
            "value_usd": 1_630_000_000_000,
            "type": "public_market_cap",
            "as_of": "2026-03-17",
            "source_url": "https://www.google.com/finance/quote/META:NASDAQ",
            "confidence": "high",
        },
    ],
}

# Known cutoff context (used when DB/log does not carry full detail yet).
CUTOFF_CONTEXT_OVERRIDES: dict[str, dict[str, str]] = {
    "Elon Musk": {
        "startup": "Zip2",
        "funding_received": "1995 (first institutional funding milestone proxy)",
    },
    "Mark Zuckerberg": {
        "startup": "Facebook",
        "funding_received": "2004 (pre-seed/first institutional milestone proxy)",
    },
}


def _parse_env_file(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not path.exists():
        return parsed
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                match = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", raw)
                if not match:
                    continue
                key = match.group(1).strip()
                value = match.group(2).strip().strip('"').strip("'")
                if value:
                    parsed[key] = value
    except Exception:
        return {}
    return parsed


def _bootstrap_env_from_local_files() -> None:
    """
    Load env vars from .env/.env.template if not already set.
    This helps Streamlit runs where shell env vars are not injected.
    """
    candidates = [
        BASE_DIR / ".env",
        BASE_DIR / ".env.template",
        Path.cwd() / ".env",
        Path.cwd() / ".env.template",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        values = _parse_env_file(candidate)
        for env_key, env_value in values.items():
            if not os.getenv(env_key, "").strip():
                os.environ[env_key] = env_value


def _discover_log_files() -> list[str]:
    pattern = "*_ingestion_detailed_log.json"
    roots = [
        Path(LOG_DIR),
        BASE_DIR / "data",
        Path.cwd() / "data",
    ]
    files: list[str] = []
    seen: set[str] = set()
    for root in roots:
        try:
            if root.exists():
                for file_path in root.glob(pattern):
                    resolved = str(file_path.resolve())
                    if resolved not in seen:
                        seen.add(resolved)
                        files.append(resolved)
        except Exception:
            continue
    # Last-resort recursive sweep under repo root.
    try:
        for file_path in BASE_DIR.rglob(pattern):
            resolved = str(file_path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    except Exception:
        pass
    return files


_bootstrap_env_from_local_files()


def title_from_slug(slug: str) -> str:
    clean = slug.replace("_", " ").replace("-", " ").strip()
    return " ".join(part.capitalize() for part in clean.split())


def slug_from_name(name: str) -> str:
    return name.lower().replace(" ", "-")


def canonical_name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def get_db_connection() -> Optional[psycopg2.extensions.connection]:
    database_url = os.getenv("DATABASE_URL", "").strip()
    try:
        if database_url:
            return psycopg2.connect(database_url)
        return psycopg2.connect(
            dbname=os.getenv("PGDATABASE", "seed_founder_intel"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"),
            host=os.getenv("PGHOST", "localhost"),
            port=os.getenv("PGPORT", "5432"),
        )
    except Exception:
        # Silent fallback to JSON logs to keep UI clean.
        return None


def get_entrepreneurs_from_db() -> list[str]:
    conn = get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT name FROM entrepreneurs ORDER BY name;")
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def get_entrepreneurs_from_logs() -> list[str]:
    index = build_log_index()
    if index:
        return sorted(index.keys())
    # If logs are missing in deployment, still provide founder selector
    # from the latest pattern report.
    return sorted(_founders_from_pattern_report())


def build_log_index() -> dict[str, str]:
    """
    Build a deduplicated map: display_name -> best matching log file path.
    If duplicate aliases exist (e.g. Elon vs Elon Musk), keep the richer file
    by accepted_count and then latest modified time.
    """
    candidates: dict[str, tuple[str, int, float]] = {}
    log_files = _discover_log_files()
    for file_path in log_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        base_name = os.path.basename(file_path).replace("_ingestion_detailed_log.json", "")
        display_name = str(payload.get("entrepreneur_name", "")).strip() or title_from_slug(base_name)
        accepted = int(payload.get("accepted_count", 0))
        mtime = os.path.getmtime(file_path)

        # Use canonical key to collapse aliases.
        key = canonical_name_key(display_name)
        existing = candidates.get(key)
        if not existing:
            candidates[key] = (file_path, accepted, mtime)
            continue
        _, accepted_old, mtime_old = existing
        if (accepted > accepted_old) or (accepted == accepted_old and mtime > mtime_old):
            candidates[key] = (file_path, accepted, mtime)

    resolved: dict[str, str] = {}
    for _, (path, _, _) in candidates.items():
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            name = str(payload.get("entrepreneur_name", "")).strip()
        except Exception:
            name = ""
        if not name:
            base_name = os.path.basename(path).replace("_ingestion_detailed_log.json", "")
            name = title_from_slug(base_name)
        resolved[name] = path
    return resolved


def parse_sources_to_links(raw_source: str) -> list[str]:
    text = str(raw_source or "").strip()
    if not text:
        return []
    # Prefer URL extraction first so links remain clickable even when
    # the source field contains mixed separators or prose.
    url_matches = re.findall(r"https?://[^\s\]|)]+", text)
    if url_matches:
        seen: set[str] = set()
        deduped: list[str] = []
        for url in url_matches:
            clean = url.strip().rstrip(".,;")
            if clean and clean not in seen:
                seen.add(clean)
                deduped.append(clean)
        return deduped
    # Fallback split for non-URL citation labels.
    parts = re.split(r"[,\n|]+", text)
    return [part.strip() for part in parts if part.strip()]


def _normalize_fact_text_for_dedup(text: str) -> str:
    cleaned = str(text or "").lower()
    cleaned = re.sub(r"\[[0-9,\s]+\]", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def deduplicate_fact_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate timeline facts and merge sources.
    Dedupe key: normalized fact text + timestamp.
    """
    if df.empty:
        return df

    best_rows: dict[tuple[str, str], dict] = {}
    source_sets: dict[tuple[str, str], set[str]] = {}

    for _, row in df.iterrows():
        fact = str(row.get("fact", "")).strip()
        timestamp = str(row.get("timestamp", "")).strip()
        source = str(row.get("source", "")).strip()
        confidence = str(row.get("confidence", "low")).strip().lower()
        norm = _normalize_fact_text_for_dedup(fact)
        if not norm:
            continue
        key = (norm, timestamp)

        if key not in best_rows:
            best_rows[key] = row.to_dict()
            source_sets[key] = set(parse_sources_to_links(source))
            continue

        current = best_rows[key]
        current_conf = str(current.get("confidence", "low")).strip().lower()
        confidence_rank = {"high": 3, "medium": 2, "low": 1}
        replace = False
        if confidence_rank.get(confidence, 1) > confidence_rank.get(current_conf, 1):
            replace = True
        elif len(fact) > len(str(current.get("fact", ""))):
            replace = True
        if replace:
            best_rows[key] = row.to_dict()

        source_sets[key].update(parse_sources_to_links(source))

    deduped_rows: list[dict] = []
    for key, row in best_rows.items():
        merged_sources = sorted(source_sets.get(key, set()))
        row["source"] = ", ".join(merged_sources) if merged_sources else str(row.get("source", ""))
        deduped_rows.append(row)

    deduped_df = pd.DataFrame(deduped_rows)
    if deduped_df.empty:
        return deduped_df
    return deduped_df.reset_index(drop=True)


def _looks_post_cutoff_context_text(text: str, cutoff_year: int) -> bool:
    low = str(text or "").lower()
    years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", low)]
    if any(y >= cutoff_year for y in years):
        return True
    if re.search(r"@[a-z0-9_]{2,}", low):
        if any(
            token in low
            for token in (
                "government",
                "administration",
                "policy",
                "public company",
                "public companies",
                "ceo",
                "regulatory",
                "sec filing",
                "mission",
                "shareholders",
                "earnings",
            )
        ):
            return True
    blocked_tokens = (
        # General post-scale context markers (non-founder-specific).
        "in a post",
        "sec filing",
        "regulatory filing",
        "news study",
        "researchers stated",
        "investigations revealed",
        "government work",
        "political activities",
        "administration",
        "president",
        "market cap",
        "public company",
        "as ceo",
        "billionaire",
        "animal testing",
        "clinical trial",
    )
    return any(token in low for token in blocked_tokens)


def filter_pre_cutoff_timeline(df: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
    if df.empty:
        return df
    rows: list[dict] = []
    for _, row in df.iterrows():
        fact = str(row.get("fact", ""))
        timestamp = str(row.get("timestamp", ""))
        if _looks_post_cutoff_context_text(fact, cutoff_year):
            continue
        if _looks_post_cutoff_context_text(timestamp, cutoff_year):
            continue
        rows.append(row.to_dict())
    return pd.DataFrame(rows)


def timing_badge(inference: str) -> tuple[str, str]:
    text = str(inference or "").lower()
    lifecycle_markers = (
        "college_phase_before_startup_inference",
        "college_phase_inference",
        "school_phase_inference",
        "teen_highschool_inference",
        "childhood_inference",
    )
    if any(marker in text for marker in lifecycle_markers):
        return "Lifecycle inferred", "badge-lifecycle"
    if text.strip():
        return "Context inferred", "badge-context"
    return "Explicitly dated", "badge-explicit"


def fetch_from_db(entrepreneur_name: str) -> pd.DataFrame:
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT
                    a.category,
                    COALESCE(a.fact, a.attribute_text) AS fact,
                    COALESCE(a.timestamp::text, a.event_date::text, '') AS timestamp,
                    COALESCE(a.source, a.source_url, '') AS source,
                    CASE WHEN a.before_cutoff THEN '' ELSE 'excluded_in_db' END AS inference,
                    a.before_cutoff
                FROM attributes a
                JOIN entrepreneurs e ON a.entrepreneur_id = e.id
                WHERE e.name = %s
                ORDER BY a.category, COALESCE(a.timestamp, a.event_date) NULLS LAST;
            """
            cur.execute(query, (entrepreneur_name,))
            rows = list(cur.fetchall())
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df[df["before_cutoff"] == True].copy()  # noqa: E712
    finally:
        conn.close()


def fetch_cutoff_context(entrepreneur_name: str) -> tuple[str, str]:
    """
    Returns cutoff_date, cutoff_reason for display.
    """
    # First try DB.
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT first_institutional_investment_date
                    FROM entrepreneurs
                    WHERE name = %s
                    LIMIT 1;
                    """,
                    (entrepreneur_name,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    cutoff = str(row[0])
                    override = CUTOFF_CONTEXT_OVERRIDES.get(entrepreneur_name, {})
                    startup = override.get("startup", "Founder first startup")
                    funding = override.get("funding_received", cutoff)
                    reason = (
                        f"Startup at cutoff: {startup}. "
                        f"Funding timing used for cutoff: {funding}. "
                        "Only facts anchored strictly before this date are shown."
                    )
                    return cutoff, reason
        except Exception:
            pass
        finally:
            conn.close()

    # Fallback to JSON log.
    _, log_data = _open_log_for_name(entrepreneur_name)
    if log_data:
        cutoff = str(log_data.get("cutoff_date", "unknown"))
        startup = str(log_data.get("cutoff_startup", "")).strip()
        funding = str(log_data.get("cutoff_funding_received", "")).strip()
        if not startup or not funding:
            override = CUTOFF_CONTEXT_OVERRIDES.get(entrepreneur_name, {})
            startup = startup or override.get("startup", "Founder first startup")
            funding = funding or override.get("funding_received", cutoff)
        reason = (
            f"Startup at cutoff: {startup}. "
            f"Funding timing used for cutoff: {funding}. "
            "Only facts anchored strictly before this date are shown."
        )
        return cutoff, reason

    override = CUTOFF_CONTEXT_OVERRIDES.get(entrepreneur_name, {})
    startup = override.get("startup", "Founder first startup")
    funding = override.get("funding_received", "unknown")
    return "unknown", f"Startup at cutoff: {startup}. Funding timing used for cutoff: {funding}."


def _open_log_for_name(entrepreneur_name: str) -> tuple[str, dict] | tuple[None, None]:
    index = build_log_index()
    selected = index.get(entrepreneur_name)
    if selected and os.path.exists(selected):
        with open(selected, "r", encoding="utf-8") as f:
            return selected, json.load(f)

    # Canonical-key lookup for aliases like "Elon" vs "Elon Musk".
    wanted_key = canonical_name_key(entrepreneur_name)
    if wanted_key:
        for display_name, file_path in index.items():
            if canonical_name_key(display_name) == wanted_key and os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    return file_path, json.load(f)

    # Prefix-token fallback, e.g. "Mark" -> "Mark Zuckerberg".
    tokens = [tok for tok in re.split(r"\s+", str(entrepreneur_name).strip().lower()) if tok]
    if tokens:
        for display_name, file_path in index.items():
            low = str(display_name).lower()
            if all(tok in low for tok in tokens) and os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    return file_path, json.load(f)

    # File-name based fallback from discovered logs.
    for file_path in _discover_log_files():
        base = os.path.basename(file_path).lower()
        slug = slug_from_name(entrepreneur_name)
        underscored = str(entrepreneur_name).lower().replace(" ", "_")
        if ((slug and slug in base) or (underscored and underscored in base)) and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return file_path, json.load(f)
    return None, None


def fetch_from_json(entrepreneur_name: str) -> pd.DataFrame:
    _, log_data = _open_log_for_name(entrepreneur_name)
    if not log_data:
        return pd.DataFrame()

    accepted = log_data.get("accepted_claims", [])
    rows: list[dict] = []
    for fact in accepted:
        merged_sources = fact.get("merged_sources", [])
        source_value = ", ".join(merged_sources) if merged_sources else str(fact.get("source", fact.get("source_url", "")))
        timestamps = fact.get("timestamps") or fact.get("merged_timestamps") or []
        timestamp = str(fact.get("event_date", "")) or ", ".join(str(ts) for ts in timestamps)
        rows.append(
            {
                "category": fact.get("category", "uncategorized"),
                "fact": fact.get("fact") or fact.get("attribute_text", ""),
                "narrative_fact": fact.get("narrative_fact", ""),
                "timestamp": timestamp,
                "source": source_value,
                "inference": fact.get("inference")
                or fact.get("narrative_inference")
                or fact.get("inference_reason")
                or fact.get("timestamp_inference", ""),
            }
        )
    return pd.DataFrame(rows)


def fetch_comprehensive_from_json(entrepreneur_name: str) -> pd.DataFrame:
    _, log_data = _open_log_for_name(entrepreneur_name)
    if not log_data:
        return pd.DataFrame()
    facts = log_data.get("comprehensive_pre_cutoff_facts", [])
    if not isinstance(facts, list) or not facts:
        # Fallback to accepted claims when comprehensive list is absent.
        facts = log_data.get("accepted_claims", [])
    rows: list[dict] = []
    for fact in facts:
        sources_value = fact.get("sources", [])
        if isinstance(sources_value, list):
            source_text = ", ".join(str(s) for s in sources_value if str(s).strip())
        else:
            source_text = str(fact.get("source", "")).strip()
        rows.append(
            {
                "category": fact.get("category", "uncategorized"),
                "fact": fact.get("fact", "") or fact.get("attribute_text", ""),
                "timestamp": fact.get("timestamp", "") or fact.get("event_date", ""),
                "source": source_text or str(fact.get("source", "")),
                "inference": fact.get("inference", "") or fact.get("inference_reason", ""),
                "confidence": fact.get("confidence", "low"),
            }
        )
    return pd.DataFrame(rows)


def fetch_organized_analysis_from_json(entrepreneur_name: str) -> dict[str, dict]:
    _, log_data = _open_log_for_name(entrepreneur_name)
    if not log_data:
        return {}
    analysis = log_data.get("organized_analysis", {}) or log_data.get("founder_analysis", {})
    if isinstance(analysis, dict):
        return analysis
    return {}


def fetch_comprehensiveness_stats(entrepreneur_name: str) -> dict[str, object]:
    _, log_data = _open_log_for_name(entrepreneur_name)
    if not log_data:
        return {}
    return {
        "source_count": int(log_data.get("source_count", 0)),
        "total_source_words": int(log_data.get("total_source_words", 0)),
        "targets": log_data.get("comprehensiveness_targets", {}),
        "comprehensive_fact_count": int(log_data.get("comprehensive_fact_count", 0)),
    }


def fetch_excluded_from_json(entrepreneur_name: str) -> pd.DataFrame:
    _, log_data = _open_log_for_name(entrepreneur_name)
    if not log_data:
        return pd.DataFrame()
    excluded = log_data.get("excluded_claims", [])
    rows: list[dict] = []
    for fact in excluded:
        rows.append(
            {
                "reason": fact.get("discard_reason") or fact.get("reason", "unknown"),
                "fact": fact.get("attribute_text") or fact.get("fact", ""),
                "source": fact.get("source")
                or fact.get("source_url")
                or ", ".join(fact.get("sources", [])),
            }
        )
    return pd.DataFrame(rows)


def load_pattern_report() -> dict[str, object]:
    path = os.path.join(MODELS_DIR, "founder_pattern_report.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _normalize_name_key(name: str) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def _format_usd(value: float) -> str:
    n = float(value or 0.0)
    if n >= 1_000_000_000_000:
        return f"${n / 1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"${n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"${n / 1_000_000:.2f}M"
    return f"${n:,.0f}"


def get_founder_company_values(founder_name: str) -> tuple[list[dict[str, Any]], float]:
    merged: dict[str, list[dict[str, Any]]] = {
        _normalize_name_key(name): entries for name, entries in DEFAULT_FOUNDER_COMPANY_VALUES_USD.items()
    }
    path = os.path.join(LOG_DIR, "founder_company_values.json")
    if os.path.exists(path):
        try:
            payload = json.load(open(path, "r", encoding="utf-8"))
            if isinstance(payload, dict):
                founder_map = payload.get("founders", payload)
                if isinstance(founder_map, dict):
                    for name, entries in founder_map.items():
                        if isinstance(entries, list):
                            merged[_normalize_name_key(name)] = entries
        except Exception:
            pass
    companies = list(merged.get(_normalize_name_key(founder_name), []))
    total = 0.0
    for item in companies:
        try:
            raw = item.get("value_usd", 0.0)
            if raw in (None, "", "n/a"):
                continue
            total += float(raw)
        except Exception:
            continue
    return companies, total


def _get_env_or_dotenv(key: str) -> str:
    val = os.getenv(key, "").strip()
    if val:
        return val
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(base_dir, ".env"),
        os.path.join(base_dir, ".env.template"),
        ".env",
        ".env.template",
    ]
    for env_path in candidate_paths:
        if not os.path.exists(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw or raw.startswith("#") or "=" not in raw:
                        continue
                    k, v = raw.split("=", 1)
                    if k.strip() == key:
                        resolved = v.strip().strip('"').strip("'")
                        if resolved:
                            return resolved
        except Exception:
            continue
    return ""


def _resolve_deepseek_api_key() -> tuple[str, str]:
    # Streamlit Cloud standard secret channel.
    try:
        secret_key = str(st.secrets.get("DEEPSEEK_API_KEY", "")).strip()
        if secret_key:
            return secret_key, "streamlit_secrets:DEEPSEEK_API_KEY"
    except Exception:
        pass

    direct = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if direct:
        return direct, "env:DEEPSEEK_API_KEY"
    dotenv_value = _get_env_or_dotenv("DEEPSEEK_API_KEY")
    if dotenv_value:
        return dotenv_value, "dotenv_loader:DEEPSEEK_API_KEY"
    # Streamlit can execute with different working directories; search robustly.
    roots: list[Path] = []
    try:
        roots.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    try:
        roots.append(Path.cwd().resolve())
    except Exception:
        pass
    env_root = os.getenv("CURSOR_WORKSPACE_PATH", "").strip()
    if env_root:
        try:
            roots.append(Path(env_root).resolve())
        except Exception:
            pass
    unique_roots: list[Path] = []
    seen_roots: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen_roots:
            seen_roots.add(key)
            unique_roots.append(root)

    candidate_paths: list[Path] = []
    for root in unique_roots:
        for base in [root, *list(root.parents)[:4]]:
            candidate_paths.append(base / ".env")
            candidate_paths.append(base / ".env.template")

    seen_files: set[str] = set()
    for env_path in candidate_paths:
        env_key = str(env_path)
        if env_key in seen_files:
            continue
        seen_files.add(env_key)
        if not env_path.exists():
            continue
        try:
            with open(env_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    raw = line.strip()
                    if not raw or raw.startswith("#") or "=" not in raw:
                        continue
                    k, v = raw.split("=", 1)
                    if k.strip().upper() == "DEEPSEEK_API_KEY":
                        resolved = v.strip().strip('"').strip("'")
                        if resolved:
                            return resolved, f"file:{env_path}"
        except Exception:
            continue
    return "", "not_found"


def _load_ui_translation_cache() -> dict[str, str]:
    if not os.path.exists(UI_TRANSLATION_CACHE_PATH):
        return {}
    try:
        payload = json.load(open(UI_TRANSLATION_CACHE_PATH, "r", encoding="utf-8"))
        if isinstance(payload, dict):
            return {str(k): str(v) for k, v in payload.items()}
    except Exception:
        return {}
    return {}


def _founders_from_pattern_report() -> list[str]:
    report = load_pattern_report()
    if not report:
        return []
    names: list[str] = []
    for cluster in report.get("cluster_summary", []) or []:
        for founder in cluster.get("founders", []) or []:
            name = str(founder or "").strip()
            if name:
                names.append(name)
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = canonical_name_key(name)
        if key and key not in seen:
            seen.add(key)
            deduped.append(name)
    return deduped


def _persist_ui_translation_cache(cache: dict[str, str]) -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        if len(cache) > 800:
            keys = list(cache.keys())[-650:]
            cache = {k: cache[k] for k in keys if k in cache}
        with open(UI_TRANSLATION_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _translate_text_with_deepseek(text: str, ui_lang: str) -> str:
    if ui_lang != "zh":
        return text
    raw = str(text or "").strip()
    if not raw:
        return raw
    api_key, _ = _resolve_deepseek_api_key()
    if not api_key:
        return raw
    model_name = _get_env_or_dotenv("DEEPSEEK_MODEL") or "deepseek-chat"
    cache_key = hashlib.sha256((f"zh|{model_name}|{raw}").encode("utf-8")).hexdigest()
    cache = _load_ui_translation_cache()
    if cache_key in cache and cache.get(cache_key):
        return cache[cache_key]
    prompt = (
        "Translate the following text into Simplified Chinese for a professional investment app UI. "
        "Keep meaning precise, preserve numbers/symbols/URLs, and keep markdown structure if present. "
        "Return translated text only.\n\n"
        f"{raw}"
    )
    try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a precise bilingual financial UI translator."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": min(1200, max(240, int(len(raw) * 1.8))),
            "temperature": 0.0,
        }
        base_url = _get_env_or_dotenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=max(35, int(_get_env_or_dotenv("DEEPSEEK_TIMEOUT_SECONDS") or "25")),
        )
        if not response.ok:
            return raw
        data = response.json()
        out = str(data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        if not out:
            return raw
        cache[cache_key] = out
        _persist_ui_translation_cache(cache)
        return out
    except Exception:
        return raw


def _split_translation_chunks(text: str, max_chars: int = 1800) -> list[str]:
    raw = str(text or "")
    if not raw.strip():
        return []
    paragraphs = raw.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        # If paragraph itself is too long, hard-split it.
        if len(para) > max_chars:
            for i in range(0, len(para), max_chars):
                part = para[i : i + max_chars]
                if part.strip():
                    chunks.append(part)
            current = ""
        else:
            current = para
    if current.strip():
        chunks.append(current)
    return chunks


def _translate_markdown_snapshot(text: str, ui_lang: str) -> str:
    """
    Translate long LLM markdown by snapshot-chunking English content first,
    then translating chunk-by-chunk with cache.
    """
    if ui_lang != "zh":
        return str(text or "")
    raw = str(text or "").strip()
    if not raw:
        return raw
    api_key, _ = _resolve_deepseek_api_key()
    if not api_key:
        return raw

    model_name = _get_env_or_dotenv("DEEPSEEK_MODEL") or "deepseek-chat"
    base_url = _get_env_or_dotenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    timeout_seconds = max(35, int(_get_env_or_dotenv("DEEPSEEK_TIMEOUT_SECONDS") or "25"))
    chunks = _split_translation_chunks(raw, max_chars=1800)
    if not chunks:
        return raw

    cache = _load_ui_translation_cache()
    out_chunks: list[str] = []
    for idx, chunk in enumerate(chunks):
        cache_key = hashlib.sha256((f"zh-md-v2|{model_name}|{idx}|{chunk}").encode("utf-8")).hexdigest()
        cached = cache.get(cache_key, "")
        if cached:
            out_chunks.append(cached)
            continue
        prompt = (
            "Translate the following markdown into Simplified Chinese.\n"
            "Rules:\n"
            "- Preserve markdown structure (headings, bullets, numbering, emphasis, tables).\n"
            "- Preserve numbers, percentages, code spans, URLs, and proper nouns.\n"
            "- Do not omit any content.\n"
            "- Return translation only.\n\n"
            f"{chunk}"
        )
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a precise bilingual translator for investor content."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": min(1600, max(400, int(len(chunk) * 2.2))),
                "temperature": 0.0,
            }
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout_seconds,
            )
            if not response.ok:
                out_chunks.append(chunk)
                continue
            data = response.json()
            translated = str(data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
            if not translated:
                out_chunks.append(chunk)
                continue
            cache[cache_key] = translated
            out_chunks.append(translated)
        except Exception:
            out_chunks.append(chunk)
    _persist_ui_translation_cache(cache)
    return "\n\n".join(out_chunks)


def _load_narrative_cache() -> dict[str, dict[str, str]]:
    if not os.path.exists(DEEPSEEK_NARRATIVE_CACHE_PATH):
        return {}
    try:
        payload = json.load(open(DEEPSEEK_NARRATIVE_CACHE_PATH, "r", encoding="utf-8"))
        if isinstance(payload, dict):
            return {
                str(k): dict(v) for k, v in payload.items() if isinstance(v, dict)
            }
    except Exception:
        return {}
    return {}


def _persist_narrative_cache(cache: dict[str, dict[str, str]]) -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        # Keep cache bounded for production stability.
        if len(cache) > 80:
            keys = list(cache.keys())[-60:]
            cache = {k: cache[k] for k in keys if k in cache}
        with open(DEEPSEEK_NARRATIVE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _sanitize_investor_memo_heading(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    lines = [ln.rstrip() for ln in raw.splitlines()]
    out: list[str] = []
    replaced_title = False
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if i == 0 and "seed investor memo" in low:
            out.append("## Identifying Ultra-Success Founders in the AI Era")
            replaced_title = True
            continue
        if low.startswith("date:") or low.startswith("to:") or low.startswith("from:") or low.startswith("subject:"):
            continue
        out.append(line)
    cleaned = "\n".join(out).strip()
    if replaced_title and cleaned and not cleaned.lower().startswith("## identifying ultra-success founders in the ai era"):
        cleaned = "## Identifying Ultra-Success Founders in the AI Era\n\n" + cleaned
    return cleaned or raw


def generate_pattern_narrative_with_deepseek(
    pattern_report: dict[str, Any],
    refresh_nonce: int = 0,
) -> tuple[str, str, str]:
    api_key, key_source = _resolve_deepseek_api_key()
    if not api_key:
        return "", "missing_api_key", key_source
    try:
        top_signals = pattern_report.get("top_signal_importance", [])
        clusters = pattern_report.get("cluster_summary", [])
        findings = pattern_report.get("findings", [])
        prompt = (
            "You are writing a seed investor memo. Use the model outputs below and produce an expansive but practical "
            "narrative (650-1000 words) that explains:\n"
            "1) what founder patterns are emerging,\n"
            "2) what these patterns imply for identifying future ultra-success founders in the AI era,\n"
            "3) which false positives to avoid,\n"
            "4) a concrete diligence playbook.\n"
            "Use clear sections and bullet points where useful. Do not repeat raw numeric tables line-by-line; interpret them.\n\n"
            f"Top signals: {json.dumps(top_signals)[:4500]}\n"
            f"Cluster summary: {json.dumps(clusters)[:4500]}\n"
            f"Findings: {json.dumps(findings)[:4500]}"
        )
        base_url = _get_env_or_dotenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        timeout_seconds = max(45, int(_get_env_or_dotenv("DEEPSEEK_TIMEOUT_SECONDS") or "25"))
        model_name = _get_env_or_dotenv("DEEPSEEK_MODEL") or "deepseek-chat"
        prompt_fingerprint = hashlib.sha256(
            (
                model_name
                + "|v3|"
                + json.dumps(
                    {
                        "top_signals": top_signals,
                        "clusters": clusters,
                        "findings": findings,
                    },
                    sort_keys=True,
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_key = f"{model_name}:{prompt_fingerprint}"
        if int(refresh_nonce or 0) <= 0:
            cache = _load_narrative_cache()
            cached = cache.get(cache_key, {})
            cached_text = str(cached.get("content", "")).strip() if isinstance(cached, dict) else ""
            if cached_text:
                return _sanitize_investor_memo_heading(cached_text), "", "cache:deepseek_narrative"
        attempts = [
            {"max_tokens": 1800, "temperature": 0.2},
            {"max_tokens": 1100, "temperature": 0.1},
        ]
        last_error = "request_failed"
        for attempt in attempts:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a top-tier venture research partner."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": int(attempt["max_tokens"]),
                "temperature": float(attempt["temperature"]),
            }
            try:
                response = requests.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout_seconds,
                )
            except requests.exceptions.Timeout:
                last_error = "timeout"
                continue
            except requests.exceptions.RequestException:
                last_error = "request_exception"
                continue

            if not response.ok:
                last_error = f"http_{response.status_code}"
                continue
            try:
                data = response.json()
            except ValueError:
                last_error = "invalid_json_response"
                continue
            content = str(data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
            if not content:
                last_error = "empty_response"
                continue
            # Guard against truncated completions (common when the model hits token limits).
            lower_content = content.lower()
            needs_continuation = (
                lower_content.endswith("task:")
                or lower_content.endswith("*   **task:**")
                or ("concrete diligence playbook" in lower_content and "**b." not in lower_content)
            )
            if needs_continuation:
                try:
                    continuation_prompt = (
                        "Continue the memo from where it stopped. "
                        "Do not repeat prior sections. Complete the full 'Concrete Diligence Playbook' "
                        "with sections A/B/C/D and finish cleanly.\n\n"
                        f"Current partial memo:\n{content[-3500:]}"
                    )
                    continuation_payload = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": "Continue unfinished venture memo output only."},
                            {"role": "user", "content": continuation_prompt},
                        ],
                        "max_tokens": 1200,
                        "temperature": 0.1,
                    }
                    continuation_response = requests.post(
                        f"{base_url.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json=continuation_payload,
                        timeout=timeout_seconds,
                    )
                    if continuation_response.ok:
                        continuation_data = continuation_response.json()
                        continuation = str(
                            continuation_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        ).strip()
                        if continuation:
                            content = f"{content.rstrip()}\n\n{continuation.lstrip()}"
                except Exception:
                    pass
            content = _sanitize_investor_memo_heading(content)
            cache = _load_narrative_cache()
            cache[cache_key] = {
                "model": model_name,
                "key_source": key_source,
                "content": content,
            }
            _persist_narrative_cache(cache)
            return content, "", key_source
        return "", last_error, key_source
    except Exception:
        return "", "request_failed", key_source


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def generate_scoring_matrix_with_deepseek(
    pattern_report: dict[str, Any],
    refresh_nonce: int = 0,
) -> tuple[str, str, str]:
    api_key, key_source = _resolve_deepseek_api_key()
    if not api_key:
        return "", "missing_api_key", key_source
    try:
        top_signals = pattern_report.get("top_signal_importance", [])
        findings = pattern_report.get("findings", [])
        prompt = (
            "Create a practical founder scoring matrix for seed investors evaluating young AI-era founders.\n"
            "Output in markdown with these sections:\n"
            "1) 'Scoring Matrix' as a table with columns: Field | Weight (%) | Why it matters | Evidence to collect now.\n"
            "2) 'How to score in practice' with 8-12 bullet points.\n"
            "3) 'Public-data heuristics' with 10-15 bullet points for what to look for online.\n"
            "4) 'Red flags / false positives' with 6-10 bullet points.\n"
            "Weights must sum to exactly 100.\n"
            "Keep it logically structured and investor-usable.\n\n"
            f"Model signals: {json.dumps(top_signals)[:4200]}\n"
            f"Findings: {json.dumps(findings)[:4200]}"
        )
        payload = {
            "model": _get_env_or_dotenv("DEEPSEEK_MODEL") or "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a venture partner designing a scoring framework."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1300,
            "temperature": 0.2,
        }
        base_url = _get_env_or_dotenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        timeout_seconds = max(45, int(_get_env_or_dotenv("DEEPSEEK_TIMEOUT_SECONDS") or "25"))
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout_seconds,
        )
        if not response.ok:
            return "", f"http_{response.status_code}", key_source
        data = response.json()
        content = str(data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        if not content:
            return "", "empty_response", key_source
        return content, "", key_source
    except Exception:
        return "", "request_failed", key_source


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_pre_cutoff_image_candidates(entrepreneur_name: str, cutoff_year: int) -> list[dict[str, str]]:
    """
    Pull likely early-years image candidates from public Wikimedia endpoints.
    """
    out: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    headers = {
        "User-Agent": "FounderAIModel/0.1 (research tool; contact: local-user)"
    }
    early_tokens = (
        "child",
        "childhood",
        "young",
        "teen",
        "student",
        "school",
        "college",
        "university",
        "yearbook",
        "graduation",
    )
    late_tokens = (
        "conference",
        "summit",
        "award",
        "official",
        "press",
        "interview",
        "portrait",
        "talk",
        "speech",
    )

    def _audit_candidate(label: str, image_url: str, source_url: str, query_hint: str) -> dict[str, str]:
        blob = f"{label} {image_url} {source_url} {query_hint}".lower()
        years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", blob)]
        pre_years = [y for y in years if y <= cutoff_year]
        post_years = [y for y in years if y > cutoff_year]
        score = 0
        reasons: list[str] = []
        if pre_years:
            score += 3
            reasons.append(f"contains pre-cutoff year(s): {sorted(set(pre_years))[:2]}")
        if post_years:
            score -= 3
            reasons.append(f"contains post-cutoff year(s): {sorted(set(post_years))[:2]}")
        if any(tok in blob for tok in early_tokens):
            score += 2
            reasons.append("contains early-life token")
        if any(tok in blob for tok in late_tokens):
            score -= 1
            reasons.append("contains modern/public-appearance token")
        if "thumb" in image_url.lower():
            score -= 1
            reasons.append("thumbnail derivative (weak temporal signal)")

        verdict = "uncertain"
        if score >= 2:
            verdict = "likely_pre_cutoff"
        elif score <= -2:
            verdict = "likely_post_cutoff"

        return {
            "image_url": image_url,
            "label": label,
            "source_url": source_url,
            "audit_score": str(score),
            "audit_verdict": verdict,
            "audit_reasons": "; ".join(reasons) if reasons else "no clear temporal cues",
            "query_hint": query_hint,
        }

    # Wikipedia summary thumbnail (quick baseline image).
    try:
        page = entrepreneur_name.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.ok:
            payload = r.json()
            thumb = str(payload.get("thumbnail", {}).get("source", "")).strip()
            if thumb and thumb not in seen_urls:
                seen_urls.add(thumb)
                out.append(
                    _audit_candidate(
                        label=f"{entrepreneur_name} (Wikipedia thumbnail)",
                        image_url=thumb,
                        source_url=f"https://en.wikipedia.org/wiki/{page}",
                        query_hint="wikipedia summary",
                    )
                )
    except Exception:
        pass

    # Wikimedia Commons search for early-life/pre-startup image cues.
    queries = [
        f"{entrepreneur_name} childhood",
        f"{entrepreneur_name} young",
        f"{entrepreneur_name} student",
        f"{entrepreneur_name} {max(cutoff_year - 5, 1970)}",
        f"{entrepreneur_name} {max(cutoff_year - 1, 1970)}",
    ]
    for q in queries:
        if len(out) >= 8:
            break
        try:
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": q,
                "gsrlimit": 8,
                "prop": "imageinfo",
                "iiprop": "url",
                "format": "json",
            }
            r = requests.get("https://commons.wikimedia.org/w/api.php", params=params, headers=headers, timeout=12)
            if not r.ok:
                continue
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                title = str(page.get("title", ""))
                info = page.get("imageinfo", [])
                if not info:
                    continue
                img_url = str(info[0].get("url", "")).strip()
                low = img_url.lower()
                if not img_url or img_url in seen_urls:
                    continue
                if not any(low.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                    continue
                if any(token in low for token in ("logo", "icon", "signature", "seal")):
                    continue
                seen_urls.add(img_url)
                out.append(
                    _audit_candidate(
                        label=title.replace("File:", ""),
                        image_url=img_url,
                        source_url=f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                        query_hint=q,
                    )
                )
                if len(out) >= 8:
                    break
        except Exception:
            continue
    return out[:8]


lang_col, _ = st.columns([1.2, 4.8])
with lang_col:
    ui_lang_pick = st.selectbox(
        "Language / 语言",
        ["English", "简体中文"],
        index=0,
        key="ui_language_toggle",
    )
UI_LANG = "zh" if ui_lang_pick == "简体中文" else "en"


def tr(text: str) -> str:
    fixed_zh = {
        "AI Founder Model": "AI 创始人模型",
        "Founder Signal AI Pattern Recognition Model": "创始人信号 AI 模型",
        "1. Data log viewer": "1. 数据日志查看器",
        "2. Founder Signal AI Pattern Recognition Model": "2. 创始人信号 AI 模型",
        "3. Modern Founder Screen (Placeholder - TBD)": "3. 现代创始人筛选（占位 - 待定）",
    }
    if UI_LANG == "zh" and text in fixed_zh:
        return fixed_zh[text]
    return _translate_text_with_deepseek(text, UI_LANG)


def _translate_analysis_category(category: str) -> str:
    zh_map = {
        "Decision making": "决策方式",
        "Early hardships / challenges and resilience indicators": "早期困难/挑战与韧性信号",
        "Early technical proficiency": "早期技术能力",
        "Personality": "人格特质",
        "Crazy things done": "非常规行为/大胆举动",
        "Values": "价值观",
        "Sense of security / insecurity": "安全感/不安全感",
        "Motivation for starting business": "创业动机",
        "Intellectual curiousity": "求知欲与智识好奇心",
    }
    raw = str(category or "").strip()
    if UI_LANG == "zh" and raw in zh_map:
        return zh_map[raw]
    return _translate_text_with_deepseek(raw, UI_LANG)


st.title(tr("AI Founder Model"))
st.markdown(
    f"""
<div class="hero-wrap">
  <div class="hero-title">{tr("Investor Demo Mode")}</div>
  <div class="hero-sub">
    {tr("Evidence-first founder intelligence with strict pre-cutoff timelines, pattern learning, and practical scoring guidance for identifying potential outlier founders in the AI era.")}
  </div>
</div>
""",
    unsafe_allow_html=True,
)
view_mode = st.radio(
    tr("View mode"),
    [
        tr("1. Data log viewer"),
        tr("2. Founder Signal AI Pattern Recognition Model"),
        tr("3. Modern Founder Screen (Placeholder - TBD)"),
    ],
    index=0,
    horizontal=True,
    help=tr("Switch between founder data logs and model-level pattern intelligence."),
)

entrepreneurs = get_entrepreneurs_from_db()
log_entrepreneurs = get_entrepreneurs_from_logs()
all_names = {str(name).strip() for name in [*entrepreneurs, *log_entrepreneurs] if str(name).strip()}
entrepreneurs = sorted(all_names)

if view_mode.startswith("1.") or "Data log viewer" in view_mode or "数据日志" in view_mode:
    if not entrepreneurs:
        st.warning(tr("No entrepreneurs found in DB or logs. Run ingestion first."))
    else:
        selected_entrepreneur = st.selectbox(
            tr("Select founder"),
            entrepreneurs,
            index=0,
            help=tr("Choose a founder to view full pre-cutoff evidence and LLM analysis."),
        )

        data_df = fetch_from_db(selected_entrepreneur)
        if data_df.empty:
            data_df = fetch_from_json(selected_entrepreneur)
        comprehensive_df = fetch_comprehensive_from_json(selected_entrepreneur)
        organized_analysis = fetch_organized_analysis_from_json(selected_entrepreneur)
        coverage_stats = fetch_comprehensiveness_stats(selected_entrepreneur)
        excluded_df = fetch_excluded_from_json(selected_entrepreneur)
        cutoff_date, cutoff_reason = fetch_cutoff_context(selected_entrepreneur)
        cutoff_year_match = re.search(r"\b((?:19|20)\d{2})\b", str(cutoff_date))
        cutoff_year = int(cutoff_year_match.group(1)) if cutoff_year_match else 9999
        comprehensive_df = filter_pre_cutoff_timeline(comprehensive_df, cutoff_year=cutoff_year)
        comprehensive_df = deduplicate_fact_dataframe(comprehensive_df)

        st.markdown("---")
        st.header(f"🧠 {selected_entrepreneur}")
        st.markdown(
            f"""
<p class="section-intro">
{_translate_text_with_deepseek(f"This section summarizes pre-cutoff evidence for {selected_entrepreneur}. All accepted facts are intended to represent information that could have been observed before the founder's first institutional funding milestone.", UI_LANG)}
</p>
""",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric(tr("Accepted Facts"), f"{len(data_df):,}")
        c2.metric(tr("Comprehensive Facts"), f"{coverage_stats.get('comprehensive_fact_count', len(comprehensive_df)):,}")
        c3.metric(tr("Excluded Facts"), f"{len(excluded_df):,}")

        companies, total_company_value = get_founder_company_values(selected_entrepreneur)
        if companies:
            st.markdown(f"📈 **{tr('Founder company value context (for success modeling):')}**")
            for item in companies:
                label = str(item.get("company", "Unknown company"))
                raw_value = item.get("value_usd", 0.0)
                value_text = _format_usd(float(raw_value or 0.0)) if raw_value not in (None, "", "n/a") else "undisclosed"
                basis = str(item.get("type", "valuation")).replace("_", " ")
                as_of = str(item.get("as_of", "unknown"))
                confidence = str(item.get("confidence", "unknown"))
                src = str(item.get("source_url", "")).strip()
                notes = str(item.get("notes", "")).strip()
                source_part = f" ([source]({src}))" if src.startswith("http") else ""
                st.markdown(
                    f"- `{label}` — {value_text} ({basis}), as of `{as_of}`, confidence `{confidence}`{source_part}"
                )
                if notes:
                    st.caption(f"  note: {notes}")
            st.markdown(f"- **{tr('Total founded-company value:')}** `{_format_usd(total_company_value)}`")

        st.info(f"⏱️ **{tr('DATE CUTOFF:')}** `{cutoff_date}`  \n**{tr('Why this cutoff:')}** {tr(cutoff_reason)}")
        if coverage_stats:
            targets = coverage_stats.get("targets", {})
            min_sources = targets.get("min_sources", "n/a")
            min_words = targets.get("min_words", "n/a")
            min_words_text = f"{int(min_words):,}" if isinstance(min_words, int) else str(min_words)
            st.caption(
                tr("Coverage targets:")
                + " "
                f"sources={coverage_stats.get('source_count', 0)} "
                + tr(f"(target {min_sources}), ")
                + " "
                f"words={coverage_stats.get('total_source_words', 0):,} "
                + tr(f"(target {min_words_text}).")
            )

        tab_analysis, tab_facts = st.tabs(
            [
                tr("Tab 1 - LLM Analysis & Categorization"),
                tr("Tab 2 - Comprehensive Pre-Cutoff Fact Library"),
            ],
            on_change="ignore",
        )

        with tab_analysis:
            st.markdown(
                f"""
<p class="section-intro">
{_translate_text_with_deepseek('This tab is a synthesized founder analysis built from the comprehensive pre-cutoff fact library. It focuses on inferred behavior patterns and investment-relevant interpretation, not simple fact repetition.', UI_LANG)}
</p>
""",
                unsafe_allow_html=True,
            )
            if not organized_analysis:
                st.warning(tr("No organized analysis found yet in log output."))
            else:
                for category, payload in organized_analysis.items():
                    category_label = _translate_analysis_category(str(category))
                    st.markdown(
                        f"<div class='category-header'>🧩 {category_label}</div>",
                        unsafe_allow_html=True,
                    )
                    if not isinstance(payload, dict):
                        st.markdown(f"- {tr('No analysis payload generated.')}")
                        continue
                    analysis_text = str(payload.get("analysis", "")).strip()
                    confidence = str(payload.get("confidence", "low")).strip()
                    evidence_count = int(payload.get("evidence_count", 0))
                    key_signals = payload.get("key_signals", [])
                    if analysis_text:
                        st.markdown(_translate_markdown_snapshot(analysis_text, UI_LANG))
                    st.markdown(f"- {tr('Confidence')}: `{confidence}`")
                    st.markdown(f"- {tr('Evidence points used')}: `{evidence_count}`")
                    if isinstance(key_signals, list) and key_signals:
                        st.markdown(f"**{tr('Key signals inferred:')}**")
                        for signal in key_signals:
                            st.markdown(f"- {_translate_text_with_deepseek(str(signal), UI_LANG)}")

        with tab_facts:
            st.markdown(
                f"""
<p class="section-intro">
{_translate_text_with_deepseek('This tab is the maximum-recall evidence dump. It aims to include as many pre-cutoff facts as possible, even when confidence is mixed, so that no potentially relevant early signal is hidden. Facts are shown in chronological order as a biography-style timeline up to the DATE CUTOFF.', UI_LANG)}
</p>
""",
                unsafe_allow_html=True,
            )
            if comprehensive_df.empty:
                st.warning(tr("No comprehensive pre-cutoff fact library found in logs yet."))
            else:
                image_candidates = fetch_pre_cutoff_image_candidates(selected_entrepreneur, cutoff_year=cutoff_year)
                if image_candidates:
                    st.markdown(f"### {tr('Early Appearance References')}")
                    st.caption(
                        _translate_text_with_deepseek(
                            "Public image candidates are automatically audited for pre-cutoff likelihood. Only likely pre-cutoff images are shown below.",
                            UI_LANG,
                        )
                    )
                    likely_pre = [x for x in image_candidates if str(x.get("audit_verdict", "")) == "likely_pre_cutoff"]
                    if likely_pre:
                        columns = st.columns(3)
                        for i, img in enumerate(likely_pre):
                            col = columns[i % 3]
                            with col:
                                st.image(img["image_url"], use_container_width=True)
                                st.markdown(f"*{img['label']}*")
                                st.markdown(f"[source]({img['source_url']})")
                    else:
                        st.info(tr("No high-confidence pre-cutoff images were found automatically. Review audit table below."))

                    with st.expander(tr("Image Candidate Audit (all candidates)")):
                        audit_rows = []
                        for img in image_candidates:
                            audit_rows.append(
                                {
                                    "label": img.get("label", ""),
                                    "verdict": img.get("audit_verdict", ""),
                                    "score": img.get("audit_score", ""),
                                    "reasons": img.get("audit_reasons", ""),
                                    "query_hint": img.get("query_hint", ""),
                                    "source_url": img.get("source_url", ""),
                                    "image_url": img.get("image_url", ""),
                                }
                            )
                        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)
                timeline = comprehensive_df.copy()
                timeline["timestamp_sort"] = pd.to_datetime(timeline["timestamp"], errors="coerce")
                timeline = timeline.sort_values(by=["timestamp_sort", "timestamp"], ascending=[True, True])
                st.markdown(
                    f"<div class='category-header'>📜 {tr('Chronological Fact Timeline')} ({len(timeline)} {tr('facts')})</div>",
                    unsafe_allow_html=True,
                )
                metadata_rows: list[dict[str, str]] = []
                for idx, (_, row) in enumerate(timeline.iterrows(), start=1):
                    fact_text = str(row.get("fact", "")).strip()
                    timestamp = str(row.get("timestamp", "")).strip() or "N/A"
                    source = str(row.get("source", "")).strip()
                    inference = str(row.get("inference", "")).strip()
                    confidence = str(row.get("confidence", "low")).strip()
                    badge_text, badge_class = timing_badge(inference)

                    st.markdown(
                        f"- **{timestamp}** — {fact_text} "
                        f"<a href='#fact-meta-{idx}'>details #{idx}</a>",
                        unsafe_allow_html=True,
                    )
                    links = parse_sources_to_links(source)
                    metadata_rows.append(
                        {
                            "fact_idx": str(idx),
                            "provenance": badge_text,
                            "provenance_class": badge_class,
                            "confidence": confidence,
                            "source_urls": links,
                            "inference": inference,
                        }
                    )

                st.markdown("<a id='fact-metadata--sources'></a>", unsafe_allow_html=True)
                st.markdown(f"### {tr('Fact Metadata & Sources')}")
                st.markdown(
                    f"<p class='section-intro'>{_translate_text_with_deepseek('Per-fact provenance, confidence, and source links are listed below to keep the biography timeline clean.', UI_LANG)}</p>",
                    unsafe_allow_html=True,
                )
                for meta in metadata_rows:
                    st.markdown(
                        f"<a id='fact-meta-{meta['fact_idx']}'></a>"
                        f"<a href='#fact-metadata--sources' style='float:right; font-size:0.85rem;'>{tr('back to top of metadata')}</a>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"- **{tr('Fact')} #{meta['fact_idx']}** "
                        f"<span class='badge {meta['provenance_class']}'>{meta['provenance']}</span>",
                        unsafe_allow_html=True,
                    )
                    if meta["inference"]:
                        st.markdown(f"  - {tr('Inference')}: {_translate_text_with_deepseek(str(meta['inference']), UI_LANG)}")
                    st.markdown(f"  - {tr('Confidence')}: `{meta['confidence']}`")
                    source_urls = meta.get("source_urls", [])
                    if isinstance(source_urls, list) and source_urls:
                        st.markdown(f"  - {tr('Sources')}:")
                        for i, url in enumerate(source_urls, start=1):
                            if str(url).startswith("http"):
                                st.markdown(f"    - [source {i}]({url})")
                            else:
                                st.markdown(f"    - `{url}`")
                    else:
                        st.markdown(f"  - {tr('Sources')}: N/A")

        st.subheader(f"🚫 {tr('Excluded Facts Review')}")
        if excluded_df.empty:
            st.info(tr("No excluded facts are currently available in logs for this entrepreneur."))
        else:
            st.markdown(
                f"""
<div class="excluded-box">
{_translate_text_with_deepseek('Facts below were excluded by the ingestion rules (e.g., post-cutoff, weak temporal anchors, or insufficient corroboration). This section is useful for auditing recall vs. precision trade-offs.', UI_LANG)}
</div>
""",
                unsafe_allow_html=True,
            )
            st.dataframe(excluded_df, use_container_width=True)
elif view_mode.startswith("2.") or "Pattern Recognition Model" in view_mode or "模式识别模型" in view_mode:
    st.header(tr("Founder Signal AI Pattern Recognition Model"))
    st.markdown(
        f"""
<p class="section-intro">
{_translate_text_with_deepseek('This section trains an MVP pattern-learning layer from the current historical founder set and extracts reusable founder signals for modern seed diligence. The framework is designed to continuously update as more historical founders are added.', UI_LANG)}
</p>
""",
        unsafe_allow_html=True,
    )

    all_training_founders = entrepreneurs if entrepreneurs else sorted(get_entrepreneurs_from_logs())
    auto_signature = "|".join(sorted(all_training_founders))
    if auto_signature and st.session_state.get("pattern_auto_run_signature") != auto_signature:
        try:
            db_dsn = os.getenv("DATABASE_URL", "").strip() or None
            service = PatternLearningService(db_dsn=db_dsn)
            summary = service.run_full_training(
                n_clusters=3,
                data_source="auto",
                founder_subset=all_training_founders,
            )
            st.session_state["pattern_auto_run_signature"] = auto_signature
            st.caption(
                tr("Auto-refreshed pattern model for all founders:")
                + f" rows={summary.rows_used}, clusters={summary.n_clusters}, source={summary.data_source}"
            )
        except Exception as exc:
            st.warning(f"{tr('Auto-refresh for all founders failed')}: {exc}")

    with st.expander(tr("Run / refresh pattern learning model")):
        st.caption(
            _translate_text_with_deepseek(
                "Runs SentenceTransformer embeddings + K-means clustering + XGBoost scoring. If DB is unavailable, it automatically trains from the JSON logs.",
                UI_LANG,
            )
        )
        selected_training_founders = st.multiselect(
            tr("Founders included in this training run"),
            options=all_training_founders,
            default=all_training_founders,
            help=tr("Default is all founders. Select a subset and click run to manually retrain."),
        )
        if st.button(tr("Train pattern model for selected founders"), key="train_pattern_model"):
            try:
                if not selected_training_founders:
                    st.error(tr("Select at least one founder to run pattern training."))
                else:
                    db_dsn = os.getenv("DATABASE_URL", "").strip() or None
                    service = PatternLearningService(db_dsn=db_dsn)
                    summary = service.run_full_training(
                        n_clusters=3,
                        data_source="auto",
                        founder_subset=list(selected_training_founders),
                    )
                    st.success(
                        f"{tr('Pattern model trained.')} rows={summary.rows_used}, "
                        f"clusters={summary.n_clusters}, source={summary.data_source}"
                    )
            except Exception as exc:
                st.error(f"{tr('Pattern training failed')}: {exc}")

    pattern_report = load_pattern_report()
    if not pattern_report:
        st.info(
            tr("No pattern-learning report found yet. Click the training button above to generate `models/founder_pattern_report.json`.")
        )
    else:
        goal = str(pattern_report.get("goal", "")).strip()
        if goal:
            st.markdown(f"**{tr('Goal')}:** {_translate_text_with_deepseek(goal, UI_LANG)}")
        st.markdown(
            f"- {tr('Data source')}: `{pattern_report.get('data_source', 'unknown')}`\n"
            f"- {tr('Historical founders used')}: `{pattern_report.get('founder_count', 0)}`"
        )
        tab_conclusion, tab_model_analysis, tab_questions, tab_scoring = st.tabs(
            [tr("AI Model Conclusion"), tr("AI Model Analysis"), tr("Founder Interview Questions"), tr("Founder Scoring Matrix")]
        )

        with tab_conclusion:
            if "pattern_narrative_refresh" not in st.session_state:
                st.session_state["pattern_narrative_refresh"] = 0
            if st.button(tr("Retry DeepSeek narrative"), key="retry_pattern_narrative"):
                st.session_state["pattern_narrative_refresh"] += 1

            llm_narrative, narrative_error, narrative_key_source = generate_pattern_narrative_with_deepseek(
                pattern_report,
                refresh_nonce=int(st.session_state.get("pattern_narrative_refresh", 0)),
            )
            if llm_narrative:
                st.markdown(_translate_markdown_snapshot(llm_narrative, UI_LANG))
            else:
                st.info(
                    "DeepSeek narrative is unavailable "
                    f"(`{narrative_error or 'unknown_error'}`). "
                    "If your key is set, click Retry DeepSeek narrative. "
                    "Showing fallback findings below."
                )
                st.caption(f"DeepSeek key source check: `{narrative_key_source}`")
                findings = pattern_report.get("findings", [])
                if isinstance(findings, list) and findings:
                    for finding in findings:
                        st.markdown(f"- {_translate_text_with_deepseek(str(finding), UI_LANG)}")

        with tab_model_analysis:
            methodology = pattern_report.get("methodology", [])
            if isinstance(methodology, list) and methodology:
                st.markdown(f"### {tr('Methodology')}")
                for step in methodology:
                    st.markdown(f"- {_translate_text_with_deepseek(str(step), UI_LANG)}")

            value_methodology = pattern_report.get("company_value_methodology", [])
            if isinstance(value_methodology, list) and value_methodology:
                st.markdown(f"### {tr('Market Value Methodology')}")
                for step in value_methodology:
                    st.markdown(f"- {_translate_text_with_deepseek(str(step), UI_LANG)}")

            top_signals = pattern_report.get("top_signal_importance", [])
            if isinstance(top_signals, list) and top_signals:
                st.markdown(f"### {tr('Patterns & model signals')}")
                signal_df = pd.DataFrame(top_signals).sort_values("importance", ascending=False)
                signal_df["rank"] = range(1, len(signal_df) + 1)
                signal_df["feature_code"] = signal_df["feature"].astype(str).str.slice(0, 18)
                fig_signals = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=signal_df["rank"],
                            y=signal_df["importance"],
                            z=signal_df["importance"].cumsum(),
                            mode="markers+text",
                            text=signal_df["feature_code"],
                            textposition="top center",
                            marker=dict(
                                size=7,
                                color=signal_df["importance"],
                                colorscale="Viridis",
                                opacity=0.9,
                                colorbar=dict(title="Importance"),
                            ),
                        )
                    ]
                )
                fig_signals.update_layout(
                    scene=dict(
                        xaxis_title="Signal Rank",
                        yaxis_title="Importance",
                        zaxis_title="Cumulative Importance",
                    ),
                    height=520,
                    margin=dict(l=0, r=0, t=30, b=0),
                    title="3D Signal Topography",
                )
                st.plotly_chart(fig_signals, use_container_width=True)
                st.dataframe(signal_df, use_container_width=True)

            clusters = pattern_report.get("cluster_summary", [])
            if isinstance(clusters, list) and clusters:
                st.markdown(f"### {tr('Founder archetype clusters')}")
                cluster_df = pd.DataFrame(clusters)
                if not cluster_df.empty:
                    if {
                        "cluster_id",
                        "total_cluster_value_usd",
                        "value_weighted_success_score",
                    }.issubset(cluster_df.columns):
                        fig_clusters = go.Figure(
                            data=[
                                go.Scatter3d(
                                    x=cluster_df["cluster_id"],
                                    y=cluster_df["total_cluster_value_usd"] / 1_000_000_000,
                                    z=cluster_df["value_weighted_success_score"],
                                    mode="markers+text",
                                    text=cluster_df["founders"].astype(str),
                                    textposition="top center",
                                    marker=dict(
                                        size=10 + cluster_df.get("size", 1).astype(float) * 3,
                                        color=cluster_df["value_weighted_success_score"],
                                        colorscale="Plasma",
                                        opacity=0.9,
                                    ),
                                )
                            ]
                        )
                        fig_clusters.update_layout(
                            scene=dict(
                                xaxis_title="Cluster ID",
                                yaxis_title="Total Cluster Value (USD B)",
                                zaxis_title="Value-Weighted Success Score",
                            ),
                            height=520,
                            margin=dict(l=0, r=0, t=30, b=0),
                            title="3D Founder Cluster Map",
                        )
                        st.plotly_chart(fig_clusters, use_container_width=True)
                    if "total_cluster_value_usd" in cluster_df.columns:
                        cluster_df["total_cluster_value"] = cluster_df["total_cluster_value_usd"].map(_format_usd)
                    st.dataframe(cluster_df, use_container_width=True)

            founder_values = pattern_report.get("founder_company_values", [])
            if isinstance(founder_values, list) and founder_values:
                st.markdown(f"### {tr('Founder company value basis')}")
                rows: list[dict[str, str]] = []
                chart_points: list[dict[str, Any]] = []
                founder_index_map: dict[str, int] = {}
                founder_idx = 1
                for item in founder_values:
                    founder = str(item.get("founder", ""))
                    if founder not in founder_index_map:
                        founder_index_map[founder] = founder_idx
                        founder_idx += 1
                    companies = item.get("companies", [])
                    total = float(item.get("total_company_value_usd", 0.0) or 0.0)
                    if isinstance(companies, list) and companies:
                        for company in companies:
                            raw_value = company.get("value_usd", 0.0)
                            as_of = str(company.get("as_of", ""))
                            year_match = re.search(r"\b(19|20)\d{2}\b", as_of)
                            as_of_year = int(year_match.group(0)) if year_match else 2000
                            rows.append(
                                {
                                    "founder": founder,
                                    "company": str(company.get("company", "")),
                                    "value": (
                                        _format_usd(float(raw_value or 0.0))
                                        if raw_value not in (None, "", "n/a")
                                        else "undisclosed"
                                    ),
                                    "basis": str(company.get("type", "")).replace("_", " "),
                                    "as_of": as_of,
                                    "confidence": str(company.get("confidence", "unknown")),
                                    "founder_total": _format_usd(total),
                                }
                            )
                            if raw_value not in (None, "", "n/a"):
                                chart_points.append(
                                    {
                                        "founder_idx": founder_index_map[founder],
                                        "founder": founder,
                                        "company": str(company.get("company", "")),
                                        "value_b": float(raw_value) / 1_000_000_000,
                                        "as_of_year": as_of_year,
                                    }
                                )
                if chart_points:
                    cp = pd.DataFrame(chart_points)
                    fig_values = go.Figure()
                    fig_values.add_trace(
                        go.Scatter(
                            x=cp["founder"],
                            y=cp["value_b"],
                            mode="markers+text",
                            text=cp["company"],
                            textposition="top center",
                            marker=dict(
                                size=(cp["value_b"].clip(lower=1) ** 0.5).clip(8, 70),
                                color=cp["as_of_year"],
                                colorscale="Turbo",
                                showscale=True,
                                colorbar=dict(title="As-of year"),
                                opacity=0.72,
                                line=dict(width=1, color="rgba(30,30,30,0.35)"),
                            ),
                        )
                    )
                    fig_values.update_layout(
                        xaxis_title="Founder",
                        yaxis_title="Company Value (USD B)",
                        height=500,
                        margin=dict(l=0, r=0, t=30, b=0),
                        title="Founder Company Value Bubble Map",
                    )
                    st.plotly_chart(fig_values, use_container_width=True)
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with tab_questions:
            interview_questions = pattern_report.get("interview_questions", [])
            if isinstance(interview_questions, list) and interview_questions:
                for q in interview_questions:
                    st.markdown(f"- {_translate_text_with_deepseek(str(q), UI_LANG)}")
            else:
                st.info(tr("No interview questions found in the current model report."))

        with tab_scoring:
            st.caption(
                _translate_text_with_deepseek(
                    "DeepSeek-generated scoring framework for evaluating modern early-stage founders from meetings and public data.",
                    UI_LANG,
                )
            )
            if "scoring_matrix_refresh" not in st.session_state:
                st.session_state["scoring_matrix_refresh"] = 0
            if st.button(tr("Regenerate scoring matrix"), key="retry_scoring_matrix"):
                st.session_state["scoring_matrix_refresh"] += 1
            scoring_md, scoring_error, scoring_key_source = generate_scoring_matrix_with_deepseek(
                pattern_report,
                refresh_nonce=int(st.session_state.get("scoring_matrix_refresh", 0)),
            )
            if scoring_md:
                st.markdown(_translate_markdown_snapshot(scoring_md, UI_LANG))
            else:
                st.info(
                    tr("DeepSeek scoring matrix is unavailable")
                    + f" (`{scoring_error or 'unknown_error'}`). "
                    + tr("Showing fallback starter matrix.")
                )
                st.caption(f"DeepSeek key source check: `{scoring_key_source}`")
                st.markdown(
                    _translate_text_with_deepseek(
                        """
| Field | Weight (%) | Why it matters | Evidence to collect now |
|---|---:|---|---|
| Founder-market obsession | 18 | Drives persistence and differentiated insight | Founder posts, interviews, product iteration logs |
| Technical velocity | 16 | Predicts speed of learning and compounding | GitHub commits, shipped features, demo cadence |
| Decision quality under uncertainty | 14 | Determines survival in ambiguous early phases | Pivots, kill-decisions, postmortems |
| Resourcefulness / resilience | 12 | Needed for sparse-resource execution | Prior failures recovered, low-capital wins |
| Distribution instinct | 12 | Great products fail without distribution | Early user growth channels, community traction |
| Talent magnetism | 10 | Attracting A-players is a force multiplier | Early hires, advisor quality, referrals |
| AI-native leverage | 10 | AI-era founders need asymmetrical build leverage | AI workflow integration, model usage depth |
| Integrity / trustworthiness | 8 | Long-term partnership and execution reliability | Reference checks, consistency across narratives |
""",
                        UI_LANG,
                    )
                )
                st.markdown(
                    _translate_text_with_deepseek(
                        """
**Public-data suggestions (today's young founders):**
- Track build cadence via GitHub, changelogs, release notes.
- Check technical depth through engineering blog quality and architecture decisions.
- Look for sustained niche community engagement (not vanity social spikes).
- Prioritize evidence of user-love signals (retention/organic referrals) over follower counts.
- Assess clarity of founder thesis across interviews/posts over time.
- Watch for fast learning loops: hypothesis -> build -> measure -> iterate.
- Validate whether early team quality exceeds company stage.
- Cross-check claim consistency across LinkedIn, X, product docs, and talks.
- Identify contrarian insight with falsifiable logic, not slogans.
- Flag hype-heavy profiles with weak shipping evidence.
""",
                        UI_LANG,
                    )
                )
else:
    st.header(tr("Modern Founder Screen (Placeholder - TBD)"))
    st.markdown(
        f"""
<p class="section-intro">
{_translate_text_with_deepseek('This section is reserved for the upcoming modern founder scanning workflow. It will support live screening, ranking, and shortlisting of current founders against the historical pattern model.', UI_LANG)}
</p>
""",
        unsafe_allow_html=True,
    )
    st.info(tr("Placeholder active. Product requirements and scoring UX to be finalized."))

with st.sidebar:
    st.title(tr("⚙️ App Controls"))
    st.markdown(tr("Configure DB via `DATABASE_URL` (or PG env vars) for production."))
    st.info(tr("Optimized for partner/investor demos: mobile-friendly layouts, clear tabs, and cached AI narratives."))

st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: #888;'>{tr('Powered by Founder AI model • Deploy to Streamlit Cloud for sharing.')}</p>",
    unsafe_allow_html=True,
)
