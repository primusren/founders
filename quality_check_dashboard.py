from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
import streamlit as st
from psycopg2.extras import RealDictCursor


def get_db_connection():
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return psycopg2.connect(database_url)
    return psycopg2.connect(
        dbname=os.getenv("PGDATABASE", "seed_founder_intel"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "postgres"),
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
    )


def ensure_suggestions_table() -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_suggestions (
                    id SERIAL PRIMARY KEY,
                    entrepreneur_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    existing_fact TEXT,
                    suggestion TEXT NOT NULL,
                    new_source TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
                """
            )


def fetch_entrepreneurs() -> list[dict]:
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, name, first_institutional_investment_date
                FROM entrepreneurs
                ORDER BY name;
                """
            )
            return list(cur.fetchall())


def fetch_ingested_data(entrepreneur_name: str) -> pd.DataFrame:
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
            SELECT
                a.category,
                COALESCE(a.fact, a.attribute_text) AS fact,
                COALESCE(a.timestamp, a.event_date) AS timestamp,
                COALESCE(a.source, a.source_url) AS source,
                a.source_url,
                a.before_cutoff,
                COALESCE(a.extraction_method, '') AS extraction_method
            FROM attributes a
            JOIN entrepreneurs e ON a.entrepreneur_id = e.id
            WHERE e.name = %s
            ORDER BY a.category, COALESCE(a.timestamp, a.event_date) NULLS LAST;
            """
            cur.execute(query, (entrepreneur_name,))
            rows = cur.fetchall()
            return pd.DataFrame(rows)


def parse_log_payload(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    try:
        return json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def accepted_df_from_log(payload: dict) -> pd.DataFrame:
    accepted = payload.get("accepted_claims", [])
    rows: list[dict] = []
    for item in accepted:
        merged_sources = item.get("merged_sources") or [item.get("source", item.get("source_url", ""))]
        rows.append(
            {
                "category": item.get("category", "uncategorized"),
                "fact": item.get("attribute_text", ""),
                "timestamp": item.get("event_date", ""),
                "source": ", ".join(str(s) for s in merged_sources if str(s).strip()),
                "source_url": item.get("source_url", ""),
                "before_cutoff": bool(item.get("before_cutoff", True)),
                "extraction_method": item.get("extraction_method", ""),
                "timestamp_inference": item.get("timestamp_inference", ""),
                "inference_reason": item.get("inference_reason", ""),
                "verification_source_count": item.get("verification_source_count", 0),
            }
        )
    return pd.DataFrame(rows)


def excluded_df_from_log(payload: dict) -> pd.DataFrame:
    excluded = payload.get("excluded_claims", [])
    rows: list[dict] = []
    for item in excluded:
        rows.append(
            {
                "category": item.get("category", "uncategorized"),
                "fact": item.get("attribute_text", ""),
                "timestamp": item.get("event_date", ""),
                "source": item.get("source", item.get("source_url", "")),
                "source_url": item.get("source_url", ""),
                "discard_reason": item.get("discard_reason", ""),
            }
        )
    return pd.DataFrame(rows)


def save_suggestion(
    entrepreneur_name: str,
    category: str,
    existing_fact: str,
    suggestion: str,
    new_source: str,
) -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_suggestions (
                    entrepreneur_name, category, existing_fact, suggestion, new_source, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (entrepreneur_name, category, existing_fact, suggestion, new_source, datetime.utcnow()),
            )


st.set_page_config(page_title="Ingestion Quality Check Dashboard", layout="wide")
st.title("Super Seed Investor AI - Ingestion Quality Check Dashboard")

st.markdown(
    """
This dashboard helps audit ingested historical founder facts.
Each fact is shown by category with timestamp and citation source.
Use the suggestions form to capture corrections and iteration notes.
"""
)

log_file_path = Path("data/elon_ingestion_detailed_log.json")
db_mode = True

try:
    ensure_suggestions_table()
    entrepreneurs = fetch_entrepreneurs()
except Exception as exc:
    db_mode = False
    entrepreneurs = []
    st.warning(
        "Database connection unavailable. Falling back to local ingestion log data.\n\n"
        f"Reason: {exc}"
    )

if db_mode and not entrepreneurs:
    st.info("DB is reachable but has no entrepreneurs yet. Falling back to local ingestion log if available.")
    db_mode = False

if not db_mode:
    payload = parse_log_payload(log_file_path)
    if not payload:
        st.error("No DB data and no local log found at `data/elon_ingestion_detailed_log.json`.")
        st.stop()
    selected_name = str(payload.get("entrepreneur_name", "Elon Musk"))
    cutoff_date = payload.get("cutoff_date", "unknown")
    data_df = accepted_df_from_log(payload)
    excluded_df = excluded_df_from_log(payload)
    st.subheader(f"{selected_name} - DATE CUTOFF: {cutoff_date} (log fallback)")
else:
    entrepreneur_options = {e["name"]: e for e in entrepreneurs}
    selected_name = st.selectbox("Select Entrepreneur", list(entrepreneur_options.keys()))
    selected = entrepreneur_options[selected_name]
    cutoff_date = selected.get("first_institutional_investment_date")
    st.subheader(f"{selected_name} - DATE CUTOFF: {cutoff_date}")
    db_df = fetch_ingested_data(selected_name)
    data_df = db_df[db_df["before_cutoff"] == True].copy()  # noqa: E712
    excluded_df = db_df[db_df["before_cutoff"] == False].copy()  # noqa: E712
    if "discard_reason" not in excluded_df.columns:
        excluded_df["discard_reason"] = "not_available_in_attributes_table"

if not data_df.empty:
    categories = sorted(data_df["category"].dropna().unique().tolist())
    for category in categories:
        st.markdown(f"### {str(category).replace('_', ' ').title()}")
        category_data = data_df[data_df["category"] == category]
        for _, row in category_data.iterrows():
            fact = row.get("fact", "")
            timestamp = row.get("timestamp", "")
            source = row.get("source", "")
            st.markdown(f"- **Fact**: {fact}")
            st.markdown(f"  - **Timestamp**: {timestamp}")
            if isinstance(source, str) and source.startswith("http"):
                st.markdown(f"  - **Citation**: [{source}]({source})")
            else:
                st.markdown(f"  - **Citation**: {source}")
            inference = row.get("timestamp_inference", "")
            inference_reason = row.get("inference_reason", "")
            if inference:
                st.markdown(f"  - **Inference**: {inference} ({inference_reason})")
else:
    st.warning("No ingested data found for this entrepreneur.")
    categories = []

st.subheader("Excluded Facts (Audit)")
if excluded_df.empty:
    st.info("No excluded facts available.")
else:
    show_cols = [c for c in ["category", "fact", "timestamp", "source", "discard_reason"] if c in excluded_df.columns]
    st.dataframe(excluded_df[show_cols], use_container_width=True)

if db_mode:
    st.subheader("Audit and Suggestions")
    with st.form(key="suggestion_form"):
        category_select = st.selectbox("Category to Suggest/Edit", categories if categories else ["N/A"])
        fact_text = st.text_area("Existing Fact (if editing) or New Fact")
        suggestion = st.text_area("Your Suggestion/Iteration")
        new_source = st.text_input("New Citation/Source (if applicable)")
        submitted = st.form_submit_button("Submit Suggestion")
        if submitted:
            if not suggestion.strip():
                st.warning("Please provide a suggestion before submitting.")
            else:
                save_suggestion(
                    entrepreneur_name=selected_name,
                    category=str(category_select),
                    existing_fact=fact_text,
                    suggestion=suggestion,
                    new_source=new_source,
                )
                st.success(f"Suggestion submitted for {category_select}.")
else:
    st.caption("Suggestion form is disabled in log fallback mode (DB not connected).")

st.sidebar.title("Controls")
st.sidebar.markdown("Use ingestion to add more entrepreneurs and refresh.")
if st.sidebar.button("Refresh Data"):
    st.rerun()

st.markdown("---")
st.markdown("To publish: deploy this Streamlit app to Streamlit Community Cloud, Render, or Heroku.")

