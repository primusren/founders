from __future__ import annotations

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_BASE = "http://127.0.0.1:8000"


def musk_style_tag(pattern_label: str) -> str:
    label = pattern_label.lower()
    if "technical" in label or "contrarian" in label or "ai-native" in label:
        return "High Musk-style technical intensity"
    if "network" in label:
        return "Moderate Musk-style signal (network-operator)"
    return "Low Musk-style similarity"


def post_json(api_base: str, path: str, payload: dict) -> dict:
    resp = requests.post(f"{api_base}{path}", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


st.set_page_config(page_title="Seed Founder Intelligence Dashboard", layout="wide")
st.title("Seed Founder Intelligence Dashboard")
st.caption("Scan and score early-stage AI founders against historical founder patterns.")

api_base = st.sidebar.text_input("API Base URL", value=DEFAULT_API_BASE)
top_k = st.sidebar.slider("Top Matches", min_value=3, max_value=25, value=10, step=1)

query = st.text_input("Scan Query", value="AI startups seed stage")
run_scan = st.button("Run Scan")

if run_scan:
    try:
        payload = {"query": query, "top_k": top_k}
        data = post_json(api_base, "/scan", payload)
        matches = data.get("matches", [])
        if not matches:
            st.warning("No matches found.")
        else:
            rows = []
            for m in matches:
                top_pattern = (m.get("pattern_matches") or [{}])[0]
                pattern_label = top_pattern.get("pattern_label", "Unknown")
                rows.append(
                    {
                        "candidate": m.get("full_name", ""),
                        "score": m.get("score", 0.0),
                        "location": m.get("location", ""),
                        "stage": m.get("funding_stage", ""),
                        "focus": m.get("company_focus", ""),
                        "historical_pattern_match": pattern_label,
                        "pattern_similarity": round(float(top_pattern.get("similarity", 0.0)), 4),
                        "musk_style_signal": musk_style_tag(pattern_label),
                        "explanation": " | ".join(m.get("explanations", [])),
                    }
                )

            df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
            st.subheader("Ranked Entrepreneurs")
            st.dataframe(df, use_container_width=True)
    except Exception as exc:
        st.error(f"Scan failed: {exc}")

st.markdown("---")
st.subheader("Score a Specific Profile")
with st.form("predict_form"):
    full_name = st.text_input("Full Name", value="Sample Founder")
    location = st.text_input("Location", value="San Francisco, United States")
    estimated_age = st.number_input("Estimated Age", min_value=16, max_value=80, value=28, step=1)
    funding_stage = st.selectbox("Funding Stage", ["seed", "series_a", "series_b"])
    first_time_founder = st.checkbox("First-Time Founder", value=True)
    company_focus = st.text_input("Company Focus", value="AI developer tools")
    bio = st.text_area(
        "Bio / Summary",
        value=(
            "First-time founder building AI developer tooling with strong technical background "
            "and evidence of early shipped ML projects."
        ),
    )
    github_ml_project_count = st.number_input("GitHub ML Projects", min_value=0, max_value=100, value=3, step=1)
    github_total_stars = st.number_input("GitHub Total Stars", min_value=0, max_value=100000, value=500, step=10)
    submit_predict = st.form_submit_button("Predict Profile Score")

if submit_predict:
    profile = {
        "full_name": full_name,
        "location": location,
        "estimated_age": int(estimated_age),
        "funding_stage": funding_stage,
        "first_time_founder": first_time_founder,
        "company_focus": company_focus,
        "bio": bio,
        "github_ml_project_count": int(github_ml_project_count),
        "github_total_stars": int(github_total_stars),
    }
    try:
        result = post_json(api_base, "/predict", {"profile": profile}).get("result", {})
        st.success(f"Predicted Score: {result.get('score', 0.0)}")
        pattern_matches = result.get("pattern_matches", [])
        if pattern_matches:
            pm_df = pd.DataFrame(pattern_matches)
            st.write("Historical Pattern Matches")
            st.dataframe(pm_df, use_container_width=True)
            top_label = pattern_matches[0].get("pattern_label", "")
            st.info(f"Musk-style signal: {musk_style_tag(top_label)}")
        explanations = result.get("explanations", [])
        if explanations:
            st.write("Explanations")
            for item in explanations:
                st.write(f"- {item}")
    except Exception as exc:
        st.error(f"Predict failed: {exc}")

