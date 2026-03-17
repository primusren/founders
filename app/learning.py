from __future__ import annotations

import glob
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from xgboost import XGBClassifier


CATEGORY_ALIAS = {
    "early_technical_proficiency": "early_technical_proficiency",
    "decision_making": "decision_making",
    "education": "education",
    "family_hardship": "family_hardship",
    "personality": "personality",
    "values": "values",
    "security_insecurity": "security_insecurity",
    "motivation": "motivation",
    "network_mentorship": "network_mentorship",
    "resilience": "resilience",
    "intellectual_curiosity": "intellectual_curiosity",
    "ai_specific": "ai_specific",
    "risk_tolerance": "risk_tolerance",
    "crazy_things_done": "risk_tolerance",
    "sense_of_security__insecurity": "security_insecurity",
    "motivation_for_starting_business": "motivation",
}

FEATURE_CATEGORIES = [
    "security_insecurity",
    "early_technical_proficiency",
    "risk_tolerance",
    "ai_specific",
    "decision_making",
    "education",
    "family_hardship",
    "personality",
    "values",
    "motivation",
    "network_mentorship",
    "resilience",
    "intellectual_curiosity",
]

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
            "notes": "Merged valuation estimate from public reporting; update as new disclosures appear.",
        },
        {
            "company": "The Boring Company",
            "value_usd": 5_700_000_000,
            "type": "private_valuation",
            "as_of": "2025-01-01",
            "source_url": "https://www.reuters.com/business/autos-transportation/musks-boring-company-valued-56-bln-after-latest-funding-round-2022-04-21/",
            "confidence": "low",
            "notes": "Latest publicly disclosed-style estimate; stale vs public comps.",
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
            "notes": "No reliable, recent public valuation disclosed.",
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


@dataclass(slots=True)
class TrainingSummary:
    rows_used: int
    feature_count: int
    model_name: str
    status: str
    n_clusters: int
    model_path: str
    cluster_path: str
    report_path: str
    data_source: str


class PatternLearningService:
    """Pattern learning layer with clustering + supervised scoring."""

    def __init__(
        self,
        db_dsn: str | None = None,
        model_dir: str = "models",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        random_state: int = 42,
        log_dir: str = "data",
    ) -> None:
        self.db_dsn = db_dsn or ""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model_name
        self.random_state = random_state
        self.log_dir = Path(log_dir)
        self._embedder: SentenceTransformer | None = None
        self.company_values_path = self.log_dir / "founder_company_values.json"

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _connect(self):
        if not self.db_dsn:
            raise ValueError("DB DSN is not set.")
        return psycopg2.connect(self.db_dsn)

    @staticmethod
    def _normalize_name_key(name: str) -> str:
        return "".join(ch for ch in str(name or "").lower() if ch.isalnum())

    def _load_company_values_lookup(self) -> dict[str, list[dict[str, Any]]]:
        merged: dict[str, list[dict[str, Any]]] = {}
        for founder, entries in DEFAULT_FOUNDER_COMPANY_VALUES_USD.items():
            merged[self._normalize_name_key(founder)] = list(entries)
        if self.company_values_path.exists():
            try:
                payload = json.loads(self.company_values_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    founder_map = payload.get("founders", payload)
                    if isinstance(founder_map, dict):
                        for founder, entries in founder_map.items():
                            if isinstance(entries, list):
                                merged[self._normalize_name_key(founder)] = entries
            except Exception:
                pass
        return merged

    def _companies_for_founder(self, founder_name: str) -> list[dict[str, Any]]:
        lookup = self._load_company_values_lookup()
        return list(lookup.get(self._normalize_name_key(founder_name), []))

    def _company_value_methodology(self) -> list[str]:
        fallback = [
            "Public companies: use latest available market cap snapshot from primary quote pages (Google Finance/Yahoo/official exchange data).",
            "Private companies: use latest publicly disclosed valuation from credible sources (Reuters/WSJ/major financial media or company filings).",
            "Merged entities: use reported combined valuation only when clearly disclosed; otherwise keep separate with confidence notes.",
            "Undisclosed private values are recorded as null and excluded from total value math to avoid false precision.",
            "Each entry stores as_of date, source URL, and confidence; refresh values regularly as markets and rounds move.",
        ]
        if not self.company_values_path.exists():
            return fallback
        try:
            payload = json.loads(self.company_values_path.read_text(encoding="utf-8"))
            steps = payload.get("methodology", {}).get("steps", [])
            if isinstance(steps, list) and steps:
                return [str(x) for x in steps]
        except Exception:
            pass
        return fallback

    @staticmethod
    def _total_company_value(companies: list[dict[str, Any]]) -> float:
        total = 0.0
        for item in companies:
            try:
                raw = item.get("value_usd", 0.0)
                if raw in (None, "", "n/a"):
                    continue
                total += float(raw)
            except Exception:
                continue
        return total

    def ensure_learning_tables(self) -> None:
        if not self.db_dsn:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS entrepreneur_labels (
                        entrepreneur_id INTEGER PRIMARY KEY REFERENCES entrepreneurs(id) ON DELETE CASCADE,
                        success_label INTEGER NOT NULL CHECK (success_label IN (0, 1)),
                        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pattern_clusters (
                        entrepreneur_id INTEGER PRIMARY KEY REFERENCES entrepreneurs(id) ON DELETE CASCADE,
                        cluster_id INTEGER NOT NULL,
                        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                    );
                    """
                )

    @staticmethod
    def _normalize_category(category: str) -> str:
        key = str(category or "").strip().lower().replace(" ", "_").replace("/", "_")
        key = key.replace("-", "_")
        return CATEGORY_ALIAS.get(key, key)

    def load_historical_data_from_db(self) -> pd.DataFrame:
        query = """
            SELECT
                e.id AS entrepreneur_id,
                e.name AS entrepreneur_name,
                a.category,
                COALESCE(a.fact, a.attribute_text) AS attribute_text,
                COALESCE(a.event_date, a.timestamp) AS event_date,
                a.source_url,
                COALESCE(l.success_label, 1) AS success_label
            FROM entrepreneurs e
            JOIN attributes a ON a.entrepreneur_id = e.id
            LEFT JOIN entrepreneur_labels l ON l.entrepreneur_id = e.id
            WHERE a.before_cutoff = TRUE;
        """
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn)
        if not df.empty:
            df["category"] = df["category"].map(self._normalize_category)
        return df

    def load_historical_data_from_logs(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        files = sorted(self.log_dir.glob("*_ingestion_detailed_log.json"))
        for fp in files:
            try:
                payload = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            founder = str(payload.get("entrepreneur_name", "")).strip()
            if not founder:
                continue
            facts = payload.get("comprehensive_pre_cutoff_facts", []) or payload.get("accepted_claims", [])
            for fact in facts:
                text = str(fact.get("fact") or fact.get("attribute_text") or "").strip()
                if not text:
                    continue
                records.append(
                    {
                        "entrepreneur_id": founder,
                        "entrepreneur_name": founder,
                        "category": self._normalize_category(str(fact.get("category", ""))),
                        "attribute_text": text,
                        "event_date": str(fact.get("timestamp") or fact.get("event_date") or ""),
                        "source_url": (
                            (fact.get("sources", [""])[0] if isinstance(fact.get("sources"), list) else "")
                            or str(fact.get("source_url", ""))
                            or str(fact.get("source_primary", ""))
                        ),
                        # Existing founders in this project are success exemplars.
                        "success_label": 1,
                    }
                )
        df = pd.DataFrame(records)
        if df.empty:
            return df
        return df

    def load_historical_data(
        self,
        data_source: str = "auto",
        founder_subset: list[str] | None = None,
    ) -> tuple[pd.DataFrame, str]:
        data_source = data_source.strip().lower()
        if data_source not in {"auto", "db", "logs"}:
            raise ValueError("data_source must be one of: auto, db, logs")
        if data_source in {"auto", "db"} and self.db_dsn:
            try:
                df = self.load_historical_data_from_db()
                if not df.empty:
                    return df, "db"
            except Exception:
                if data_source == "db":
                    raise
        df = self.load_historical_data_from_logs()
        if founder_subset:
            allowed = {self._normalize_name_key(x) for x in founder_subset if str(x).strip()}
            if allowed and not df.empty:
                df = df[df["entrepreneur_name"].map(lambda n: self._normalize_name_key(str(n)) in allowed)].copy()
        return df, "logs"

    def _apply_founder_subset(self, df: pd.DataFrame, founder_subset: list[str] | None) -> pd.DataFrame:
        if not founder_subset or df.empty:
            return df
        allowed = {self._normalize_name_key(x) for x in founder_subset if str(x).strip()}
        if not allowed:
            return df
        return df[df["entrepreneur_name"].map(lambda n: self._normalize_name_key(str(n)) in allowed)].copy()

    def _subset_label(self, founder_subset: list[str] | None) -> str:
        if not founder_subset:
            return "all"
        compact = sorted({str(x).strip() for x in founder_subset if str(x).strip()})
        return "|".join(compact)[:240]

    def aggregate_founder_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("No historical founder data found.")
        df = df.copy()
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        grouped = df.groupby(["entrepreneur_id", "entrepreneur_name", "success_label"], as_index=False).agg(
            attribute_count=("attribute_text", "count"),
            unique_sources=("source_url", "nunique"),
            avg_event_year=("event_date", lambda s: float(s.dt.year.dropna().mean()) if not s.dropna().empty else 0.0),
        )

        for category in FEATURE_CATEGORIES:
            category_count = (
                df.assign(is_cat=(df["category"] == category).astype(int))
                .groupby("entrepreneur_id")["is_cat"]
                .sum()
                .rename(f"{category}_count")
            )
            grouped = grouped.merge(category_count, on="entrepreneur_id", how="left")
            grouped[f"{category}_count"] = grouped[f"{category}_count"].fillna(0.0)

        grouped["source_density"] = grouped["attribute_count"] / grouped["unique_sources"].clip(lower=1)
        grouped["technical_risk_interaction"] = (
            grouped["early_technical_proficiency_count"] * grouped["risk_tolerance_count"]
        )
        grouped["ai_curiosity_interaction"] = (
            grouped["ai_specific_count"] * grouped["intellectual_curiosity_count"]
        )
        return grouped

    def build_text_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        founder_text = (
            df.groupby(["entrepreneur_id", "entrepreneur_name"], as_index=False)["attribute_text"]
            .apply(lambda s: " ".join(s.tolist()))
            .rename(columns={"attribute_text": "attribute_corpus"})
        )
        vectors = self.embedder.encode(
            founder_text["attribute_corpus"].tolist(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        emb_cols = [f"emb_{i}" for i in range(vectors.shape[1])]
        emb_df = pd.DataFrame(vectors, columns=emb_cols)
        emb_df["entrepreneur_id"] = founder_text["entrepreneur_id"].values
        emb_df["entrepreneur_name"] = founder_text["entrepreneur_name"].values
        return emb_df

    def run_kmeans_clustering(self, embedding_df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
        emb_cols = [c for c in embedding_df.columns if c.startswith("emb_")]
        n_clusters = max(2, min(n_clusters, len(embedding_df)))
        model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        clusters = model.fit_predict(embedding_df[emb_cols])
        output = embedding_df[["entrepreneur_id", "entrepreneur_name"]].copy()
        output["cluster_id"] = clusters
        return output

    def _synthesize_control_profiles(self, feature_df: pd.DataFrame, n_controls: int = 6) -> pd.DataFrame:
        if feature_df.empty:
            return feature_df
        rng = random.Random(self.random_state)
        controls: list[dict[str, Any]] = []
        count_cols = [c for c in feature_df.columns if c.endswith("_count")]
        avg_attr = float(feature_df["attribute_count"].mean())
        for idx in range(n_controls):
            base = {
                "entrepreneur_id": f"control_{idx + 1}",
                "entrepreneur_name": f"Synthetic Control {idx + 1}",
                "success_label": 0,
                "attribute_count": max(3.0, avg_attr * rng.uniform(0.35, 0.75)),
                "unique_sources": max(1.0, float(feature_df["unique_sources"].mean() * rng.uniform(0.4, 0.8))),
                "avg_event_year": float(feature_df["avg_event_year"].median() or 1990.0),
                "source_density": 0.0,
                "technical_risk_interaction": 0.0,
                "ai_curiosity_interaction": 0.0,
            }
            for col in count_cols:
                if col in {"ai_specific_count", "early_technical_proficiency_count"}:
                    base[col] = max(0.0, float(feature_df[col].mean() * rng.uniform(0.05, 0.45)))
                elif col in {"risk_tolerance_count", "decision_making_count", "resilience_count"}:
                    base[col] = max(0.0, float(feature_df[col].mean() * rng.uniform(0.15, 0.65)))
                else:
                    base[col] = max(0.0, float(feature_df[col].mean() * rng.uniform(0.25, 0.8)))
            base["source_density"] = base["attribute_count"] / max(1.0, base["unique_sources"])
            base["technical_risk_interaction"] = (
                base.get("early_technical_proficiency_count", 0.0) * base.get("risk_tolerance_count", 0.0)
            )
            base["ai_curiosity_interaction"] = (
                base.get("ai_specific_count", 0.0) * base.get("intellectual_curiosity_count", 0.0)
            )
            controls.append(base)
        return pd.DataFrame(controls)

    def train_xgboost_classifier(self, feature_df: pd.DataFrame) -> tuple[XGBClassifier, list[str]]:
        feature_cols = [
            c
            for c in feature_df.columns
            if c not in {"entrepreneur_id", "entrepreneur_name", "success_label"}
        ]
        x = feature_df[feature_cols].fillna(0.0)
        y = feature_df["success_label"].astype(int)
        if y.nunique() < 2:
            raise ValueError("Need both success and non-success labels for supervised training.")

        model = XGBClassifier(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.random_state,
        )
        model.fit(x, y)
        return model, feature_cols

    def _build_pattern_report(
        self,
        raw_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        cluster_df: pd.DataFrame,
        xgb_model: XGBClassifier,
        feature_cols: list[str],
        data_source: str,
    ) -> dict[str, Any]:
        importances = list(zip(feature_cols, xgb_model.feature_importances_.tolist()))
        if not importances or max(score for _, score in importances) <= 1e-12:
            # Tiny-cohort fallback: derive heuristic importances from success/control separation.
            success_df = feature_df[feature_df["success_label"] == 1].copy()
            control_df = feature_df[feature_df["success_label"] == 0].copy()
            derived: list[tuple[str, float]] = []
            for feat in feature_cols:
                s_avg = float(success_df[feat].mean()) if feat in success_df else 0.0
                c_avg = float(control_df[feat].mean()) if (not control_df.empty and feat in control_df) else 0.0
                s_std = float(feature_df[feat].std()) if feat in feature_df else 0.0
                denom = max(1e-6, s_std)
                derived.append((feat, abs((s_avg - c_avg) / denom)))
            importances = derived
        importances.sort(key=lambda x: x[1], reverse=True)
        top_features = importances[:10]
        success_df = feature_df[feature_df["success_label"] == 1].copy()
        control_df = feature_df[feature_df["success_label"] == 0].copy()

        findings: list[str] = []
        for feat, score in top_features[:6]:
            s_avg = float(success_df[feat].mean()) if feat in success_df else 0.0
            c_avg = float(control_df[feat].mean()) if (not control_df.empty and feat in control_df) else 0.0
            delta = s_avg - c_avg
            findings.append(
                f"{feat}: higher in exemplar founders by {delta:.2f} on average (importance {score:.3f})."
            )

        founder_company_values: list[dict[str, Any]] = []
        for founder in sorted(raw_df["entrepreneur_name"].astype(str).dropna().unique().tolist()):
            companies = self._companies_for_founder(founder)
            founder_company_values.append(
                {
                    "founder": founder,
                    "companies": companies,
                    "total_company_value_usd": self._total_company_value(companies),
                }
            )
        founder_value_map = {
            str(item.get("founder", "")): float(item.get("total_company_value_usd", 0.0))
            for item in founder_company_values
        }

        cluster_summary = []
        merged = cluster_df.merge(
            feature_df[["entrepreneur_id", "entrepreneur_name", "success_label"]],
            on=["entrepreneur_id", "entrepreneur_name"],
            how="left",
        )
        for cluster_id, frame in merged.groupby("cluster_id"):
            names = sorted(frame["entrepreneur_name"].astype(str).tolist())
            founder_values = [founder_value_map.get(name, 0.0) for name in names]
            total_cluster_value = float(sum(founder_values))
            avg_founder_value = total_cluster_value / max(1, len(founder_values))
            # Smooth 0-1 scaling using log market-value magnitude.
            # ~1B -> ~0.69, ~100B -> ~0.85, ~1T -> ~0.92.
            value_weighted_success_score = 0.0
            if avg_founder_value > 0:
                value_weighted_success_score = max(
                    0.0,
                    min(1.0, (math.log10(avg_founder_value) - 6.0) / 7.0),
                )
            cluster_summary.append(
                {
                    "cluster_id": int(cluster_id),
                    "size": int(len(frame)),
                    "founders": names,
                    "total_cluster_value_usd": total_cluster_value,
                    "avg_founder_value_usd": avg_founder_value,
                    "value_weighted_success_score": round(value_weighted_success_score, 6),
                }
            )

        methodology = [
            "Input data: pre-cutoff fact libraries for historical founders from DB or JSON logs.",
            "Feature construction: category counts, source density, timing, and interaction terms (technical x risk, AI x curiosity).",
            "Unsupervised step: SentenceTransformer embeddings + K-means to detect founder archetype clusters.",
            "Supervised step: XGBoost classifier on labeled success vs control founders to rank predictive signals.",
            "MVP strategy: with only 3 successful examples, synthetic control profiles are generated to allow supervised fitting; this is a bootstrap baseline until real negatives are added.",
        ]
        interview_questions = [
            "What high-conviction technical project did you build before external funding, and why did you choose it?",
            "Describe a major pre-funding risk you took that could have failed badly. Why was it worth it?",
            "Which early hardship most changed your operating behavior as a founder?",
            "How quickly do you make irreversible decisions under uncertainty, and what evidence do you require?",
            "What ideas in AI do you study deeply beyond your current product roadmap?",
            "Which mentors or peers materially changed your trajectory before founding?",
            "Tell me about a time you persisted through failure when quitting was rational.",
            "What contrarian belief do you hold about your market that most smart people disagree with?",
            "How do you balance speed versus quality when technical debt and customer pressure conflict?",
            "If you had zero capital for 18 months, what is your exact build-and-go-to-market plan?",
        ]
        return {
            "title": "Founder Signal AI Pattern Recognition Model",
            "goal": "Identify pre-funding founder traits that correlate with future ultra-success outcomes and use them in seed-stage diligence.",
            "data_source": data_source,
            "founder_count": int(raw_df["entrepreneur_id"].nunique()),
            "company_value_methodology": self._company_value_methodology(),
            "methodology": methodology,
            "top_signal_importance": [
                {"feature": feat, "importance": round(float(score), 6)}
                for feat, score in top_features
            ],
            "cluster_summary": cluster_summary,
            "founder_company_values": founder_company_values,
            "findings": findings,
            "interview_questions": interview_questions,
        }

    def save_artifacts(
        self,
        xgb_model: XGBClassifier,
        feature_columns: list[str],
        cluster_df: pd.DataFrame,
        summary: TrainingSummary,
        report: dict[str, Any],
    ) -> None:
        model_path = Path(summary.model_path)
        cluster_path = Path(summary.cluster_path)
        metadata_path = self.model_dir / "pattern_training_summary.json"
        report_path = Path(summary.report_path)

        joblib.dump({"model": xgb_model, "feature_columns": feature_columns}, model_path)
        cluster_df.to_csv(cluster_path, index=False)
        metadata_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def build_feature_matrix(self, claims: list[dict[str, Any]]) -> list[dict[str, float]]:
        row = {
            "early_technical_proficiency_count": 0.0,
            "risk_tolerance_count": 0.0,
            "ai_specific_count": 0.0,
            "security_insecurity_count": 0.0,
            "attribute_count": float(len(claims)),
        }
        for claim in claims:
            category = self._normalize_category(str(claim.get("category", "")))
            key = f"{category}_count"
            if key in row:
                row[key] += 1.0
        return [row]

    def train_model(self, feature_rows: list[dict[str, float]], labels: list[int]) -> dict[str, Any]:
        if not feature_rows or not labels:
            return {"model_name": "xgboost_classifier", "status": "skipped_no_data"}
        df = pd.DataFrame(feature_rows)
        df["success_label"] = labels[: len(df)]
        model, cols = self.train_xgboost_classifier(df)
        temp_path = self.model_dir / "xgboost_legacy.joblib"
        joblib.dump({"model": model, "feature_columns": cols}, temp_path)
        return {"model_name": "xgboost_classifier", "status": "trained", "path": str(temp_path)}

    def predict_founder_score(self, feature_row: dict[str, float]) -> float:
        model_path = self.model_dir / "xgboost_founder_success.joblib"
        if not model_path.exists():
            return 0.0
        obj = joblib.load(model_path)
        model: XGBClassifier = obj["model"]
        feature_columns: list[str] = obj["feature_columns"]
        x = pd.DataFrame([{col: float(feature_row.get(col, 0.0)) for col in feature_columns}])
        proba = model.predict_proba(x)[0][1]
        return float(round(proba, 6))

    def run_full_training(
        self,
        n_clusters: int = 3,
        data_source: str = "auto",
        founder_subset: list[str] | None = None,
    ) -> TrainingSummary:
        self.ensure_learning_tables()
        raw_df, used_source = self.load_historical_data(data_source=data_source, founder_subset=founder_subset)
        raw_df = self._apply_founder_subset(raw_df, founder_subset)
        if raw_df.empty:
            raise ValueError("No historical founder data available from DB/logs.")

        founder_count = raw_df["entrepreneur_id"].nunique()
        if founder_count < 3:
            raise ValueError("Need at least 3 historical founders for MVP training.")

        n_clusters = min(max(2, n_clusters), founder_count)
        embedding_df = self.build_text_embeddings(raw_df)
        cluster_df = self.run_kmeans_clustering(embedding_df, n_clusters=n_clusters)

        feature_df = self.aggregate_founder_features(raw_df).merge(
            cluster_df[["entrepreneur_id", "cluster_id"]], on="entrepreneur_id", how="left"
        )
        if feature_df["success_label"].nunique() < 2:
            controls = self._synthesize_control_profiles(feature_df, n_controls=max(4, founder_count))
            feature_df = pd.concat([feature_df, controls], ignore_index=True)

        model, feature_cols = self.train_xgboost_classifier(feature_df)
        report = self._build_pattern_report(raw_df, feature_df, cluster_df, model, feature_cols, used_source)

        summary = TrainingSummary(
            rows_used=len(feature_df),
            feature_count=len(feature_cols),
            model_name="xgboost_classifier",
            status="trained",
            n_clusters=n_clusters,
            model_path=str(self.model_dir / "xgboost_founder_success.joblib"),
            cluster_path=str(self.model_dir / "founder_clusters.csv"),
            report_path=str(self.model_dir / "founder_pattern_report.json"),
            data_source=f"{used_source}:{self._subset_label(founder_subset)}",
        )
        self.save_artifacts(model, feature_cols, cluster_df, summary, report)
        return summary

    def run_training(self, claims: list[dict[str, Any]], labels: list[int]) -> TrainingSummary:
        features = self.build_feature_matrix(claims)
        model_meta = self.train_model(features, labels)
        return TrainingSummary(
            rows_used=len(features),
            feature_count=len(features[0]) if features else 0,
            model_name=str(model_meta.get("model_name", "unknown")),
            status=str(model_meta.get("status", "unknown")),
            n_clusters=0,
            model_path=str(model_meta.get("path", "")),
            cluster_path="",
            report_path="",
            data_source="legacy",
        )

