from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.learning import PatternLearningService


def ensure_base_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entrepreneurs (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                first_institutional_investment_date DATE NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS attributes (
                id SERIAL PRIMARY KEY,
                entrepreneur_id INTEGER NOT NULL REFERENCES entrepreneurs(id) ON DELETE CASCADE,
                source_url TEXT NOT NULL,
                category TEXT NOT NULL,
                attribute_text TEXT NOT NULL,
                event_date DATE,
                source_pub_date DATE,
                before_cutoff BOOLEAN NOT NULL,
                extraction_method TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entrepreneur_labels (
                entrepreneur_id INTEGER PRIMARY KEY REFERENCES entrepreneurs(id) ON DELETE CASCADE,
                success_label INTEGER NOT NULL CHECK (success_label IN (0, 1)),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
            """
        )


def seed_sample_historical_data(conn) -> None:
    """
    Seed 8 historical entrepreneurs (5-10 target) with pre-cutoff attributes.
    """
    founders = [
        ("Elon Musk", "1996-01-01", 1),
        ("Jeff Bezos", "1995-07-01", 1),
        ("Larry Page", "1998-08-01", 1),
        ("Sergey Brin", "1998-08-01", 1),
        ("Mark Zuckerberg", "2004-09-01", 1),
        ("Founder Control One", "2008-01-01", 0),
        ("Founder Control Two", "2010-01-01", 0),
        ("Founder Control Three", "2012-01-01", 0),
    ]

    category_templates = [
        ("security_insecurity", "Early insecurity drove intense achievement behavior."),
        ("early_technical_proficiency", "Built technical projects early with self-taught coding."),
        ("risk_tolerance", "Made high-risk career moves before institutional support."),
        ("ai_specific", "Explored machine-learning ideas before mainstream adoption."),
        ("decision_making", "Demonstrated fast decision cycles under uncertainty."),
        ("network_mentorship", "Built mentorship links and leveraged network effects."),
        ("resilience", "Recovered from setbacks and continued execution."),
        ("intellectual_curiosity", "Read deeply across physics, economics, and computing."),
    ]

    with conn.cursor() as cur:
        for idx, (name, cutoff, label) in enumerate(founders):
            cur.execute(
                """
                INSERT INTO entrepreneurs (name, first_institutional_investment_date)
                VALUES (%s, %s)
                ON CONFLICT (name)
                DO UPDATE SET first_institutional_investment_date = EXCLUDED.first_institutional_investment_date
                RETURNING id;
                """,
                (name, cutoff),
            )
            entrepreneur_id = int(cur.fetchone()[0])

            cur.execute(
                """
                INSERT INTO entrepreneur_labels (entrepreneur_id, success_label, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (entrepreneur_id)
                DO UPDATE SET success_label = EXCLUDED.success_label, updated_at = NOW();
                """,
                (entrepreneur_id, label),
            )

            for offset, (category, template) in enumerate(category_templates):
                if label == 0 and category in {"ai_specific", "early_technical_proficiency"} and offset % 2 == 0:
                    # Make control profiles weaker on deep-tech/AI traits.
                    text = "Limited evidence of early technical depth or AI-focused exploration."
                else:
                    text = template
                year = 1985 + ((idx + offset) % 10)
                cur.execute(
                    """
                    INSERT INTO attributes (
                        entrepreneur_id, source_url, category, attribute_text, event_date,
                        source_pub_date, before_cutoff, extraction_method
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        entrepreneur_id,
                        f"sample://historical/{name.replace(' ', '_').lower()}",
                        category,
                        text,
                        f"{year}-01-01",
                        "2020-01-01",
                        True,
                        "seed_script",
                    ),
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pattern learning layer from DB historical data.")
    parser.add_argument(
        "--db-dsn",
        default=os.getenv("DATABASE_URL", ""),
        help="PostgreSQL DSN, e.g. postgresql://user:pass@localhost:5432/dbname",
    )
    parser.add_argument(
        "--seed-sample",
        action="store_true",
        help="Seed sample data for 5-10 historical entrepreneurs before training.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "db", "logs"],
        default="auto",
        help="Training data source: auto (prefer DB, fallback logs), db only, or logs only.",
    )
    parser.add_argument("--clusters", type=int, default=3, help="Target number of K-means clusters.")
    args = parser.parse_args()

    if args.seed_sample:
        if not args.db_dsn:
            raise ValueError("--seed-sample requires --db-dsn or DATABASE_URL.")
        with psycopg2.connect(args.db_dsn) as conn:
            ensure_base_tables(conn)
            seed_sample_historical_data(conn)
    elif args.db_dsn and args.source in {"auto", "db"}:
        with psycopg2.connect(args.db_dsn) as conn:
            ensure_base_tables(conn)

    service = PatternLearningService(db_dsn=args.db_dsn or None)
    summary = service.run_full_training(n_clusters=args.clusters, data_source=args.source)

    print("Training completed.")
    print(f"Rows used: {summary.rows_used}")
    print(f"Feature count: {summary.feature_count}")
    print(f"Clusters: {summary.n_clusters}")
    print(f"Data source: {summary.data_source}")
    print(f"Model path: {summary.model_path}")
    print(f"Cluster path: {summary.cluster_path}")
    print(f"Pattern report path: {summary.report_path}")


if __name__ == "__main__":
    main()

