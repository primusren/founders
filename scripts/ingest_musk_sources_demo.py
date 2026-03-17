from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion import DataIngestionService


def main() -> None:
    db_dsn = os.getenv("DATABASE_URL", "")
    service = DataIngestionService(db_dsn=db_dsn or None)
    # Keep demo deterministic and compact.
    service._llm_attribute_hint = lambda s: s  # type: ignore[attr-defined]

    source_texts = [
        {
            "source_url": "sample://ashlee_vance_biography",
            "source": "Ashlee Vance Biography",
            "source_pub_date": "2015-05-19",
            "text": (
                "In 1989, Elon Musk moved from South Africa to Canada. "
                "In 1992, he transferred to the University of Pennsylvania to study physics and economics. "
                "In 2002, he founded SpaceX."
            ),
        },
        {
            "source_url": "https://en.wikipedia.org/wiki/Elon_Musk",
            "source": "https://en.wikipedia.org/wiki/Elon_Musk",
            "source_pub_date": "2026-03-14",
            "text": (
                "In 1989, Elon Musk moved from South Africa to Canada. "
                "In 1992, he transferred to the University of Pennsylvania to study physics and economics. "
                "In 2002, he founded SpaceX."
            ),
        },
    ]

    result = service.process_text_sources(
        entrepreneur_name="Elon Musk",
        first_institutional_investment_date="1995-01-01",
        source_texts=source_texts,
    )

    # Optionally persist demo output into DB for dashboard inspection.
    if db_dsn:
        service.persist_claims(
            entrepreneur_name="Elon Musk",
            cutoff_date=date.fromisoformat("1995-01-01"),
            valid_claims=result["accepted_claims"],
            invalid_claims=result["excluded_claims"],
        )

    out_path = ROOT / "data" / "musk_sources_demo_output.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Accepted={result['accepted_count']} Excluded={result['excluded_count']}")


if __name__ == "__main__":
    main()

