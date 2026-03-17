from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion import DataIngestionService


def main() -> None:
    service = DataIngestionService(llm_model="gpt2", anomaly_log_path="data/ingestion_anomalies.log")

    # Sample biography-like excerpts for dry validation.
    source_texts = [
        {
            "source_url": "sample://musk_bio_source_1",
            "source_pub_date": "2015-05-19",
            "text": (
                "In 1983, Elon Musk sold a game called Blastar as a teenager. "
                "In 1989, he moved from South Africa to Canada. "
                "In 1992, he transferred to the University of Pennsylvania to study physics and economics. "
                "In 2002, he founded SpaceX."
            ),
        },
        {
            "source_url": "sample://musk_bio_source_2",
            "source_pub_date": "2023-09-12",
            "text": (
                "In 1983, Elon Musk sold a game called Blastar as a teenager. "
                "In 1989, he moved from South Africa to Canada to pursue opportunity. "
                "In 1992, he transferred to the University of Pennsylvania to study physics and economics. "
                "In 1999, Zip2 was acquired by Compaq."
            ),
        },
    ]

    result = service.process_text_sources(
        entrepreneur_name="Elon Musk",
        first_institutional_investment_date="1995-01-01",
        source_texts=source_texts,
    )

    output_path = Path("data/musk_pre_1995_validation_output.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote: {output_path}")
    print(f"Accepted: {result['accepted_count']}, Excluded: {result['excluded_count']}")


if __name__ == "__main__":
    main()

