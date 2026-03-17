from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion import DataIngestionService


@pytest.fixture()
def ingestion_service(tmp_path: Path) -> DataIngestionService:
    service = DataIngestionService(
        llm_model="gpt2",
        anomaly_log_path=str(tmp_path / "ingestion_anomalies.log"),
    )
    # Make tests deterministic and fast: skip generative rewrite noise.
    service._llm_attribute_hint = lambda sentence: sentence  # type: ignore[attr-defined]
    return service


def _sample_elon_sources() -> list[dict[str, str]]:
    """
    Two-source sample with:
    - repeated pre-1995 facts (should pass, 2-source corroborated),
    - repeated post-cutoff fact (should be discarded),
    - repeated no-date fact (should be discarded and logged).
    """
    source_1 = {
        "source_url": "sample://elon_bio_source_1",
        "source_pub_date": "2015-05-19",
        "text": (
            "In 1989, Elon Musk moved from South Africa to Canada after childhood hardship. "
            "In 1992, he transferred to the University of Pennsylvania to study physics and economics. "
            "In 1998, he decided to focus on scaling Zip2 after funding expansion. "
            "His personality was intense and obsessive during early startup work."
        ),
    }
    source_2 = {
        "source_url": "sample://elon_bio_source_2",
        "source_pub_date": "2023-09-12",
        "text": (
            "In 1989, Elon Musk moved from South Africa to Canada after childhood hardship. "
            "In 1992, he transferred to the University of Pennsylvania to study physics and economics. "
            "In 1998, he decided to focus on scaling Zip2 after funding expansion. "
            "His personality was intense and obsessive during early startup work."
        ),
    }
    return [source_1, source_2]


def test_elon_pre_1995_cutoff_filters_post_cutoff_and_unanchored_facts(
    ingestion_service: DataIngestionService,
) -> None:
    result = ingestion_service.process_text_sources(
        entrepreneur_name="Elon Musk",
        first_institutional_investment_date="1995-01-01",
        source_texts=_sample_elon_sources(),
    )

    accepted = result["accepted_claims"]
    excluded = result["excluded_claims"]

    # We expect two pre-1995 facts, each corroborated by both sources.
    # The pipeline keeps source-level records, so count is 4.
    assert result["accepted_count"] == 4
    assert result["excluded_count"] >= 2

    for claim in accepted:
        assert claim["before_cutoff"] is True
        assert claim["verification_source_count"] >= 2
        assert claim["event_date"] <= "1995-01-01"

    discard_reasons = {claim.get("discard_reason", "") for claim in excluded}
    assert "post_cutoff_event" in discard_reasons
    assert "no_parseable_event_date" in discard_reasons


def test_elon_cutoff_validation_outputs_structured_json(
    ingestion_service: DataIngestionService,
) -> None:
    payload = ingestion_service.process_text_sources(
        entrepreneur_name="Elon Musk",
        first_institutional_investment_date="1995-01-01",
        source_texts=_sample_elon_sources(),
    )
    raw = json.dumps(payload, indent=2)
    parsed = json.loads(raw)

    assert parsed["entrepreneur_name"] == "Elon Musk"
    assert parsed["cutoff_date"] == "1995-01-01"
    assert isinstance(parsed["accepted_claims"], list)
    assert isinstance(parsed["excluded_claims"], list)

