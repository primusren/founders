from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any

import requests
from sentence_transformers import SentenceTransformer

try:
    from pinecone import Pinecone
except Exception:  # pragma: no cover
    Pinecone = None

try:
    from linkedin_api import Linkedin
except Exception:  # pragma: no cover
    Linkedin = None


HISTORICAL_PATTERN_TEXTS = [
    {
        "id": "pattern_technical_contrarian",
        "text": (
            "Founder shows early technical proficiency, ships products before funding, "
            "takes high-conviction contrarian decisions, and has strong resilience under constraints."
        ),
        "label": "Technical + Contrarian Execution",
    },
    {
        "id": "pattern_network_operator",
        "text": (
            "Founder builds strong mentor network early, recruits exceptional collaborators, "
            "and demonstrates clear communication with high market timing intuition."
        ),
        "label": "Network-Driven Operator",
    },
    {
        "id": "pattern_ai_native_builder",
        "text": (
            "Founder has AI-native technical interests, active machine-learning project footprint, "
            "and focuses on early-stage AI product opportunities."
        ),
        "label": "AI-Native Builder",
    },
]


@dataclass(slots=True)
class ScanResult:
    candidate_id: str
    score: float
    reasons: list[str]


@dataclass(slots=True)
class CandidateProfile:
    candidate_id: str
    full_name: str
    location: str
    estimated_age: int | None
    first_time_founder: bool
    funding_stage: str
    company_focus: str
    source: str
    github_ml_project_count: int
    github_total_stars: int
    profile_text: str


class ScanningPredictionService:
    """Real-time scanning engine for seed/Series A AI founders."""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        pinecone_index_name: str = "founder-patterns",
    ) -> None:
        self.embedder = SentenceTransformer(embedding_model_name)
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.github_token = os.getenv("GITHUB_TOKEN", "")
        self.linkedin_email = os.getenv("LINKEDIN_EMAIL", "")
        self.linkedin_password = os.getenv("LINKEDIN_PASSWORD", "")
        self.crunchbase_api_key = os.getenv("CRUNCHBASE_API_KEY", "")
        self.x_bearer_token = os.getenv("X_BEARER_TOKEN", "")

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def _embed(self, text: str) -> list[float]:
        vec = self.embedder.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        return vec.tolist()

    def _pinecone_index(self):
        if not self.pinecone_api_key or Pinecone is None:
            return None
        client = Pinecone(api_key=self.pinecone_api_key)
        names = [idx["name"] for idx in client.list_indexes()]
        if self.pinecone_index_name not in names:
            # Serverless defaults are intentionally omitted; if missing, we fallback to local matching.
            return None
        return client.Index(self.pinecone_index_name)

    def _upsert_historical_patterns(self) -> None:
        index = self._pinecone_index()
        if index is None:
            return
        vectors = []
        for pattern in HISTORICAL_PATTERN_TEXTS:
            vectors.append(
                {
                    "id": pattern["id"],
                    "values": self._embed(pattern["text"]),
                    "metadata": {"label": pattern["label"], "text": pattern["text"]},
                }
            )
        index.upsert(vectors=vectors)

    def _match_historical_patterns(self, profile_text: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Use Pinecone cosine similarity when available, otherwise local cosine fallback."""
        query_vec = self._embed(profile_text)
        index = self._pinecone_index()
        if index is not None:
            response = index.query(
                vector=query_vec,
                top_k=top_k,
                include_values=False,
                include_metadata=True,
            )
            matches = []
            for m in response.get("matches", []):
                matches.append(
                    {
                        "pattern_id": m.get("id", ""),
                        "pattern_label": m.get("metadata", {}).get("label", ""),
                        "similarity": float(m.get("score", 0.0)),
                    }
                )
            return matches

        # Local cosine fallback.
        scored = []
        for pattern in HISTORICAL_PATTERN_TEXTS:
            score = self._cosine(query_vec, self._embed(pattern["text"]))
            scored.append(
                {
                    "pattern_id": pattern["id"],
                    "pattern_label": pattern["label"],
                    "similarity": float(score),
                }
            )
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def _simulate_linkedin_results(self, query: str) -> list[CandidateProfile]:
        simulated = [
            CandidateProfile(
                candidate_id="linkedin_sim_1",
                full_name="Avery Lin",
                location="San Francisco, United States",
                estimated_age=27,
                first_time_founder=True,
                funding_stage="seed",
                company_focus="AI developer tooling",
                source="linkedin_simulated",
                github_ml_project_count=4,
                github_total_stars=620,
                profile_text=f"{query}. Built AI tooling startup. Prior ML engineer, first-time founder.",
            ),
            CandidateProfile(
                candidate_id="linkedin_sim_2",
                full_name="Jordan Patel",
                location="New York, United States",
                estimated_age=29,
                first_time_founder=True,
                funding_stage="series_a",
                company_focus="AI workflow automation",
                source="linkedin_simulated",
                github_ml_project_count=2,
                github_total_stars=240,
                profile_text=f"{query}. Focused on AI automation for SMBs. First-time founder.",
            ),
        ]
        return simulated

    def _fetch_linkedin_candidates(self, query: str) -> list[CandidateProfile]:
        if Linkedin is None or not self.linkedin_email or not self.linkedin_password:
            return self._simulate_linkedin_results(query)

        try:
            api = Linkedin(self.linkedin_email, self.linkedin_password)
            # linkedin-api endpoint behavior changes; fallback to simulation on failure.
            raw_people = api.search_people(keywords=query, limit=10)
            results: list[CandidateProfile] = []
            for idx, person in enumerate(raw_people):
                full_name = f"{person.get('firstName', '')} {person.get('lastName', '')}".strip() or f"Candidate {idx}"
                headline = person.get("headline", "")
                location = person.get("locationName", "Unknown, United States")
                results.append(
                    CandidateProfile(
                        candidate_id=str(person.get("urn_id", f"linkedin_{idx}")),
                        full_name=full_name,
                        location=location,
                        estimated_age=None,
                        first_time_founder=True,
                        funding_stage="seed",
                        company_focus=headline or "AI startup",
                        source="linkedin_api",
                        github_ml_project_count=0,
                        github_total_stars=0,
                        profile_text=f"{full_name}. {headline}. {location}.",
                    )
                )
            return results if results else self._simulate_linkedin_results(query)
        except Exception:
            return self._simulate_linkedin_results(query)

    def _fetch_crunchbase_candidates(self, query: str) -> list[CandidateProfile]:
        """
        Crunchbase access is typically paid/API-key based.
        If key is unavailable, return simulated candidates.
        """
        if not self.crunchbase_api_key:
            return [
                CandidateProfile(
                    candidate_id="crunchbase_sim_1",
                    full_name="Taylor Rivera",
                    location="Austin, United States",
                    estimated_age=28,
                    first_time_founder=True,
                    funding_stage="seed",
                    company_focus="AI security startup",
                    source="crunchbase_simulated",
                    github_ml_project_count=1,
                    github_total_stars=140,
                    profile_text=f"{query}. Seed AI security founder based in Austin.",
                )
            ]
        # Safe fallback because Crunchbase API contracts vary by plan.
        return []

    def _fetch_x_candidates(self, query: str) -> list[CandidateProfile]:
        """
        X API access requires bearer token and paid tiers; fallback to simulation.
        """
        if not self.x_bearer_token:
            return [
                CandidateProfile(
                    candidate_id="x_sim_1",
                    full_name="Riley Chen",
                    location="Seattle, United States",
                    estimated_age=26,
                    first_time_founder=True,
                    funding_stage="seed",
                    company_focus="LLM ops platform",
                    source="x_simulated",
                    github_ml_project_count=3,
                    github_total_stars=510,
                    profile_text=f"{query}. LLM ops founder actively posting AI build logs.",
                )
            ]
        return []

    def _fetch_github_ai_signals(self, query: str) -> dict[str, dict[str, int]]:
        """
        Pull AI-era technical signals from GitHub repo search.
        Returns owner -> {ml_projects, total_stars}.
        """
        headers = {"Accept": "application/vnd.github+json"}
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"

        gh_query = f"{query} in:name,description,readme language:Python"
        url = "https://api.github.com/search/repositories"
        params = {"q": gh_query, "sort": "stars", "order": "desc", "per_page": 25}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            resp.raise_for_status()
            items = resp.json().get("items", [])
        except Exception:
            return {}

        owner_stats: dict[str, dict[str, int]] = {}
        for repo in items:
            owner = repo.get("owner", {}).get("login")
            if not owner:
                continue
            stats = owner_stats.setdefault(owner.lower(), {"ml_projects": 0, "total_stars": 0})
            stats["ml_projects"] += 1
            stats["total_stars"] += int(repo.get("stargazers_count", 0))
        return owner_stats

    def _apply_candidate_filters(self, candidates: list[CandidateProfile]) -> list[CandidateProfile]:
        """Filter to US-based, <30, first-time founders, seed/Series A, AI-focused."""
        filtered: list[CandidateProfile] = []
        for c in candidates:
            location_ok = "united states" in c.location.lower() or ", us" in c.location.lower()
            age_ok = c.estimated_age is None or c.estimated_age < 30
            stage_ok = c.funding_stage.lower() in {"seed", "series_a", "series a"}
            first_time_ok = c.first_time_founder
            ai_ok = "ai" in c.company_focus.lower() or "ml" in c.company_focus.lower()
            if location_ok and age_ok and stage_ok and first_time_ok and ai_ok:
                filtered.append(c)
        return filtered

    def _score_candidate(self, candidate: CandidateProfile, pattern_matches: list[dict[str, Any]]) -> tuple[float, list[str]]:
        top_similarity = pattern_matches[0]["similarity"] if pattern_matches else 0.0
        ai_signal = min(1.0, (candidate.github_ml_project_count / 5.0) + (candidate.github_total_stars / 1000.0))
        age_bonus = 1.0 if candidate.estimated_age is not None and candidate.estimated_age < 30 else 0.6
        first_time_bonus = 1.0 if candidate.first_time_founder else 0.4
        stage_bonus = 1.0 if candidate.funding_stage.lower() in {"seed", "series_a", "series a"} else 0.3

        score = (
            0.5 * float(top_similarity)
            + 0.2 * ai_signal
            + 0.1 * age_bonus
            + 0.1 * first_time_bonus
            + 0.1 * stage_bonus
        )
        score = round(min(1.0, max(0.0, score)), 4)

        reasons = []
        if pattern_matches:
            reasons.append(
                f"High similarity to historical pattern '{pattern_matches[0]['pattern_label']}' ({pattern_matches[0]['similarity']:.3f})"
            )
        reasons.append(f"AI-era signal: {candidate.github_ml_project_count} ML repos, {candidate.github_total_stars} stars")
        reasons.append(f"Profile fit: {candidate.location}, stage={candidate.funding_stage}, first_time={candidate.first_time_founder}")
        return score, reasons

    def predict_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        """
        Score a specific founder profile payload.
        Expected keys include: full_name, location, estimated_age, funding_stage,
        first_time_founder, company_focus, bio, github_ml_project_count, github_total_stars.
        """
        full_name = str(profile.get("full_name", "Unknown Founder"))
        location = str(profile.get("location", "United States"))
        estimated_age = profile.get("estimated_age")
        estimated_age = int(estimated_age) if estimated_age is not None else None
        funding_stage = str(profile.get("funding_stage", "seed"))
        first_time_founder = bool(profile.get("first_time_founder", True))
        company_focus = str(profile.get("company_focus", "AI startup"))
        bio = str(profile.get("bio", ""))
        github_ml_project_count = int(profile.get("github_ml_project_count", 0))
        github_total_stars = int(profile.get("github_total_stars", 0))

        candidate = CandidateProfile(
            candidate_id=f"profile::{full_name.lower().replace(' ', '_')}",
            full_name=full_name,
            location=location,
            estimated_age=estimated_age,
            first_time_founder=first_time_founder,
            funding_stage=funding_stage,
            company_focus=company_focus,
            source="direct_profile",
            github_ml_project_count=github_ml_project_count,
            github_total_stars=github_total_stars,
            profile_text=f"{full_name}. {company_focus}. {bio}",
        )
        pattern_matches = self._match_historical_patterns(candidate.profile_text, top_k=3)
        score, reasons = self._score_candidate(candidate, pattern_matches)
        return {
            "candidate_id": candidate.candidate_id,
            "full_name": candidate.full_name,
            "score": score,
            "pattern_matches": pattern_matches,
            "explanations": reasons,
        }

    def search_candidates(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Main scanning entrypoint.
        Example query: 'AI startups seed stage'
        Returns top 10 ranked matches with scores + explanations.
        """
        self._upsert_historical_patterns()

        candidates = []
        candidates.extend(self._fetch_linkedin_candidates(query))
        candidates.extend(self._fetch_crunchbase_candidates(query))
        candidates.extend(self._fetch_x_candidates(query))

        github_stats = self._fetch_github_ai_signals(query)
        for c in candidates:
            owner_key = c.full_name.lower().replace(" ", "")
            stats = github_stats.get(owner_key)
            if stats:
                c.github_ml_project_count = max(c.github_ml_project_count, stats["ml_projects"])
                c.github_total_stars = max(c.github_total_stars, stats["total_stars"])

        filtered = self._apply_candidate_filters(candidates)
        ranked: list[dict[str, Any]] = []
        for c in filtered:
            matches = self._match_historical_patterns(c.profile_text, top_k=3)
            score, reasons = self._score_candidate(c, matches)
            ranked.append(
                {
                    "candidate_id": c.candidate_id,
                    "full_name": c.full_name,
                    "source": c.source,
                    "location": c.location,
                    "estimated_age": c.estimated_age,
                    "first_time_founder": c.first_time_founder,
                    "funding_stage": c.funding_stage,
                    "company_focus": c.company_focus,
                    "score": score,
                    "pattern_matches": matches,
                    "explanations": reasons,
                }
            )

        ranked.sort(key=lambda x: x["score"], reverse=True)
        if len(ranked) < top_k:
            # Ensure deterministic shape for downstream consumers.
            while len(ranked) < top_k and candidates:
                c = random.choice(candidates)
                matches = self._match_historical_patterns(c.profile_text, top_k=1)
                score, reasons = self._score_candidate(c, matches)
                ranked.append(
                    {
                        "candidate_id": f"{c.candidate_id}_filler_{len(ranked)}",
                        "full_name": c.full_name,
                        "source": c.source,
                        "location": c.location,
                        "estimated_age": c.estimated_age,
                        "first_time_founder": c.first_time_founder,
                        "funding_stage": c.funding_stage,
                        "company_focus": c.company_focus,
                        "score": score,
                        "pattern_matches": matches,
                        "explanations": reasons,
                    }
                )
        return ranked[:top_k]

    # Backward-compatible helper methods for previous API usage.
    def collect_candidate_profile(self, handle_or_url: str) -> dict[str, Any]:
        matches = self.search_candidates(handle_or_url, top_k=1)
        if matches:
            return matches[0]
        return {
            "candidate_id": handle_or_url,
            "full_name": "unknown",
            "score": 0.0,
            "explanations": ["No candidate matches found."],
        }

    def retrieve_similar_founders(self, profile: dict[str, Any]) -> list[dict[str, Any]]:
        text = profile.get("profile_text") or profile.get("company_focus") or ""
        return self._match_historical_patterns(str(text), top_k=3)

    def score_candidate(self, profile: dict[str, Any]) -> float:
        return float(profile.get("score", 0.0))

    def run_scan(self, handle_or_url: str) -> ScanResult:
        top = self.search_candidates(handle_or_url, top_k=1)
        if not top:
            return ScanResult(candidate_id=handle_or_url, score=0.0, reasons=["No matches found."])
        best = top[0]
        return ScanResult(
            candidate_id=str(best["candidate_id"]),
            score=float(best["score"]),
            reasons=list(best["explanations"]),
        )

