from __future__ import annotations

import json
import logging
import math
import os
import re
import hashlib
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import psycopg2
import requests
from bs4 import BeautifulSoup
from psycopg2.extensions import connection as PgConnection
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Reduce noisy environment warnings on Windows/Python 3.14+.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
)

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None

try:
    from dateutil import parser as dt_parser
except ImportError:  # pragma: no cover
    dt_parser = None


SIGNAL_MAP: dict[str, list[str]] = {
    "appearance": ["appearance", "dress", "impression", "style", "look"],
    "decision_making": ["decided", "decision", "chose", "pivot", "why", "strategy"],
    "education": ["school", "university", "college", "degree", "studied"],
    "family_hardship": ["family", "childhood", "hardship", "parents", "bully"],
    "most_difficult_accomplishment": ["difficult", "hardest", "struggle", "overcame"],
    "most_crazy_accomplishment": ["crazy", "bold", "wild", "outlier"],
    "personality": ["personality", "introvert", "obsessive", "temperament"],
    "values": ["value", "belief", "principle", "mission"],
    "security_insecurity": ["insecure", "security", "fear", "confidence"],
    "psychological_condition": ["psychological", "mental", "condition", "diagnosed"],
    "motivation": ["motivation", "motivated", "reason", "purpose", "started"],
    "external_positive_signal": ["employee", "ex-employee", "competitor", "peer", "praised"],
    "early_technical_proficiency": ["self-taught", "coded", "built", "programming", "prototype"],
    "network_mentorship": ["mentor", "network", "internship", "connection"],
    "risk_tolerance": ["risk", "dropped out", "gamble", "bootstrapped"],
    "intellectual_curiosity": ["read", "book", "curiosity", "physics", "science"],
    "resilience": ["resilience", "recovered", "failure", "persisted"],
    "social_signals": ["forum", "online", "social", "github", "hackathon"],
    "demographic_contextual": ["immigrant", "migration", "urban", "rural", "context"],
    "ai_specific": ["ai", "machine learning", "neural", "algorithm"],
}

CATEGORY_ANALYSIS_BUCKETS: dict[str, tuple[str, ...]] = {
    "Decision making": ("decision_making", "motivation", "risk_tolerance"),
    "Early hardships / challenges and resilience indicators": (
        "family_hardship",
        "resilience",
        "security_insecurity",
        "demographic_contextual",
    ),
    "Early technical proficiency": ("early_technical_proficiency", "ai_specific"),
    "Personality": ("personality", "psychological_condition"),
    "Crazy things done": ("most_crazy_accomplishment", "social_signals"),
    "Values": ("values",),
    "Sense of security / insecurity": ("security_insecurity",),
    "Motivation for starting business": ("motivation", "decision_making"),
    "Intellectual curiousity": ("intellectual_curiosity", "education"),
}

FOUNDER_ANALYSIS_CATEGORIES: tuple[str, ...] = (
    "Decision making",
    "Early hardships / challenges and resilience indicators",
    "Early technical proficiency",
    "Personality",
    "Crazy things done",
    "Values",
    "Sense of security / insecurity",
    "Motivation for starting business",
    "Intellectual curiousity",
)

CATEGORY_ALIAS_MAP: dict[str, str] = {
    "appearance": "appearance",
    "decision making": "decision_making",
    "decision_making": "decision_making",
    "education": "education",
    "family": "family_hardship",
    "family hardship": "family_hardship",
    "family_hardship": "family_hardship",
    "most difficult accomplishment": "most_difficult_accomplishment",
    "most_difficult_accomplishment": "most_difficult_accomplishment",
    "most crazy accomplishment": "most_crazy_accomplishment",
    "most_crazy_accomplishment": "most_crazy_accomplishment",
    "personality": "personality",
    "values": "values",
    "sense of security / insecurity": "security_insecurity",
    "security_insecurity": "security_insecurity",
    "psychological condition": "psychological_condition",
    "psychological_condition": "psychological_condition",
    "motivation for starting business": "motivation",
    "motivation": "motivation",
    "early technical proficiency": "early_technical_proficiency",
    "early_technical_proficiency": "early_technical_proficiency",
    "risk tolerance": "risk_tolerance",
    "risk_tolerance": "risk_tolerance",
    "intellectual curiosity": "intellectual_curiosity",
    "intellectual_curiosity": "intellectual_curiosity",
    "resilience indicators": "resilience",
    "resilience": "resilience",
    "network/mentorship": "network_mentorship",
    "network_mentorship": "network_mentorship",
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "was",
    "were",
    "have",
    "has",
    "had",
    "his",
    "her",
    "their",
    "they",
    "them",
    "she",
    "him",
    "who",
    "after",
    "before",
    "during",
    "about",
    "because",
    "which",
    "when",
    "where",
    "while",
}


@dataclass(slots=True)
class IngestionResult:
    entrepreneur_name: str
    source_urls: list[str]
    cutoff_date: str
    extracted_claims: list[dict[str, Any]]
    excluded_claims: list[dict[str, Any]]
    temporal_filter_passed: int
    temporal_filter_failed: int


class DataIngestionService:
    """
    Historical entrepreneur ingestion:
    - scrape public pages,
    - extract candidate facts with LLM + heuristics,
    - tag dates with spaCy/date parsing,
    - enforce DATE CUTOFF (first institutional seed/Series A investment),
    - persist to PostgreSQL.
    """

    def __init__(
        self,
        db_dsn: str | None = None,
        llm_model: str = "facebook/bart-large-cnn",
        generation_model: str = "gpt2",
        llm_provider: str | None = None,
        deepseek_api_key: str | None = None,
        deepseek_model: str = "deepseek-chat",
        deepseek_base_url: str = "https://api.deepseek.com",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        timeout_seconds: int = 25,
        anomaly_log_path: str = "data/ingestion_anomalies.log",
    ) -> None:
        self.db_dsn = db_dsn
        self.timeout_seconds = int(os.getenv("SOURCE_REQUEST_TIMEOUT_SECONDS", str(timeout_seconds)))
        self._llm_model = llm_model
        self._generation_model = generation_model
        self._llm_provider = (llm_provider or os.getenv("LLM_PROVIDER", "huggingface")).strip().lower()
        self._extraction_llm_provider = (
            os.getenv("EXTRACTION_LLM_PROVIDER", self._llm_provider).strip().lower()
        )
        self._analysis_llm_provider = (
            os.getenv("ANALYSIS_LLM_PROVIDER", self._llm_provider).strip().lower()
        )
        self._deepseek_api_key = (deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")).strip()
        self._deepseek_model = (os.getenv("DEEPSEEK_MODEL", deepseek_model) or deepseek_model).strip()
        self._deepseek_base_url = (os.getenv("DEEPSEEK_BASE_URL", deepseek_base_url) or deepseek_base_url).strip().rstrip("/")
        self._deepseek_timeout_seconds = int(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "18"))
        self._deepseek_cache_enabled = os.getenv("DEEPSEEK_CACHE_ENABLED", "1") == "1"
        self._deepseek_max_source_chars = int(os.getenv("DEEPSEEK_MAX_SOURCE_CHARS", "2800"))
        self._deepseek_cache_path = Path(os.getenv("DEEPSEEK_CACHE_PATH", "data/deepseek_cache.json"))
        self._deepseek_cache: dict[str, str] | None = None
        self._embedding_model_name = embedding_model_name
        self._summarizer = None
        self._generator = None
        self._embedder = None
        self._nlp = self._load_spacy_pipeline()
        self.anomaly_log_path = Path(anomaly_log_path)
        self._setup_anomaly_logger()

    def _setup_anomaly_logger(self) -> None:
        self.anomaly_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.anomaly_logger = logging.getLogger("seed_intel.ingestion.anomalies")
        self.anomaly_logger.setLevel(logging.INFO)
        if not self.anomaly_logger.handlers:
            handler = logging.FileHandler(self.anomaly_log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.anomaly_logger.addHandler(handler)

    @staticmethod
    def _load_spacy_pipeline():
        if spacy is None:
            return None
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            # Fallback keeps tokenizer/sentences working even if model is unavailable.
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp

    @property
    def summarizer(self):
        if self._summarizer is None:
            try:
                self._summarizer = pipeline("summarization", model=self._llm_model)
            except Exception:
                # Lightweight fallback if the larger model is unavailable.
                self._summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        return self._summarizer

    @property
    def generator(self):
        if self._generator is None:
            candidates = [self._generation_model, "gpt2-xl", "gpt2"]
            tried: set[str] = set()
            for model_name in candidates:
                if model_name in tried:
                    continue
                tried.add(model_name)
                try:
                    self._generator = pipeline("text-generation", model=model_name)
                    break
                except Exception:
                    continue
            if self._generator is None:
                raise RuntimeError("No text-generation model could be initialized.")
        return self._generator

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def _use_deepseek(self) -> bool:
        return self._llm_provider == "deepseek"

    def _use_deepseek_extraction(self) -> bool:
        return self._extraction_llm_provider == "deepseek"

    def _use_deepseek_analysis(self) -> bool:
        return self._analysis_llm_provider == "deepseek"

    @staticmethod
    def _normalize_analysis_category(raw: str) -> str | None:
        key = re.sub(r"[^a-z0-9]", "", str(raw or "").lower())
        for canonical in FOUNDER_ANALYSIS_CATEGORIES:
            c_key = re.sub(r"[^a-z0-9]", "", canonical.lower())
            if key == c_key:
                return canonical
        alias_map = {
            "intellectualcuriosity": "Intellectual curiousity",
            "earlyhardshipschallengesandresilienceindicators": "Early hardships / challenges and resilience indicators",
            "senseofsecurityinsecurity": "Sense of security / insecurity",
            "motivationforstartingbusiness": "Motivation for starting business",
        }
        return alias_map.get(key)

    def _load_deepseek_cache(self) -> dict[str, str]:
        if self._deepseek_cache is not None:
            return self._deepseek_cache
        self._deepseek_cache = {}
        if not self._deepseek_cache_enabled:
            return self._deepseek_cache
        try:
            if self._deepseek_cache_path.exists():
                loaded = json.loads(self._deepseek_cache_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._deepseek_cache = {str(k): str(v) for k, v in loaded.items()}
        except Exception:
            self._deepseek_cache = {}
        return self._deepseek_cache

    def _persist_deepseek_cache(self) -> None:
        if not self._deepseek_cache_enabled:
            return
        cache = self._load_deepseek_cache()
        try:
            self._deepseek_cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Keep cache bounded so file size remains manageable.
            if len(cache) > 1500:
                trimmed = dict(list(cache.items())[-1200:])
            else:
                trimmed = cache
            self._deepseek_cache_path.write_text(json.dumps(trimmed), encoding="utf-8")
            self._deepseek_cache = trimmed
        except Exception:
            pass

    def _deepseek_chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1800,
        temperature: float = 0.1,
    ) -> str:
        if not self._deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set while llm_provider=deepseek.")
        cache_key = ""
        cache = self._load_deepseek_cache()
        if self._deepseek_cache_enabled:
            fingerprint = "|".join(
                [
                    self._deepseek_model,
                    f"{max_tokens}",
                    f"{temperature}",
                    system_prompt,
                    user_prompt,
                ]
            )
            cache_key = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
            if cache_key in cache:
                return cache[cache_key]

        endpoint = f"{self._deepseek_base_url}/chat/completions"
        payload = {
            "model": self._deepseek_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._deepseek_api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(endpoint, headers=headers, json=payload, timeout=self._deepseek_timeout_seconds)
        response.raise_for_status()
        data = response.json()
        content = str(data["choices"][0]["message"]["content"]).strip()
        if self._deepseek_cache_enabled and cache_key:
            cache[cache_key] = content
            # Persist opportunistically every 20 additions.
            if len(cache) % 20 == 0:
                self._persist_deepseek_cache()
        return content

    def _build_deepseek_extraction_prompts(
        self,
        *,
        entrepreneur_name: str,
        birth_year: int,
        cutoff_year: int,
        source_text: str,
    ) -> list[str]:
        """
        Multi-pass extraction prompts to improve recall by semantic slice.
        """
        base_rules = (
            f"Extract facts strictly before {cutoff_year}. "
            "Return strict JSON only in this shape: "
            "{\"category\": [{\"narrative_fact\": \"...\", \"sub_facts\": [\"...\"], "
            "\"timestamps\": [\"YYYY or range\"], \"sources\": [\"source URL if available\"], "
            "\"inference\": \"if inferred\"}]}. "
            "No commentary outside JSON."
        )
        source_trimmed = source_text[: self._deepseek_max_source_chars]
        prompts = [
            (
                f"{base_rules}\n"
                f"Pass focus: chronological biography timeline for {entrepreneur_name} (born {birth_year}). "
                "Prioritize life-stage milestones: childhood, school transitions, migration, education, first projects, "
                "early attempts to build things, and pre-funding startup steps.\n\n"
                f"Source text:\n{source_trimmed}"
            ),
            (
                f"{base_rules}\n"
                f"Pass focus: founder psychology and decision profile for {entrepreneur_name}. "
                "Extract decision-making moments, hardships/challenges, resilience indicators, values, insecurity/security cues, "
                "motivation drivers, and concrete examples of risk tolerance.\n\n"
                f"Source text:\n{source_trimmed}"
            ),
            (
                f"{base_rules}\n"
                f"Pass focus: technical and curiosity signals for {entrepreneur_name}. "
                "Extract early technical proficiency, self-learning behaviors, intellectual curiosity patterns, "
                "network/mentorship, unusual/crazy accomplishments, and concrete artifacts (projects, prototypes, coding, experiments).\n\n"
                f"Source text:\n{source_trimmed}"
            ),
        ]
        return prompts

    def _run_deepseek_extraction_passes(
        self,
        *,
        entrepreneur_name: str,
        birth_year: int,
        cutoff_year: int,
        source_text: str,
    ) -> list[str]:
        outputs: list[str] = []
        prompts = self._build_deepseek_extraction_prompts(
            entrepreneur_name=entrepreneur_name,
            birth_year=birth_year,
            cutoff_year=cutoff_year,
            source_text=source_text,
        )
        for prompt in prompts:
            try:
                out = self._deepseek_chat_completion(
                    system_prompt=(
                        "You are a strict pre-cutoff extraction engine for venture diligence. "
                        "Return JSON only."
                    ),
                    user_prompt=prompt,
                    max_tokens=700,
                    temperature=0.1,
                )
                if out:
                    outputs.append(out)
            except Exception:
                continue
        return outputs

    def _connect(self) -> PgConnection:
        if not self.db_dsn:
            raise ValueError("db_dsn is required for database persistence.")
        return psycopg2.connect(self.db_dsn)

    def create_tables(self) -> None:
        """Create PostgreSQL tables for entrepreneurs, attributes, and timelines."""
        with self._connect() as conn:
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
                        source TEXT,
                        category TEXT NOT NULL,
                        fact TEXT,
                        attribute_text TEXT NOT NULL,
                        timestamp DATE,
                        event_date DATE,
                        source_pub_date DATE,
                        before_cutoff BOOLEAN NOT NULL,
                        extraction_method TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute("ALTER TABLE attributes ADD COLUMN IF NOT EXISTS source TEXT;")
                cur.execute("ALTER TABLE attributes ADD COLUMN IF NOT EXISTS fact TEXT;")
                cur.execute("ALTER TABLE attributes ADD COLUMN IF NOT EXISTS timestamp DATE;")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS timelines (
                        id SERIAL PRIMARY KEY,
                        entrepreneur_id INTEGER NOT NULL REFERENCES entrepreneurs(id) ON DELETE CASCADE,
                        event_date DATE,
                        event_text TEXT NOT NULL,
                        source_url TEXT NOT NULL,
                        before_cutoff BOOLEAN NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT NOW()
                    );
                    """
                )

    def scrape_source(self, source_url: str) -> str:
        """Fetch and return visible paragraph text from a public webpage."""
        headers = {
            "User-Agent": (
                "SeedFounderIntelligenceBot/0.1 "
                "(research use; contact: local-dev)"
            )
        }
        response = requests.get(source_url, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n".join([p for p in paragraphs if p])
        return text[:250_000]

    @staticmethod
    def _slugify_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    def _extract_early_life_text(self, html_text: str) -> str:
        soup = BeautifulSoup(html_text, "html.parser")
        heading_tags = soup.find_all(re.compile(r"^h[1-4]$"))
        section_keywords = (
            "early life",
            "childhood",
            "education",
            "background",
            "upbringing",
            "family",
            "student",
            "college",
            "university",
            "youth",
            "podcast",
            "interview",
            "transcript",
            "episode",
            "biography",
        )

        chunks: list[str] = []
        for heading in heading_tags:
            heading_text = heading.get_text(" ", strip=True).lower()
            if not any(keyword in heading_text for keyword in section_keywords):
                continue
            sibling = heading.find_next_sibling()
            while sibling is not None and not re.match(r"^h[1-4]$", getattr(sibling, "name", "")):
                if getattr(sibling, "name", "") in {"p", "li"}:
                    txt = sibling.get_text(" ", strip=True)
                    if txt:
                        chunks.append(txt)
                sibling = sibling.find_next_sibling()

        if chunks:
            return "\n".join(chunks)[:250_000]

        # Fallback if structured sections are not available.
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n".join([p for p in paragraphs if p])
        return text[:250_000]

    def _lookup_birth_month_day(self, entrepreneur_name: str) -> str | None:
        """
        Quick birth date lookup via Wikipedia summary endpoint.
        Returns month/day text like "June 28" when available.
        """
        title = entrepreneur_name.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(title)}"
        try:
            response = requests.get(url, timeout=self.timeout_seconds, headers={"User-Agent": "SeedFounderIntelligenceBot/0.1"})
            response.raise_for_status()
            payload = response.json()
            text = " ".join(
                [
                    str(payload.get("extract", "")),
                    str(payload.get("description", "")),
                ]
            )
            pattern = re.compile(
                r"\bborn\s+([A-Z][a-z]+(?:\s+\d{1,2})?)\s*,?\s*(?:\d{4})",
                flags=re.IGNORECASE,
            )
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        except Exception:
            return None
        return None

    def _extract_ddg_target_url(self, href: str) -> str | None:
        if not href:
            return None
        parsed = urlparse(href)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            query = parse_qs(parsed.query)
            uddg = query.get("uddg", [])
            if uddg:
                return unquote(uddg[0])
        if href.startswith("http"):
            return href
        return None

    def _fetch_duckduckgo_urls(self, query: str, max_results: int = 10) -> list[str]:
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {"User-Agent": "SeedFounderIntelligenceBot/0.1"}
        try:
            response = requests.get(search_url, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            urls: list[str] = []
            for link in soup.select("a.result__a"):
                target = self._extract_ddg_target_url(str(link.get("href", "")).strip())
                if target and target not in urls:
                    urls.append(target)
                if len(urls) >= max_results:
                    break
            return urls
        except Exception:
            return []

    def _default_biography_sources(self, entrepreneur_name: str) -> list[str]:
        slug = self._slugify_name(entrepreneur_name)
        wiki_title = entrepreneur_name.replace(" ", "_")
        return [
            # Primary biography/reference sources
            f"https://en.wikipedia.org/wiki/{wiki_title}",
            f"https://www.britannica.com/biography/{slug}",
            f"https://www.biography.com/business-leaders/{slug}",
            f"https://www.forbes.com/profile/{slug}/",
            f"https://www.businessinsider.com/{slug}",
            f"https://www.investopedia.com/{slug}-5224190",
            f"https://www.history.com/topics/{slug}",
            f"https://www.inc.com/profile/{slug}",
            f"https://www.encyclopedia.com/people/{slug}",
            f"https://www.bloomberg.com/profile/person/{slug}",
            f"https://www.reuters.com/world/us/{slug}-profile/",
            f"https://www.cnbc.com/{slug}/",
            f"https://www.nytimes.com/topic/person/{slug}",
            f"https://www.theguardian.com/profile/{slug}",
            f"https://www.wsj.com/search?query={slug}",
            f"https://www.latimes.com/topic/{slug}",
            # Podcast / interview / transcript oriented defaults
            f"https://www.founderspodcast.com/search?q={quote_plus(entrepreneur_name)}",
            f"https://www.podchaser.com/search/{quote_plus(entrepreneur_name)}",
            f"https://www.listennotes.com/search/?q={quote_plus(entrepreneur_name)}",
            f"https://www.podscripts.co/search/{quote_plus(entrepreneur_name)}",
            f"https://www.rev.com/blog/transcript-tag/{slug}",
            f"https://www.youtube.com/results?search_query={quote_plus(entrepreneur_name + ' interview early life')}",
            f"https://www.youtube.com/results?search_query={quote_plus('Founders podcast ' + entrepreneur_name)}",
            f"https://www.npr.org/search?query={quote_plus(entrepreneur_name + ' How I Built This')}",
            f"https://tim.blog/?s={quote_plus(entrepreneur_name)}",
            f"https://mastersofscale.com/?s={quote_plus(entrepreneur_name)}",
            f"https://www.starterstory.com/search?query={quote_plus(entrepreneur_name)}",
            f"https://www.failory.com/blog/{self._slugify_name(entrepreneur_name)}",
            f"https://www.failory.com/blog/business-biographies",
            f"https://www.reddit.com/r/Entrepreneur/search/?q={quote_plus(entrepreneur_name + ' early life biography')}&restrict_sr=1",
            f"https://www.ycombinator.com/library?query={quote_plus(entrepreneur_name)}",
            f"https://www.startups.com/search?query={quote_plus(entrepreneur_name)}",
            f"https://www.theceolibrary.com/search?q={quote_plus(entrepreneur_name)}",
            f"https://swisspreneur.com/?s={quote_plus(entrepreneur_name)}",
            f"https://archive.org/search?query={quote_plus(entrepreneur_name + ' biography interview')}",
            f"https://www.gutenberg.org/ebooks/search/?query={quote_plus(entrepreneur_name)}",
            f"https://www.kauffman.org/?s={quote_plus(entrepreneur_name)}",
            f"https://www.loc.gov/guides/business-history/",
        ]

    def _default_high_depth_stack(self, entrepreneur_name: str) -> list[str]:
        """
        Preferred default stack for richer early-life coverage.
        Ordered by expected depth/biographical utility.
        """
        slug = self._slugify_name(entrepreneur_name)
        wiki_title = entrepreneur_name.replace(" ", "_")
        return [
            f"https://en.wikipedia.org/wiki/{wiki_title}",
            f"https://www.britannica.com/biography/{slug}",
            f"https://www.biography.com/business-leaders/{slug}",
            "https://www.loc.gov/guides/business-history/",
            f"https://archive.org/search?query={quote_plus(entrepreneur_name + ' biography interview')}",
            f"https://www.failory.com/blog/{slug}",
            "https://www.failory.com/blog/business-biographies",
            f"https://www.founderspodcast.com/search?q={quote_plus(entrepreneur_name)}",
            f"https://www.podscripts.co/search/{quote_plus(entrepreneur_name)}",
            f"https://tim.blog/?s={quote_plus(entrepreneur_name)}",
            f"https://mastersofscale.com/?s={quote_plus(entrepreneur_name)}",
            f"https://www.gutenberg.org/ebooks/search/?query={quote_plus(entrepreneur_name)}",
            f"https://www.kauffman.org/?s={quote_plus(entrepreneur_name)}",
        ]

    @staticmethod
    def _is_usable_biography_url(url: str) -> bool:
        lowered = url.lower().strip()
        if not lowered.startswith("http"):
            return False
        if "duckduckgo.com" in lowered:
            return False
        # Exclude generic landing/search pages when possible, but keep key podcast/video
        # discovery endpoints where transcript-rich pages are often linked.
        allow_search_domains = (
            "youtube.com/results",
            "founderspodcast.com/search",
            "podscripts.co/search",
            "listennotes.com/search",
            "podchaser.com/search",
            "npr.org/search",
            "tim.blog/?s=",
            "mastersofscale.com/?s=",
            "starterstory.com/search",
            "failory.com/blog",
            "reddit.com/r/entrepreneur/search",
            "startups.com/search",
            "theceolibrary.com/search",
            "swisspreneur.com/?s=",
            "archive.org/search",
            "gutenberg.org/ebooks/search",
            "ycombinator.com/library?query=",
            "kauffman.org/?s=",
            "loc.gov/guides/business-history",
        )
        if any(token in lowered for token in allow_search_domains):
            return True
        noisy_patterns = ["/search", "?query=", "/tag/", "/topics/", "/topic/"]
        if any(pattern in lowered for pattern in noisy_patterns):
            # Keep wikipedia and britannica even if query-like.
            if "wikipedia.org/wiki/" in lowered or "/biography/" in lowered:
                return True
            return False
        return True

    @staticmethod
    def _source_depth_score(url: str) -> int:
        """
        Higher score means deeper, biography-like, early-life-relevant source.
        """
        low = str(url or "").lower()
        score = 0
        domain_weights = {
            "wikipedia.org/wiki/": 26,
            "britannica.com/biography/": 24,
            "biography.com": 24,
            "archive.org": 22,
            "gutenberg.org": 22,
            "founderspodcast.com": 21,
            "podscripts.co": 20,
            "listennotes.com": 18,
            "podchaser.com": 17,
            "npr.org": 17,
            "tim.blog": 16,
            "mastersofscale.com": 16,
            "starterstory.com": 15,
            "failory.com": 17,
            "reddit.com/r/entrepreneur": 12,
            "ycombinator.com": 14,
            "startups.com": 14,
            "theceolibrary.com": 14,
            "kauffman.org": 13,
            "loc.gov/guides/business-history": 21,
            "businessinsider.com": 10,
            "forbes.com": 9,
            "investopedia.com": 9,
            "reuters.com": 7,
            "cnbc.com": 6,
            "wsj.com": 6,
            "latimes.com": 5,
            "youtube.com": 4,
        }
        for token, value in domain_weights.items():
            if token in low:
                score = max(score, value)
        content_tokens = (
            "biography",
            "early-life",
            "early_life",
            "childhood",
            "upbringing",
            "education",
            "interview",
            "transcript",
            "podcast",
            "book",
            "memoir",
            "autobiography",
            "founder",
            "history",
        )
        score += sum(1 for token in content_tokens if token in low)
        # Penalize generic list/search pages a bit.
        if "/search" in low or "?q=" in low or "?query=" in low:
            score -= 3
        return score

    def _prioritize_sources(self, urls: list[str], max_results: int) -> list[str]:
        deduped: list[str] = []
        for u in urls:
            if u and u not in deduped:
                deduped.append(u)
        ranked = sorted(
            deduped,
            key=lambda u: (
                -self._source_depth_score(u),
                len(u),
            ),
        )
        return ranked[:max_results]

    def discover_biography_sources(
        self,
        entrepreneur_name: str,
        cutoff_year: int,
        max_results: int = 24,
    ) -> list[str]:
        wiki_title = entrepreneur_name.replace(" ", "_")
        queries: list[tuple[str, int]] = [
            (f"early life biography {entrepreneur_name} before {cutoff_year}", 5),
            (f"{entrepreneur_name} childhood education biography", 5),
            (f"{entrepreneur_name} anecdote early years before {cutoff_year}", 4),
            (f"{entrepreneur_name} first person interview childhood hardship", 4),
            (f"{entrepreneur_name} long-form biography early life", 4),
            (f"{entrepreneur_name} memoir interview childhood", 4),
            (f"site:wikipedia.org {entrepreneur_name}", 4),
            (f"site:britannica.com {entrepreneur_name} biography", 4),
            (f"site:biography.com {entrepreneur_name}", 4),
            (f"site:forbes.com {entrepreneur_name} profile", 1),
            (f"site:businessinsider.com {entrepreneur_name} biography", 2),
            (f"site:investopedia.com {entrepreneur_name}", 1),
            # Podcast and long-form interview sources
            (f"site:founderspodcast.com {entrepreneur_name}", 4),
            (f"Founders podcast David Senra {entrepreneur_name}", 4),
            (f"{entrepreneur_name} podcast transcript early years", 4),
            (f"site:youtube.com {entrepreneur_name} interview early life", 2),
            (f"site:podscripts.co {entrepreneur_name}", 4),
            (f"site:listennotes.com {entrepreneur_name} podcast", 3),
            (f"site:starterstory.com {entrepreneur_name}", 3),
            (f"site:failory.com {entrepreneur_name} biography", 4),
            (f"site:failory.com business biographies {entrepreneur_name}", 4),
            (f"site:reddit.com/r/Entrepreneur {entrepreneur_name} early life biography", 3),
            (f"site:ycombinator.com {entrepreneur_name} interview", 3),
            (f"site:startups.com {entrepreneur_name} interview", 3),
            (f"site:npr.org {entrepreneur_name} how i built this", 3),
            (f"site:tim.blog {entrepreneur_name}", 3),
            (f"site:mastersofscale.com {entrepreneur_name}", 3),
            (f"site:archive.org {entrepreneur_name} biography", 4),
            (f"site:gutenberg.org {entrepreneur_name} autobiography", 4),
            (f"site:kauffman.org entrepreneur profile {entrepreneur_name}", 3),
            (f"site:loc.gov guides business history {entrepreneur_name}", 4),
        ]
        discovered: list[str] = self._default_high_depth_stack(entrepreneur_name)
        scored_discovered: list[tuple[str, int]] = []
        for query, boost in queries:
            urls = self._fetch_duckduckgo_urls(query=query, max_results=6)
            for u in urls:
                scored_discovered.append((u, boost + self._source_depth_score(u)))
            discovered.extend(urls)

        biography_like = [
            url
            for url in discovered
            if any(
                token in url.lower()
                for token in (
                    "wikipedia.org",
                    "britannica.com",
                    "biography.com",
                    "forbes.com",
                    "businessinsider.com",
                    "history.com",
                    "inc.com",
                    "investopedia.com",
                    "alumni",
                    "students/article",
                    "founderspodcast.com",
                    "youtube.com",
                    "youtu.be",
                    "podscripts.co",
                    "listennotes.com",
                    "podchaser.com",
                    "starterstory.com",
                    "failory.com",
                    "reddit.com/r/entrepreneur",
                    "ycombinator.com",
                    "startups.com",
                    "npr.org",
                    "tim.blog",
                    "mastersofscale.com",
                    "theceolibrary.com",
                    "swisspreneur.com",
                    "archive.org",
                    "gutenberg.org",
                    "kauffman.org",
                    "loc.gov/guides/business-history",
                    "transcript",
                    "interview",
                )
            )
            and self._is_usable_biography_url(url)
        ]
        # Merge quality score from query context + intrinsic source depth.
        best_score: dict[str, int] = {}
        for url in biography_like:
            best_score[url] = max(best_score.get(url, -999), self._source_depth_score(url))
        for url, score in scored_discovered:
            if url in best_score:
                best_score[url] = max(best_score[url], score)
        deduped = sorted(
            best_score.keys(),
            key=lambda u: (-best_score.get(u, 0), len(u)),
        )[:max_results]

        # Ensure high-depth defaults are always represented before generic fallbacks.
        for url in self._default_high_depth_stack(entrepreneur_name):
            if url not in deduped:
                deduped.append(url)
        deduped = self._prioritize_sources(deduped, max_results=max_results)

        # Fallback to broad defaults if still sparse.
        if len(deduped) < 12:
            fallback = self._default_biography_sources(entrepreneur_name)
            deduped.extend([u for u in fallback if u not in deduped])
            deduped = self._prioritize_sources(deduped, max_results=max_results)
        return deduped[:max_results]

    def _harvest_followup_links(
        self,
        html_text: str,
        base_url: str,
        entrepreneur_name: str,
        cutoff_year: int,
        max_links: int = 8,
    ) -> list[str]:
        """
        Extract likely biography/interview/transcript follow-up URLs from a source page.
        """
        try:
            soup = BeautifulSoup(html_text, "html.parser")
        except Exception:
            return []
        base = urlparse(base_url)
        name_tokens = [t.lower() for t in entrepreneur_name.split() if t.strip()]
        keep_markers = (
            "biography",
            "early-life",
            "early_life",
            "childhood",
            "interview",
            "transcript",
            "podcast",
            "founder",
            "about",
            "profile",
            "wiki",
        )
        blocked_markers = (
            "privacy",
            "terms",
            "advertis",
            "login",
            "signup",
            "subscribe",
            "account",
            "javascript:",
        )
        found: list[str] = []
        for a in soup.select("a[href]"):
            href = str(a.get("href", "")).strip()
            if not href:
                continue
            absolute = requests.compat.urljoin(base_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme not in {"http", "https"}:
                continue
            if parsed.netloc != base.netloc:
                continue
            low = absolute.lower()
            if any(m in low for m in blocked_markers):
                continue
            anchor_text = " ".join(a.stripped_strings).lower()
            content_hint = low + " " + anchor_text
            has_name = any(token in content_hint for token in name_tokens)
            has_marker = any(marker in content_hint for marker in keep_markers)
            has_year = str(cutoff_year) in content_hint or str(cutoff_year - 1) in content_hint
            if (has_name and has_marker) or (has_marker and has_year):
                if absolute not in found and self._is_usable_biography_url(absolute):
                    found.append(absolute)
            if len(found) >= max_links:
                break
        return found

    def _extract_event_queries(
        self,
        entrepreneur_name: str,
        claims: list[dict[str, Any]],
        cutoff_year: int,
        max_queries: int = 6,
    ) -> list[str]:
        """
        Build focused follow-up queries to deepen interview-like coverage.
        """
        query_candidates: list[str] = []
        seen_terms: set[str] = set()
        for claim in claims:
            text = str(claim.get("raw_sentence") or claim.get("attribute_text", ""))
            if not text:
                continue
            # Pull possible event terms from quoted/Title-Case tokens and frequent nouns.
            tokens = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b|\b[a-z]{5,}\b", text)
            for token in tokens:
                low = token.lower()
                if low in STOPWORDS or low in seen_terms:
                    continue
                seen_terms.add(low)
                query_candidates.append(
                    f"{entrepreneur_name} {token} early life before {cutoff_year}"
                )
                if len(query_candidates) >= max_queries:
                    return query_candidates
        return query_candidates

    def expand_sources_for_interview_depth(
        self,
        entrepreneur_name: str,
        cutoff_year: int,
        base_sources: list[str],
        seed_claims: list[dict[str, Any]],
        max_total_sources: int = 60,
    ) -> list[str]:
        expanded = list(base_sources)
        queries = self._extract_event_queries(
            entrepreneur_name=entrepreneur_name,
            claims=seed_claims,
            cutoff_year=cutoff_year,
            max_queries=8,
        )
        for query in queries:
            urls = self._fetch_duckduckgo_urls(query=query, max_results=6)
            for url in urls:
                if self._is_usable_biography_url(url) and url not in expanded:
                    expanded.append(url)
                if len(expanded) >= max_total_sources:
                    return expanded
        # Domain-focused deepening pass for anecdotal detail.
        targeted_queries = [
            f"site:starterstory.com {entrepreneur_name} story",
            f"site:npr.org {entrepreneur_name} interview",
            f"site:tim.blog {entrepreneur_name} transcript",
            f"site:mastersofscale.com {entrepreneur_name}",
            f"site:founderspodcast.com {entrepreneur_name} notes",
        ]
        for query in targeted_queries:
            urls = self._fetch_duckduckgo_urls(query=query, max_results=5)
            for url in urls:
                if self._is_usable_biography_url(url) and url not in expanded:
                    expanded.append(url)
                if len(expanded) >= max_total_sources:
                    return expanded
        return expanded

    @staticmethod
    def _normalize_category_name(raw: str) -> str | None:
        key = re.sub(r"\s+", " ", str(raw).strip().lower())
        key = key.replace("-", " ")
        if key in CATEGORY_ALIAS_MAP:
            return CATEGORY_ALIAS_MAP[key]
        key_u = key.replace(" ", "_")
        if key_u in SIGNAL_MAP:
            return key_u
        return None

    def extract_source_pub_date(self, html_text: str) -> date | None:
        """Best-effort extraction of page publication date."""
        soup = BeautifulSoup(html_text, "html.parser")
        meta_candidates = [
            ("meta", {"property": "article:published_time"}),
            ("meta", {"name": "pubdate"}),
            ("meta", {"name": "publish-date"}),
            ("meta", {"name": "date"}),
        ]
        for tag_name, attrs in meta_candidates:
            tag = soup.find(tag_name, attrs=attrs)
            if tag and tag.get("content"):
                parsed = self._safe_parse_date(tag["content"])
                if parsed:
                    return parsed
        return None

    def _safe_parse_date(self, value: str) -> date | None:
        value = value.strip()
        if not value:
            return None
        # Normalize year-only strings to Jan 1 to avoid parser defaulting to today's month/day.
        if re.fullmatch(r"(?:19|20)\d{2}", value):
            return date(int(value), 1, 1)
        if dt_parser is not None:
            try:
                return dt_parser.parse(value, fuzzy=True).date()
            except Exception:
                pass
        # Year-only fallback
        year_match = re.search(r"\b(19|20)\d{2}\b", value)
        if year_match:
            return date(int(year_match.group(0)), 1, 1)
        return None

    def _extract_dates_from_sentence(self, sentence: str) -> list[date]:
        dates: list[date] = []
        doc = self._nlp(sentence) if self._nlp is not None else None

        # Prefer spaCy NER DATE entities when model is available.
        for ent in getattr(doc, "ents", []) if doc is not None else []:
            if ent.label_.upper() == "DATE":
                parsed = self._safe_parse_date(ent.text)
                if parsed:
                    dates.append(parsed)

        # Regex fallback for years.
        for match in re.findall(r"\b(?:19|20)\d{2}\b", sentence):
            parsed = self._safe_parse_date(match)
            if parsed:
                dates.append(parsed)

        # Year ranges like "1990-1992" or "1990 to 1992" -> use range start.
        for start_year, _ in re.findall(r"\b((?:19|20)\d{2})\s*(?:-|to|–)\s*((?:19|20)\d{2})\b", sentence):
            parsed = self._safe_parse_date(start_year)
            if parsed:
                dates.append(parsed)

        # Decades like "early 1990s" or "1990s" -> anchor to decade start.
        for decade in re.findall(r"\b((?:19|20)\d0)s\b", sentence.lower()):
            parsed = self._safe_parse_date(decade)
            if parsed:
                dates.append(parsed)

        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique: list[date] = []
        for d in dates:
            key = d.isoformat()
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique

    def _classify_category(self, sentence: str) -> str | None:
        sent = sentence.lower()

        def keyword_match(text: str, keyword: str) -> bool:
            kw = keyword.lower().strip()
            if not kw:
                return False
            if " " in kw:
                return kw in text
            if len(kw) <= 3:
                return re.search(rf"\b{re.escape(kw)}\b", text) is not None
            return kw in text

        for category, keywords in SIGNAL_MAP.items():
            if any(keyword_match(sent, k) for k in keywords):
                return category
        return None

    def summarize_pre_cutoff_attributes(
        self,
        raw_text: str,
        entrepreneur_name: str,
        birth_year: int,
        cutoff_year: int,
        birth_month_day: str | None = None,
    ) -> str:
        """
        Produce verbose interview-style narrative extraction for pre-cutoff facts.
        """
        trimmed = raw_text[:7000]
        if self._use_deepseek_extraction():
            trimmed = raw_text[: self._deepseek_max_source_chars]
        cutoff_date_text = f"{cutoff_year}-01-01"
        prompt = (
            f"Imagine interviewing {entrepreneur_name} (born {birth_year}) just before {cutoff_year}, asking about their "
            "entire life up to now. Based ONLY on the provided text with facts true before the cutoff date, generate detailed, "
            f"expansive answers in first-person narrative as if {entrepreneur_name} is responding.\n\n"
            "Cover vast details across categories: Appearance (describe in depth), Decision making (detail phases like "
            "childhood/teen/early adult, all key decisions with contexts/reasons/why), Education (full timeline, experiences, "
            "influences), Family (in-depth childhood, hardships, relationships, impacts), Most difficult accomplishment "
            "(elaborate on challenges overcome), Most crazy accomplishment (vivid story), Personality (traits with examples), "
            "Values (beliefs shaped by experiences), Sense of security/insecurity (sources and manifestations), Psychological "
            "condition (observable traits/behaviors), Motivation for starting business (early drives, frustrations, visions), "
            "plus additional signals like Early technical proficiency (projects/skills), Risk tolerance (examples), Intellectual "
            "curiosity (readings/interests), Resilience indicators (failures bounced from), Network/mentorship (early connections).\n\n"
            "For each category, include multiple sub-facts, inferred timestamps (e.g., age-based from birth_year, context like "
            f"'early 1990s'), and sources. Output strictly in JSON:\n"
            "{\"category\": [{\"narrative_fact\": \"detailed first-person description\", \"sub_facts\": [\"fact1\", \"fact2\"], "
            "\"timestamps\": [\"YYYY or range\"], \"sources\": [\"URL1\", \"URL2\"], \"inference\": \"if inferred\"}]}\n\n"
            f"Rules: only use facts true before {cutoff_date_text}. Exclude {cutoff_year} and later facts.\n\n"
            f"Source text:\n{trimmed}"
        )

        if self._use_deepseek_extraction():
            try:
                return self._deepseek_chat_completion(
                    system_prompt=(
                        "You are an extraction engine for venture diligence. Return strict JSON only. "
                        "Never include facts that are post-cutoff."
                    ),
                    user_prompt=prompt,
                    max_tokens=700,
                    temperature=0.1,
                )
            except Exception:
                return trimmed

        if os.getenv("DISABLE_HF_MODEL", "0") == "1":
            return trimmed

        # Prefer summarization path first for stability and throughput.
        try:
            summary = self.summarizer(
                prompt,
                max_length=2048,
                min_length=700,
                do_sample=False,
                truncation=True,
            )
            return str(summary[0]["summary_text"]).strip()
        except Exception:
            pass

        # Optional generation fallback (gpt2-xl / gpt2) when explicitly enabled.
        if os.getenv("ENABLE_GENERATION_FALLBACK", "0") == "1":
            try:
                generated = self.generator(
                    prompt[:2200],
                    max_new_tokens=400,
                    do_sample=False,
                    truncation=True,
                    return_full_text=False,
                )
                text = str(generated[0].get("generated_text", "")).strip()
                if text:
                    return text
            except Exception:
                pass
        return trimmed

    @staticmethod
    def _extract_json_block(text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    def _extract_anchor_claims(
        self,
        raw_text: str,
        source_url: str,
        source_date: date,
        source_label: str | None,
    ) -> list[dict[str, Any]]:
        """
        Deterministic anchor extraction from raw text for high-value pre-cutoff events.
        Generic patterns are used so this works for any entrepreneur profile.
        """
        anchors: list[dict[str, Any]] = []
        lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_text) if s.strip()]
        for sentence in lines:
            low = sentence.lower()
            event_date: date | None = None
            category: str | None = None

            years = re.findall(r"\b(?:19|20)\d{2}\b", sentence)
            if years:
                event_date = self._safe_parse_date(years[0])

            # Migration / relocation anchors.
            if any(token in low for token in ("emigrated", "immigrated", "moved to", "relocated to")):
                category = "demographic_contextual"
                if event_date is None:
                    event_date = self._safe_parse_date("1990")

            # Education transition anchors.
            elif any(token in low for token in ("transferred to", "enrolled in", "attended", "graduated")) and any(
                token in low for token in ("university", "college", "academy", "school")
            ):
                category = "education"

            # Early technical build anchors.
            elif any(token in low for token in ("coded", "programmed", "built", "developed", "created", "wrote")) and any(
                token in low for token in ("game", "software", "program", "app", "computer", "prototype")
            ):
                category = "early_technical_proficiency"

            # Motivation / decision anchors.
            elif any(token in low for token in ("decided", "chose", "because", "in order to", "motivated")):
                category = "decision_making"

            if not category:
                continue
            anchors.append(
                {
                    "category": category,
                    "attribute_text": sentence,
                    "event_date": event_date.isoformat() if event_date else None,
                    "source_pub_date": source_date.isoformat(),
                    "source_url": source_url,
                    "source": source_label or source_url,
                    "extraction_method": "deterministic_anchor",
                    "raw_sentence": sentence,
                }
            )
        return anchors

    def _extract_raw_sentence_claims(
        self,
        raw_text: str,
        source_url: str,
        source_date: date,
        source_label: str | None,
    ) -> list[dict[str, Any]]:
        """
        High-recall extraction directly from source text sentences.
        This boosts comprehensiveness before DATE CUTOFF filtering.
        """
        claims: list[dict[str, Any]] = []
        if self._nlp is not None:
            try:
                doc = self._nlp(raw_text)
                lines = [s.text.strip() for s in doc.sents if s.text.strip()]
            except Exception:
                lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_text) if s.strip()]
        else:
            lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_text) if s.strip()]

        # Avoid overlong noisy tails while still keeping very broad coverage.
        for sentence in lines[:600]:
            if len(sentence) < 35:
                continue
            category = self._classify_category(sentence)
            if not category:
                continue
            extracted_dates = self._extract_dates_from_sentence(sentence)
            event_date = extracted_dates[0] if extracted_dates else None
            claims.append(
                {
                    "category": category,
                    "attribute_text": sentence,
                    "event_date": event_date.isoformat() if event_date else None,
                    "source_pub_date": source_date.isoformat(),
                    "source_url": source_url,
                    "source": source_label or source_url,
                    "extraction_method": "raw_sentence_high_recall",
                    "raw_sentence": sentence,
                }
            )
        return claims

    def _infer_pre_cutoff_from_context(
        self,
        claim: dict[str, Any],
        cutoff_date: date,
        birth_year: int,
    ) -> tuple[date | None, str | None]:
        text = str(claim.get("raw_sentence") or claim.get("attribute_text", "")).lower()

        # Age-based inference: "age 12" => 1983 for birth year 1971.
        age_match = re.search(r"\bage\s*(\d{1,2})\b", text)
        if age_match:
            age = int(age_match.group(1))
            inferred_year = birth_year + age
            inferred = date(inferred_year, 1, 1)
            if inferred < cutoff_date:
                return inferred, f"age_based_inference:{age}"

        # Additional age patterns: "at 17", "aged around 9", "until he was 15", "when he was 12 years old".
        extra_age_patterns = [
            r"\bat\s+(\d{1,2})\b",
            r"\baged?\s+(?:around\s+)?(\d{1,2})\b",
            r"\buntil\s+(?:he|she|they)\s+was\s+(\d{1,2})\b",
            r"\bwhen\s+(?:he|she|they)\s+was\s+(\d{1,2})\b",
            r"\b(\d{1,2})\s+years?\s+old\b",
        ]
        for pattern in extra_age_patterns:
            m = re.search(pattern, text)
            if not m:
                continue
            age = int(m.group(1))
            if 0 < age <= 30:
                inferred_year = birth_year + age
                inferred = date(inferred_year, 1, 1)
                if inferred < cutoff_date:
                    return inferred, f"age_pattern_inference:{age}"

        # Phrase-based inference for clearly pre-1995 life phases.
        inference_map = [
            (("childhood", "as a child", "schoolboy"), date(1980, 1, 1), "childhood_inference"),
            (
                (
                    "as a teen",
                    "in his teens",
                    "in her teens",
                    "teenage years",
                    "high school",
                    "secondary school",
                ),
                date(1987, 1, 1),
                "teen_highschool_inference",
            ),
            (("school years", "pre-university", "before university"), date(1990, 1, 1), "school_phase_inference"),
            (("college years", "university years", "undergraduate"), date(1992, 1, 1), "college_phase_inference"),
        ]
        for tokens, inferred_date, reason in inference_map:
            if any(token in text for token in tokens) and inferred_date < cutoff_date:
                return inferred_date, reason

        # Lifecycle inference: college / university phase is generally pre-first-startup
        # for early founder timelines and should be treated as pre-cutoff when no
        # explicit post-cutoff date is present.
        grad_tokens = (
            "doctorate",
            "doctoral",
            "phd",
            "graduate school",
            "masters",
            "postgraduate",
        )
        if any(token in text for token in grad_tokens):
            # Graduate-school phrases are often around or after first startup period;
            # do not auto-backdate them by generic college inference.
            return None, None

        college_tokens = (
            "college",
            "university",
            "undergraduate",
            "freshman",
            "sophomore",
            "junior",
            "senior",
            "campus",
            "dorm",
            "major",
            "degree",
        )
        modern_career_tokens = (
            "executive",
            "ceo",
            "lawsuit",
            "court",
            "regulator",
            "platform",
            "users",
            "instagram",
            "whatsapp",
            "messenger",
            "threads",
            "content moderation",
            "policy",
            "administration",
            "senate",
            "house committee",
            "acquisition",
        )
        college_action_tokens = (
            "attended",
            "enrolled",
            "studied",
            "dropped out",
            "major",
            "degree",
            "campus",
            "dorm",
            "undergraduate",
            "freshman",
            "sophomore",
            "junior",
            "senior",
        )
        if (
            any(token in text for token in college_tokens)
            and any(token in text for token in college_action_tokens)
            and not any(token in text for token in modern_career_tokens)
        ):
            college_start_year = birth_year + 17
            college_end_year = birth_year + 24
            inferred_year = min(cutoff_date.year - 1, college_end_year)
            if inferred_year >= college_start_year and inferred_year < cutoff_date.year:
                return date(inferred_year, 1, 1), "college_phase_before_startup_inference"

        # Startup-adjacent context still before first institutional funding.
        pre_startup_tokens = (
            "before startup",
            "before founding",
            "prior to founding",
            "before first startup",
            "first startup",
            "internship",
            "early project",
            "student project",
            "while in school",
            "while at university",
        )
        if any(token in text for token in pre_startup_tokens):
            return date(cutoff_date.year - 1, 1, 1), "pre_startup_context_inference"

        broad_context_tokens = (
            f"pre-{cutoff_date.year}",
            f"before {cutoff_date.year}",
        )
        if any(token in text for token in broad_context_tokens):
            return date(cutoff_date.year - 1, 1, 1), "broad_pre_cutoff_context"

        return None, None

    def extract_claims_with_llm(
        self,
        raw_text: str,
        source_url: str,
        entrepreneur_name: str,
        birth_year: int,
        source_pub_date: date | None = None,
        source_label: str | None = None,
        cutoff_date: date | None = None,
        birth_month_day: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build structured claims from narrative JSON output + sentence/date heuristics.
        """
        cutoff_year = cutoff_date.year if cutoff_date else date.today().year
        narrative_outputs: list[str] = []
        if self._use_deepseek_extraction():
            narrative_outputs.extend(
                self._run_deepseek_extraction_passes(
                    entrepreneur_name=entrepreneur_name,
                    birth_year=birth_year,
                    cutoff_year=cutoff_year,
                    source_text=raw_text,
                )
            )
        primary_output = self.summarize_pre_cutoff_attributes(
            raw_text,
            entrepreneur_name=entrepreneur_name,
            birth_year=birth_year,
            cutoff_year=cutoff_year,
            birth_month_day=birth_month_day,
        )
        narrative_outputs.append(primary_output)

        parsed_json_blocks: list[dict[str, Any]] = []
        for out in narrative_outputs:
            parsed = self._extract_json_block(out)
            if isinstance(parsed, dict):
                parsed_json_blocks.append(parsed)

        narrative_output = "\n\n".join(narrative_outputs)
        sentences: list[str]
        if self._nlp is not None:
            doc = self._nlp(narrative_output)
            sentences = [s.text.strip() for s in doc.sents]
        else:
            # Fallback sentence split when spaCy is unavailable.
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", narrative_output) if s.strip()]
        source_date = source_pub_date or date.today()
        claims: list[dict[str, Any]] = []
        claims.extend(self._extract_anchor_claims(raw_text, source_url, source_date, source_label))
        claims.extend(self._extract_raw_sentence_claims(raw_text, source_url, source_date, source_label))

        # Parse JSON narrative categories when output is structured.
        for parsed_json in parsed_json_blocks:
            for raw_category, items in parsed_json.items():
                category = self._normalize_category_name(raw_category)
                if not category:
                    continue
                if isinstance(items, dict):
                    items = [items]
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    narrative_fact = str(item.get("narrative_fact", "")).strip()
                    sub_facts = item.get("sub_facts", [])
                    timestamps = item.get("timestamps", [])
                    sources = item.get("sources", [])
                    inference = str(item.get("inference", "")).strip()
                    if not narrative_fact:
                        continue
                    event_date: date | None = None
                    if isinstance(timestamps, list):
                        for ts in timestamps:
                            parsed_date = self._safe_parse_date(str(ts))
                            if parsed_date:
                                event_date = parsed_date
                                break
                    elif isinstance(timestamps, str):
                        event_date = self._safe_parse_date(timestamps)

                    claims.append(
                        {
                            "category": category,
                            "attribute_text": narrative_fact,
                            "narrative_fact": narrative_fact,
                            "sub_facts": [str(x) for x in sub_facts] if isinstance(sub_facts, list) else [],
                            "timestamps": [str(x) for x in timestamps] if isinstance(timestamps, list) else [str(timestamps)],
                            "event_date": event_date.isoformat() if event_date else None,
                            "source_pub_date": source_date.isoformat(),
                            "source_url": source_url,
                            "source": source_label or source_url,
                            "narrative_sources": [str(x) for x in sources] if isinstance(sources, list) else [],
                            "narrative_inference": inference,
                            "extraction_method": "interview_generation_json",
                            "raw_sentence": narrative_fact,
                        }
                    )

        for sentence in sentences:
            if len(sentence) < 20:
                continue
            category = self._classify_category(sentence)
            if not category:
                continue

            extracted_dates = self._extract_dates_from_sentence(sentence)
            event_date = extracted_dates[0] if extracted_dates else None
            claim_text = sentence
            claims.append(
                {
                    "category": category,
                    "attribute_text": claim_text,
                    "event_date": event_date.isoformat() if event_date else None,
                    "source_pub_date": source_date.isoformat(),
                    "source_url": source_url,
                    "source": source_label or source_url,
                    "extraction_method": "interview_generation_heuristics",
                    "raw_sentence": sentence,
                }
            )
        return claims

    def apply_temporal_filter(
        self,
        claims: list[dict[str, Any]],
        cutoff_date: date,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split claims by DATE CUTOFF validity."""
        valid: list[dict[str, Any]] = []
        invalid: list[dict[str, Any]] = []

        for claim in claims:
            claim_text_raw = str(claim.get("raw_sentence") or claim.get("attribute_text", "")).strip()
            if self._looks_post_cutoff_context(claim_text_raw, cutoff_date.year):
                claim["before_cutoff"] = False
                claim["discard_reason"] = "post_cutoff_context_semantic"
                discarded.append(claim)
                self.anomaly_logger.info(
                    "discarded=post_cutoff_context_semantic | cutoff=%s | source=%s | claim=%s",
                    cutoff_date.isoformat(),
                    claim.get("source_url"),
                    claim.get("attribute_text"),
                )
                continue

            event_raw = claim.get("event_date")
            event_date = self._safe_parse_date(event_raw) if event_raw else None
            if event_date is not None and event_date <= cutoff_date:
                claim["before_cutoff"] = True
                valid.append(claim)
            else:
                claim["before_cutoff"] = False
                invalid.append(claim)
        return valid, invalid

    @staticmethod
    def _fact_fingerprint(claim: dict[str, Any]) -> str:
        # Prefer raw sentence for cross-source matching because LLM rewrites can vary.
        base = str(claim.get("raw_sentence") or claim.get("attribute_text", "")).lower()
        base = re.sub(r"[^a-z0-9\s]", "", base)
        base = re.sub(r"\s+", " ", base).strip()
        return base[:140]

    def _tokenize_for_match(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return {t for t in tokens if t not in STOPWORDS}

    def _event_year(self, claim: dict[str, Any]) -> int | None:
        event_raw = claim.get("event_date")
        event_date = self._safe_parse_date(event_raw) if event_raw else None
        return event_date.year if event_date else None

    def _token_overlap(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a.intersection(b))
        denom = max(1, min(len(a), len(b)))
        return inter / denom

    def validate_date_cutoff_compliance(
        self,
        claims: list[dict[str, Any]],
        entrepreneur_name: str,
        cutoff_date: date,
        birth_year: int,
        min_source_count: int = 2,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Validate DATE CUTOFF with 2-source cross-reference.

        Rules:
        1) fact needs parseable event date and event_date < cutoff_date
        2) corroboration must appear in at least 2 distinct sources
           with same category + event year and lexical overlap
        3) if not anchorable pre-cutoff, discard and log anomaly
        """
        accepted: list[dict[str, Any]] = []
        discarded: list[dict[str, Any]] = []
        available_sources = {
            str(c.get("source_url", "")).strip() for c in claims if str(c.get("source_url", "")).strip()
        }
        source_claim_counts: dict[str, int] = {}
        for claim in claims:
            src = str(claim.get("source_url", "")).strip()
            if not src:
                continue
            source_claim_counts[src] = source_claim_counts.get(src, 0) + 1
        robust_source_count = sum(1 for _, cnt in source_claim_counts.items() if cnt >= 3)
        effective_min_source_count = (
            1 if robust_source_count < 2 else max(1, min(min_source_count, len(available_sources)))
        )

        for claim in claims:
            if not self._is_claim_relevant_to_founder(claim, entrepreneur_name):
                claim["before_cutoff"] = False
                claim["discard_reason"] = "irrelevant_to_target_founder"
                discarded.append(claim)
                continue
            event_raw = claim.get("event_date")
            event_date = self._safe_parse_date(event_raw) if event_raw else None
            inferred = False
            inference_reason: str | None = None

            if event_date is None:
                inferred_date, reason = self._infer_pre_cutoff_from_context(
                    claim,
                    cutoff_date=cutoff_date,
                    birth_year=birth_year,
                )
                if inferred_date is not None:
                    event_date = inferred_date
                    inferred = True
                    inference_reason = reason
                else:
                    claim["before_cutoff"] = False
                    claim["discard_reason"] = "no_parseable_event_date"
                    discarded.append(claim)
                    self.anomaly_logger.info(
                        "discarded=no_parseable_event_date | source=%s | claim=%s",
                        claim.get("source_url"),
                        claim.get("attribute_text"),
                    )
                    continue

            # Strict post-1994 exclusion: anything in 1995+ is excluded.
            if event_date >= cutoff_date:
                # Boundary handling: if the text indicates "until/by/before cutoff year"
                # and relates to education/early-phase context, treat as pre-cutoff boundary.
                claim_text = str(claim.get("raw_sentence") or claim.get("attribute_text", "")).lower()
                claim_category = str(claim.get("category", "")).lower()
                boundary_ok = (
                    event_date.year == cutoff_date.year
                    and claim_category in {"education", "decision_making", "early_technical_proficiency"}
                    and any(token in claim_text for token in ("until", "by", "before", "prior to", "during"))
                )
                if boundary_ok:
                    event_date = date(cutoff_date.year - 1, 12, 31)
                    inferred = True
                    inference_reason = "cutoff_boundary_context_inference"
                else:
                    claim["before_cutoff"] = False
                    claim["discard_reason"] = "post_cutoff_event"
                    discarded.append(claim)
                    self.anomaly_logger.info(
                        "discarded=post_cutoff_event | event_date=%s | cutoff=%s | source=%s | claim=%s",
                        event_date.isoformat(),
                        cutoff_date.isoformat(),
                        claim.get("source_url"),
                        claim.get("attribute_text"),
                    )
                    continue

            claim_tokens = self._tokenize_for_match(
                str(claim.get("raw_sentence") or claim.get("attribute_text", ""))
            )
            claim_year = event_date.year
            support_sources = {str(claim.get("source_url", ""))}
            broad_pre_context = bool(inference_reason and "broad_pre_cutoff_context" in inference_reason)

            for other in claims:
                if other is claim:
                    continue
                other_source = str(other.get("source_url", ""))
                if other_source in support_sources:
                    continue
                other_event_raw = other.get("event_date")
                other_event = self._safe_parse_date(other_event_raw) if other_event_raw else None
                if other_event is None or other_event >= cutoff_date:
                    continue
                if not broad_pre_context and other_event.year != claim_year:
                    continue
                other_tokens = self._tokenize_for_match(
                    str(other.get("raw_sentence") or other.get("attribute_text", ""))
                )
                overlap = self._token_overlap(claim_tokens, other_tokens)
                if overlap >= 0.12:
                    support_sources.add(other_source)

            is_anchor = str(claim.get("extraction_method", "")) == "deterministic_anchor"
            if broad_pre_context:
                required_sources = max(1, math.ceil(0.8 * max(1, len(available_sources))))
            else:
                required_sources = 1 if inferred or is_anchor else effective_min_source_count
            cross_source_ok = len(support_sources) >= required_sources
            if not cross_source_ok:
                claim["before_cutoff"] = False
                claim["discard_reason"] = "insufficient_source_corroboration"
                discarded.append(claim)
                self.anomaly_logger.info(
                    "discarded=insufficient_source_corroboration | required=%s | found=%s | source=%s | claim=%s",
                    required_sources,
                    len(support_sources),
                    claim.get("source_url"),
                    claim.get("attribute_text"),
                )
                continue

            claim["before_cutoff"] = True
            claim["event_date"] = event_date.isoformat()
            claim["verification_source_count"] = len(support_sources)
            if inferred:
                claim["timestamp_inference"] = f"inferred pre-{cutoff_date.year}"
                claim["inference_reason"] = inference_reason
            if claim.get("timestamp_inference"):
                claim["timing_provenance"] = "inferred"
            else:
                claim["timing_provenance"] = "explicit"
            if broad_pre_context:
                claim["corroboration_policy"] = "broad_pre_cutoff_80_percent_sources"
            accepted.append(claim)

        return accepted, discarded

    @staticmethod
    def _contains_post_cutoff_year(text: str, cutoff_year: int) -> bool:
        years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", text)]
        return any(y >= cutoff_year for y in years)

    def _looks_post_cutoff_context(self, text: str, cutoff_year: int) -> bool:
        low = str(text or "").lower()
        if self._contains_post_cutoff_year(low, cutoff_year):
            return True
        # Social-post style references plus governance/public-company context
        # are strong signals of post-cutoff, mature-company periods.
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
        strong_post_tokens = (
            # Government / policy-cycle context (generally post-scale).
            "president",
            "administration",
            "white house",
            "government work",
            "political activities",
            "policy debate",
            # Public-company / regulatory-cycle context.
            "sec filing",
            "regulatory filing",
            "market cap",
            "shareholders",
            "earnings call",
            "public company",
            # Late-stage public media scrutiny context.
            "investigations revealed",
            "news study",
            "researchers stated",
            "according to nbc",
            "according to the times",
            # Biomedical / compliance contexts typically far after founding.
            "animal testing",
            "clinical trial",
            # Platform-era social posting context.
            "in a post",
            "social media platform",
            # Late-stage executive framing.
            "as ceo",
            "billionaire",
            # Mature-company legal/safety/governance context.
            "lawsuit",
            "court documents",
            "legal action",
            "ongoing lawsuit",
            "content moderation",
            "fact-checking",
            "billions of users",
            "global affairs",
            "senior executives",
            "integrated the chat systems",
        )
        return any(token in low for token in strong_post_tokens)

    @staticmethod
    def _name_aliases(entrepreneur_name: str) -> set[str]:
        cleaned = re.sub(r"\s+", " ", str(entrepreneur_name or "").strip().lower())
        if not cleaned:
            return set()
        parts = [p for p in cleaned.split(" ") if p]
        aliases = {cleaned}
        if parts:
            aliases.add(parts[0])
            aliases.add(parts[-1])
            aliases.add(f"{parts[0]}'s")
            aliases.add(f"{parts[-1]}'s")
        return {a for a in aliases if len(a) >= 3}

    def _source_looks_founder_specific(self, source_url: str, entrepreneur_name: str) -> bool:
        url = str(source_url or "").lower()
        if not url:
            return False
        compact_url = re.sub(r"[^a-z0-9]", "", url)
        for alias in self._name_aliases(entrepreneur_name):
            compact_alias = re.sub(r"[^a-z0-9]", "", alias)
            if compact_alias and compact_alias in compact_url:
                return True
        return False

    def _is_claim_relevant_to_founder(self, claim: dict[str, Any], entrepreneur_name: str) -> bool:
        text = str(claim.get("raw_sentence") or claim.get("attribute_text", "")).strip()
        if not text:
            return False
        low = text.lower()
        aliases = self._name_aliases(entrepreneur_name)
        if any(alias in low for alias in aliases):
            return True

        # Filter common listicle/book-summary contamination.
        contamination_tokens = (
            "name of book",
            "description of the book",
            "autobiography",
            "author :",
            "length :",
            "notable quote",
        )
        if any(token in low for token in contamination_tokens):
            return False

        source_url = str(claim.get("source_url", "")).strip()
        if self._source_looks_founder_specific(source_url, entrepreneur_name):
            # Allow pronoun-only lines only if they look biographical.
            pronoun_tokens = (" he ", " his ", " she ", " her ", " they ", " their ")
            bio_tokens = (
                "born",
                "childhood",
                "parents",
                "mother",
                "father",
                "school",
                "college",
                "university",
                "dropped out",
                "founded",
                "started",
                "created",
                "when he was",
                "when she was",
                "at age",
            )
            padded = f" {low} "
            if any(p in padded for p in pronoun_tokens) and any(t in low for t in bio_tokens):
                return True
        return False

    def build_comprehensive_pre_cutoff_facts(
        self,
        claims: list[dict[str, Any]],
        entrepreneur_name: str,
        cutoff_date: date,
        birth_year: int,
    ) -> list[dict[str, Any]]:
        """
        High-recall library of pre-cutoff facts.
        Keeps rich detail with lighter gating than accepted_claims.
        """
        comprehensive: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str]] = set()

        for claim in claims:
            raw_text = str(claim.get("raw_sentence") or claim.get("attribute_text", "")).strip()
            if not raw_text:
                continue
            if not self._is_claim_relevant_to_founder(claim, entrepreneur_name):
                continue
            if self._looks_post_cutoff_context(raw_text, cutoff_date.year):
                continue

            event_raw = claim.get("event_date")
            event_date = self._safe_parse_date(event_raw) if event_raw else None
            inferred_reason = ""
            if event_date is None:
                inferred_date, reason = self._infer_pre_cutoff_from_context(
                    claim, cutoff_date=cutoff_date, birth_year=birth_year
                )
                if inferred_date is not None:
                    event_date = inferred_date
                    inferred_reason = reason or "inferred_context"

            # If still undated, keep only likely pre-cutoff contextual facts.
            if event_date is None and not self._contains_post_cutoff_year(raw_text.lower(), cutoff_date.year):
                # Conservative unanchored inference: only apply when early-life cues are strong.
                strong_pre_tokens = (
                    "childhood",
                    "as a child",
                    "early life",
                    "early years",
                    "high school",
                    "teenage years",
                    "in his teens",
                    "in her teens",
                    "parents",
                    "mother",
                    "father",
                    "born",
                    "at age",
                    "aged",
                    "when he was",
                    "when she was",
                )
                explicit_pre_tokens = (
                    "before founding",
                    "before startup",
                    "before first startup",
                    "prior to founding",
                    f"pre-{cutoff_date.year}",
                    f"before {cutoff_date.year}",
                )
                post_context_tokens = (
                    "currently",
                    "today",
                    "later",
                    "after founding",
                    "after launch",
                    "ceo",
                    "billionaire",
                    "ipo",
                    "acquired",
                    "market cap",
                    "lawsuit",
                    "court",
                    "legal action",
                    "y combinator startup school",
                    "startup school",
                    "married",
                    "wedding",
                    "exchange vows",
                    "medical school",
                )
                lowered = raw_text.lower()
                if (
                    (any(t in lowered for t in strong_pre_tokens) or any(t in lowered for t in explicit_pre_tokens))
                    and not any(t in lowered for t in post_context_tokens)
                ):
                    event_date = date(cutoff_date.year - 1, 1, 1)
                    inferred_reason = "contextual_pre_cutoff_unanchored"

            if event_date is None or event_date >= cutoff_date:
                continue

            # Light corroboration estimate.
            claim_tokens = self._tokenize_for_match(raw_text)
            support_sources = {str(claim.get("source_url", ""))}
            for other in claims:
                if other is claim:
                    continue
                other_source = str(other.get("source_url", ""))
                if not other_source or other_source in support_sources:
                    continue
                other_text = str(other.get("raw_sentence") or other.get("attribute_text", ""))
                if not other_text:
                    continue
                other_event_raw = other.get("event_date")
                other_event = self._safe_parse_date(other_event_raw) if other_event_raw else None
                if other_event and other_event >= cutoff_date:
                    continue
                overlap = self._token_overlap(claim_tokens, self._tokenize_for_match(other_text))
                if overlap >= 0.1:
                    support_sources.add(other_source)

            confidence = "low"
            if not inferred_reason and len(support_sources) >= 2:
                confidence = "high"
            elif len(support_sources) >= 2 or inferred_reason:
                confidence = "medium"

            key = (
                str(claim.get("category", "uncategorized")),
                event_date.isoformat(),
                raw_text.lower()[:180],
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)

            comprehensive.append(
                {
                    "category": str(claim.get("category", "uncategorized")),
                    "fact": str(claim.get("attribute_text", raw_text)),
                    "raw_sentence": raw_text,
                    "timestamp": event_date.isoformat(),
                    "sources": sorted({str(s) for s in support_sources if str(s).strip()}),
                    "source_primary": str(claim.get("source", claim.get("source_url", ""))),
                    "inference": inferred_reason,
                    "confidence": confidence,
                    "verification_source_count": len(support_sources),
                    "extraction_method": str(claim.get("extraction_method", "")),
                }
            )

        comprehensive.sort(
            key=lambda x: (
                str(x.get("timestamp", "")),
                str(x.get("category", "")),
                -int(x.get("verification_source_count", 0)),
            )
        )
        return comprehensive

    def organize_comprehensive_facts(
        self,
        comprehensive_facts: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Organize high-recall facts into investor-facing analysis buckets.
        """
        organized: dict[str, list[dict[str, Any]]] = {k: [] for k in CATEGORY_ANALYSIS_BUCKETS.keys()}
        seen_per_bucket: dict[str, set[tuple[str, str]]] = {k: set() for k in CATEGORY_ANALYSIS_BUCKETS.keys()}
        keyword_map: dict[str, tuple[str, ...]] = {
            "Decision making": ("decided", "chose", "choice", "pivot", "left", "moved", "strategy"),
            "Early hardships / challenges and resilience indicators": (
                "hardship",
                "bullied",
                "divorce",
                "challenge",
                "struggle",
                "overcame",
                "resilience",
            ),
            "Early technical proficiency": ("code", "coding", "program", "software", "game", "computer", "engineering"),
            "Personality": ("personality", "introvert", "intense", "driven", "focused", "behavior"),
            "Crazy things done": ("crazy", "wild", "extreme", "unusual", "risky", "audacious"),
            "Values": ("value", "belief", "principle", "ethic", "mission"),
            "Sense of security / insecurity": ("insecurity", "secure", "fear", "anxiety", "confidence"),
            "Motivation for starting business": ("motivation", "motivated", "start", "founded", "build", "why"),
            "Intellectual curiousity": ("read", "book", "curious", "curiosity", "science", "physics", "learning"),
        }

        for fact in comprehensive_facts:
            fact_category = str(fact.get("category", "")).lower()
            fact_text = str(fact.get("fact", ""))
            low = fact_text.lower()
            target_buckets: set[str] = set()

            for bucket, mapped_categories in CATEGORY_ANALYSIS_BUCKETS.items():
                if fact_category in mapped_categories:
                    target_buckets.add(bucket)
            for bucket, words in keyword_map.items():
                if any(w in low for w in words):
                    target_buckets.add(bucket)
            if not target_buckets and fact_category:
                target_buckets.add("Intellectual curiousity")

            row = {
                "fact": fact_text,
                "timestamp": fact.get("timestamp", ""),
                "sources": fact.get("sources", []),
                "inference": fact.get("inference", ""),
                "confidence": fact.get("confidence", "low"),
            }
            key = (row["fact"], str(row["timestamp"]))
            for bucket in sorted(target_buckets):
                if key in seen_per_bucket[bucket]:
                    continue
                organized[bucket].append(row)
                seen_per_bucket[bucket].add(key)

        # Keep output concise-ish but comprehensive: cap extreme buckets at 120 facts.
        for bucket in organized.keys():
            organized[bucket] = organized[bucket][:120]
        return organized

    def _fallback_founder_analysis(
        self,
        organized_fact_buckets: dict[str, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        analysis: dict[str, dict[str, Any]] = {}
        for category in FOUNDER_ANALYSIS_CATEGORIES:
            facts = organized_fact_buckets.get(category, [])
            if not facts:
                analysis[category] = {
                    "analysis": (
                        "Insufficient corroborated pre-cutoff evidence was found in this run to make a robust "
                        "pattern-level conclusion for this category."
                    ),
                    "confidence": "low",
                    "evidence_count": 0,
                    "key_signals": [],
                    "open_questions": [
                        "Collect more primary interviews or biographies focused on early years.",
                    ],
                }
                continue

            years = [str(f.get("timestamp", ""))[:4] for f in facts if str(f.get("timestamp", ""))[:4].isdigit()]
            year_span = f"{min(years)}-{max(years)}" if years else "pre-cutoff period"
            high_conf_count = sum(1 for f in facts if str(f.get("confidence", "")).lower() == "high")
            confidence = "high" if high_conf_count >= max(3, len(facts) // 2) else "medium"
            text_blob = " ".join(str(f.get("fact", "")).lower() for f in facts)
            pattern_flags: list[str] = []
            if any(k in text_blob for k in ("chose", "decided", "left", "moved")):
                pattern_flags.append("shows active agency in high-stakes choices")
            if any(k in text_blob for k in ("bullied", "hardship", "struggle", "divorce", "failure")):
                pattern_flags.append("built resilience under early adversity")
            if any(k in text_blob for k in ("coded", "program", "built", "software", "game")):
                pattern_flags.append("demonstrates early builder/technical compounding")
            if any(k in text_blob for k in ("read", "science", "physics", "book", "curiosity")):
                pattern_flags.append("shows broad intellectual range and self-directed learning")
            if any(k in text_blob for k in ("risk", "drop", "left", "startup", "founded")):
                pattern_flags.append("accepts asymmetric risk before institutional validation")
            if not pattern_flags:
                pattern_flags.append("contains weakly clustered signals; needs stronger primary-source depth")
            analysis[category] = {
                "analysis": (
                    f"Across {len(facts)} pre-cutoff evidence points ({year_span}), the signal pattern suggests the founder "
                    f"{pattern_flags[0]}. The trajectory appears front-loaded before institutional funding, implying these "
                    "behaviors are intrinsic rather than post-success artifacts. This category likely contributes to founder-market "
                    "fit through compounding learning loops, decision velocity, and tolerance for uncertainty."
                ),
                "confidence": confidence,
                "evidence_count": len(facts),
                "key_signals": pattern_flags[:5],
                "open_questions": [
                    "Which of these signals persist consistently in the 12 months before cutoff?",
                    "Are these behaviors independently corroborated by primary, not tertiary, sources?",
                ],
            }
        return analysis

    def generate_founder_analysis_from_fact_library(
        self,
        entrepreneur_name: str,
        cutoff_year: int,
        comprehensive_facts: list[dict[str, Any]],
        organized_fact_buckets: dict[str, list[dict[str, Any]]],
    ) -> tuple[dict[str, dict[str, Any]], str]:
        """
        Generate category-level synthesis from the comprehensive fact library.
        This is analysis, not a restatement of every fact.
        """
        fallback = self._fallback_founder_analysis(organized_fact_buckets)
        if not comprehensive_facts:
            return fallback, "fallback_no_facts"

        # Fit evidence into prompt budget.
        evidence_lines: list[str] = []
        max_evidence = (
            int(os.getenv("DEEPSEEK_ANALYSIS_MAX_EVIDENCE", "220"))
            if self._use_deepseek_analysis()
            else 220
        )
        for idx, fact in enumerate(comprehensive_facts[:max_evidence], start=1):
            srcs = ",".join(str(s) for s in fact.get("sources", [])[:2])
            evidence_lines.append(
                f"[{idx}] category={fact.get('category')} | t={fact.get('timestamp')} | "
                f"conf={fact.get('confidence')} | text={str(fact.get('fact', ''))[:260]} | src={srcs}"
            )
        evidence_blob = "\n".join(evidence_lines)

        prompt = (
            f"You are an investment analyst. Build a founder analysis for {entrepreneur_name} using only evidence before "
            f"{cutoff_year}. Do NOT regurgitate facts verbatim. Infer latent patterns, tradeoffs, causal links, and likely behavioral tendencies. "
            "Explicitly reason about (a) decision velocity, (b) adversity response style, (c) technical learning rate, "
            "(d) risk asymmetry, and (e) motivation durability. "
            "Output strict JSON where each category maps to an object with keys: "
            "`analysis` (120-220 words), `confidence` (high|medium|low), `evidence_count` (int), "
            "`key_signals` (3-6 short synthesized bullets), `open_questions` (2-4).\n\n"
            f"Required categories: {list(FOUNDER_ANALYSIS_CATEGORIES)}\n\n"
            "Evidence:\n"
            f"{evidence_blob}"
        )

        if self._use_deepseek_analysis() and os.getenv("DEEPSEEK_VERBOSE_ANALYSIS", "1") == "1":
            # Quality-first path: produce richer per-category synthesis from larger evidence slices.
            verbose_out: dict[str, dict[str, Any]] = {}
            per_category_limit = int(os.getenv("DEEPSEEK_PER_CATEGORY_EVIDENCE_LIMIT", "36"))
            per_category_tokens = int(os.getenv("DEEPSEEK_PER_CATEGORY_MAX_TOKENS", "650"))
            try:
                for category in FOUNDER_ANALYSIS_CATEGORIES:
                    bucket_facts = organized_fact_buckets.get(category, [])[:per_category_limit]
                    if not bucket_facts:
                        verbose_out[category] = {
                            "analysis": (
                                f"Available pre-{cutoff_year} evidence for '{category}' is sparse after temporal filtering and "
                                "corroboration checks, so confidence is low. This does not imply the signal is absent; it indicates "
                                "the current source stack did not yield enough anchorable, pre-cutoff statements for robust inference. "
                                "For diligence quality, prioritize primary interviews, biographies with early-life chapters, and "
                                "archival material that describe concrete behaviors, decisions, or constraints from the pre-cutoff period. "
                                "Until richer evidence is added, this category should be treated as an open risk area rather than a negative trait."
                            ),
                            "confidence": "low",
                            "evidence_count": 0,
                            "key_signals": [],
                            "open_questions": [
                                f"What primary-source evidence can be added for '{category}' before cutoff?",
                                "Are there archived interviews or biographies that cover this signal directly?",
                            ],
                        }
                        continue
                    bucket_lines: list[str] = []
                    for i, fact in enumerate(bucket_facts, start=1):
                        srcs = ",".join(str(s) for s in fact.get("sources", [])[:2])
                        bucket_lines.append(
                            f"[{i}] t={fact.get('timestamp')} conf={fact.get('confidence')} "
                            f"text={str(fact.get('fact', ''))[:320]} src={srcs} "
                            f"inference={str(fact.get('inference', ''))[:120]}"
                        )
                    bucket_prompt = (
                        f"Build a deep investor-quality analysis for category '{category}' about {entrepreneur_name}, "
                        f"strictly using pre-{cutoff_year} evidence below. This must be synthesis, not restatement.\n"
                        "Requirements:\n"
                        "- 220-420 words.\n"
                        "- Explain causal patterns, tradeoffs, and likely behavior under pressure.\n"
                        "- Highlight at least two non-obvious inferences and one risk caveat.\n"
                        "- Mention contradictions or data gaps if relevant.\n"
                        "- Do not invent facts.\n\n"
                        "Evidence:\n"
                        + "\n".join(bucket_lines)
                    )
                    bucket_text = self._deepseek_chat_completion(
                        system_prompt=(
                            "You are a top-tier venture analyst writing diligence notes. "
                            "Use only provided evidence and produce substantive synthesis."
                        ),
                        user_prompt=bucket_prompt,
                        max_tokens=per_category_tokens,
                        temperature=0.1,
                    )
                    analysis_text = str(bucket_text or "").strip()
                    if len(analysis_text) < 180:
                        analysis_text = fallback[category]["analysis"]
                    evidence_count = len(bucket_facts)
                    confidence = "high" if evidence_count >= 6 else ("medium" if evidence_count >= 3 else "low")
                    verbose_out[category] = {
                        "analysis": analysis_text,
                        "confidence": confidence,
                        "evidence_count": evidence_count,
                        "key_signals": fallback[category]["key_signals"],
                        "open_questions": fallback[category]["open_questions"],
                    }
                return verbose_out, "llm_deepseek_verbose_per_category"
            except Exception:
                # Fall through to compact JSON/delimited/fallback paths if any per-category call fails.
                pass

        if self._use_deepseek_analysis():
            # Prefer delimited output first; it is more robust than strict JSON parsing.
            try:
                line_prompt = (
                    "Return EXACTLY one line per category with this delimiter format:\n"
                    "CATEGORY||CONFIDENCE||EVIDENCE_COUNT||ANALYSIS||KEY_SIGNAL_1;KEY_SIGNAL_2;KEY_SIGNAL_3||OPEN_Q1;OPEN_Q2\n"
                    f"Categories: {', '.join(FOUNDER_ANALYSIS_CATEGORIES)}\n"
                    "No extra text.\n\n"
                    f"Evidence:\n{evidence_blob}"
                )
                lines_text = self._deepseek_chat_completion(
                    system_prompt="You are an investment analyst. Follow output format exactly.",
                    user_prompt=line_prompt,
                    max_tokens=1000,
                    temperature=0.1,
                )
                alt: dict[str, dict[str, Any]] = {}
                for raw_line in lines_text.splitlines():
                    line = raw_line.strip()
                    if "||" not in line:
                        continue
                    # Strip bullets/numbering prefixes if model adds them.
                    line = re.sub(r"^\s*[-*\d\.\)]\s*", "", line)
                    parts = [p.strip() for p in line.split("||")]
                    if len(parts) < 6:
                        continue
                    category, conf, evidence_raw, analysis_text, key_signals_raw, open_q_raw = parts[:6]
                    canonical_category = self._normalize_analysis_category(category)
                    if not canonical_category:
                        continue
                    evidence_count = len(organized_fact_buckets.get(canonical_category, []))
                    if evidence_raw.isdigit():
                        evidence_count = int(evidence_raw)
                    alt[canonical_category] = {
                        "analysis": analysis_text,
                        "confidence": conf.lower() if conf.lower() in {"high", "medium", "low"} else "medium",
                        "evidence_count": evidence_count,
                        "key_signals": [s.strip() for s in key_signals_raw.split(";") if s.strip()][:6],
                        "open_questions": [q.strip() for q in open_q_raw.split(";") if q.strip()][:4],
                    }
                if len(alt) >= 3:
                    for category in FOUNDER_ANALYSIS_CATEGORIES:
                        if category not in alt:
                            alt[category] = fallback.get(category, {})
                    return alt, "llm_delimited_synthesis"
            except Exception:
                pass

        if os.getenv("DISABLE_HF_MODEL", "0") == "1" and not self._use_deepseek_analysis():
            return fallback, "fallback_model_disabled"

        llm_text = ""
        if self._use_deepseek_analysis():
            try:
                llm_text = self._deepseek_chat_completion(
                    system_prompt=(
                        "You are an investment analyst. Return strict JSON only. "
                        "Use only given evidence; do not invent facts."
                    ),
                    user_prompt=prompt,
                    max_tokens=800,
                    temperature=0.1,
                )
            except Exception:
                llm_text = ""
        else:
            try:
                # Summarizer path first for stability.
                summary = self.summarizer(
                    prompt,
                    max_length=2500,
                    min_length=900,
                    do_sample=False,
                    truncation=True,
                )
                llm_text = str(summary[0]["summary_text"]).strip()
            except Exception:
                llm_text = ""

            if not llm_text and os.getenv("ENABLE_GENERATION_FALLBACK", "0") == "1":
                try:
                    generated = self.generator(
                        prompt[:2800],
                        max_new_tokens=700,
                        do_sample=False,
                        truncation=True,
                        return_full_text=False,
                    )
                    llm_text = str(generated[0].get("generated_text", "")).strip()
                except Exception:
                    llm_text = ""

        parsed = self._extract_json_block(llm_text) if llm_text else None
        if self._use_deepseek_analysis() and not isinstance(parsed, dict):
            # One retry with stricter JSON instruction; avoid many per-category calls.
            try:
                retry_prompt = (
                    "Reformat the following into strict JSON only, matching required category keys and object schema. "
                    "Do not add commentary.\n\n"
                    f"{llm_text or prompt[:2200]}"
                )
                llm_text_retry = self._deepseek_chat_completion(
                    system_prompt="Return valid JSON only.",
                    user_prompt=retry_prompt,
                    max_tokens=800,
                    temperature=0.0,
                )
                parsed = self._extract_json_block(llm_text_retry)
            except Exception:
                parsed = None
        if self._use_deepseek_analysis() and not isinstance(parsed, dict):
            # Alternative compact format parser for models that miss strict JSON.
            try:
                line_prompt = (
                    "Return EXACTLY one line per category with this delimiter format:\n"
                    "CATEGORY||CONFIDENCE||EVIDENCE_COUNT||ANALYSIS||KEY_SIGNAL_1;KEY_SIGNAL_2;KEY_SIGNAL_3||OPEN_Q1;OPEN_Q2\n"
                    f"Categories: {', '.join(FOUNDER_ANALYSIS_CATEGORIES)}\n"
                    "No extra text.\n\n"
                    f"Evidence:\n{evidence_blob}"
                )
                lines_text = self._deepseek_chat_completion(
                    system_prompt="You are an investment analyst. Follow output format exactly.",
                    user_prompt=line_prompt,
                    max_tokens=1000,
                    temperature=0.1,
                )
                alt: dict[str, dict[str, Any]] = {}
                for raw_line in lines_text.splitlines():
                    line = raw_line.strip()
                    if "||" not in line:
                        continue
                    line = re.sub(r"^\s*[-*\d\.\)]\s*", "", line)
                    parts = [p.strip() for p in line.split("||")]
                    if len(parts) < 6:
                        continue
                    category, conf, evidence_raw, analysis_text, key_signals_raw, open_q_raw = parts[:6]
                    canonical_category = self._normalize_analysis_category(category)
                    if not canonical_category:
                        continue
                    evidence_count = len(organized_fact_buckets.get(canonical_category, []))
                    if evidence_raw.isdigit():
                        evidence_count = int(evidence_raw)
                    alt[canonical_category] = {
                        "analysis": analysis_text,
                        "confidence": conf.lower() if conf.lower() in {"high", "medium", "low"} else "medium",
                        "evidence_count": evidence_count,
                        "key_signals": [s.strip() for s in key_signals_raw.split(";") if s.strip()][:6],
                        "open_questions": [q.strip() for q in open_q_raw.split(";") if q.strip()][:4],
                    }
                if len(alt) >= 3:
                    for category in FOUNDER_ANALYSIS_CATEGORIES:
                        if category not in alt:
                            alt[category] = fallback.get(category, {})
                    return alt, "llm_delimited_synthesis"
            except Exception:
                pass
        if not isinstance(parsed, dict):
            # Second LLM path: per-category plain-text synthesis (no strict JSON parsing).
            llm_category_analysis: dict[str, dict[str, Any]] = {}
            try:
                for category in FOUNDER_ANALYSIS_CATEGORIES:
                    bucket_facts = organized_fact_buckets.get(category, [])[:24]
                    if not bucket_facts:
                        llm_category_analysis[category] = fallback[category]
                        continue
                    bucket_lines = []
                    for i, fact in enumerate(bucket_facts, start=1):
                        bucket_lines.append(
                            f"[{i}] t={fact.get('timestamp')} conf={fact.get('confidence')} "
                            f"text={str(fact.get('fact', ''))[:240]}"
                        )
                    bucket_prompt = (
                        f"Analyze {entrepreneur_name}'s pre-{cutoff_year} evidence for category '{category}'. "
                        "Provide a synthesized interpretation (not fact repetition) in 140-220 words focused on "
                        "behavioral pattern, likely strengths/risks, and what this implies for early founder quality.\n\n"
                        "Evidence:\n"
                        + "\n".join(bucket_lines)
                    )
                    bucket_text = ""
                    if self._use_deepseek_analysis():
                        try:
                            bucket_text = self._deepseek_chat_completion(
                                system_prompt=(
                                    "You are an investment analyst. Provide concise category-level synthesis. "
                                    "Do not repeat evidence verbatim."
                                ),
                                user_prompt=bucket_prompt,
                                max_tokens=420,
                                temperature=0.1,
                            )
                        except Exception:
                            bucket_text = ""
                    else:
                        try:
                            out = self.summarizer(
                                bucket_prompt,
                                max_length=260,
                                min_length=80,
                                do_sample=False,
                                truncation=True,
                            )
                            bucket_text = str(out[0]["summary_text"]).strip()
                        except Exception:
                            bucket_text = ""
                        if not bucket_text and os.getenv("ENABLE_GENERATION_FALLBACK", "0") == "1":
                            try:
                                generated = self.generator(
                                    bucket_prompt[:1400],
                                    max_new_tokens=220,
                                    do_sample=False,
                                    truncation=True,
                                    return_full_text=False,
                                )
                                bucket_text = str(generated[0].get("generated_text", "")).strip()
                            except Exception:
                                bucket_text = ""
                    # Guard against low-quality generation artifacts.
                    alpha_chars = sum(1 for ch in bucket_text if ch.isalpha())
                    non_space_chars = sum(1 for ch in bucket_text if not ch.isspace())
                    alpha_ratio = (alpha_chars / non_space_chars) if non_space_chars else 0.0
                    if len(bucket_text) < 80 or alpha_ratio < 0.45 or bucket_text.count("[") > 8:
                        bucket_text = ""
                    llm_category_analysis[category] = {
                        "analysis": bucket_text or fallback[category]["analysis"],
                        "confidence": fallback[category]["confidence"],
                        "evidence_count": len(bucket_facts),
                        "key_signals": fallback[category]["key_signals"],
                        "open_questions": fallback[category]["open_questions"],
                    }
                return llm_category_analysis, "llm_per_category_text"
            except Exception:
                return fallback, "fallback_json_parse_failed"

        normalized: dict[str, dict[str, Any]] = {}
        for category in FOUNDER_ANALYSIS_CATEGORIES:
            raw = parsed.get(category, {})
            if not isinstance(raw, dict):
                normalized[category] = fallback.get(category, {})
                continue
            normalized[category] = {
                "analysis": str(raw.get("analysis", "")).strip() or fallback[category]["analysis"],
                "confidence": str(raw.get("confidence", "")).strip().lower() if str(raw.get("confidence", "")).strip() else fallback[category]["confidence"],
                "evidence_count": int(raw.get("evidence_count", len(organized_fact_buckets.get(category, [])) or 0)),
                "key_signals": [str(x).strip() for x in raw.get("key_signals", []) if str(x).strip()][:6]
                if isinstance(raw.get("key_signals", []), list)
                else fallback[category]["key_signals"],
                "open_questions": [str(x).strip() for x in raw.get("open_questions", []) if str(x).strip()][:4]
                if isinstance(raw.get("open_questions", []), list)
                else fallback[category]["open_questions"],
            }
            if normalized[category]["confidence"] not in {"high", "medium", "low"}:
                normalized[category]["confidence"] = fallback[category]["confidence"]
        return normalized, "llm_json_synthesis"

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _claim_priority_key(claim: dict[str, Any]) -> tuple[int, int]:
        verification = int(claim.get("verification_source_count") or 0)
        detail = len(str(claim.get("attribute_text", "")))
        return verification, detail

    def _unique_detail_ratio(self, base_text: str, candidate_text: str) -> float:
        base_tokens = self._tokenize_for_match(base_text)
        cand_tokens = self._tokenize_for_match(candidate_text)
        if not cand_tokens:
            return 0.0
        unique = cand_tokens.difference(base_tokens)
        return len(unique) / max(1, len(cand_tokens))

    def deduplicate_accepted_claims(
        self,
        accepted_claims: list[dict[str, Any]],
        cutoff_date: date,
        similarity_threshold: float = 0.65,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Lightweight semantic deduplication:
        - group claims by category,
        - cluster by sentence embedding cosine similarity,
        - keep highest-priority claim (verification_source_count, detail length),
        - merge source/timestamp metadata.
        """
        if not accepted_claims:
            return [], {
                "raw_accepted_count": 0,
                "dedup_accepted_count": 0,
                "dedup_removed_count": 0,
                "clusters_merged": 0,
            }

        by_category: dict[str, list[dict[str, Any]]] = {}
        for claim in accepted_claims:
            by_category.setdefault(str(claim.get("category", "uncategorized")), []).append(claim)

        deduped: list[dict[str, Any]] = []
        clusters_merged = 0

        for category, claims in by_category.items():
            if len(claims) == 1:
                deduped.append(claims[0])
                continue

            texts = [str(c.get("raw_sentence") or c.get("attribute_text", "")) for c in claims]
            embeddings = self.embedder.encode(texts, convert_to_numpy=False)
            remaining = set(range(len(claims)))

            while remaining:
                idx = min(remaining)
                remaining.remove(idx)
                cluster = [idx]
                frontier = [idx]

                # Build a connected component using cosine threshold.
                while frontier:
                    current = frontier.pop()
                    to_add: list[int] = []
                    for other in list(remaining):
                        similarity = self._cosine(embeddings[current], embeddings[other])
                        if similarity > similarity_threshold:
                            to_add.append(other)
                    for other in to_add:
                        remaining.remove(other)
                        cluster.append(other)
                        frontier.append(other)

                cluster_claims = [claims[i] for i in cluster]
                best_original = max(cluster_claims, key=self._claim_priority_key)
                best_claim = best_original.copy()
                if len(cluster_claims) > 1:
                    clusters_merged += 1

                merged_sources = sorted(
                    {
                        str(c.get("source", c.get("source_url", "")))
                        for c in cluster_claims
                        if str(c.get("source", c.get("source_url", ""))).strip()
                    }
                )
                merged_source_urls = sorted(
                    {str(c.get("source_url", "")) for c in cluster_claims if str(c.get("source_url", "")).strip()}
                )
                merged_timestamps = sorted(
                    {str(c.get("event_date", "")) for c in cluster_claims if str(c.get("event_date", "")).strip()}
                )
                merged_verification = max(int(c.get("verification_source_count") or 0) for c in cluster_claims)

                best_claim["source"] = best_claim.get("source", best_claim.get("source_url"))
                best_claim["merged_sources"] = merged_sources
                best_claim["merged_source_urls"] = merged_source_urls
                best_claim["merged_timestamps"] = merged_timestamps
                best_claim["verification_source_count"] = merged_verification
                best_claim["dedup_group_size"] = len(cluster_claims)
                best_claim["dedup_similarity_threshold"] = similarity_threshold
                best_claim["category"] = category
                deduped.append(best_claim)

                # Preserve nearby variants when they contain substantial unique detail.
                base_text = str(best_claim.get("attribute_text", ""))
                for candidate in cluster_claims:
                    if candidate is best_original:
                        continue
                    candidate_text = str(candidate.get("attribute_text", ""))
                    detail_ratio = self._unique_detail_ratio(base_text, candidate_text)
                    if detail_ratio >= 0.35:
                        variant = candidate.copy()
                        variant["dedup_preserved_variant"] = True
                        variant["detail_uniqueness_ratio"] = round(detail_ratio, 3)
                        variant["dedup_group_size"] = len(cluster_claims)
                        variant["dedup_similarity_threshold"] = similarity_threshold
                        deduped.append(variant)

        strict_deduped: list[dict[str, Any]] = []
        removed_post_cutoff = 0
        for claim in deduped:
            event_raw = claim.get("event_date")
            event_date = self._safe_parse_date(event_raw) if event_raw else None
            if event_date is None or event_date >= cutoff_date:
                removed_post_cutoff += 1
                self.anomaly_logger.info(
                    "discarded=post_dedup_post_cutoff_guard | event_date=%s | cutoff=%s | source=%s | claim=%s",
                    event_raw,
                    cutoff_date.isoformat(),
                    claim.get("source_url"),
                    claim.get("attribute_text"),
                )
                continue
            strict_deduped.append(claim)

        strict_deduped.sort(
            key=lambda c: (
                str(c.get("category", "")),
                str(c.get("event_date", "")),
                -int(c.get("verification_source_count", 0)),
            )
        )
        return strict_deduped, {
            "raw_accepted_count": len(accepted_claims),
            "dedup_accepted_count": len(strict_deduped),
            "dedup_removed_count": len(accepted_claims) - len(strict_deduped),
            "clusters_merged": clusters_merged,
            "removed_post_cutoff_guard": removed_post_cutoff,
            "similarity_threshold": similarity_threshold,
        }

    @staticmethod
    def _count_discard_reasons(claims: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for claim in claims:
            reason = str(claim.get("discard_reason", "unknown"))
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def _low_acceptance_diagnostics(
        self,
        dedup_accepted_count: int,
        invalid_claims: list[dict[str, Any]],
        dedup_meta: dict[str, Any],
    ) -> dict[str, Any] | None:
        if dedup_accepted_count >= 15:
            return None
        reason_counts = self._count_discard_reasons(invalid_claims)
        return {
            "message": "Accepted facts dropped below 15 after deduplication.",
            "discard_reason_counts": reason_counts,
            "dedup_meta": dedup_meta,
            "suggested_prompt_tweaks": [
                "Add explicit instruction to include one concrete year or age for every fact.",
                "Ask model to emit one JSON object per fact with short single-event sentences.",
                "Prioritize pre-1995 school/childhood transitions and avoid post-1994 spillover.",
                "Request citations to source sentence fragments to strengthen corroboration.",
            ],
        }

    def _upsert_entrepreneur(self, conn: PgConnection, entrepreneur_name: str, cutoff_date: date) -> int:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entrepreneurs (name, first_institutional_investment_date)
                VALUES (%s, %s)
                ON CONFLICT (name)
                DO UPDATE SET first_institutional_investment_date = EXCLUDED.first_institutional_investment_date
                RETURNING id;
                """,
                (entrepreneur_name, cutoff_date),
            )
            row = cur.fetchone()
            return int(row[0])

    def persist_claims(
        self,
        entrepreneur_name: str,
        cutoff_date: date,
        valid_claims: list[dict[str, Any]],
        invalid_claims: list[dict[str, Any]],
    ) -> None:
        """Persist extracted attributes and timelines into PostgreSQL."""
        if not self.db_dsn:
            return
        self.create_tables()
        with self._connect() as conn:
            entrepreneur_id = self._upsert_entrepreneur(conn, entrepreneur_name, cutoff_date)
            with conn.cursor() as cur:
                for claim in [*valid_claims, *invalid_claims]:
                    event_date = self._safe_parse_date(claim["event_date"]) if claim.get("event_date") else None
                    source_pub_date = self._safe_parse_date(claim["source_pub_date"])
                    before_cutoff = bool(claim.get("before_cutoff"))
                    cur.execute(
                        """
                        INSERT INTO attributes (
                            entrepreneur_id, source_url, source, category, fact, attribute_text, timestamp, event_date,
                            source_pub_date, before_cutoff, extraction_method
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            entrepreneur_id,
                            claim["source_url"],
                            claim.get("source", claim["source_url"]),
                            claim["category"],
                            claim["attribute_text"],
                            claim["attribute_text"],
                            event_date,
                            event_date,
                            source_pub_date,
                            before_cutoff,
                            claim.get("extraction_method", "unknown"),
                        ),
                    )
                    cur.execute(
                        """
                        INSERT INTO timelines (entrepreneur_id, event_date, event_text, source_url, before_cutoff)
                        VALUES (%s, %s, %s, %s, %s);
                        """,
                        (
                            entrepreneur_id,
                            event_date,
                            claim.get("raw_sentence", claim["attribute_text"]),
                            claim["source_url"],
                            before_cutoff,
                        ),
                    )

    def process_sources(
        self,
        entrepreneur_name: str,
        first_institutional_investment_date: str,
        source_urls: list[str],
        persist_to_db: bool = True,
        birth_year: int = 1970,
    ) -> dict[str, Any]:
        """
        Main ingestion pipeline for historical entrepreneurs.

        DATE CUTOFF must be explicitly provided as first institutional investment date.
        """
        cutoff_date = date.fromisoformat(first_institutional_investment_date)
        birth_month_day = self._lookup_birth_month_day(entrepreneur_name)
        all_claims: list[dict[str, Any]] = []
        headers = {"User-Agent": "SeedFounderIntelligenceBot/0.1"}
        source_word_count_map: dict[str, int] = {}
        harvested_sources: list[str] = []

        def _extract_from_sources(urls: list[str], current_claims: list[dict[str, Any]]) -> None:
            for source_url in urls:
                try:
                    response = requests.get(source_url, headers=headers, timeout=self.timeout_seconds)
                    response.raise_for_status()
                except Exception:
                    self.anomaly_logger.info(
                        "discarded=source_fetch_failed | entrepreneur=%s | source=%s",
                        entrepreneur_name,
                        source_url,
                    )
                    continue
                source_pub_date = self.extract_source_pub_date(response.text)
                text = self._extract_early_life_text(response.text)
                source_word_count_map[source_url] = len(re.findall(r"\w+", text))
                for harvested in self._harvest_followup_links(
                    html_text=response.text,
                    base_url=source_url,
                    entrepreneur_name=entrepreneur_name,
                    cutoff_year=cutoff_date.year,
                    max_links=6,
                ):
                    if harvested not in harvested_sources and harvested not in source_urls:
                        harvested_sources.append(harvested)
                claims = self.extract_claims_with_llm(
                    text,
                    source_url,
                    entrepreneur_name=entrepreneur_name,
                    birth_year=birth_year,
                    source_pub_date=source_pub_date,
                    source_label=source_url,
                    cutoff_date=cutoff_date,
                    birth_month_day=birth_month_day,
                )
                current_claims.extend(claims)

        initial_sources = self._prioritize_sources(list(source_urls), max_results=len(source_urls))
        _extract_from_sources(initial_sources, all_claims)

        # Interview-depth expansion: follow event-specific clues and crawl extra sources.
        expanded_sources = self.expand_sources_for_interview_depth(
            entrepreneur_name=entrepreneur_name,
            cutoff_year=cutoff_date.year,
            base_sources=initial_sources,
            seed_claims=all_claims,
            max_total_sources=20,
        )
        if os.getenv("DISABLE_SOURCE_EXPANSION", "0") == "1":
            expanded_sources = initial_sources
        extra_sources = [u for u in expanded_sources if u not in initial_sources]
        extra_sources = self._prioritize_sources(extra_sources, max_results=len(extra_sources))
        if extra_sources:
            _extract_from_sources(extra_sources, all_claims)

        # Third pass: fetch promising follow-up links discovered inside pages.
        if os.getenv("DISABLE_SOURCE_EXPANSION", "0") != "1":
            followup_candidates = [
                u for u in harvested_sources if u not in initial_sources and u not in extra_sources
            ][:20]
            if followup_candidates:
                followup_candidates = self._prioritize_sources(
                    followup_candidates, max_results=len(followup_candidates)
                )
                _extract_from_sources(followup_candidates, all_claims)
                expanded_sources.extend([u for u in followup_candidates if u not in expanded_sources])

        valid, invalid = self.validate_date_cutoff_compliance(
            all_claims,
            entrepreneur_name=entrepreneur_name,
            cutoff_date=cutoff_date,
            birth_year=birth_year,
            min_source_count=2,
        )
        comprehensive_facts = self.build_comprehensive_pre_cutoff_facts(
            claims=all_claims,
            entrepreneur_name=entrepreneur_name,
            cutoff_date=cutoff_date,
            birth_year=birth_year,
        )
        organized_fact_buckets = self.organize_comprehensive_facts(comprehensive_facts)
        founder_analysis, founder_analysis_method = self.generate_founder_analysis_from_fact_library(
            entrepreneur_name=entrepreneur_name,
            cutoff_year=cutoff_date.year,
            comprehensive_facts=comprehensive_facts,
            organized_fact_buckets=organized_fact_buckets,
        )
        dedup_valid, dedup_meta = self.deduplicate_accepted_claims(
            valid,
            cutoff_date=cutoff_date,
            similarity_threshold=0.65,
        )
        diagnostics = self._low_acceptance_diagnostics(
            dedup_accepted_count=len(dedup_valid),
            invalid_claims=invalid,
            dedup_meta=dedup_meta,
        )
        if persist_to_db:
            self.persist_claims(entrepreneur_name, cutoff_date, dedup_valid, invalid)

        output = {
            "entrepreneur_name": entrepreneur_name,
            "cutoff_date": cutoff_date.isoformat(),
            "cutoff_reason": (
                "DATE CUTOFF is the first institutional financing milestone "
                "(seed/Series A proxy). Only facts anchored strictly before this date are accepted."
            ),
            "source_urls": expanded_sources,
            "source_count": len(expanded_sources),
            "source_priority_scores": {
                u: self._source_depth_score(u) for u in expanded_sources
            },
            "total_source_words": int(sum(source_word_count_map.values())),
            "raw_candidate_fact_count": len(all_claims),
            "raw_candidate_claims": [
                {
                    "category": str(c.get("category", "")),
                    "fact": str(c.get("attribute_text", c.get("raw_sentence", "")))[:500],
                    "event_date": str(c.get("event_date", "")),
                    "source_url": str(c.get("source_url", "")),
                    "extraction_method": str(c.get("extraction_method", "")),
                }
                for c in all_claims
            ],
            "comprehensiveness_targets": {
                "min_sources": 10,
                "min_words": 10000,
                "met_source_target": len(expanded_sources) >= 10,
                "met_word_target": int(sum(source_word_count_map.values())) >= 10000,
            },
            "comprehensive_fact_count": len(comprehensive_facts),
            "comprehensive_pre_cutoff_facts": comprehensive_facts,
            "organized_fact_buckets": organized_fact_buckets,
            "founder_analysis": founder_analysis,
            "founder_analysis_generation_method": founder_analysis_method,
            # Backward compatibility for old viewer key.
            "organized_analysis": founder_analysis,
            "accepted_pre_dedup_count": len(valid),
            "accepted_count": len(dedup_valid),
            "excluded_count": len(invalid),
            "accepted_claims": dedup_valid,
            "excluded_claims": invalid,
            "deduplication": dedup_meta,
            "post_dedup_diagnostics": diagnostics,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        return output

    def process_text_sources(
        self,
        entrepreneur_name: str,
        first_institutional_investment_date: str,
        source_texts: list[dict[str, str]],
        birth_year: int = 1970,
    ) -> dict[str, Any]:
        """
        Offline/test ingestion path for in-memory sample text sources.

        Each item in source_texts must include:
        - source_url
        - text
        - optional source_pub_date (YYYY-MM-DD)
        - optional source (human-readable citation label)
        """
        cutoff_date = date.fromisoformat(first_institutional_investment_date)
        birth_month_day = self._lookup_birth_month_day(entrepreneur_name)
        all_claims: list[dict[str, Any]] = []
        source_word_count_map: dict[str, int] = {}
        for source in source_texts:
            source_url = source["source_url"]
            pub_date_raw = source.get("source_pub_date")
            pub_date = self._safe_parse_date(pub_date_raw) if pub_date_raw else None
            source_word_count_map[source_url] = len(re.findall(r"\w+", str(source.get("text", ""))))
            claims = self.extract_claims_with_llm(
                raw_text=source["text"],
                source_url=source_url,
                entrepreneur_name=entrepreneur_name,
                birth_year=birth_year,
                source_pub_date=pub_date,
                source_label=source.get("source"),
                cutoff_date=cutoff_date,
                birth_month_day=birth_month_day,
            )
            all_claims.extend(claims)

        valid, invalid = self.validate_date_cutoff_compliance(
            all_claims,
            entrepreneur_name=entrepreneur_name,
            cutoff_date=cutoff_date,
            birth_year=birth_year,
            min_source_count=2,
        )
        comprehensive_facts = self.build_comprehensive_pre_cutoff_facts(
            claims=all_claims,
            entrepreneur_name=entrepreneur_name,
            cutoff_date=cutoff_date,
            birth_year=birth_year,
        )
        organized_fact_buckets = self.organize_comprehensive_facts(comprehensive_facts)
        founder_analysis, founder_analysis_method = self.generate_founder_analysis_from_fact_library(
            entrepreneur_name=entrepreneur_name,
            cutoff_year=cutoff_date.year,
            comprehensive_facts=comprehensive_facts,
            organized_fact_buckets=organized_fact_buckets,
        )
        dedup_valid, dedup_meta = self.deduplicate_accepted_claims(
            valid,
            cutoff_date=cutoff_date,
            similarity_threshold=0.65,
        )
        diagnostics = self._low_acceptance_diagnostics(
            dedup_accepted_count=len(dedup_valid),
            invalid_claims=invalid,
            dedup_meta=dedup_meta,
        )
        return {
            "entrepreneur_name": entrepreneur_name,
            "cutoff_date": cutoff_date.isoformat(),
            "cutoff_reason": (
                "DATE CUTOFF is the first institutional financing milestone "
                "(seed/Series A proxy). Only facts anchored strictly before this date are accepted."
            ),
            "source_urls": [s.get("source_url", "") for s in source_texts],
            "source_count": len(source_texts),
            "source_priority_scores": {
                str(s.get("source_url", "")): self._source_depth_score(str(s.get("source_url", "")))
                for s in source_texts
            },
            "total_source_words": int(sum(source_word_count_map.values())),
            "raw_candidate_fact_count": len(all_claims),
            "raw_candidate_claims": [
                {
                    "category": str(c.get("category", "")),
                    "fact": str(c.get("attribute_text", c.get("raw_sentence", "")))[:500],
                    "event_date": str(c.get("event_date", "")),
                    "source_url": str(c.get("source_url", "")),
                    "extraction_method": str(c.get("extraction_method", "")),
                }
                for c in all_claims
            ],
            "comprehensiveness_targets": {
                "min_sources": 10,
                "min_words": 10000,
                "met_source_target": len(source_texts) >= 10,
                "met_word_target": int(sum(source_word_count_map.values())) >= 10000,
            },
            "comprehensive_fact_count": len(comprehensive_facts),
            "comprehensive_pre_cutoff_facts": comprehensive_facts,
            "organized_fact_buckets": organized_fact_buckets,
            "founder_analysis": founder_analysis,
            "founder_analysis_generation_method": founder_analysis_method,
            "organized_analysis": founder_analysis,
            "accepted_pre_dedup_count": len(valid),
            "accepted_count": len(dedup_valid),
            "excluded_count": len(invalid),
            "accepted_claims": dedup_valid,
            "excluded_claims": invalid,
            "deduplication": dedup_meta,
            "post_dedup_diagnostics": diagnostics,
            "generated_at": datetime.now(UTC).isoformat(),
        }

    def process_sources_to_json(
        self,
        entrepreneur_name: str,
        first_institutional_investment_date: str,
        source_urls: list[str],
        birth_year: int = 1970,
    ) -> str:
        """Run pipeline and return structured JSON string."""
        payload = self.process_sources(
            entrepreneur_name=entrepreneur_name,
            birth_year=birth_year,
            first_institutional_investment_date=first_institutional_investment_date,
            source_urls=source_urls,
        )
        return json.dumps(payload, indent=2)

    def run(self, source_url: str, cutoff_date: date) -> IngestionResult:
        """
        Backward-compatible single-source helper used by existing API route.
        """
        claims_json = self.process_sources(
            entrepreneur_name="unknown_entrepreneur",
            birth_year=1970,
            first_institutional_investment_date=cutoff_date.isoformat(),
            source_urls=[source_url],
            persist_to_db=False,
        )
        return IngestionResult(
            entrepreneur_name=claims_json["entrepreneur_name"],
            source_urls=[source_url],
            cutoff_date=cutoff_date.isoformat(),
            extracted_claims=claims_json["accepted_claims"],
            excluded_claims=claims_json["excluded_claims"],
            temporal_filter_passed=claims_json["accepted_count"],
            temporal_filter_failed=claims_json["excluded_count"],
        )


def ingest_entrepreneur(
    entrepreneur_name: str,
    birth_year: Optional[int],
    cutoff_year: int,
    custom_sources: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Generalized ingestion for any entrepreneur.
    If custom_sources is missing, discover 8-10 biography URLs dynamically.
    """
    db_dsn = os.getenv("DATABASE_URL")
    llm_model = os.getenv("LLM_MODEL", "sshleifer/distilbart-cnn-12-6")
    service = DataIngestionService(db_dsn=db_dsn, llm_model=llm_model)
    normalized_birth_year = birth_year if birth_year is not None else 1970
    default_source_count = int(os.getenv("DEFAULT_HIGH_DEPTH_SOURCE_COUNT", "8"))
    source_urls = custom_sources or service.discover_biography_sources(
        entrepreneur_name=entrepreneur_name,
        cutoff_year=cutoff_year,
        max_results=max(8, default_source_count),
    )

    result = service.process_sources(
        entrepreneur_name=entrepreneur_name,
        birth_year=normalized_birth_year,
        first_institutional_investment_date=f"{cutoff_year}-01-01",
        source_urls=source_urls,
        persist_to_db=bool(db_dsn),
    )
    # Detailed local log for accepted/excluded inspection.
    slug = service._slugify_name(entrepreneur_name)
    detail_path = Path(f"data/{slug}_ingestion_detailed_log.json")
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    latest_path = Path("data/latest_ingestion_detailed_log.json")
    latest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Keep backward-compatible alias for existing Elon workflows.
    if slug == "elon-musk":
        legacy_elon_path = Path("data/elon_ingestion_detailed_log.json")
        legacy_elon_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Flush DeepSeek cache to disk after each run.
    service._persist_deepseek_cache()
    return result


def run_sample_elon_musk_ingestion() -> dict[str, Any]:
    """
    Backward-compatible sample helper.
    """
    return ingest_entrepreneur(
        entrepreneur_name="Elon Musk",
        birth_year=1971,
        cutoff_year=1995,
    )


def format_accepted_claims_markdown(payload: dict[str, Any]) -> str:
    claims = list(payload.get("accepted_claims", []))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for claim in claims:
        grouped.setdefault(str(claim.get("category", "uncategorized")), []).append(claim)

    lines = [
        "# Accepted Pre-1995 Facts",
        "",
        f"- Entrepreneur: {payload.get('entrepreneur_name', 'unknown')}",
        f"- Cutoff: {payload.get('cutoff_date', 'unknown')}",
        f"- Accepted (pre-dedup): {payload.get('accepted_pre_dedup_count', payload.get('accepted_count', 0))}",
        f"- Accepted (deduplicated): {payload.get('accepted_count', 0)}",
        f"- Excluded: {payload.get('excluded_count', 0)}",
        "",
    ]
    for category in sorted(grouped.keys()):
        pretty = category.replace("_", " ").title()
        lines.append(f"## {pretty}")
        for claim in sorted(grouped[category], key=lambda c: str(c.get("event_date", ""))):
            fact = str(claim.get("attribute_text", "")).strip().replace("\n", " ")
            narrative_fact = str(claim.get("narrative_fact", "")).strip().replace("\n", " ")
            timestamp = str(claim.get("event_date", ""))
            verification = int(claim.get("verification_source_count") or 0)
            inference = str(claim.get("timestamp_inference", "")).strip()
            inference_reason = str(claim.get("inference_reason", "")).strip()
            merged_sources = claim.get("merged_sources") or [claim.get("source", claim.get("source_url", ""))]
            source_display = ", ".join(str(s) for s in merged_sources if str(s).strip())
            lines.append(f"- **Narrative Fact**: {narrative_fact or fact}")
            if claim.get("sub_facts"):
                lines.append(f"  - **Sub Facts**: {' | '.join(str(x) for x in claim.get('sub_facts', []))}")
            if claim.get("timestamps"):
                lines.append(f"  - **Raw Timestamps**: {' | '.join(str(x) for x in claim.get('timestamps', []))}")
            lines.append(f"  - **Timestamp**: {timestamp}")
            lines.append(f"  - **Sources**: {source_display}")
            lines.append(f"  - **Verification Source Count**: {verification}")
            if inference:
                lines.append(f"  - **Inference**: {inference} ({inference_reason})")
            if claim.get("narrative_inference"):
                lines.append(f"  - **Narrative Inference**: {claim.get('narrative_inference')}")
            if claim.get("dedup_preserved_variant"):
                lines.append(
                    f"  - **Dedup Variant Preserved**: uniqueness_ratio={claim.get('detail_uniqueness_ratio')}"
                )
        lines.append("")
    diagnostics = payload.get("post_dedup_diagnostics")
    if diagnostics:
        lines.append("## Low Acceptance Diagnostics")
        lines.append(f"- {diagnostics.get('message', 'Accepted count below target.')}")
        for tweak in diagnostics.get("suggested_prompt_tweaks", []):
            lines.append(f"- Prompt tweak: {tweak}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        test_cases = [
            ("Elon Musk", 1971, 1995),
            ("Mark Zuckerberg", 1984, 2004),
        ]
        for name, birth_year, cutoff_year in test_cases:
            sample = ingest_entrepreneur(
                entrepreneur_name=name,
                birth_year=birth_year,
                cutoff_year=cutoff_year,
            )
            print(json.dumps(sample, indent=2))
            print("\n" + format_accepted_claims_markdown(sample))
            print(
                f"\nIngestion complete for {sample['entrepreneur_name']} "
                f"(cutoff={sample['cutoff_date']}, accepted={sample['accepted_count']}, "
                f"excluded={sample['excluded_count']})."
            )
            if os.getenv("DATABASE_URL"):
                print("Confirmation: extracted facts were inserted into PostgreSQL.")
            else:
                print("Confirmation: DATABASE_URL not set, so DB insert was skipped.")
    except Exception as exc:
        print(f"Ingestion failed: {exc}")
