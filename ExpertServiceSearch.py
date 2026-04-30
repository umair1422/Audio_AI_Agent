from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from math import log, sqrt
from typing import Any, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rapidfuzz import fuzz, process


def _load_sentence_transformers() -> None:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except Exception:
        return None
    return SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
INTENT_THRESHOLD = 0.22
RESULT_THRESHOLD = 0.25
SEMANTIC_WEIGHT = 0.4
BM25_WEIGHT = 0.4
FUZZY_WEIGHT = 0.2

# When the query exactly matches a service name/synonym/category/subcategory (after normalization),
# we should always prefer that result. This is common in this dataset (e.g. "GP Consultation").
EXACT_NAME_MATCH_BOOST = 0.35
FACET_MATCH_BOOST = 0.10

# Minimal abbreviation normalization (keep small; embeddings reduce need for large maps).
ABBREVIATIONS = {
    "gp": "general practitioner",
    "sti": "sexually transmitted infection",
    "std": "sexually transmitted disease",
    "ac": "air conditioning",
    "aircon": "air conditioning",
}

QUERY_NORMALIZATION = {
    "fix": "repair",
    "repairs": "repair",
    "broken": "repair",
    "issue": "repair",
    "light": "lights",
    "lighting": "lights",
    "electrical": "electric",
    "electric": "electric",
    "ac": "air conditioning",
    "aircon": "air conditioning",
    "install": "installation",
    "instal": "installation",
    "leak": "leak",
    "leaks": "leak",
    "plumbing": "plumbing",
}

PHRASE_NORMALIZATION = {
    "skin fix": "skin treatment",
    "skin repair": "skin treatment",
    "face fix": "facial treatment",
    "hair fix": "hair treatment",
    "light fix": "lights repair",
    "ac fix": "air conditioning repair",
    "aircon fix": "air conditioning repair",
    "socket fix": "electrical repair",
    "power fix": "electrical repair",
    "electric fix": "electrical repair",
}

GENERIC_QUERY_TERMS = {
    "fix",
    "repair",
    "repairs",
    "issue",
    "issues",
    "broken",
    "problem",
    "problems",
    "service",
    "services",
    "help",
}

# Extra-generic tokens that cause cross-domain leakage in this dataset.
EXTRA_GENERIC_TOKENS = {
    "general",
    "consultation",
}

GREETING_TERMS = {
    "hello",
    "hi",
    "hey",
    "morning",
    "evening",
    "afternoon",
}

HELP_TERMS = {
    "help",
    "how",
    "use",
    "usage",
    "what",
    "can",
    "do",
    "search",
}

HELP_STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "this",
    "that",
    "it",
    "me",
    "my",
    "your",
    "you",
    "we",
    "i",
}


def normalize_phrase_query(query: str) -> str:
    normalized = query.lower()
    for phrase in sorted(PHRASE_NORMALIZATION, key=len, reverse=True):
        if phrase in normalized:
            return PHRASE_NORMALIZATION[phrase]
    return query


def remove_generic_terms(query: str) -> str:
    filtered_tokens = [token for token in tokenize(query) if token not in GENERIC_QUERY_TERMS]
    return " ".join(filtered_tokens)


@dataclass(frozen=True)
class Service:
    service_id: str
    service_name: str
    description: str
    category: str | None
    subcategory: str | None
    tags: tuple[str, ...]
    synonyms: tuple[str, ...]
    embed_text: str

    @property
    def display_name(self) -> str:
        return self.service_name

    @property
    def search_text(self) -> str:
        if self.embed_text:
            return self.embed_text
        return " ".join(
            [
                self.service_name,
                self.description,
                " ".join(self.tags),
                " ".join(self.synonyms),
            ]
        ).strip()


@dataclass(frozen=True)
class SearchResult:
    service: Service
    score: float
    match_type: str = "hybrid"


class SearchIntentDetector:
    def __init__(self, embedder: "SemanticEmbedder", service_examples: Sequence[str] | None = None) -> None:
        self.embedder = embedder
        service_search_examples = list(service_examples) if service_examples else []
        if not service_search_examples:
            service_search_examples = [
                "find a service",
                "looking for a service",
                "I need",
                "search for",
                "help me find",
                "service name",
                "service for",
            ]

        self.intent_examples = {
            "greeting": [
                "hello",
                "hi",
                "good morning",
                "hey there",
                "good evening",
            ],
            "service_search": service_search_examples,
            "help": [
                "how do I use this",
                "what can you do",
                "help me search services",
                "how to search",
            ],
        }
        self.intent_vectors = {intent: self.embedder.encode_batch(examples) for intent, examples in self.intent_examples.items()}

    def detect(self, query: str) -> tuple[str, float]:
        query_vector = self.embedder.encode(query)
        best_intent = "unknown"
        best_score = -1.0

        for intent, vectors in self.intent_vectors.items():
            scores = cosine_similarity(query_vector, vectors)
            score = max(scores) if scores else 0.0
            if score > best_score:
                best_intent = intent
                best_score = score

        if best_score < INTENT_THRESHOLD:
            return "unknown", best_score
        return best_intent, best_score


class SemanticEmbedder:
    def encode(self, text: str) -> Any:
        raise NotImplementedError

    def encode_batch(self, texts: Sequence[str]) -> list[Any]:
        raise NotImplementedError

    @property
    def is_dense(self) -> bool:
        return False


class SentenceTransformerEmbedder(SemanticEmbedder):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        sentence_transformers = _load_sentence_transformers()
        if sentence_transformers is None:
            raise RuntimeError(
                "sentence-transformers is not available or incompatible in this Python environment. "
                "Install a compatible version or use the keyword fallback mode."
            )
        self.model = sentence_transformers(model_name)

    def encode(self, text: str) -> Counter[str]:
        # Requires numpy under the hood; kept for compatibility but should be disabled by default
        # via ENABLE_SENTENCE_TRANSFORMERS.
        raise RuntimeError("SentenceTransformerEmbedder is disabled in pure-Python mode")

    def encode_batch(self, texts: Sequence[str]) -> list[Counter[str]]:
        raise RuntimeError("SentenceTransformerEmbedder is disabled in pure-Python mode")


class KeywordFallbackEmbedder(SemanticEmbedder):
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def fit(self, texts: Iterable[str]) -> None:
        vocabulary: dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)
        self.vocabulary = vocabulary

    def encode(self, text: str) -> Counter[str]:
        if not self.vocabulary:
            raise RuntimeError("Fallback embedder must be fitted before encoding")

        counts: Counter[str] = Counter()
        for token in tokenize(text):
            if token in self.vocabulary:
                counts[token] += 1
        return counts

    def encode_batch(self, texts: Sequence[str]) -> list[Counter[str]]:
        return [self.encode(text) for text in texts]


class HttpEmbeddingEmbedder(SemanticEmbedder):
    """
    Dense embeddings via HTTP (no numpy required).

    Configure with:
      - EMBEDDINGS_HTTP_URL (required)
      - EMBEDDINGS_BEARER_TOKEN (optional)

    Supported response formats:
      - {"embedding": [...]}  (single)
      - {"embeddings": [[...], ...]} (batch)
      - OpenAI-style: {"data":[{"embedding":[...]} ...]}
    """

    def __init__(self, url: str, bearer_token: str | None = None, timeout_s: float = 20.0) -> None:
        self.url = url
        self.bearer_token = bearer_token
        self.timeout_s = timeout_s

    @property
    def is_dense(self) -> bool:
        return True

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = Request(self.url, data=data, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"Embedding HTTP request failed: {exc}") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Embedding HTTP response was not valid JSON") from exc

    @staticmethod
    def _as_vector(value: Any) -> list[float]:
        if not isinstance(value, list) or not value:
            raise RuntimeError("Embedding was not a list")
        return [float(x) for x in value]

    def encode(self, text: str) -> list[float]:  # type: ignore[override]
        obj = self._post({"input": text})
        if "embedding" in obj:
            return self._as_vector(obj["embedding"])
        if "data" in obj and isinstance(obj["data"], list) and obj["data"]:
            return self._as_vector(obj["data"][0].get("embedding"))
        raise RuntimeError("Embedding response missing embedding")

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:  # type: ignore[override]
        obj = self._post({"input": list(texts)})
        if "embeddings" in obj and isinstance(obj["embeddings"], list):
            return [self._as_vector(v) for v in obj["embeddings"]]
        if "data" in obj and isinstance(obj["data"], list) and obj["data"]:
            return [self._as_vector(row.get("embedding")) for row in obj["data"]]
        # Fallback for APIs without batch: call per-text.
        return [self.encode(text) for text in texts]


class BM25Simple:
    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = [list(doc) for doc in corpus]
        self.doc_freq: Counter[str] = Counter()
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        for doc in self.corpus:
            for term in set(doc):
                self.doc_freq[term] += 1
        self.N = len(self.corpus)

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        # Standard BM25 idf with +1 to avoid negative idf
        return log(1.0 + (self.N - df + 0.5) / (df + 0.5)) if self.N else 0.0

    def get_scores(self, query_tokens: Sequence[str]) -> list[float]:
        if not self.corpus:
            return []
        q_terms = list(query_tokens)
        scores = [0.0 for _ in range(self.N)]
        for i, doc in enumerate(self.corpus):
            tf = Counter(doc)
            dl = self.doc_len[i]
            denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl)) if self.avgdl else 1.0
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                freq = tf[term]
                numer = freq * (self.k1 + 1.0)
                denom = freq + denom_norm
                score += self.idf(term) * (numer / denom)
            scores[i] = score
        return scores


class ServiceSearchEngine:
    def __init__(self, services: Sequence[Service], embedder: SemanticEmbedder) -> None:
        self.services = list(services)
        self.embedder = embedder
        self.search_texts = [service.search_text for service in self.services]
        self.service_vectors = self.embedder.encode_batch(self.search_texts)
        self.bm25 = BM25Simple([tokenize(text) for text in self.search_texts])
        self.intent_detector = SearchIntentDetector(self.embedder, service_examples=self._build_service_search_examples())

        vocabulary: set[str] = set()
        exact_name_keys: list[set[str]] = []
        facet_keys: list[set[str]] = []
        service_key_tokens: list[set[str]] = []
        service_terms: set[str] = set()
        for service in self.services:
            name_tokens = tokenize(service.service_name)
            desc_tokens = tokenize(service.description)
            search_tokens = tokenize(service.search_text)
            synonym_tokens = tokenize(" ".join(service.synonyms))
            vocab_tokens = set(name_tokens) | set(desc_tokens) | set(search_tokens) | set(synonym_tokens) | set(service.tags)

            vocabulary.update(vocab_tokens)

            # For intent heuristics: any overlap with these terms suggests it's a service search.
            service_terms.update(name_tokens)
            service_terms.update(synonym_tokens)
            if service.category:
                service_terms.update(tokenize(service.category))
            if service.subcategory:
                service_terms.update(tokenize(service.subcategory))

            # For ranking: exact match keys per service.
            name_keys = {normalize_key(service.service_name)}
            for syn in service.synonyms:
                if syn:
                    name_keys.add(normalize_key(syn))
            exact_name_keys.append(name_keys)

            facets: set[str] = set()
            if service.category:
                facets.add(normalize_key(service.category))
            if service.subcategory:
                facets.add(normalize_key(service.subcategory))
            facet_keys.append(facets)

            # Tokens used for lexical gating (name/synonyms/category/subcategory only).
            key_blob_parts = [service.service_name, *service.synonyms]
            if service.category:
                key_blob_parts.append(service.category)
            if service.subcategory:
                key_blob_parts.append(service.subcategory)
            service_key_tokens.append(set(tokenize(" ".join(key_blob_parts))))
        self.vocabulary = sorted(vocabulary)
        self.service_terms = service_terms
        self._exact_name_keys = exact_name_keys
        self._facet_keys = facet_keys
        self._service_key_tokens = service_key_tokens

    def _build_service_search_examples(self) -> list[str]:
        examples: list[str] = []
        seen: set[str] = set()
        for service in self.services:
            if service.service_name not in seen:
                examples.append(service.service_name)
                seen.add(service.service_name)
            for synonym in service.synonyms:
                if synonym and synonym not in seen:
                    examples.append(synonym)
                    seen.add(synonym)
            if service.category and service.category not in seen:
                examples.append(service.category)
                seen.add(service.category)
            if service.subcategory and service.subcategory not in seen:
                examples.append(service.subcategory)
                seen.add(service.subcategory)
            if len(examples) >= 200:
                break
        return examples

    def detect_intent(self, query: str) -> tuple[str, float]:
        # Fast, dataset-backed heuristics first.
        tokens = tokenize(query)
        if not tokens:
            return "unknown", 0.0

        if any(token in GREETING_TERMS for token in tokens):
            return "greeting", 1.0

        # If the user is asking meta-questions (help/usage), treat it as help.
        # We keep this rule strong because this dataset includes generic tokens like "use/search"
        # inside embed_text, which would otherwise incorrectly force service_search.
        token_set = set(tokens)
        helpish = any(token in HELP_TERMS for token in tokens)
        if helpish:
            trimmed = {t for t in tokens if t not in HELP_STOPWORDS}
            # e.g. "how to use this", "search for services"
            if trimmed.issubset(HELP_TERMS | {"services", "service"}):
                return "help", 1.0
            if "search" in token_set and ("service" in token_set or "services" in token_set) and len(tokens) <= 4:
                return "help", 1.0

        # If the query overlaps known service/category/subcategory tokens, it's a service search.
        if set(tokens) & self.service_terms:
            return "service_search", 1.0

        # Fallback to semantic intent detection.
        return self.intent_detector.detect(query)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        # For exact/facet matching we want to preserve the user's wording (e.g. "GP Consultation"),
        # so we compute those keys from the original tokens (after spelling correction).
        raw_tokens = tokenize(query)
        raw_corrected_tokens = [correct_spelling(token, self.vocabulary) for token in raw_tokens]
        raw_corrected_query = " ".join(raw_corrected_tokens)

        # For ranking we can expand abbreviations to improve semantic/keyword recall.
        abbrev_expanded = " ".join(ABBREVIATIONS.get(t, t) for t in raw_corrected_tokens)
        query_tokens = tokenize(abbrev_expanded)
        corrected_tokens = [correct_spelling(token, self.vocabulary) for token in query_tokens]
        corrected_query = " ".join(corrected_tokens)
        expanded_query = corrected_query

        # If embeddings are NOT enabled, keep the older normalization expansions
        # to help keyword-only matching. With embeddings, these maps can be minimized.
        if not self.embedder.is_dense:
            normalized_tokens = [QUERY_NORMALIZATION.get(token, token) for token in corrected_tokens]
            normalized_query = " ".join(normalized_tokens)
            phrase_normalized_query = normalize_phrase_query(corrected_query)

            if normalized_query and normalized_query != corrected_query:
                expanded_query = f"{expanded_query} {normalized_query}"
            if (
                phrase_normalized_query
                and phrase_normalized_query != corrected_query
                and phrase_normalized_query != normalized_query
            ):
                expanded_query = f"{expanded_query} {phrase_normalized_query}"

        query_vector = self.embedder.encode(expanded_query)
        semantic_scores = cosine_similarity(query_vector, self.service_vectors)

        bm25_query = remove_generic_terms(expanded_query)
        if not bm25_query:
            bm25_query = expanded_query

        bm25_scores = [float(x) for x in self.bm25.get_scores(tokenize(bm25_query))]
        bm25_max = max(bm25_scores) if bm25_scores else 0.0
        if bm25_max > 0:
            bm25_scores = [x / bm25_max for x in bm25_scores]

        fuzzy_query = bm25_query
        fuzzy_scores = [self._fuzzy_score(fuzzy_query, service) for service in self.services]

        # Exact match boost (service_name/synonyms) + smaller facet boost (category/subcategory).
        query_key = normalize_key(raw_corrected_query)
        exact_name_boosts = [EXACT_NAME_MATCH_BOOST if query_key in keys else 0.0 for keys in self._exact_name_keys]
        facet_boosts = [FACET_MATCH_BOOST if query_key in keys else 0.0 for keys in self._facet_keys]

        combined_scores = [
            semantic_scores[i] * SEMANTIC_WEIGHT
            + bm25_scores[i] * BM25_WEIGHT
            + fuzzy_scores[i] * FUZZY_WEIGHT
            + exact_name_boosts[i]
            + facet_boosts[i]
            for i in range(len(self.services))
        ]

        ranked_indices = sorted(range(len(combined_scores)), key=combined_scores.__getitem__, reverse=True)
        results: list[SearchResult] = []

        # Lexical gating: for "service-name-like" queries, require at least one meaningful token
        # overlap with service key fields (name/synonyms/category/subcategory). This prevents
        # cross-domain hits from generic words like "general".
        q_gate_tokens = set(tokenize(raw_corrected_query)) - GENERIC_QUERY_TERMS - EXTRA_GENERIC_TOKENS
        for index in ranked_indices[: top_k * 3]:
            score = float(combined_scores[index])
            if score < RESULT_THRESHOLD:
                continue
            if q_gate_tokens and not (q_gate_tokens & self._service_key_tokens[index]):
                continue
            semantic = float(semantic_scores[index])
            bm25 = float(bm25_scores[index]) if index < len(bm25_scores) else 0.0
            fuzzy = float(fuzzy_scores[index]) if index < len(fuzzy_scores) else 0.0
            if exact_name_boosts[index] > 0:
                match_type = "exact"
            elif facet_boosts[index] > 0:
                match_type = "facet"
            else:
                match_type = self._choose_match_type(semantic, bm25, fuzzy)
            results.append(
                SearchResult(
                    service=self.services[index],
                    score=score,
                    match_type=match_type,
                )
            )
            if len(results) >= top_k:
                break
        return results

    def _choose_match_type(self, semantic: float, bm25: float, fuzzy: float) -> str:
        # NOTE: `max((score, label), ...)` breaks ties by comparing the label string,
        # which can mislabel exact matches as "fuzzy" simply because "fuzzy" sorts
        # after "bm25"/"semantic". We handle ties deterministically instead.
        scores: dict[str, float] = {"semantic": semantic, "bm25": bm25, "fuzzy": fuzzy}
        best_score = max(scores.values())

        # Treat near-equal best scores as hybrid.
        eps = 1e-6
        tied = [label for label, value in scores.items() if abs(value - best_score) <= eps]
        if len(tied) > 1:
            return "hybrid"

        # Otherwise return the true argmax label (stable priority if something is weird).
        priority = ("bm25", "semantic", "fuzzy")
        for label in priority:
            if scores[label] == best_score:
                return label
        return "hybrid"

    def _fuzzy_score(self, query: str, service: Service) -> float:
        # Fuzzy matching should be constrained to "name-like" fields; using the full `search_text`
        # (which may include long `embed_text`) can create spurious matches for unrelated services.
        candidates: list[str] = [service.service_name, *list(service.synonyms)]
        if service.category:
            candidates.append(service.category)
        if service.subcategory:
            candidates.append(service.subcategory)

        # Gate: if the query shares zero tokens with the name/synonyms, downweight hard.
        # This prevents generic words like "general" from matching unrelated domains.
        q_tokens = set(tokenize(query))
        name_blob = " ".join([service.service_name, *service.synonyms])
        if q_tokens and not (q_tokens & set(tokenize(name_blob))):
            return 0.0

        best_score = 0.0
        for candidate in candidates:
            score = fuzz.token_set_ratio(query, candidate) / 100.0
            if score > best_score:
                best_score = score
        return best_score


def tokenize(text: str) -> list[str]:
    cleaned = "".join(character.lower() if character.isalnum() else " " for character in text)
    return [token for token in cleaned.split() if token]


def normalize_key(text: str) -> str:
    # Stronger normalization for exact matches: token-based, order-insensitive.
    tokens = tokenize(text)
    return " ".join(sorted(tokens))


def correct_spelling(word: str, vocabulary: Sequence[str], threshold: float = 0.75) -> str:
    if not word or word in vocabulary:
        return word
    match = process.extractOne(word, vocabulary, scorer=fuzz.ratio)
    if match is None:
        return word
    corrected, score, _ = match
    return corrected if score >= threshold * 100 else word


def _counter_dot(a: Counter[str], b: Counter[str]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return float(sum(value * b.get(term, 0) for term, value in a.items()))


def _counter_norm(a: Counter[str]) -> float:
    return sqrt(sum(float(v * v) for v in a.values()))

def _dense_dot(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    return float(sum(float(a[i]) * float(b[i]) for i in range(n)))


def _dense_norm(a: Sequence[float]) -> float:
    return sqrt(sum(float(x) * float(x) for x in a))


def cosine_similarity(vector: Any, matrix: Sequence[Any]) -> list[float]:
    # Sparse path (keyword fallback)
    if isinstance(vector, Counter):
        v_norm = _counter_norm(vector)
        if v_norm == 0.0:
            return [0.0 for _ in matrix]
        scores: list[float] = []
        for row in matrix:
            if not isinstance(row, Counter):
                scores.append(0.0)
                continue
            r_norm = _counter_norm(row)
            if r_norm == 0.0:
                scores.append(0.0)
                continue
            scores.append(_counter_dot(vector, row) / (v_norm * r_norm))
        return scores

    # Dense path (HTTP embeddings)
    if not isinstance(vector, list):
        return [0.0 for _ in matrix]
    v_norm = _dense_norm(vector)
    if v_norm == 0.0:
        return [0.0 for _ in matrix]
    scores: list[float] = []
    for row in matrix:
        if not isinstance(row, list):
            scores.append(0.0)
            continue
        r_norm = _dense_norm(row)
        if r_norm == 0.0:
            scores.append(0.0)
            continue
        scores.append(_dense_dot(vector, row) / (v_norm * r_norm))
    return scores


def load_services(json_path: Path) -> list[Service]:
    with json_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if not isinstance(raw_data, list):
        raise ValueError("JSON must contain a list of services")

    services: list[Service] = []
    for index, item in enumerate(raw_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Service at index {index} must be an object")

        service_id = item.get("id", index)
        service_name = str(item.get("service_name") or item.get("title") or "").strip()
        description = str(item.get("description") or "").strip()
        category = str(item.get("category") or "").strip() or None
        subcategory = str(item.get("subcategory") or "").strip() or None
        raw_tags = item.get("tags") or []
        raw_synonyms = item.get("synonyms") or []
        embed_text = str(item.get("embed_text") or "").strip()

        if not service_name:
            raise ValueError(f"Service at index {index} is missing 'service_name' or 'title'")
        if not isinstance(raw_tags, list):
            raise ValueError(f"Tags for service '{service_name}' must be a list")
        if not isinstance(raw_synonyms, list):
            raise ValueError(f"Synonyms for service '{service_name}' must be a list")

        tags = tuple(str(tag).strip() for tag in raw_tags if str(tag).strip())
        synonyms = tuple(str(syn).strip() for syn in raw_synonyms if str(syn).strip())
        services.append(
            Service(
                service_id=str(service_id),
                service_name=service_name,
                description=description,
                category=category,
                subcategory=subcategory,
                tags=tags,
                synonyms=synonyms,
                embed_text=embed_text,
            )
        )

    return services


def build_search_engine(json_path: Path, model_name: str = DEFAULT_MODEL_NAME) -> ServiceSearchEngine:
    services = load_services(json_path)
    if not services:
        raise ValueError("No services found in the JSON file")

    embeddings_http_url = os.getenv("EMBEDDINGS_HTTP_URL", "").strip()
    embeddings_bearer = os.getenv("EMBEDDINGS_BEARER_TOKEN", "").strip() or None

    if embeddings_http_url:
        embedder = HttpEmbeddingEmbedder(url=embeddings_http_url, bearer_token=embeddings_bearer)
    else:
        # Safety: some environments can segfault when initializing sentence-transformers.
        # Default to the keyword fallback unless explicitly enabled.
        enable_sentence_transformers = os.getenv("ENABLE_SENTENCE_TRANSFORMERS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

        if enable_sentence_transformers:
            try:
                embedder = SentenceTransformerEmbedder(model_name)
            except RuntimeError:
                embedder = KeywordFallbackEmbedder()
                embedder.fit(service.search_text for service in services)
        else:
            embedder = KeywordFallbackEmbedder()
            embedder.fit(service.search_text for service in services)

    return ServiceSearchEngine(services=services, embedder=embedder)


def print_results(query: str, engine: ServiceSearchEngine, top_k: int) -> None:
    intent, confidence = engine.detect_intent(query)
    print(f"Intent: {intent} (confidence={confidence:.3f})")

    if intent == "greeting":
        print("Hello! Ask for a service, for example: 'hair loss treatment' or 'book a consultation'.")
        return
    if intent == "help":
        print("Use a keyword, service name, or natural language request to find the top 5 relevant services.")
        return

    # Proceed to search for service_query and service_search intents

    query_tokens = tokenize(query)
    corrected_tokens = [correct_spelling(token, engine.vocabulary) for token in query_tokens]
    if corrected_tokens != query_tokens:
        print(f"Correction: {' '.join(query_tokens)} -> {' '.join(corrected_tokens)}")

    results = engine.search(query, top_k=top_k)
    if not results:
        print("No relevant service found.")
        return

    print("Top relevant services:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result.service.display_name} (score={result.score:.3f}, match={result.match_type})"
        )
        if result.service.category:
            print(f"   Category: {result.service.category} / {result.service.subcategory}")
        if result.service.description:
            print(f"   Description: {result.service.description}")
        if result.service.tags:
            print(f"   Tags: {', '.join(result.service.tags)}")
        if result.service.synonyms:
            print(f"   Synonyms: {', '.join(result.service.synonyms[:5])}{'...' if len(result.service.synonyms) > 5 else ''}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI search over ExpertServices JSON data")
    parser.add_argument("json_path", type=Path, help="Path to the service catalog JSON file")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = build_search_engine(args.json_path, model_name=args.model)
    if args.query:
        print_results(args.query, engine, top_k=args.top_k)
        return

    print("Interactive expert service search. Type a query or Ctrl+C to exit.")
    while True:
        try:
            query = input("Search query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not query:
            continue
        print_results(query, engine, top_k=args.top_k)


if __name__ == "__main__":
    main()
