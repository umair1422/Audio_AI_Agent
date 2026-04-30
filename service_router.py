from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_ollama import OllamaConfig, ollama_generate_json


@dataclass(frozen=True)
class RoutedService:
    service_id: str
    service_name: str
    description: str
    category: str | None
    subcategory: str | None
    score: float
    match_type: str


@dataclass(frozen=True)
class RouteResult:
    query: str
    best: RoutedService | None
    reason: str | None
    top: list[RoutedService]


def _build_rerank_prompt(user_query: str, candidates: list[RoutedService]) -> str:
    compact = []
    for c in candidates:
        compact.append(
            {
                "service_id": c.service_id,
                "service_name": c.service_name,
                "description": (c.description or "")[:240],
                "category": c.category,
                "subcategory": c.subcategory,
            }
        )

    return (
        "You are a service router. Pick the single best service for the user.\n"
        "Return STRICT JSON only (no markdown, no extra text).\n"
        "If none is a good fit, return best_service_id as null.\n\n"
        f"User query: {user_query!r}\n\n"
        "Candidates (JSON array):\n"
        f"{compact}\n\n"
        "Output JSON schema:\n"
        "{\n"
        '  "best_service_id": string | null,\n'
        '  "reason": string\n'
        "}\n"
    )


def bm25_candidates(engine: Any, query: str, *, candidate_k: int = 12) -> list[RoutedService]:
    raw_results = engine.search(query, top_k=candidate_k)
    out: list[RoutedService] = []
    for r in raw_results:
        s = r.service
        out.append(
            RoutedService(
                service_id=s.service_id,
                service_name=s.service_name,
                description=s.description,
                category=s.category,
                subcategory=s.subcategory,
                score=float(r.score),
                match_type=r.match_type,
            )
        )
    return out


def route_service(
    *,
    engine: Any,
    query: str,
    top_k: int = 5,
    candidate_k: int = 12,
    use_llm_rerank: bool = True,
    ollama_config: OllamaConfig | None = None,
) -> RouteResult:
    q = (query or "").strip()
    if not q:
        return RouteResult(query=q, best=None, reason="Empty query.", top=[])

    candidates = bm25_candidates(engine, q, candidate_k=candidate_k)
    top = candidates[:top_k]

    if not candidates:
        return RouteResult(query=q, best=None, reason="No candidates.", top=[])

    # Fast path: BM25 top-1
    if not use_llm_rerank:
        return RouteResult(query=q, best=candidates[0], reason="BM25 top match (fast mode).", top=top)

    prompt = _build_rerank_prompt(q, candidates)
    llm = ollama_generate_json(prompt, config=ollama_config)
    best_id = llm.get("best_service_id")
    best_id = str(best_id) if best_id is not None else None
    reason = (llm.get("reason") or "").strip() or None

    best = None
    if best_id:
        for c in candidates:
            if c.service_id == best_id:
                best = c
                break

    return RouteResult(query=q, best=best, reason=reason, top=top)

