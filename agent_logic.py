from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ExpertServiceSearch import GREETING_TERMS, HELP_TERMS, tokenize

from service_router import RouteResult, route_service


@dataclass(frozen=True)
class AgentReply:
    reply_text: str
    route: RouteResult | None = None


def build_agent_reply(*, text: str, engine: Any) -> AgentReply:
    """
    Lightweight dialog policy:
    - Greeting -> greet back immediately
    - Help / "I need help" -> ask what they need
    - Otherwise -> route to best service (BM25 + Ollama rerank)
    """
    raw = (text or "").strip()
    token_list = tokenize(raw)
    tokens = set(token_list)

    if not raw:
        return AgentReply(reply_text="I didn't catch that. Please say it again.")

    has_greeting = any(t in GREETING_TERMS for t in tokens)
    # If the user only greets ("hi", "hello"), greet back.
    # If they greet + ask something in the same utterance, continue to routing.
    if has_greeting:
        non_greeting_tokens = {t for t in tokens if t not in GREETING_TERMS}
        if len(non_greeting_tokens) == 0:
            return AgentReply(reply_text="Hello! How can I help you today?")

    helpish = any(t in HELP_TERMS for t in tokens) or ("need" in tokens and "help" in tokens)
    # If the user says "help" without specifying the domain, ask a follow-up.
    if helpish and len(tokens) <= 6 and not any(t for t in tokens if t not in (HELP_TERMS | GREETING_TERMS)):
        return AgentReply(
            reply_text="Sure — tell me what you need help with. For example: website development, SEO, or a mobile app."
        )

    routed = route_service(engine=engine, query=raw, candidate_k=12, top_k=5, use_llm_rerank=True)
    if routed.best:
        top = routed.best
        desc = (top.description or "") if top else ""
        desc = " ".join(desc.split())
        if len(desc) > 220:
            desc = desc[:220].rsplit(" ", 1)[0] + "…"
        category = top.category
        subcategory = top.subcategory

        domain_phrase = ""
        if has_greeting:
            domain_phrase = "Yes — "
        reply = f"{domain_phrase}we have a service for that. The best match is {top.service_name}."
        if category or subcategory:
            reply += f" Category: {category or ''} {('- ' + subcategory) if subcategory else ''}."
        if desc:
            reply += f" Description: {desc}"
    else:
        reply = "I couldn't find an exact match. Can you rephrase what you need in one short sentence?"
    return AgentReply(reply_text=reply, route=routed)

