from __future__ import annotations

from typing import Iterable, List

from backend.agents.base import AgentResponse
from .summarizer import (
    summarize_competitors,
    summarize_filings,
    summarize_funding,
    summarize_news,
)


def build_report(responses: Iterable[AgentResponse]) -> str:
    funding = summarize_funding(responses)
    competitors = summarize_competitors(responses)
    news = summarize_news(responses)
    filings = summarize_filings(responses)

    paragraphs: List[str] = []

    if funding["highlights"]:
        funding_intro = "Funding landscape: " + "; ".join(funding["highlights"][:3]) + "."
        paragraphs.append(funding_intro)
    if funding["top_investors"]:
        investors = ", ".join(funding["top_investors"])
        paragraphs.append(f"Active investors include {investors}.")

    if competitors["competitors"]:
        competitor_list = ", ".join(competitors["competitors"][:5])
        text = f"Competitive field: key players include {competitor_list}."
        if competitors["growth_notes"]:
            text += " Notable metrics: " + "; ".join(competitors["growth_notes"])
        paragraphs.append(text)

    if news["articles"]:
        news_items = "; ".join(news["articles"])
        paragraphs.append(f"Market signals: {news_items}.")

    if filings["filings"]:
        filing_items = "; ".join(filings["filings"])
        paragraphs.append(f"Recent Form D filings worth watching: {filing_items}.")

    if not paragraphs:
        return "No synthesised insights available."

    return "\n\n".join(paragraphs)

