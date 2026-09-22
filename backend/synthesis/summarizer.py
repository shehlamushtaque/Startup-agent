from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

from backend.agents.base import AgentResponse, Evidence


def _format_money(value: Optional[float]) -> Optional[str]:
    if value in (None, "", "nan"):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    suffixes = [
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ]
    for threshold, suffix in suffixes:
        if abs(amount) >= threshold:
            return f"${amount / threshold:.1f}{suffix}"
    return f"${amount:,.0f}"


def _normalize_investors(raw: Optional[Sequence[str]]) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw_list = [part.strip() for part in raw.split(",") if part.strip()]
        return raw_list
    return [str(item).strip() for item in raw if item]


def _extract_evidence(responses: Iterable[AgentResponse], agent_name: str) -> List[Evidence]:
    for response in responses:
        if response.agent_name == agent_name:
            return response.evidence
    return []


def summarize_funding(responses: Iterable[AgentResponse]) -> Dict[str, List[str]]:
    evidence = _extract_evidence(responses, "FundingIntelligenceAgent")
    funding_highlights: List[str] = []
    qa_highlights: List[str] = []
    investors: Counter[str] = Counter()

    for item in evidence:
        metadata = item.metadata or {}
        record = metadata if isinstance(metadata, dict) else {}
        investor_candidates = record.get("investor_names") or record.get("investors")
        for investor in _normalize_investors(investor_candidates):
            investors[investor] += 1

        source = (item.source or "").lower()
        if source == "companyfundingtool":
            company = record.get("name_cb") or record.get("normalized_name")
            funding = _format_money(record.get("funding_total_usd"))
            rounds = record.get("funding_round_count")
            latest = record.get("latest_investment_at")
            if not company or not funding:
                continue
            parts: List[str] = [f"{company}: {funding} raised"]
            if isinstance(rounds, (int, float)):
                if not math.isnan(rounds):
                    parts.append(f"across {int(rounds)} rounds")
            elif rounds:
                parts.append(f"across {rounds} rounds")
            if latest:
                parts.append(f"latest check {latest}")
            growth = record.get("growjo_growth_percent")
            if isinstance(growth, (int, float)) and not math.isnan(growth):
                parts.append(f"growth {growth}%")
            highlight = ", ".join(parts)
            if highlight not in funding_highlights:
                funding_highlights.append(highlight)
        elif source == "companyqaindextool":
            company = record.get("metadata", {}).get("company") if record.get("metadata") else record.get("company")
            snippet = item.content
            if company and snippet:
                qa_entry = f"{company}: {snippet}"
                if qa_entry not in qa_highlights:
                    qa_highlights.append(qa_entry)

    top_investors = [name for name, _ in investors.most_common(5)]
    highlights = funding_highlights or qa_highlights
    return {
        "highlights": highlights[:5],
        "top_investors": top_investors,
    }


def summarize_competitors(responses: Iterable[AgentResponse]) -> Dict[str, List[str]]:
    evidence = _extract_evidence(responses, "CompetitorGrowthAgent")
    competitors: List[str] = []
    growth_notes: List[str] = []

    for item in evidence:
        raw_title = item.title or ""
        clean_title = raw_title.replace("Growth snapshot:", "").strip()
        if clean_title and clean_title not in competitors:
            competitors.append(clean_title)
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        growth = metadata.get("growjo_growth_percent")
        revenue = metadata.get("growjo_estimated_revenue_usd")
        employees = metadata.get("growjo_employees")
        pieces = []
        if isinstance(growth, (int, float)) and math.isnan(growth):
            growth = None
        if isinstance(revenue, (int, float)) and math.isnan(revenue):
            revenue = None
        if isinstance(employees, (int, float)) and math.isnan(employees):
            employees = None
        if isinstance(growth, (int, float)) and not math.isnan(growth):
            pieces.append(f"growth {growth:.0f}%")
        if isinstance(revenue, (int, float)) and not math.isnan(revenue):
            money = _format_money(revenue)
            if money:
                pieces.append(f"revenue {money}")
        if isinstance(employees, (int, float)) and not math.isnan(employees):
            pieces.append(f"{int(employees):,} employees")
        if pieces:
            note = f"{clean_title}: " + ", ".join(pieces)
            if note not in growth_notes:
                growth_notes.append(note)

    return {
        "competitors": competitors[:10],
        "growth_notes": growth_notes[:5],
    }


def summarize_news(responses: Iterable[AgentResponse]) -> Dict[str, List[str]]:
    serps = _extract_evidence(responses, "SerpSearchAgent")
    web = _extract_evidence(responses, "WebResearchAgent")
    combined = serps + web
    articles: List[str] = []

    for item in combined:
        title = item.title or ""
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        source = metadata.get("source")
        link = item.source or metadata.get("link")
        if title and link:
            entry = f"{title} ({link})"
        elif title and source:
            entry = f"{title} - {source}"
        else:
            entry = ""
        if entry and entry not in articles:
            articles.append(entry)

    return {"articles": articles[:5]}


def summarize_filings(responses: Iterable[AgentResponse]) -> Dict[str, List[str]]:
    evidence = _extract_evidence(responses, "SecFilingsAgent")
    filings: List[str] = []
    for item in evidence:
        company = item.title or ""
        # Remove redundant CIK info from title if present
        if " (CIK" in company:
            company = company.split(" (CIK")[0]
        
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        content = item.content or ""
        
        # Build summary from content or metadata
        parts = [company]
        
        # Extract from content if available (already formatted by SecFilingsAgent)
        if content and content != "Form D filing":
            # Content already has formatted info like "Filed: 2025-01-15 | Amount raised: $5M | Investors: X, Y"
            # Parse it to make it more readable
            if "Filed:" in content:
                # Extract just the date part for cleaner display
                content_parts = content.split(" | ")
                formatted_parts = []
                for part in content_parts:
                    if part.startswith("Filed:"):
                        formatted_parts.append(part.replace("Filed:", "filed").strip())
                    elif part.startswith("Amount raised:") or part.startswith("Offering amount:"):
                        formatted_parts.append(part.replace("Amount raised:", "raised").replace("Offering amount:", "offering").strip())
                    elif part.startswith("Investors:"):
                        formatted_parts.append(part.strip())
                    else:
                        formatted_parts.append(part.strip())
                parts.extend(formatted_parts)
            else:
                parts.append(content)
        else:
            # Fallback to metadata extraction
            if metadata.get("filed_at"):
                parts.append(f"filed {metadata.get('filed_at')}")
            
            amount = metadata.get("amount_raised") or metadata.get("total_offering_amount") or metadata.get("amount_sold")
            if amount:
                formatted = _format_money(amount)
                if formatted:
                    parts.append(f"raised {formatted}")
            
            investors = metadata.get("investors")
            if investors:
                inv_list = _normalize_investors(investors)
                if inv_list:
                    inv_str = ", ".join(inv_list[:2])
                    if len(inv_list) > 2:
                        inv_str += f" and {len(inv_list) - 2} more"
                    parts.append(f"investors: {inv_str}")
        
        entry = " | ".join(parts)
        if entry and entry not in filings:
            filings.append(entry)
    
    return {"filings": filings[:5]}

