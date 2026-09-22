"""SEC filings agent fetching Form D disclosures via sec-api.io."""

from __future__ import annotations

from typing import Dict, List

from .base import AgentResponse, BaseAgent, Evidence, Task
from .tools import SecFormDTool


class SecFilingsAgent(BaseAgent):
    name = "SecFilingsAgent"

    def __init__(self, tools: Dict[str, object]):
        super().__init__(tools)
        if "SecFormDTool" not in tools:
            raise ValueError("SecFilingsAgent requires SecFormDTool")

    def execute(self, task: Task) -> AgentResponse:
        sec_tool: SecFormDTool = self.tools["SecFormDTool"]  # type: ignore
        query = {
            "search": task.inputs.get("search") or task.instruction,
            "form_type": task.inputs.get("form_type", "D"),
            "limit": task.inputs.get("limit", 10),
            "from": task.inputs.get("from"),
            "to": task.inputs.get("to"),
            "cik": task.inputs.get("cik"),
        }
        result = sec_tool.run(query)

        if result.get("error"):
            return AgentResponse(
                agent_name=self.name,
                task_id=task.task_id,
                summary=f"SEC filings lookup failed: {result['error']}",
                confidence="low",
                evidence=[],
                citations=[],
                raw_output=result,
            )

        filings = result.get("results", [])
        
        # Filter out irrelevant filings (real estate funds, etc.)
        exclude_terms = task.inputs.get("exclude_terms", [])
        # Always filter, even if exclude_terms is empty (we have built-in filters)
        filtered_filings = []
        for filing in filings:
            company_name = (filing.get("company") or "").lower()
            description = (filing.get("description") or "").lower()
            
            # Skip if company name contains exclude terms
            if exclude_terms and any(exclude_term.lower() in company_name for exclude_term in exclude_terms):
                continue
            
            # AGGRESSIVE: Skip if it's clearly an investment fund
            # Check for LP/LLC/LTD suffixes (with or without comma, case-insensitive)
            company_lower = company_name.lower()
            is_lp_or_llc = any(company_lower.endswith(suffix) for suffix in [
                " lp", " l.p.", ", lp", ", l.p.", " llc", ", llc", " ltd", ", ltd",
                " l.p", ", l.p", " lp.", ", lp."
            ])
            
            # Rule 1: If it has "fund" in name (case-insensitive) and ends with LP/LLC, it's almost certainly a fund
            if (" fund" in company_lower or "fund " in company_lower or company_lower.startswith("fund ")):
                if is_lp_or_llc:
                    continue  # Definitely a fund
            
            # Rule 2: "Partners" + LP/LLC is almost always a fund (e.g., "Patient Square Equity Partners II-A, L.P.")
            # Check for "partners" anywhere in name (not just at end)
            if "partners" in company_lower:
                if is_lp_or_llc:
                    continue  # Almost certainly a fund
            
            # Rule 3: If it's LP/LLC with fund-related keywords, likely a fund
            has_fund_keywords = any(term in company_lower for term in [
                " capital", " opportunity", " equity", " investment", 
                " acquisitions", " holdings", " ventures", " group lp", " group llc",
                "capital fund", "opportunity fund", "equity fund", "capital opportunity",
                "discretionary partners", "equity partners"
            ])
            
            if is_lp_or_llc and has_fund_keywords:
                # Exception: Only allow if it's clearly an OPERATING company (not a fund)
                # Must have strong operating company indicators
                has_strong_operating_indicators = any(term in company_lower for term in [
                    "software", "tech", "technology", "ai ", "health", "healthcare", "medical", 
                    "fintech", "payment", "payments", "digital ", "platform", "solutions", "systems",
                    "inc", "corp", "corporation", "labs", "technologies", "systems inc"
                ])
                
                # If it doesn't have strong operating indicators, it's a fund
                if not has_strong_operating_indicators:
                    continue
            
            # Skip common fund/real estate patterns (universal filters)
            fund_patterns = ["realty", "real estate", "property fund", "retail fund", "opportunistic", 
                            "development fund", "investment fund", "holdings llc", "partners lp",
                            "fund ii", "fund iii", "fund iv", "fund v", "credit fund", "credit ii",
                            "reit", "real estate investment", "mortgage", "housing fund", "workforce housing",
                            "residential fund", "commercial property", "property management", "property llc",
                            "acquisition fund", "equity fund", "debt fund", "bridge fund", "opportunity fund",
                            "capital opportunity", "patient capital", "patient partners"]
            
            if any(pattern in company_name for pattern in fund_patterns):
                continue
            
            # Skip if it's clearly a fund/real estate entity by suffix
            fund_suffixes = (" realty llc", " realty lp", " realty l.p.", " properties llc",
                            " property llc", " holdings llc", " fund lp", " fund l.p.", " fund, lp",
                            " credit lp", " credit llc", " reit llc", " reit lp", " capital lp",
                            " capital llc", " partners lp", " partners llc", " investments lp")
            
            if company_name.endswith(fund_suffixes):
                continue
            
            # Skip if description suggests it's a fund/real estate
            if description and any(term in description for term in ["real estate", "property", "fund", 
                                                                   "investment vehicle", "limited partnership"]):
                # But allow if it's clearly a tech/healthcare/fintech company
                if not any(term in company_name.lower() for term in ["software", "tech", "ai", "health", 
                                                                     "medical", "fintech", "payment", "digital"]):
                    continue
            
            filtered_filings.append(filing)
        filings = filtered_filings[:10]  # Limit to top 10 after filtering
        
        evidence: List[Evidence] = []
        for filing in filings:
            company = filing.get("company") or "Unknown Company"
            cik = filing.get("cik")
            title = f"{company} (CIK {cik}) Form D" if cik else f"{company} Form D"
            
            content_parts = []
            if filing.get("filed_at"):
                content_parts.append(f"Filed: {filing.get('filed_at')}")
            if filing.get("amount_raised"):
                # Format money
                amount = filing.get("amount_raised")
                if amount >= 1_000_000:
                    formatted = f"${amount / 1_000_000:.1f}M"
                else:
                    formatted = f"${amount:,.0f}"
                content_parts.append(f"Amount raised: {formatted}")
            elif filing.get("total_offering_amount"):
                amount = filing.get("total_offering_amount")
                if amount >= 1_000_000:
                    formatted = f"${amount / 1_000_000:.1f}M"
                else:
                    formatted = f"${amount:,.0f}"
                content_parts.append(f"Offering amount: {formatted}")
            if filing.get("investors"):
                investors = filing.get("investors")
                if isinstance(investors, list):
                    investor_str = ", ".join(investors[:3])
                    if len(investors) > 3:
                        investor_str += f" and {len(investors) - 3} more"
                    content_parts.append(f"Investors: {investor_str}")
            
            # Add XML parsing status indicator
            if filing.get("xml_parsed"):
                content_parts.append("[XML data extracted]")
            elif filing.get("xml_parsed") is False:
                # XML parsing was attempted but failed
                pass  # Don't show error to user, just silently fail
            
            content = " | ".join(filter(None, content_parts)) or "Form D filing"
            evidence.append(
                Evidence(
                    title=title,
                    content=content,
                    source=filing.get("link") or "",
                    metadata=filing,
                )
            )

        if evidence:
            top = [ev.metadata.get("company") or ev.title for ev in evidence[:3]]
            summary = f"Recent Form D filings include: {', '.join(top)}"
            confidence = "medium"
            citations = [ev.source for ev in evidence if ev.source][:3]
        else:
            summary = "No Form D filings found for the given query."
            confidence = "low"
            citations = []

        return AgentResponse(
            agent_name=self.name,
            task_id=task.task_id,
            summary=summary,
            confidence=confidence,
            evidence=evidence,
            citations=citations,
            raw_output={"results": filings},
        )

