"""Funding intelligence agent implementation."""

from __future__ import annotations

from typing import Dict, List

from .base import AgentResponse, BaseAgent, Evidence, Task
from .tools import CompanyFundingTool, CompanyQAIndexTool


class FundingIntelligenceAgent(BaseAgent):
    name = "FundingIntelligenceAgent"

    def __init__(self, tools: Dict[str, object]):
        super().__init__(tools)
        if "CompanyFundingTool" not in tools or "CompanyQAIndexTool" not in tools:
            raise ValueError("Funding agent requires CompanyFundingTool and CompanyQAIndexTool")

    def execute(self, task: Task) -> AgentResponse:
        funding_tool: CompanyFundingTool = self.tools["CompanyFundingTool"]  # type: ignore
        qa_tool: CompanyQAIndexTool = self.tools["CompanyQAIndexTool"]  # type: ignore

        funding_result = funding_tool.run(task.inputs)
        qa_result = qa_tool.run({"question": task.instruction, "top_k": task.inputs.get("qa_top_k", 3)})

        evidence: List[Evidence] = []
        geography = (task.inputs.get("geography") or task.inputs.get("region") or "").strip()
        for record in funding_result.get("results", []):
            company_name = record.get("name_cb") or record.get("normalized_name")
            # Format content as readable text
            content_parts = []
            funding = record.get("funding_total_usd")
            if funding:
                try:
                    funding_float = float(funding)
                    if funding_float >= 1_000_000_000:
                        formatted = f"${funding_float / 1_000_000_000:.1f}B"
                    elif funding_float >= 1_000_000:
                        formatted = f"${funding_float / 1_000_000:.1f}M"
                    else:
                        formatted = f"${funding_float:,.0f}"
                    content_parts.append(f"Total funding: {formatted}")
                except (TypeError, ValueError):
                    pass
            investors = record.get("investor_names")
            if investors:
                if isinstance(investors, list):
                    inv_str = ", ".join(str(inv) for inv in investors[:3])
                    if len(investors) > 3:
                        inv_str += f" and {len(investors) - 3} more"
                else:
                    inv_str = str(investors)
                content_parts.append(f"Investors: {inv_str}")
            rounds = record.get("funding_round_count")
            if rounds:
                content_parts.append(f"Funding rounds: {rounds}")
            latest = record.get("latest_investment_at")
            if latest:
                content_parts.append(f"Latest investment: {latest}")
            
            content = " | ".join(content_parts) if content_parts else f"Funding data for {company_name}"
            
            # Add source information to content
            source_info = "Data source: Crunchbase (via internal company database)"
            if content:
                content += f" | {source_info}"
            else:
                content = source_info
            
            metadata = {
                **record,
                "_data_source": "Crunchbase",
                "_source_type": "internal_database",
            }
            if geography:
                metadata.setdefault("region_hint", geography)

            evidence.append(
                Evidence(
                    title=f"Funding record: {company_name}",
                    content=content,
                    source="Crunchbase Database",  # More user-friendly source name
                    metadata=metadata,
                )
            )

        for qa in qa_result.get("results", []):
            evidence.append(
                Evidence(
                    title=f"QA snippet: {qa.get('metadata', {}).get('company')}",
                    content=qa.get("response", ""),
                    source="CompanyQAIndexTool",
                    score=qa.get("score"),
                    metadata=qa,
                )
            )

        summary_parts = []
        if funding_result.get("results"):
            top = funding_result["results"][0]
            company_name = top.get('name_cb') or top.get('normalized_name')
            funding_amount = top.get('funding_total_usd')
            
            # Format funding amount (convert to readable format, not a calculation)
            try:
                funding_float = float(funding_amount)
                # This is just formatting/rounding for readability, not a calculation
                # Original: 954560000.0 USD
                # Divided by 1,000,000 to get millions: 954.56
                # Rounded to 1 decimal: 954.6M
                if funding_float >= 1_000_000_000:
                    millions = funding_float / 1_000_000_000
                    formatted = f"${millions:.1f}B"
                    exact = f"${funding_float:,.0f}"
                elif funding_float >= 1_000_000:
                    millions = funding_float / 1_000_000
                    formatted = f"${millions:.1f}M"
                    exact = f"${funding_float:,.0f}"
                else:
                    formatted = f"${funding_float:,.0f}"
                    exact = formatted
            except (TypeError, ValueError):
                formatted = str(funding_amount)
                exact = str(funding_amount)
            
            summary_parts.append(
                f"Top comparable company {company_name} shows funding total of {formatted} "
                f"(exact: {exact} USD, Source: Crunchbase database via internal company data)"
            )
        if qa_result.get("results"):
            summary_parts.append("Relevant Q&A snippets captured.")

        summary = summary_parts[0] if summary_parts else "No funding comparables found."
        
        # Add citations with data source information
        citations = []
        # Add Crunchbase as citation source for funding data
        if funding_result.get("results"):
            citations.append({
                "label": "Crunchbase - Company Funding Database",
                "url": "https://www.crunchbase.com",
                "excerpt": "Funding data sourced from Crunchbase company database via internal data processing"
            })
        
        # Add any other actual URLs from evidence
        for ev in evidence:
            source = getattr(ev, "source", None)
            if source and source not in ["CompanyFundingTool", "CompanyQAIndexTool", "Crunchbase Database"]:
                if isinstance(source, str) and (source.startswith('http://') or source.startswith('https://')):
                    if source not in [c.get("url", "") if isinstance(c, dict) else "" for c in citations]:
                        citations.append({
                            "label": getattr(ev, "title", "Source"),
                            "url": source,
                            "excerpt": getattr(ev, "content", "")[:200] if hasattr(ev, "content") else ""
                        })
        
        return AgentResponse(
            agent_name=self.name,
            task_id=task.task_id,
            summary=summary,
            confidence="medium" if evidence else "low",
            evidence=evidence,
            citations=citations,
        )


