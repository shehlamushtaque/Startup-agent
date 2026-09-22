"""LLM-powered report generation with structured evidence."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Iterable, Optional

from backend.agents.base import AgentResponse
from backend.synthesis.evidence_prep import curate_top_evidence, prepare_structured_evidence
from backend.synthesis.evidence_store import persist_structured_evidence

# Import LLM clients
try:
    from backend.llm import get_llm_client
    from backend.llm.client import DeepSeekClient
    from backend.llm.ollama_client import OllamaClient
except ImportError:
    get_llm_client = None
    DeepSeekClient = None
    OllamaClient = None


class LLMReportGenerator:
    """Generate human-like business reports using LLM synthesis."""

    def __init__(
        self,
        llm_client=None,
        max_evidence_per_category: int = 5,
    ):
        # Use provided client, or auto-detect (Ollama first, then DeepSeek)
        if llm_client:
            self.llm_client = llm_client
        elif get_llm_client:
            self.llm_client = get_llm_client()
        else:
            self.llm_client = None
        self.max_evidence_per_category = max_evidence_per_category

    def generate(
        self,
        responses: Iterable[AgentResponse],
        idea_context: Dict[str, Any],
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """Generate report from agent responses."""
        # Prepare structured evidence
        structured = prepare_structured_evidence(responses, idea_context)
        snapshot_path = None
        try:
            snapshot_path = persist_structured_evidence(structured, idea_context)
        except Exception as exc:
            print(f"[WARNING] Failed to persist structured evidence: {exc}", file=sys.stderr)

        # Curate top evidence to reduce token usage
        curated = curate_top_evidence(structured, max_per_category=self.max_evidence_per_category)

        if not use_llm:
            # Fallback to template-based if LLM disabled
            return {
                "report": self._template_fallback(curated),
                "evidence_used": curated,
                "llm_used": False,
                "evidence_snapshot_path": str(snapshot_path) if snapshot_path else None,
            }

        try:
            # Generate LLM synthesis
            llm_response = self.llm_client.synthesize_report(idea_context, curated)
            report_content = llm_response["content"]
            
            # Evidence sources section removed - not needed in report
            
            return {
                "report": report_content,
                "evidence_used": curated,
                "llm_used": True,
                "token_usage": llm_response.get("usage", {}),
                "evidence_snapshot_path": str(snapshot_path) if snapshot_path else None,
            }
        except Exception as e:
            # Fallback on error - provide user-friendly error message
            error_msg = str(e)
            user_friendly_msg = None
            if "402" in error_msg or "Insufficient Balance" in error_msg:
                user_friendly_msg = "DeepSeek API account has insufficient balance. Please add credits to enable LLM synthesis."
            elif "401" in error_msg or "Unauthorized" in error_msg:
                user_friendly_msg = "DeepSeek API key is invalid or expired. Please check your DEEPSEEK_API_KEY."
            
            # Re-raise with user-friendly message so it's caught and displayed by synthesis agent
            if user_friendly_msg:
                raise ValueError(user_friendly_msg) from e
            
            return {
                "report": self._template_fallback(curated),
                "evidence_used": curated,
                "llm_used": False,
                "error": error_msg,
                "evidence_snapshot_path": str(snapshot_path) if snapshot_path else None,
            }

    def _template_fallback(self, curated: Dict[str, Any]) -> str:
        """Template-based fallback when LLM is unavailable."""
        sections = []
        evidence_by_category = curated.get("evidence_by_category", {})

        # Helper to format money
        def _format_money(value):
            if value is None or value == "":
                return None
            try:
                amount = float(value)
                if amount >= 1_000_000_000:
                    return f"${amount / 1_000_000_000:.1f}B"
                elif amount >= 1_000_000:
                    return f"${amount / 1_000_000:.1f}M"
                elif amount >= 1_000:
                    return f"${amount / 1_000:.1f}K"
                return f"${amount:,.0f}"
            except (TypeError, ValueError):
                return None

        # Funding Landscape
        if "funding" in evidence_by_category:
            funding_items = evidence_by_category["funding"]
            if funding_items:
                sections.append("## Funding Landscape")
                for item in funding_items[:5]:
                    company = item.get("metadata", {}).get("company") or item.get("title", "") or item.get("content", "")[:50]
                    amount = item.get("metadata", {}).get("funding_total_usd")
                    investors = item.get("metadata", {}).get("investors")
                    formatted_amount = _format_money(amount)
                    if formatted_amount:
                        line = f"- {company}: {formatted_amount} raised"
                        if investors:
                            inv_str = investors if isinstance(investors, str) else ", ".join(investors[:2]) if isinstance(investors, list) else ""
                            if inv_str:
                                line += f" (investors: {inv_str})"
                        sections.append(line)
                    elif company:
                        sections.append(f"- {company}")

        # Competitive Field
        if "competitors" in evidence_by_category:
            comp_items = evidence_by_category["competitors"]
            if comp_items:
                sections.append("\n## Competitive Field")
                for item in comp_items[:5]:
                    company = item.get("metadata", {}).get("company") or item.get("title", "") or item.get("content", "")[:50]
                    if not company:
                        continue
                    meta = item.get("metadata", {})
                    growth = meta.get("growth_percent")
                    revenue = meta.get("revenue_usd")
                    employees = meta.get("employees")
                    parts = [company]
                    if growth is not None:
                        try:
                            parts.append(f"{float(growth):.0f}% growth")
                        except (TypeError, ValueError):
                            pass
                    if revenue is not None:
                        rev_str = _format_money(revenue)
                        if rev_str:
                            parts.append(f"revenue {rev_str}")
                    if employees is not None:
                        try:
                            parts.append(f"{int(float(employees)):,} employees")
                        except (TypeError, ValueError):
                            pass
                    sections.append(f"- {' | '.join(parts)}")

        # Market Trends
        if "market_trends" in evidence_by_category:
            trend_items = evidence_by_category["market_trends"]
            if trend_items:
                sections.append("\n## Market Signals")
                for item in trend_items[:7]:
                    title = item.get("title", "") or item.get("content", "")[:100]
                    source = item.get("source", "")
                    if title:
                        if source:
                            sections.append(f"- {title} ({source})")
                        else:
                            sections.append(f"- {title}")

        # Regulatory Filings
        if "regulatory" in evidence_by_category:
            reg_items = evidence_by_category["regulatory"]
            if reg_items:
                sections.append("\n## Recent Regulatory Filings (Form D)")
                for item in reg_items[:5]:
                    company = item.get("metadata", {}).get("company") or item.get("title", "")
                    # Remove redundant CIK info
                    if " (CIK" in company:
                        company = company.split(" (CIK")[0]
                    
                    meta = item.get("metadata", {})
                    content = item.get("content", "")
                    source = item.get("source", "")
                    
                    # Use content if available (already formatted), otherwise build from metadata
                    if content and content != "Form D filing":
                        line = f"- {company} | {content}"
                    else:
                        parts = [company]
                        if meta.get("filed_at"):
                            parts.append(f"filed {meta.get('filed_at')}")
                        amount = meta.get("total_offering_amount") or meta.get("amount_sold") or meta.get("amount_raised")
                        formatted_amount = _format_money(amount)
                        if formatted_amount:
                            parts.append(f"raised {formatted_amount}")
                        investors = meta.get("investors")
                        if investors:
                            inv_list = investors if isinstance(investors, list) else [investors] if isinstance(investors, str) else []
                            if inv_list:
                                inv_str = ", ".join(str(inv) for inv in inv_list[:2])
                                if len(inv_list) > 2:
                                    inv_str += f" and {len(inv_list) - 2} more"
                                parts.append(f"investors: {inv_str}")
                        line = " | ".join(parts)
                    
                    if source:
                        line += f" (source: {source})"
                    sections.append(line)

        if not sections:
            return "No evidence available for synthesis."

        report_text = "\n".join(sections)
        
        # Evidence sources section removed - not needed in report
        
        return report_text
    
    def _build_evidence_sources_section(self, curated: Dict[str, Any]) -> str:
        """Build a section listing all evidence sources with clickable links."""
        evidence_index = curated.get("evidence_index", {})
        if not evidence_index:
            return ""
        
        sections = []
        sections.append("## Evidence Sources & References")
        sections.append("")
        sections.append("The following evidence items (referenced as [E1], [E2], etc. in the report above) were used in this analysis. Click on any link to verify the information:")
        sections.append("")
        sections.append("**About the Data:**")
        sections.append("- **Evidence IDs (E1, E2, etc.)**: Sequential numbers assigned to each data point collected by research agents")
        sections.append("- **Numeric Data**: All funding amounts, revenue, growth percentages come directly from source databases (Crunchbase, Growjo, etc.)")
        sections.append("- **Formatting**: Large numbers are formatted for readability (e.g., $954,560,000 → $954.6M) but the original exact value is preserved in the database")
        sections.append("- **Source Types**: Data comes from internal databases (Crunchbase, Growjo), web searches, or SEC filings")
        sections.append("")
        
        # Sort by evidence ID (E1, E2, etc.)
        sorted_evidence = sorted(
            evidence_index.items(),
            key=lambda x: int(x[0][1:]) if x[0][1:].isdigit() else 999
        )
        
        for ev_id, ev_data in sorted_evidence:
            title = ev_data.get("title", "Untitled")
            source = ev_data.get("source", "")
            content = ev_data.get("content", "")
            category = ev_data.get("category", "")
            agent = ev_data.get("agent", "")
            metadata = ev_data.get("metadata", {})
            
            # Format the citation with evidence ID prominently
            citation_line = f"**[{ev_id}]** {title}"
            
            # Add category/agent info
            if category:
                citation_line += f" *({category})*"
            
            # Add data source explanation
            data_source_explanation = ""
            if metadata:
                if "funding_total_usd" in metadata:
                    funding = metadata.get("funding_total_usd")
                    if funding:
                        try:
                            funding_float = float(funding)
                            exact = f"${funding_float:,.0f}"
                            data_source_explanation = f"\n  💰 **Funding Data**: Exact amount from database: {exact} USD (Source: Crunchbase)"
                        except:
                            pass
                if "growjo_growth_percent" in metadata:
                    growth = metadata.get("growjo_growth_percent")
                    if growth is not None:
                        data_source_explanation = f"\n  📈 **Growth Data**: Growth percentage from Growjo database: {growth}%"
                if "growjo_estimated_revenue_usd" in metadata:
                    revenue = metadata.get("growjo_estimated_revenue_usd")
                    if revenue:
                        try:
                            rev_float = float(revenue)
                            exact = f"${rev_float:,.0f}"
                            data_source_explanation = f"\n  💵 **Revenue Data**: Estimated revenue from Growjo: {exact} USD"
                        except:
                            pass
            
            # Add source link if available
            if source and (source.startswith("http://") or source.startswith("https://")):
                citation_line += f"\n  🔗 [View Source]({source})"
            elif source:
                if "Crunchbase" in source or "Database" in source:
                    citation_line += f"\n  📊 **Data Source**: {source} (Internal company database)"
                else:
                    citation_line += f"\n  📄 Source: {source}"
            elif agent:
                citation_line += f"\n  🤖 **Collected by**: {agent}"
            
            # Add data source explanation if we have numeric data
            if data_source_explanation:
                citation_line += data_source_explanation
            
            # Add brief excerpt if available
            if content:
                # Handle dict content
                if isinstance(content, dict):
                    content_str = " | ".join([f"{k}: {v}" for k, v in list(content.items())[:3]])
                else:
                    content_str = str(content)
                
                excerpt = content_str[:200].strip()
                if len(content_str) > 200:
                    excerpt += "..."
                if excerpt:
                    citation_line += f"\n  *{excerpt}*"
            
            sections.append(citation_line)
            sections.append("")
        
        sections.append("---")
        sections.append("")
        sections.append("*Note: Evidence IDs (E1, E2, etc.) are assigned sequentially as evidence is collected from different research agents. Each ID corresponds to a specific data point, source, or finding used in the analysis above.*")
        
        return "\n".join(sections)

