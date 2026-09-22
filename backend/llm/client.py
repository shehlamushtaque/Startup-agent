"""LLM client wrapper supporting both DeepSeek (cloud) and Ollama (local)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Try to import Ollama client
try:
    from backend.llm.ollama_client import OllamaClient
except ImportError:
    OllamaClient = None


class DeepSeekClient:
    """Client for DeepSeek API (OpenAI-compatible)."""

    BASE_URL = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-chat"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if OpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
        self.model = model or self.DEFAULT_MODEL
        self.base_url = base_url or self.BASE_URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send chat completion request."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    def synthesize_report(
        self,
        idea_context: Dict[str, Any],
        structured_evidence: Dict[str, Any],
        temperature: float = 0.9,  # Increased for more variation
        max_tokens: Optional[int] = 4000,
    ) -> Dict[str, Any]:
        """Generate human-like business report from structured evidence."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(idea_context, structured_evidence)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def _build_system_prompt(self) -> str:
        return """You are a business intelligence analyst specializing in startup research and strategic analysis.

Your task is to synthesize structured evidence into a human-readable, actionable business report.

Guidelines:
1. Extract key insights from the provided evidence (funding, competitors, market trends, regulatory filings)
2. Connect patterns across different data sources to reveal strategic implications
3. Provide specific, actionable recommendations tied to evidence
4. Write in clear, professional business language that anyone can understand
5. DO NOT use evidence IDs like [E1], [E2] in the main report text - write naturally
6. Instead, mention company names, data sources, or findings directly (e.g., "Goodr raised $949 million" instead of "[E2] Goodr raises $949 million")
7. Highlight surprising or non-obvious insights that provide competitive advantage
8. Be honest about gaps or limitations in the data

Output format:
- Executive Summary: 2-3 sentences capturing the most critical findings
- Key Findings: Bullet points with natural descriptions
- Strategic Implications: What this means for the startup
- Recommended Actions: 3-5 specific next steps with rationale
- Open Questions: What additional research is needed

Write naturally - no technical IDs or codes in the main text."""

    def _build_user_prompt(self, idea_context: Dict[str, Any], structured_evidence: Dict[str, Any]) -> str:
        import json

        idea = idea_context.get("idea", "")
        problem = idea_context.get("problem", "")
        audience = idea_context.get("audience", "")
        region = idea_context.get("region", "")
        
        context_str = json.dumps(idea_context, indent=2)
        evidence_str = json.dumps(structured_evidence, indent=2)

        return f"""Analyze the following SPECIFIC startup idea and research evidence to generate a UNIQUE, tailored business report.

CRITICAL: This report must be SPECIFIC to this exact startup idea. Do NOT use generic templates or repeat the same insights for different queries.

STARTUP IDEA CONTEXT:
{context_str}

SPECIFIC DETAILS:
- Idea: {idea}
- Problem: {problem or 'Not specified'}
- Target Audience: {audience or 'Not specified'}
- Region: {region or 'Not specified'}

STRUCTURED EVIDENCE (from research agents):
{evidence_str}

Generate a report that:
1. Is SPECIFIC to this startup idea - connect the evidence directly to "{idea}"
2. Synthesizes the evidence into coherent insights that are RELEVANT to this specific business
3. Identifies patterns and connections that are UNIQUE to this query
4. Provides strategic implications SPECIFIC to this startup's problem, audience, and region
5. Recommends actionable next steps that are TAILORED to this business idea
6. Write naturally - mention company names, funding amounts, and data directly (e.g., "Goodr raised $949 million" or "Lucid shows 534% growth")
7. DO NOT use evidence IDs like [E1], [E2] in the text - write for a general business audience
8. If the evidence doesn't match the idea well, explicitly state this and explain why

IMPORTANT: 
- Do NOT use generic statements that could apply to any startup
- Do NOT use technical evidence IDs ([E1], [E2]) - write naturally
- Reference the specific idea, problem, and audience throughout
- If the evidence shows companies/trends unrelated to the idea, explain the disconnect
- Make connections between the evidence and the specific startup concept
- Write as if explaining to a business person, not a technical system

Focus on insights that are SPECIFIC to this startup idea. What does this data tell us about THIS PARTICULAR business opportunity?"""

