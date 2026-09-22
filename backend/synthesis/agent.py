from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from backend.agents.base import AgentResponse
from backend.synthesis.llm_report import LLMReportGenerator
from .report import build_report


def generate_synthesis_response(
    responses: Iterable[AgentResponse],
    idea_context: Optional[Dict[str, Any]] = None,
    use_llm: bool = True,
) -> AgentResponse:
    """Generate synthesis response using LLM or template fallback."""
    idea_context = idea_context or {}

    # Try LLM synthesis if enabled
    if use_llm:
        try:
            generator = LLMReportGenerator()
            result = generator.generate(responses, idea_context, use_llm=True)
            report = result.get("report", "")
            llm_used = result.get("llm_used", False)
            token_usage = result.get("token_usage", {})

            return AgentResponse(
                agent_name="SynthesisAgent",
                task_id="synthesis",
                summary="Compiled multi-source business intelligence using LLM synthesis." if llm_used else "Compiled multi-source business intelligence (template fallback).",
                confidence="high" if llm_used else "medium",
                evidence=[],  # SynthesisAgent doesn't collect new evidence - it synthesizes from other agents
                citations=[],
                raw_output={
                    "report": report,
                    "llm_used": llm_used,
                    "token_usage": token_usage,
                },
            )
        except Exception as e:
            # Fallback to template on error
            import sys
            error_msg = str(e)
            # Check for user-friendly messages first
            if "ollama" in error_msg.lower() and ("not available" in error_msg.lower() or "connection" in error_msg.lower()):
                msg = "\n" + "="*60 + "\n[INFO] Ollama not available. Make sure Ollama is running: 'ollama serve'\nUsing template-based report instead.\n" + "="*60 + "\n"
                print(msg, file=sys.stderr)
                print(msg)
            elif "insufficient balance" in error_msg.lower() or "needs credits" in error_msg.lower():
                msg = "\n" + "="*60 + "\n[INFO] LLM synthesis unavailable: DeepSeek account needs credits.\nUsing template-based report instead.\n" + "="*60 + "\n"
                print(msg, file=sys.stderr)
                print(msg)
            elif "invalid api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                msg = "\n" + "="*60 + "\n[WARNING] LLM synthesis failed: Invalid API key.\nUsing template-based report instead.\n" + "="*60 + "\n"
                print(msg, file=sys.stderr)
                print(msg)
            elif "402" in error_msg or "Insufficient Balance" in error_msg:
                msg = "\n" + "="*60 + "\n[INFO] LLM synthesis unavailable: DeepSeek account needs credits.\nUsing template-based report instead.\n" + "="*60 + "\n"
                print(msg, file=sys.stderr)
                print(msg)
            elif "401" in error_msg:
                msg = "\n" + "="*60 + "\n[WARNING] LLM synthesis failed: Invalid API key.\nUsing template-based report instead.\n" + "="*60 + "\n"
                print(msg, file=sys.stderr)
                print(msg)
            else:
                msg = f"\n" + "="*60 + f"\n[WARNING] LLM synthesis failed: {error_msg}\nUsing template-based report instead.\n" + "="*60 + "\n"
                print(msg, file=sys.stderr)
                print(msg)
            pass

    # Template fallback
    report = build_report(responses)
    return AgentResponse(
        agent_name="SynthesisAgent",
        task_id="synthesis",
        summary="Compiled multi-source business intelligence.",
        confidence="medium",
        evidence=[],
        citations=[],
        raw_output={"report": report, "llm_used": False},
    )

