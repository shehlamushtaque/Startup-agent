from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))
if Path.cwd() != _PROJECT_ROOT:
    os.chdir(_PROJECT_ROOT)

load_dotenv(_PROJECT_ROOT / ".env")

from backend.agents.base import AgentResponse as LegacyAgentResponse, Task as LegacyTask
from backend.agents.orchestrator import Orchestrator
from backend.intake import IdeaInterpreter
from backend.models import IdeaBrief, PlannerTask
from backend.planning import ResearchPlanner
from backend.synthesis.agent import generate_synthesis_response


def _planner_task_to_legacy(task: PlannerTask) -> LegacyTask:
    return LegacyTask(
        task_id=task.task_id,
        agent_name=task.agent,
        instruction=task.instruction,
        inputs=task.inputs,
    )


def _response_to_dict(response: LegacyAgentResponse) -> dict:
    return {
        "agent": response.agent_name,
        "task_id": response.task_id,
        "summary": response.summary,
        "confidence": response.confidence,
        "citations": response.citations,
        "evidence_count": len(response.evidence),
    }


def _print_human_readable(idea_brief: IdeaBrief, planner_tasks: Iterable[PlannerTask], responses: List[LegacyAgentResponse]) -> None:
    print("\n=== Business Research Run ===")
    print(f"Idea: {idea_brief.raw_input}")
    if idea_brief.problem:
        print(f"Problem: {idea_brief.problem}")
    if idea_brief.audience:
        print(f"Audience: {idea_brief.audience}")
    if idea_brief.region:
        print(f"Region: {idea_brief.region}")
    if idea_brief.missing_fields:
        print(f"Missing fields: {', '.join(idea_brief.missing_fields)}")

    print("\nPlanned Tasks:")
    for task in planner_tasks:
        print(f"- [{task.agent}] {task.instruction}")

    print("\nAgent Responses:")
    for resp in responses:
        print(f"[{resp.agent_name}] ({resp.confidence}) {resp.summary}")
        if resp.citations:
            print(f"  Citations: {', '.join(resp.citations)}")
        if resp.agent_name == "SynthesisAgent":
            # SynthesisAgent synthesizes from other agents, doesn't collect new evidence
            print(f"  Evidence items: 0 (synthesizes from other agents' evidence)")
        elif resp.evidence:
            print(f"  Evidence items ({len(resp.evidence)}):")
            for ev in resp.evidence:
                source = getattr(ev, "source", None) or getattr(ev, "metadata", {}).get("source")
                title = getattr(ev, "title", None) or getattr(ev, "content", "")[:100]
                source_info = f" | Source: {source}" if source else ""
                print(f"    - {title}{source_info}")
        else:
            print("  Evidence items: 0")
        print()


def _prompt_input(
    prompt: str,
    *,
    required: bool = False,
    choices: Iterable[str] | None = None,
) -> str:
    import sys
    # If not in interactive mode, return empty string for optional fields
    if not sys.stdin.isatty() and not required:
        return ""
    
    while True:
        try:
            value = input(prompt).strip()
        except EOFError:
            # Handle non-interactive mode
            if required:
                raise ValueError(f"Required field cannot be empty in non-interactive mode")
            return ""
        if not value:
            if required:
                print("This field is required.")
                continue
            return ""
        if value and choices and value not in choices:
            print(f"Value must be one of: {', '.join(choices)}")
            continue
        return value


def _collect_inputs(args: argparse.Namespace) -> IdeaBrief:
    interpreter = IdeaInterpreter()

    import sys
    idea = args.idea
    if not idea:
        if sys.stdin.isatty():
            idea = _prompt_input("Idea description: ", required=True)
        else:
            raise ValueError("--idea is required in non-interactive mode")
    
    problem = args.problem
    if not problem and sys.stdin.isatty():
        problem = _prompt_input("Problem / industry focus (optional): ")
    
    audience = args.audience
    if not audience and sys.stdin.isatty():
        audience = _prompt_input("Target audience (optional): ")
    
    region = args.region
    if not region and sys.stdin.isatty():
        region = _prompt_input("Region (optional): ")
    
    maturity_choices = ("concept", "mvp", "pre-seed", "seed", "growth")
    maturity = args.maturity
    if not maturity and sys.stdin.isatty():
        maturity = _prompt_input(
            f"Maturity stage [{', '.join(maturity_choices)}] (optional): ",
            choices=maturity_choices,
        )
        maturity = maturity or None

    goals = args.goals
    if goals is None and sys.stdin.isatty():
        goals = _prompt_input("Comma-separated business goals (optional): ")

    assumptions = args.assumptions
    if assumptions is None and sys.stdin.isatty():
        assumptions = _prompt_input("Comma-separated assumptions (optional): ")

    return interpreter.interpret(
        idea=idea,
        problem=problem or None,
        audience=audience or None,
        region=region or None,
        maturity=maturity,
        goals=goals or None,
        assumptions=assumptions or None,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the business research assistant pipeline.")
    parser.add_argument("--idea", help="Raw description of the startup idea.")
    parser.add_argument("--problem", help="Problem the idea solves / industry focus.")
    parser.add_argument("--audience", help="Target audience or customer segment.")
    parser.add_argument("--region", help="Geographic focus.")
    parser.add_argument(
        "--maturity",
        choices=["concept", "mvp", "pre-seed", "seed", "growth"],
        help="Maturity stage of the idea.",
    )
    parser.add_argument("--goals", help="Comma-separated list of business goals.")
    parser.add_argument("--assumptions", help="Comma-separated list of assumptions or hypotheses.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM synthesis (use template fallback).")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive Q&A mode after generating report.")
    return parser


def run(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    idea_brief = _collect_inputs(args)

    orchestrator = Orchestrator()
    planner = ResearchPlanner(orchestrator.agents.keys())
    planner_tasks = planner.create_plan(idea_brief)

    if not planner_tasks:
        print("No tasks generated for the provided idea. Check input fields.", file=sys.stderr)
        return 1

    legacy_tasks = [_planner_task_to_legacy(task) for task in planner_tasks]
    responses = orchestrator.run_tasks(legacy_tasks)

    # Prepare idea context for LLM synthesis
    idea_context = {
        "idea": idea_brief.raw_input,
        "problem": idea_brief.problem or "",
        "audience": idea_brief.audience or "",
        "region": idea_brief.region or "",
        "maturity": idea_brief.maturity_stage.value if idea_brief.maturity_stage else "",
        "goals": idea_brief.goals,
        "assumptions": idea_brief.assumptions,
    }

    synthesis_response = generate_synthesis_response(responses, idea_context=idea_context, use_llm=not args.no_llm)
    responses.append(synthesis_response)

    if args.json:
        payload = {
            "idea": idea_brief.dict(),
            "tasks": [task.dict() for task in planner_tasks],
            "responses": [_response_to_dict(resp) for resp in responses],
        }
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        _print_human_readable(idea_brief, planner_tasks, responses)
        report = synthesis_response.raw_output.get("report")
        llm_used = synthesis_response.raw_output.get("llm_used", False)
        if report:
            print("\n=== Synthesised Report ===")
            if llm_used:
                print("[Generated by LLM]")
            else:
                print("[Template-based report]")
            print(report)
        
        # Interactive Q&A mode - always prompt unless explicitly disabled
        if not args.json:  # Don't prompt in JSON mode
            import sys
            if sys.stdin.isatty():  # Only if interactive terminal
                if args.interactive:
                    # Explicitly requested
                    _interactive_qa(responses, use_llm=not args.no_llm)
                else:
                    # Prompt user if they want to ask questions
                    try:
                        response = input("\nWould you like to ask questions about the research? (yes/no): ").strip().lower()
                        if response in ("yes", "y", ""):
                            _interactive_qa(responses, use_llm=not args.no_llm)
                    except (EOFError, KeyboardInterrupt):
                        pass  # User cancelled, just exit

    return 0


def _interactive_qa(responses: List[LegacyAgentResponse], use_llm: bool = True) -> None:
    """Interactive Q&A session for querying research results."""
    from backend.qa import ResearchQA
    import os
    
    print("\n" + "="*60)
    print("Interactive Q&A Mode")
    print("="*60)
    
    # Check LLM availability (Ollama or DeepSeek)
    from backend.llm import get_llm_client
    llm_client = get_llm_client() if use_llm else None
    
    if use_llm and llm_client:
        if hasattr(llm_client, 'base_url') and 'localhost' in str(llm_client.base_url):
            print("[LLM Enabled - Ollama] Using local Ollama for AI analysis")
            print("Note: Make sure Ollama is running: ollama serve")
        else:
            print("[LLM Enabled - DeepSeek] Using DeepSeek API for AI analysis")
            print("Note: If you see 'needs credits' error, add credits at https://platform.deepseek.com")
    elif use_llm:
        print("[LLM Disabled] No LLM available. Using template-based answers.")
        print("To enable AI-powered analysis:")
        print("  Option 1 (Recommended - Free): Install and run Ollama")
        print("    1. Install: https://ollama.ai")
        print("    2. Run: ollama serve")
        print("    3. Pull a model: ollama pull llama3.2")
        print("  Option 2: Use DeepSeek API")
        print("    1. Get API key from https://platform.deepseek.com")
        print("    2. Set: $env:DEEPSEEK_API_KEY='your-key' or add to .env file")
    else:
        print("[LLM Disabled] Using template-based answers (--no-llm flag used)")
    
    print("Ask questions about the research results. Type 'exit' or 'quit' to end.")
    print()
    
    qa = ResearchQA(responses)
    
    while True:
        try:
            question = input("Question: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("Exiting Q&A mode.")
                break
            
            result = qa.answer_question(question, use_llm=use_llm)
            
            print("\n--- Answer ---")
            if result.get("llm_used"):
                print("[Generated by LLM - Analysis & Synthesis]")
            else:
                print("[Template-based answer - LLM not available]")
            print()
            print(result["answer"])
            
            # Show sources (but not full evidence details - LLM should have analyzed them)
            if result["sources"]:
                print("\n--- Sources Referenced ---")
                for i, source in enumerate(result["sources"][:5], 1):
                    print(f"{i}. [{source['agent']}] {source['title']}")
                    if source.get("source"):
                        print(f"   {source['source']}")
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting Q&A mode.")
            break
        except EOFError:
            print("\n\nExiting Q&A mode.")
            break


if __name__ == "__main__":
    raise SystemExit(run())

