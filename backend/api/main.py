"""
FastAPI backend server for the Startup Research Assistant.
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict as TypingDict
import uuid

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

# Import backend modules (may fail if dependencies missing)
try:
    from backend.agents.base import AgentResponse as LegacyAgentResponse, Task as LegacyTask
    from backend.agents.orchestrator import Orchestrator
    from backend.intake import IdeaInterpreter
    from backend.models import IdeaBrief, PlannerTask
    from backend.planning import ResearchPlanner
    from backend.synthesis.agent import generate_synthesis_response
    from backend.qa import ResearchQA
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    IMPORT_ERROR = str(e)

app = FastAPI(
    title="Startup Research Assistant API",
    description="API for comprehensive startup research and analysis",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for research sessions (in production, use Redis or database)
# Format: {session_id: {"responses": [...], "idea_context": {...}}}
research_sessions: TypingDict[str, Dict[str, Any]] = {}


# Helper --------------------------------------------------------------
def _json_safe(value):
    """Recursively replace NaN/Infinity with None so the payload is valid JSON.

    Evidence is sourced from pandas frames, where missing numbers arrive as
    float('nan'). json.dumps emits a bare `NaN` token for those, which is not
    valid JSON, so serialising the response raises ValueError.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _is_meaningful_text(text: Optional[str]) -> bool:
    if not text:
        return False
    cleaned = text.strip()
    if len(cleaned) < 3:
        return False
    tokens = re.findall(r"[a-zA-Z]{2,}", cleaned)
    if not tokens:
        return False
    unique_tokens = {token.lower() for token in tokens}
    # Consider extremely short repeated tokens as noise
    if len(unique_tokens) == 1 and len(next(iter(unique_tokens))) <= 3:
        return False
    return True


# Pydantic models for API
class IdeaInput(BaseModel):
    idea: str
    problem: Optional[str] = None
    audience: Optional[str] = None
    region: Optional[str] = None
    maturity: Optional[str] = None
    goals: Optional[str] = None
    assumptions: Optional[str] = None


class ResearchRequest(BaseModel):
    idea_input: IdeaInput
    use_llm: bool = True


class TaskStatus(BaseModel):
    task_id: str
    agent: str
    status: str  # "pending", "running", "completed", "error"
    progress: float = 0.0
    message: Optional[str] = None


class AgentResponseModel(BaseModel):
    agent: str
    task_id: str
    summary: str
    confidence: str
    citations: List[Dict]
    evidence_count: int


class ResearchResponse(BaseModel):
    idea_brief: Dict
    tasks: List[Dict]
    responses: List[AgentResponseModel]
    report: Optional[str] = None
    llm_used: bool = False
    session_id: Optional[str] = None


def _planner_task_to_legacy(task: PlannerTask) -> LegacyTask:
    return LegacyTask(
        task_id=task.task_id,
        agent_name=task.agent,
        instruction=task.instruction,
        inputs=task.inputs,
    )


def _response_to_dict(response: LegacyAgentResponse) -> Dict:
    # Convert citations to proper format with label, url, excerpt
    formatted_citations = []
    
    # Filter out tool names - these are internal programming functions, not real sources
    tool_names = ["CompanyFundingTool", "CompanyQAIndexTool", "CompetitorGrowthTool", 
                  "SerpApiTool", "BraveSearchTool", "SecFilingsTool"]
    
    # Create a mapping of source URLs to evidence items for better labels
    # Also collect all unique non-tool sources for citations
    source_to_evidence = {}
    all_sources = set()
    for ev in response.evidence:
        source = getattr(ev, "source", None) or ""
        if source and source not in tool_names:  # Skip tool names
            if source not in source_to_evidence:
                source_to_evidence[source] = ev
            all_sources.add(source)
    
    # If citations are empty or only contain tool names, use actual evidence sources
    if not response.citations or all(c in tool_names for c in response.citations if isinstance(c, str)):
        response.citations = list(all_sources)
    
    if response.citations:
        for citation in response.citations:
            # Skip tool names - they're not real sources
            if isinstance(citation, str) and citation in tool_names:
                continue
                
            if isinstance(citation, str):
                # If citation is a string (URL), try to find matching evidence for better label
                if citation.startswith('http://') or citation.startswith('https://'):
                    # Look for matching evidence to get title
                    matching_ev = source_to_evidence.get(citation)
                    label = getattr(matching_ev, "title", citation) if matching_ev else citation
                    content_text = ""
                    if matching_ev:
                        content = getattr(matching_ev, "content", "")
                        if isinstance(content, str):
                            content_text = content[:200]
                        else:
                            content_text = str(content)[:200]
                    excerpt = content_text
                    
                    formatted_citations.append({
                        "label": label,
                        "url": citation,
                        "excerpt": excerpt
                    })
                elif citation not in tool_names:  # Only add if not a tool name
                    # Not a URL, but might be a meaningful source name
                    # Try to find matching evidence
                    matching_ev = source_to_evidence.get(citation)
                    if matching_ev:
                        label = matching_ev.title if hasattr(matching_ev, 'title') else citation
                        excerpt = matching_ev.content[:200] if hasattr(matching_ev, 'content') else ""
                        formatted_citations.append({
                            "label": label,
                            "url": citation if citation.startswith('http') else "",
                            "excerpt": excerpt
                        })
                    # If no matching evidence and it's not a tool name, skip it
            elif isinstance(citation, dict):
                # If citation is already a dict, use it directly
                formatted_citations.append({
                    "label": citation.get("label", citation.get("title", "")),
                    "url": citation.get("url", ""),
                    "excerpt": citation.get("excerpt", citation.get("content", ""))
                })
            else:
                # Try to convert to dict if it has attributes
                try:
                    formatted_citations.append({
                        "label": getattr(citation, "label", getattr(citation, "title", str(citation))),
                        "url": getattr(citation, "url", ""),
                        "excerpt": getattr(citation, "excerpt", getattr(citation, "content", ""))
                    })
                except:
                    formatted_citations.append({
                        "label": str(citation),
                        "url": "",
                        "excerpt": ""
                    })
    
    return {
        "agent": response.agent_name,
        "task_id": response.task_id,
        "summary": response.summary,
        "confidence": response.confidence,
        "citations": formatted_citations,
        "evidence_count": len(response.evidence),
    }


@app.get("/")
async def root():
    return {"message": "Startup Research Assistant API", "version": "1.0.0"}


@app.get("/health")
async def health():
    if not BACKEND_AVAILABLE:
        return {
            "status": "degraded",
            "message": "Backend agents not available. Some features may not work.",
            "error": IMPORT_ERROR
        }
    return {"status": "healthy"}


@app.post("/api/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    """Run research pipeline synchronously."""
    if not BACKEND_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Backend agents not available. Missing dependencies: {IMPORT_ERROR}"
        )
    try:
        if not _is_meaningful_text(request.idea_input.idea):
            raise HTTPException(
                status_code=400,
                detail="Idea input looks invalid or unreadable. Please describe the startup idea in plain words."
            )

        interpreter = IdeaInterpreter()
        idea_brief = interpreter.interpret(
            idea=request.idea_input.idea,
            problem=request.idea_input.problem,
            audience=request.idea_input.audience,
            region=request.idea_input.region,
            maturity=request.idea_input.maturity,
            goals=request.idea_input.goals,
            assumptions=request.idea_input.assumptions,
        )

        orchestrator = Orchestrator()
        planner = ResearchPlanner(orchestrator.agents.keys())
        planner_tasks = planner.create_plan(idea_brief)

        if not planner_tasks:
            raise HTTPException(status_code=400, detail="No tasks generated for the provided idea.")

        legacy_tasks = [_planner_task_to_legacy(task) for task in planner_tasks]
        responses = orchestrator.run_tasks(legacy_tasks)

        idea_context = {
            "idea": idea_brief.raw_input,
            "problem": idea_brief.problem or "",
            "audience": idea_brief.audience or "",
            "region": idea_brief.region or "",
            "maturity": idea_brief.maturity_stage.value if idea_brief.maturity_stage else "",
            "goals": idea_brief.goals,
            "assumptions": idea_brief.assumptions,
        }
        
        synthesis_response = generate_synthesis_response(
            responses, idea_context=idea_context, use_llm=request.use_llm
        )
        responses.append(synthesis_response)

        report = synthesis_response.raw_output.get("report", "")
        llm_used = synthesis_response.raw_output.get("llm_used", False)

        # Store responses and idea context in session for Q&A
        session_id = str(uuid.uuid4())
        research_sessions[session_id] = {
            "responses": responses,
            "idea_context": idea_context,
        }

        return ResearchResponse(
            idea_brief=idea_brief.dict(),
            tasks=[task.dict() for task in planner_tasks],
            responses=[AgentResponseModel(**_response_to_dict(resp)) for resp in responses],
            report=report,
            llm_used=llm_used,
            session_id=session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/research")
async def websocket_research(websocket: WebSocket):
    """Run research pipeline with real-time progress updates via WebSocket."""
    try:
        await websocket.accept()
    except Exception as e:
        print(f"WebSocket accept error: {e}")
        return
    
    if not BACKEND_AVAILABLE:
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Backend agents not available. Missing dependencies: {IMPORT_ERROR}"
            })
            await websocket.close()
        except Exception as e:
            print(f"Error sending backend unavailable message: {e}")
        return

    try:
        # Receive initial request
        data = await websocket.receive_json()
        request = ResearchRequest(**data)

        if not _is_meaningful_text(request.idea_input.idea):
            await websocket.send_json({
                "type": "error",
                "message": "Idea input looks invalid or unreadable. Please describe the startup idea in plain words."
            })
            await websocket.close()
            return

        # Send initial acknowledgment
        await websocket.send_json({"type": "status", "message": "Starting research..."})

        # Step 1: Interpret idea
        await websocket.send_json({
            "type": "progress",
            "step": "interpret",
            "message": "Interpreting idea brief...",
            "progress": 10,
        })

        interpreter = IdeaInterpreter()
        idea_brief = interpreter.interpret(
            idea=request.idea_input.idea,
            problem=request.idea_input.problem,
            audience=request.idea_input.audience,
            region=request.idea_input.region,
            maturity=request.idea_input.maturity,
            goals=request.idea_input.goals,
            assumptions=request.idea_input.assumptions,
        )

        await websocket.send_json({
            "type": "progress",
            "step": "plan",
            "message": "Creating research plan...",
            "progress": 20,
            "idea_brief": idea_brief.dict(),
        })

        # Step 2: Create plan
        orchestrator = Orchestrator()
        planner = ResearchPlanner(orchestrator.agents.keys())
        planner_tasks = planner.create_plan(idea_brief)

        if not planner_tasks:
            await websocket.send_json({
                "type": "error",
                "message": "No tasks generated for the provided idea.",
            })
            return

        await websocket.send_json({
            "type": "tasks",
            "tasks": [task.dict() for task in planner_tasks],
            "progress": 30,
        })

        # Step 3: Execute tasks
        legacy_tasks = [_planner_task_to_legacy(task) for task in planner_tasks]
        total_tasks = len(legacy_tasks)
        responses: List[LegacyAgentResponse] = []

        for idx, task in enumerate(legacy_tasks):
            progress = 30 + int((idx / total_tasks) * 50)
            await websocket.send_json({
                "type": "task_start",
                "task_id": task.task_id,
                "agent": task.agent_name,
                "message": f"Running {task.agent_name}...",
                "progress": progress,
            })

            agent = orchestrator.agents.get(task.agent_name)
            if not agent:
                response = LegacyAgentResponse(
                    agent_name=task.agent_name,
                    task_id=task.task_id,
                    summary=f"Agent {task.agent_name} not available.",
                    confidence="low",
                )
            else:
                try:
                    response = agent.execute(task)
                except Exception as exc:
                    response = LegacyAgentResponse(
                        agent_name=task.agent_name,
                        task_id=task.task_id,
                        summary=f"Agent execution failed: {exc}",
                        confidence="low",
                        raw_output={"error": str(exc)},
                    )

            responses.append(response)
            await websocket.send_json({
                "type": "task_complete",
                "task_id": task.task_id,
                "agent": task.agent_name,
                "response": _response_to_dict(response),
                "progress": 30 + int(((idx + 1) / total_tasks) * 50),
            })

        # Step 4: Generate synthesis
        await websocket.send_json({
            "type": "progress",
            "step": "synthesis",
            "message": "Generating synthesis report...",
            "progress": 85,
        })

        idea_context = {
            "idea": idea_brief.raw_input,
            "problem": idea_brief.problem or "",
            "audience": idea_brief.audience or "",
            "region": idea_brief.region or "",
            "maturity": idea_brief.maturity_stage.value if idea_brief.maturity_stage else "",
            "goals": idea_brief.goals,
            "assumptions": idea_brief.assumptions,
        }

        synthesis_response = generate_synthesis_response(
            responses, idea_context=idea_context, use_llm=request.use_llm
        )
        responses.append(synthesis_response)

        report = synthesis_response.raw_output.get("report", "")
        llm_used = synthesis_response.raw_output.get("llm_used", False)

        # Store responses and idea context in session for Q&A
        session_id = str(uuid.uuid4())
        research_sessions[session_id] = {
            "responses": responses,
            "idea_context": idea_context,
        }

        # Send final result
        await websocket.send_json({
            "type": "complete",
            "session_id": session_id,  # Include session ID for Q&A
            "idea_brief": idea_brief.dict(),
            "tasks": [task.dict() for task in planner_tasks],
            "responses": [_response_to_dict(resp) for resp in responses],
            "report": report,
            "llm_used": llm_used,
            "progress": 100,
        })

    except WebSocketDisconnect:
        print("WebSocket disconnected by client")
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except Exception as send_error:
            print(f"Error sending error message: {send_error}")


class QARequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    use_llm: bool = True


@app.post("/api/qa")
async def ask_question(request: QARequest):
    """Answer questions about research results."""
    try:
        if not request.question:
            raise HTTPException(status_code=400, detail="Question is required.")

        # Get responses and idea context from session if session_id provided
        if request.session_id and request.session_id in research_sessions:
            session_data = research_sessions[request.session_id]
            responses = session_data.get("responses", [])
            idea_context = session_data.get("idea_context", {})
        else:
            raise HTTPException(
                status_code=400, 
                detail="Session ID required. Please provide session_id from research results."
            )

        # Use ResearchQA to answer the question with idea context
        qa = ResearchQA(responses)
        result = qa.answer_question(
            request.question, 
            use_llm=request.use_llm,
            idea_context=idea_context
        )

        return _json_safe({
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "evidence": result.get("evidence", []),  # Include evidence details
            "llm_used": result.get("llm_used", False),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

