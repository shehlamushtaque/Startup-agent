# Business Decision Assistant Architecture

## 1. High-Level System View

```
User Input
   ↓
Intake Layer (`IdeaInterpreter`)
   ↓ structured brief
Planning Layer (`ResearchPlanner`)
   ↓ task queue
Orchestration Layer (`Orchestrator`)
   ↘ Agent Execution Layer (specialised agents + tools)
        ↘ Knowledge/Data Sources (APIs, indices, datasets)
   ↓ aggregated evidence
Synthesis Layer (`InsightSynthesiser`)
   ↓
Final Business Report
```

The platform runs each request as a session. A session owns: structured idea brief, planner task queue, accumulated evidence, and final response. Session state is stored in a `ContextStore`.

## 2. Core Modules and Responsibilities

- **`backend/intake/idea_interpreter.py`**
  - Normalises raw user input (free text or form fields).
  - Validates required business fields (problem, audience, region, maturity, goals).
  - Outputs `IdeaBrief` model, capturing missing information.

- **`backend/planning/research_planner.py`**
  - Consumes `IdeaBrief`.
  - Calls planner LLM with `PlanPrompt`.
  - Emits ordered `PlannerTask` list.
  - Adds provenance metadata (LLM model, prompt version).

- **`backend/orchestration/task_queue.py`**
  - In-memory queue (initial) with hooks to move to Redis/SQS later.
  - Supports dependency resolution, retries, and backoff policies.

- **`backend/agents/orchestrator.py` (enhanced)**
  - Loads agent/tool configs from registry.
  - Validates task requirements before dispatch.
  - Coordinates execution order (dependency graph).
  - Records success/failure telemetry.

- **`backend/context/context_store.py`**
  - Session-scoped evidence store.
  - Persists agent outputs, raw tool payloads, citations.
  - Provides aggregation helpers for synthesis.

- **`backend/synthesis/insight_synthesiser.py`**
  - Builds synthesis prompt from aggregated evidence.
  - Calls LLM (`LLaMAClient`/GPT) with reasoning instructions.
  - Produces final `BusinessReport`.

- **`backend/interfaces/base_agent.py` / `base_tool.py`**
  - Protocol/base classes enforcing `execute(task: Task) -> AgentResponse`.
  - Provide shared logging, timing, error instrumentation.

## 3. Data Models (Pydantic)

- `IdeaBrief`
  - `idea_id: str`
  - `raw_input: str`
  - `problem: str`
  - `audience: str`
  - `region: str`
  - `maturity_stage: Literal["concept", "mvp", "pre-seed", "seed", "growth"]`
  - `goals: List[str]`
  - `missing_fields: List[str]`

- `PlannerTask`
  - `task_id: str`
  - `agent: str`
  - `instruction: str`
  - `inputs: Dict[str, Any]`
  - `dependencies: List[str]`
  - `priority: int`

- `AgentResponse`
  - `task_id: str`
  - `agent_name: str`
  - `summary: str`
  - `evidence: List[EvidenceItem]`
  - `citations: List[Citation]`
  - `confidence: Literal["low", "medium", "high"]`
  - `metadata: Dict[str, Any]`

- `EvidenceItem`
  - `id: str`
  - `content: str`
  - `source_type: Literal["academic", "news", "statistical", "linkedin", "report", "internal"]`
  - `source_ref: str` (URL or dataset identifier)
  - `published_at: datetime`
  - `relevance_score: float`

- `BusinessReport`
  - `idea_id: str`
  - `executive_summary: str`
  - `vision_analysis: Section`
  - `health_scores: Section`
  - `forecasts: Section`
  - `recommendations: List[str]`
  - `open_questions: List[str]`
  - `references: List[Citation]`

- `Citation`
  - `label: str`
  - `url: str`
  - `excerpt: str`

All models inherit from `BaseModel` with validation and `.json()` serialisation.

## 4. Agent & Tool Registry

- Registry stored in `config/agent_registry.yaml`.
- Fields:
  ```
  agents:
    - name: FundingIntelligenceAgent
      class_path: backend.agents.funding.FundingIntelligenceAgent
      tools: [CompanyFundingTool, CompanyQAIndexTool]
      requires: [industry, geography]
      outputs: [summary, funding_totals, investor_list, citations]
  tools:
    - name: CompanyFundingTool
      class_path: backend.agents.tools.CompanyFundingTool
      config:
        db_url: ${DATABASE_URL}
        cache_ttl_minutes: 60
  ```
- Loader module `backend/config/registry_loader.py` reads YAML, instantiates configs, and passes to orchestrator.
- Hot reload capability guarded behind feature flag.

## 5. Specialized Agents and Tools

### 5.1 Existing Agents (to be refactored)
- `FundingIntelligenceAgent`
- `CompetitorGrowthAgent`
  - Both must conform to new base class and output schema.

### 5.2 New Agents

- `AcademicEvidenceAgent`
  - Tools: `AcademicSearchTool` (Semantic Scholar/ArXiv API), `AcademicQAIndex` (local FAISS over curated papers).
  - Fetch top N papers, summarise findings, assess relevance.

- `NewsResearchAgent`
  - Tools: `NewsResearchTool` (NewsAPI/RSS aggregator), `TrendSummariser`.
  - Focus on recent events affecting the business idea.

- `EconomicStatsAgent`
  - Tools: `EconomicStatsTool` (World Bank, IMF APIs), `MacroDataCache`.
  - Provide macroeconomic indicators relevant to region/industry.

- `TalentIntelAgent`
  - Tools: `TalentDataTool` (people data provider), `RoleDemandEstimator`.
  - Assess hiring landscape and available expertise.

- `BusinessVisionAgent` (new business-focused capability)
  - Tools: `VisionScorer` (LLM prompt templates), `ValuePropComparator`.
  - Score clarity, differentiation, growth potential.

- `BusinessHealthAgent`
  - Tools: `FinancialBenchmarkTool`, `OperationalBenchmarkTool`.
  - Score financial, market position, operational readiness.

- `BusinessForecastAgent`
  - Tools: `ForecastModelRegistry`, `RevenueForecaster`, `GrowthForecaster`, `BudgetForecaster`.
  - Run ML models (Prophet/LSTM/XGBoost) with features assembled via `business_data_client`.

### 5.3 Tool Interfaces

- All tools inherit from `BaseTool`:
  - `name: str`
  - `config: ToolConfig`
  - `execute(payload: Dict[str, Any]) -> ToolResult`
  - Built-in retry, timeout, and caching behaviour.

- `business_data_client.py`
  - Exposes fetch methods:
    - `get_financial_metrics(company_id, industry)`
    - `get_industry_benchmarks(industry, region)`
    - `get_market_size(industry, region)`
    - `get_news_articles(keywords, since)`
    - `get_academic_papers(keywords, limit)`
  - Implements caching layer (Redis or local) and credential management.

## 6. Orchestration Flow

1. `IdeaInterpreter.handle(raw_input)` → returns `IdeaBrief`.
2. `ResearchPlanner.create_plan(idea_brief)` → returns ordered `PlannerTask` list.
3. `TaskQueue.enqueue(tasks)` initialises dependencies.
4. Orchestrator polls queue:
   - Validates required inputs (uses `requires` from agent config).
   - Collects necessary context from `ContextStore`.
   - Calls agent `execute`.
   - On success: persists `AgentResponse` → `ContextStore`.
   - On failure: retries (max N), then records `AgentError` entry.
5. After all tasks completed or exhausted:
   - `ContextStore.compile_evidence()` returns structured data for synthesis.
6. `InsightSynthesiser.generate_report(session_context)` → `BusinessReport`.
7. Response returned via API/CLI, including JSON and optional PDF render.

Failure paths:
- Missing required input → planner flagged in `missing_info`, orchestrator stops and requests user clarification.
- Tool exception → retry with exponential backoff; if still failing, agent returns low-confidence response with error metadata.
- LLM failure → fallback to backup model (configurable).

## 7. Logging, Metrics, Observability

- Structured logging with `task_id`, `agent_name`, duration, outcome.
- Metrics exported via Prometheus:
  - Task latency, agent success rate, tool error rate, LLM token usage.
- Tracing (OpenTelemetry) for cross-component visibility.
- ContextStore maintains audit log for evidence provenance.

## 8. Configuration and Deployment

- Config files under `config/` with environment overrides.
- Secrets (API keys, DB credentials) read from environment or Vault.
- Docker compose for local development:
  - App container
  - Redis (queue/cache)
  - Postgres (if needed for datasets)
  - Optional MinIO/S3 for dataset storage

- CI pipeline:
  - Linting (ruff/black/mypy)
  - Unit tests (pytest with mocks)
  - Integration tests (recorded HTTP fixtures)

## 9. Implementation Roadmap

1. **Foundation**
   - Create base agent/tool interfaces, data models, ContextStore skeleton.
   - Externalise agent/tool registry.

2. **Core Loop MVP**
   - Implement Intake → Planner → Orchestrator loop with one agent (Funding) updated to new interface.
   - Add synthesis stub that echoes evidence for validation.

3. **Business Vision & Health**
   - Build `BusinessVisionAgent`, `BusinessHealthAgent` with initial heuristic scoring (LLM + benchmarking).
   - Integrate relevant tools.

4. **Forecasting**
   - Implement `business_data_client`.
   - Train baseline Prophet/XGBoost models with available data; store models under `models/`.
   - Expose inference via `BusinessForecastAgent`.

5. **Evidence Expansion**
   - Add Academic, News, Economic agents/tools.
   - Build local FAISS indices where datasets exist.

6. **Synthesis & Reporting**
   - Implement `InsightSynthesiser` with final report template and LLM prompt.
   - Add PDF/HTML rendering pipeline if required.

7. **Hardening**
   - Add retries, caching, circuit breakers.
   - Instrument logging/metrics.
   - Expand tests and run end-to-end benchmarks.

8. **Production Readiness**
   - Containerise services, configure CI/CD.
   - Load test with representative workloads.
   - Document operations runbook.

## 10. Open Questions

- Source availability for sector research reports and LinkedIn data (licensing/compliance).
- Choice of LLM provider(s) and cost management for planner and synthesis stages.
- Data retention policies and user privacy requirements.
- Access control: multi-tenant support, per-user quotas, audit trails.

This architecture spec is the baseline. Any implementation work should reference and update this document as decisions evolve.


