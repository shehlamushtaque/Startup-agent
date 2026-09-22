# Multi-Agent Architecture Plan

## Overview
We will build a modular research assistant that orchestrates multiple specialised agents around GPT-4/LLaMA driven planning. The system runs in three layers:

1. **Idea Intake & Planning**
   - Intake agent structures the user's idea (problem, audience, geography, maturity, goals).
   - Planner (LLM) decomposes the structured brief into sub-queries and assigns them to agents.

2. **Specialised Agents**
   Each agent owns a set of tools and returns strongly typed findings:
   - `FundingIntelligenceAgent`: queries `master_companies`, `company_investors`, QA index.
   - `CompetitorGrowthAgent`: uses Crunchbase + Growjo to surface competitors, growth rates, rankings.
   - `AcademicEvidenceAgent`: searches academic APIs (Semantic Scholar) for supporting studies.
   - `MarketTrendAgent`: hits news/RSS/research APIs for current trends, regulations.
   - `TalentLinkedInAgent`: analyses talent pools, potential advisors via LinkedIn/people data.
   - `EconomicStatsAgent`: fetches official statistics (World Bank, IMF, OECD, etc.).

   Each agent:
   - Accepts `task_id`, `goal`, `inputs`, `needed_fields`.
   - Calls registered tools (SQL/RAG/search).
   - Returns `summary`, `evidence_snippets`, `citations`, `confidence`.

3. **Synthesis & Reasoning**
   - Planner aggregates agent responses into a reasoning log.
   - Final answer prompt includes explicit reasoning steps, cites agent outputs, highlights gaps.

## Tool Registry
- `CompanyFundingTool`: wrapper on processed tables (`master_companies`, `company_investors`).
- `GrowthCompetitorTool`: merges Crunchbase/Growjo outputs.
- `CompanyQAIndex`: FAISS-based semantic search over `company_qa.jsonl`.
- `AcademicSearchTool`: Semantic Scholar/ArXiv API client.
- `NewsResearchTool`: News API or RSS summariser.
- `LinkedInScoutTool`: LinkedIn or people data scraper.
- `EconomicStatsTool`: Official stats API client.
- `LLaMAClient`: calls local/remote LLaMA (LLM inference).

## Agent Config Schema
```json
{
  "name": "FundingIntelligenceAgent",
  "description": "Surface comparable funding amounts, investors.",
  "tools": ["CompanyFundingTool", "CompanyQAIndex"],
  "requires": ["industry", "region"],
  "outputs": ["summary", "funding_totals", "investor_list", "citations"]
}
```

Registry stored as JSON/YAML or Python dict to allow dynamic loading.

## Planner Prompts
1. **Plan Prompt (system -> coordinator LLM)**:
   ```
   Role: Research Planner
   Tasks:
     - Understand structured idea brief.
     - Decompose into sub-queries.
     - Assign each query to an available agent.
     - Specify required inputs and desired outputs.
     - Do NOT invent data; only plan.
   Output format:
     plan: [
       { "task_id": "...", "agent": "...", "instruction": "...", "inputs": {...} },
       ...
     ],
     missing_info: [...]
   ```

2. **Synthesis Prompt (final answer)**:
   ```
   Role: Senior Venture Analyst
   Context: <<aggregated agent outputs>>
   Instructions:
     - Summarise findings with clear reasoning steps.
     - Cite agent names/sources for each claim.
     - Highlight gaps or uncertainties.
     - Provide action-oriented recommendations.
   ```

## Execution Flow
1. User idea -> `IdeaInterpreter` -> structured brief.
2. Planner LLM -> plan with tasks -> `TaskQueue`.
3. For each task:
   - Fetch agent config.
   - Validate required inputs.
   - Execute agent: call tools -> gather evidence.
   - Store result in `ContextStore`.
4. Once all tasks complete:
   - Aggregate evidence.
   - Build final prompt.
   - Call LLaMA/GPT-4.
   - Return answer + reasoning log.

## Implementation Notes
- Use Python classes (`BaseAgent`, `Tool`) for composability.
- Logging: include `task_id`, agent name, success/failure, response size.
- Error handling: on failure, planner receives error message -> can retry or report gap.
- Future integration: add more agents or swap LLMs without altering orchestration core.


