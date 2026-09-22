# Methodology: Hybrid Multi-Agent Business Intelligence System

## 1. System Architecture Overview

The system implements a multi-agent architecture for autonomous business research and intelligence synthesis. The architecture consists of four primary components: (1) Input Interpretation and Planning, (2) Multi-Agent Execution, (3) Evidence Synthesis, and (4) Interactive Query-Answering. The system is designed to handle diverse business ideas across multiple industries and geographies through a hybrid planning approach that combines rule-based heuristics with LLM-driven task generation.

### 1.1 Core Components

**Input Layer**: Accepts structured business idea descriptions including problem statement, target audience, geographic region, and maturity stage. The `IdeaInterpreter` module normalizes and validates inputs, producing a standardized `IdeaBrief` object.

**Planning Layer**: The `ResearchPlanner` employs a hybrid approach—using a rule-based catalog for well-understood domains and an LLM-based planner for novel or out-of-catalog ideas. This dual-mode planning ensures both reliability for common cases and adaptability for edge cases.

**Execution Layer**: An `Orchestrator` manages agent lifecycle, tool instantiation, and task distribution. Agents operate independently, querying specialized data sources and returning structured evidence.

**Synthesis Layer**: Aggregates evidence from all agents, curates top-relevance items, and generates human-readable reports using either LLM-based narrative synthesis or template-based fallback.

**Query Layer**: Provides interactive Q&A over collected evidence using RAG (Retrieval-Augmented Generation) with LLM-powered answer generation.

## 2. Hybrid Planning Methodology

### 2.1 Rule-Based Planning (Catalog Mode)

For ideas matching known domains (e.g., software, retail, fintech, healthcare), the system uses a deterministic task catalog. The planner analyzes the `IdeaBrief` and applies keyword-based heuristics to select appropriate research tasks.

**Task Selection Logic**:
- **FundingIntelligenceAgent**: Triggered for all ideas; queries company funding databases filtered by industry and geography.
- **CompetitorGrowthAgent**: Activated when industry keywords are detected; searches growth metrics and revenue data.
- **WebResearchAgent**: Always active; performs web search using combined problem/audience/region terms.
- **SerpSearchAgent**: Executes news-focused search with geographic targeting.
- **SecFilingsAgent**: Industry-adaptive SEC Form D search with keyword extraction and filtering.

**Query Construction**:
The planner builds domain-specific search queries. For SEC filings, it:
1. Extracts industry keywords (tech, healthcare, fintech)
2. Combines industry terms with main business concepts
3. Applies universal exclusion filters (real estate funds, investment vehicles)
4. Constructs Boolean queries: `(industry_keywords) AND (main_terms)`

**Confidence Scoring**:
Each task is assigned a priority based on keyword match strength and domain coverage. Tasks with higher keyword overlap receive higher priority in execution order.

### 2.2 LLM-Based Planning (Adaptive Mode)

When an idea falls outside the catalog (detected via keyword coverage < 30% or ≥3 novel terms), the system invokes an LLM planner.

**Trigger Conditions**:
- Keyword coverage threshold: `coverage = matched_keywords / total_keywords < 0.3`
- Novelty detection: `unique_novel_terms ≥ 3`
- Empty or ambiguous problem statement

**LLM Planning Process**:
1. **Context Preparation**: Formats `IdeaBrief` into structured context including idea, problem, audience, region, maturity stage, and available agent list.
2. **Prompt Engineering**: System prompt instructs the LLM to generate JSON task specifications using only available agents. User prompt includes full context and example JSON schema.
3. **Task Generation**: LLM produces a JSON array of 3-6 tasks, each specifying:
   - `agent`: One of the available agent names
   - `instruction`: Natural language task description tailored to the idea
   - `inputs`: Dictionary of agent-specific parameters
   - `priority`: Integer ordering (0 = highest priority)
4. **Validation and Filtering**: 
   - Parses JSON response
   - Validates agent names against available agents
   - Filters out invalid or duplicate tasks
   - Falls back to rule-based plan if LLM output is invalid or empty

**Fallback Strategy**:
If LLM planning fails (parsing error, invalid agents, empty response), the system automatically falls back to the rule-based catalog plan, ensuring system reliability.

### 2.3 Hybrid Decision Logic

The planner uses a two-stage decision process:

```python
def create_plan(idea_brief):
    # Stage 1: Check if LLM planning should be used
    if _should_use_llm(idea_brief) and llm_available:
        tasks = _llm_plan(idea_brief)
        if tasks:  # Valid LLM tasks generated
            return tasks
    
    # Stage 2: Fallback to rule-based catalog
    return _catalog_plan(idea_brief)
```

This ensures:
- **Performance**: Common ideas use fast, deterministic planning
- **Adaptability**: Novel ideas get custom task generation
- **Reliability**: Always produces a valid plan, even if LLM fails

## 3. Multi-Agent Execution Framework

### 3.1 Agent Architecture

Each agent implements the `BaseAgent` interface with:
- `name`: Unique identifier
- `execute(task)`: Main execution method returning `AgentResponse`
- Tool dependencies: Declared at initialization

**Agent Response Structure**:
```python
AgentResponse(
    agent_name: str,
    task_id: str,
    summary: str,  # Human-readable summary
    confidence: str,  # "high", "medium", "low"
    evidence: List[Evidence],  # Structured evidence items
    citations: List[str],  # Source URLs/identifiers
    raw_output: Dict  # Raw tool responses
)
```

**Evidence Structure**:
```python
Evidence(
    title: str,  # Brief title
    content: str,  # Main content/description
    source: str,  # URL or tool identifier
    metadata: Dict,  # Structured data (funding amounts, growth %, etc.)
    score: float  # Optional relevance score
)
```

### 3.2 Specialized Agents

**FundingIntelligenceAgent**:
- **Tools**: `CompanyFundingTool`, `CompanyQAIndexTool`
- **Process**: 
  1. Queries funding database filtered by industry/geography
  2. Performs semantic search over QA index for related insights
  3. Formats results with funding amounts, investor names, round counts
- **Output**: List of comparable companies with funding metrics

**CompetitorGrowthAgent**:
- **Tools**: `CompetitorGrowthTool`
- **Process**:
  1. Searches growth database by industry keywords
  2. Applies data cleaning (handles NaN values, normalizes metrics)
  3. Prioritizes companies with actual growth/revenue data
  4. Falls back to global top performers if filters over-constrain
- **Output**: Growth snapshots with revenue, employee count, growth percentage

**WebResearchAgent**:
- **Tools**: `BraveSearchTool`
- **Process**:
  1. Constructs search query from idea components
  2. Calls Brave Search API with result limit
  3. Extracts titles, snippets, URLs
- **Output**: Web search results with relevance-ranked snippets

**SerpSearchAgent**:
- **Tools**: `SerpApiTool`
- **Process**:
  1. Performs news-focused search via SerpAPI
  2. Filters by geographic location
  3. Extracts article titles, descriptions, publication dates
- **Output**: Recent news articles related to the idea

**SecFilingsAgent**:
- **Tools**: `SecFormDTool`
- **Process**:
  1. Builds industry-adaptive SEC search query
  2. Calls SEC Full-Text Search API (POST to `/full-text-search`)
  3. Applies aggressive filtering to exclude investment funds, real estate entities
  4. Optionally enriches with XML parsing for detailed filing data
  5. Formats results with company names, filing dates, amounts raised
- **Output**: Recent Form D filings with extracted metadata

### 3.3 Tool Implementations

**CompanyFundingTool**:
- **Data Source**: Parquet file (`master_companies.parquet` or `master_companies_enriched.parquet`)
- **Query Logic**: Filters by `industry` (fuzzy match on problem statement) and `country` (geography filter)
- **Data Cleaning**: Coerces numeric columns, handles missing values
- **Output**: Top N companies sorted by funding amount

**CompetitorGrowthTool**:
- **Data Source**: Same parquet file with Growjo metrics
- **Query Logic**: Industry keyword matching, growth/revenue filtering
- **Normalization**: Handles missing `growjo_growth_percent`, `growjo_estimated_revenue_usd`, `growjo_employees`
- **Fallback**: If filters return empty, uses global metric-bearing subset

**CompanyQAIndexTool**:
- **Data Source**: FAISS vector index + JSONL metadata
- **Model**: SentenceTransformer (`all-MiniLM-L6-v2`)
- **Process**: 
  1. Encodes question into embedding
  2. Searches FAISS index for top-k similar vectors
  3. Retrieves corresponding metadata (company, QA pairs)
- **Output**: Semantic search results with similarity scores

**BraveSearchTool**:
- **API**: Brave Search API (`/v1/web/search`)
- **Authentication**: API key in `Authorization` header
- **Parameters**: Query string, result count, safe search settings
- **Output**: Web search results with titles, snippets, URLs

**SerpApiTool**:
- **API**: SerpAPI (`/search.json`)
- **Parameters**: Query, location, search type (news), result limit
- **Output**: News articles with titles, snippets, publication info

**SecFormDTool**:
- **API**: SEC Full-Text Search API (`/full-text-search`)
- **Method**: POST with JSON payload
- **Query Construction**: Boolean operators, phrase matching, date filters
- **XML Enrichment**: Optional parsing of Form D XML to extract:
  - Total offering amount
  - Amount sold
  - Investor names
  - Minimum investment
- **Filtering**: Post-processing to exclude funds, real estate entities based on company name patterns

## 4. Evidence Synthesis Methodology

### 4.1 Evidence Preparation

The synthesis layer receives `AgentResponse` objects from all agents and prepares structured evidence for LLM consumption.

**Evidence Curation**:
1. **Flattening**: Extracts all `Evidence` items from agent responses
2. **Categorization**: Groups evidence by agent type (funding, competitors, market_trends, regulatory)
3. **Metadata Extraction**: Normalizes metadata fields (funding amounts, growth percentages, dates)
4. **Relevance Scoring**: Assigns scores based on agent confidence and evidence quality
5. **Top-K Selection**: Selects top N evidence items per category to manage token limits

**Structured Evidence Format**:
```json
{
  "idea_summary": {
    "idea": "...",
    "problem": "...",
    "audience": "...",
    "region": "...",
    "maturity": "..."
  },
  "evidence_by_category": {
    "funding": [...],
    "competitors": [...],
    "market_trends": [...],
    "regulatory": [...]
  },
  "evidence_index": {
    "E1": {...},
    "E2": {...}
  },
  "key_metrics": {
    "funding": [...],
    "competitors": [...],
    "market_signals": 0,
    "regulatory_filings": 0
  },
  "total_evidence_count": N
}
```

### 4.2 LLM-Based Synthesis

**Model Selection**: 
- Primary: Ollama (local LLM, default: `llama3.1:8b`)
- Fallback: DeepSeek API (cloud-based)
- Auto-detection: System checks Ollama availability first, falls back to DeepSeek if unavailable

**Prompt Engineering**:
- **System Prompt**: Defines role as "business intelligence analyst" with guidelines for:
  - Extracting key insights from evidence
  - Connecting patterns across data sources
  - Providing actionable recommendations
  - Citing evidence IDs
  - Highlighting surprising insights
- **User Prompt**: Includes:
  - Full `IdeaBrief` context
  - Structured evidence JSON
  - Instructions for synthesis format (Executive Summary, Key Findings, Strategic Implications, Recommendations)

**Synthesis Process**:
1. Curates evidence to top 5 items per category
2. Formats evidence with IDs (E1, E2, ...)
3. Sends to LLM with temperature=0.7, max_tokens=4000
4. Parses response into report sections
5. Handles errors gracefully with template fallback

**Error Handling**:
- API failures (402 Insufficient Balance, 401 Unauthorized): Falls back to template
- Connection errors (Ollama unavailable): Falls back to DeepSeek or template
- Invalid responses: Uses template-based report

### 4.3 Template-Based Fallback

When LLM synthesis is unavailable, the system uses a deterministic template:

1. **Funding Landscape**: Lists top funding amounts with company names
2. **Competitive Field**: Shows growth metrics, revenue, employee counts
3. **Market Signals**: Displays web search and news article titles
4. **Regulatory Filings**: Lists Form D filings with dates and amounts

Template ensures system always produces a report, even without LLM.

## 5. Interactive Query-Answering System

### 5.1 Evidence Indexing

The Q&A system builds a searchable index of all collected evidence:

**Index Construction**:
1. Flattens all evidence from agent responses
2. Extracts full-text searchable content:
   - Evidence title
   - Evidence content
   - Source URLs
   - Metadata fields (converted to text)
   - Agent summary
3. Stores as list of dictionaries with `full_text` field

### 5.2 Retrieval Mechanism

**Keyword-Based Search**:
- Tokenizes query into terms (filters words < 3 characters)
- Scores evidence items based on:
  - Exact phrase match: +20 points
  - Semantic keyword matching: +3 points per related term
  - Individual term matches: +2 points per term
  - Business-relevant agent boost: +5 points
  - Concrete metrics boost: +3 points
- Returns top-k (default: 10) most relevant evidence items

**Semantic Keyword Mapping**:
Maintains a dictionary mapping query terms to related business concepts:
- "invest" → ["funding", "investment", "capital", "raise", "money"]
- "profit" → ["revenue", "growth", "success", "viable", "profitable"]
- "competitor" → ["competition", "similar", "market", "industry", "rival"]

**Context Enrichment**:
- Includes all business-relevant evidence (funding, competitors, SEC) even if not top-ranked
- Deduplicates by title
- Limits additional context to prevent token overflow

### 5.3 Answer Generation

**LLM-Powered Answers**:
- **System Prompt**: Instructs LLM to:
  - Analyze evidence and provide synthesized insights
  - Use specific numbers and metrics
  - Cite evidence naturally with [Evidence N] format
  - Provide actionable recommendations when asked
  - Be concise and fact-based
- **User Prompt**: Includes:
  - Full question
  - All relevant evidence with detailed metadata
  - Instructions for answer format (Answer + Supporting facts)
- **Output Format**: 
  ```
  Answer: <direct statement with numbers>
  
  Supporting facts:
  - <fact with number> [Evidence N]
  - ...
  ```
- **Temperature**: 0.5 (balanced creativity/accuracy)
- **Max Tokens**: 1000

**Template-Based Fallback**:
- Pattern matching for common question types (funding, competitors, trends, filings)
- Extracts relevant metrics from evidence
- Formats as simple bullet points
- Only used if LLM unavailable

### 5.4 Evidence Filtering

The Q&A system filters evidence by business relevance:
- **Always Includes**: FundingIntelligenceAgent, CompetitorGrowthAgent, SecFilingsAgent results
- **Conditionally Includes**: WebResearchAgent, SerpSearchAgent results only if they contain business keywords (funding, investment, startup, company, market, revenue, growth, etc.)
- **Purpose**: Prevents irrelevant web results (e.g., alumni stories) from polluting answers

## 6. Data Sources and Integration

### 6.1 Local Data Sources

**Master Companies Database**:
- Format: Parquet file
- Columns: Company names, funding totals, growth metrics, employee counts, investor names
- Enrichment: Merged with Growjo data for growth/revenue metrics
- Access: Pandas DataFrame queries with filtering

**QA Vector Index**:
- Format: FAISS index + JSONL metadata
- Embeddings: SentenceTransformer (`all-MiniLM-L6-v2`)
- Content: Company-specific Q&A pairs
- Access: Semantic similarity search

### 6.2 External APIs

**Brave Search API**:
- Endpoint: `https://api.search.brave.com/v1/web/search`
- Authentication: API key in header
- Rate Limits: Varies by plan
- Use Case: General web search

**SerpAPI**:
- Endpoint: `https://serpapi.com/search.json`
- Authentication: API key in query parameter
- Rate Limits: Varies by plan
- Use Case: News-focused search with geographic targeting

**SEC Full-Text Search API**:
- Endpoint: `https://api.sec-api.io/full-text-search`
- Method: POST with JSON payload
- Authentication: API key in `Authorization` header
- Query Features: Boolean operators, phrase matching, date ranges, form type filters
- Use Case: Regulatory filing discovery

**Ollama (Local LLM)**:
- Endpoint: `http://localhost:11434/api/generate` (default)
- Model: Configurable (default: `llama3.1:8b`)
- Use Case: LLM planning, synthesis, Q&A
- Advantages: Free, local, private

**DeepSeek API**:
- Endpoint: `https://api.deepseek.com/v1/chat/completions`
- Authentication: API key in header
- Model: `deepseek-chat`
- Use Case: Fallback LLM when Ollama unavailable
- Cost: Pay-per-use

### 6.3 Data Enrichment Pipeline

**SEC XML Parsing**:
- Fetches Form D XML from SEC.gov
- Parses using `xml.etree.ElementTree` and regex fallback
- Extracts: offering amounts, investor names, filing dates
- Handles: Namespace variations, malformed XML, 403 Forbidden errors
- Enriches: Base filing metadata with extracted details

**Data Normalization**:
- Funding amounts: Coerced to float, formatted as $X.YM
- Growth percentages: Normalized to float, displayed as X%
- Dates: Standardized to ISO format (YYYY-MM-DD)
- Company names: Cleaned of redundant suffixes, normalized case

## 7. Error Handling and Resilience

### 7.1 Graceful Degradation

The system implements multiple fallback layers:

1. **LLM Planning Failure** → Rule-based catalog plan
2. **LLM Synthesis Failure** → Template-based report
3. **LLM Q&A Failure** → Template-based answers
4. **API Failures** → Continue with available data, log errors
5. **Data Source Missing** → Skip agent, continue with others

### 7.2 Error Detection

**API Errors**:
- 401 Unauthorized: Invalid API key
- 402 Insufficient Balance: Account needs credits
- 403 Forbidden: Access denied (e.g., SEC.gov blocking)
- 404 Not Found: Endpoint or resource missing
- Connection errors: Network issues, service unavailable

**Data Errors**:
- Missing files: FileNotFoundError handling
- Invalid data types: Type coercion with error handling
- Empty results: Graceful handling, returns empty evidence lists

### 7.3 User Feedback

System provides clear error messages:
- LLM unavailable: Explains why (no API key, insufficient credits, Ollama not running)
- Template fallback: Indicates when template is used vs LLM
- Missing data: Reports which agents returned no results

## 8. Performance Considerations

### 8.1 Parallelization Opportunities

Currently, agents execute sequentially. Future improvements:
- Parallel agent execution (threading/async)
- Batch API calls where supported
- Caching of API responses

### 8.2 Token Management

- Evidence curation limits items per category (default: 5)
- Q&A context limited to top 10-15 evidence items
- Max tokens set per LLM call (4000 for synthesis, 1000 for Q&A)
- Truncation of long evidence content

### 8.3 Caching Strategy

- No caching currently implemented
- Potential: Cache API responses, LLM outputs for identical queries
- Trade-off: Freshness vs performance

## 9. Evaluation Metrics

### 9.1 Task Relevance

- **Catalog Mode**: Keyword coverage score (matched / total keywords)
- **LLM Mode**: Validation rate (valid tasks / total LLM tasks)

### 9.2 Evidence Quality

- **Coverage**: Number of evidence items per category
- **Relevance**: Agent confidence scores
- **Completeness**: Presence of key metrics (funding, growth, revenue)

### 9.3 Synthesis Quality

- **LLM Usage Rate**: Percentage of reports using LLM vs template
- **Evidence Citations**: Number of evidence IDs cited in report
- **Actionability**: Presence of specific recommendations

### 9.4 Q&A Performance

- **Answer Relevance**: Manual evaluation of answer quality
- **Evidence Usage**: Number of evidence items referenced
- **Response Time**: Latency of Q&A generation

## 10. Limitations and Future Work

### 10.1 Current Limitations

1. **Data Coverage**: Limited to available datasets (funding DB, Growjo metrics, SEC filings)
2. **Agent Catalog**: Fixed set of agents; new domains require code changes
3. **LLM Dependency**: Quality depends on LLM capabilities and availability
4. **Sequential Execution**: Agents run one at a time
5. **No Learning**: System doesn't improve from past runs

### 10.2 Future Enhancements

1. **Expanded Agent Catalog**: Add domain-specific agents (FDA filings, patent search, etc.)
2. **Learning from Feedback**: Incorporate user feedback to improve planning
3. **Multi-Model Ensemble**: Use multiple LLMs and combine outputs
4. **Real-Time Data**: Integrate live data feeds for market trends
5. **Confidence Calibration**: Better confidence scoring for evidence and answers
6. **Explainability**: Provide reasoning traces for LLM decisions

## 11. Reproducibility

### 11.1 Environment Setup

- Python 3.8+
- Dependencies: `requirements.txt` (pandas, requests, faiss, sentence-transformers, openai, python-dotenv)
- Environment variables: API keys for external services
- Data files: Parquet databases, FAISS indexes

### 11.2 Configuration

- Agent registry: `backend/agents/orchestrator.py` (DEFAULT_AGENT_CONFIGS, DEFAULT_TOOL_CONFIGS)
- Planner settings: `backend/planning/research_planner.py` (keyword lists, thresholds)
- LLM settings: Environment variables (OLLAMA_BASE_URL, OLLAMA_MODEL, DEEPSEEK_API_KEY)

### 11.3 Execution

- CLI: `python -m backend.cli.main --idea "..." --problem "..." ...`
- Programmatic: Import `Orchestrator`, `ResearchPlanner`, `IdeaInterpreter`

---

*This methodology document describes the system as of the hybrid planner implementation. The system continues to evolve with additional agents, data sources, and improvements.*

