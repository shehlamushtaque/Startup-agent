## DeepSeek LLM Integration Setup

### Environment Variable

Add your DeepSeek API key to your `.env` file:

```bash
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here
```

### Usage

The system will automatically use DeepSeek for report synthesis if:
1. `DEEPSEEK_API_KEY` is set in environment
2. `--no-llm` flag is NOT used

### Testing

Run with LLM synthesis (default):
```bash
python -m backend.cli --idea "your idea" --problem "problem" --audience "audience" --region "US" --maturity seed
```

Run without LLM (template fallback):
```bash
python -m backend.cli --idea "your idea" --problem "problem" --audience "audience" --region "US" --maturity seed --no-llm
```

### Token Usage

DeepSeek is cost-effective, but the system:
- Curates top 5 evidence items per category to reduce token usage
- Falls back to template-based synthesis on errors
- Reports token usage in the response metadata

### Architecture

1. **Evidence Preparation** (`backend/synthesis/evidence_prep.py`):
   - Converts agent responses to structured JSON
   - Categorizes evidence (funding, competitors, market_trends, regulatory)
   - Extracts key metrics

2. **LLM Client** (`backend/llm/client.py`):
   - DeepSeek API wrapper (OpenAI-compatible)
   - Handles chat completions
   - Builds prompts for business report synthesis

3. **Report Generator** (`backend/synthesis/llm_report.py`):
   - Orchestrates evidence preparation + LLM synthesis
   - Falls back to templates on errors
   - Returns structured report with metadata

