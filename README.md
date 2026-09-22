# Startup Research Assistant

A multi-agent research assistant for startup ideas. Describe an idea, and a set
of specialized agents gather funding history, competitor growth metrics, web
results and SEC filings in parallel, then synthesize them into a single report
you can ask follow-up questions about.

- **Multi-agent pipeline** — funding intelligence, competitor growth, web research, SERP, SEC filings
- **Live progress** — WebSocket streams each agent's status as it runs
- **Synthesis + Q&A** — LLM-generated report with a question-answering tab over the results
- **React frontend** — Vite, TypeScript and Tailwind

---

## Prerequisites

| | Version | Notes |
|---|---|---|
| Python | 3.11+ | Developed and tested on 3.13.3 |
| Node.js | 18+ | Tested on 24.19.0 |
| Disk | ~3 GB | `sentence-transformers` pulls in PyTorch |

---

## 1. Clone

```bash
git clone https://github.com/shehlamushtaque/Startup-agent.git
cd Startup-agent
```

## 2. Backend setup

**Create and activate a virtual environment:**

```bash
python -m venv startup_assistant_env

# Windows (PowerShell)
startup_assistant_env\Scripts\Activate.ps1

# Windows (Git Bash)
source startup_assistant_env/Scripts/activate

# macOS / Linux
source startup_assistant_env/bin/activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

This takes a while — `sentence-transformers` pulls in PyTorch and Transformers
(~2 GB). On first run the app also downloads the `all-MiniLM-L6-v2` embedding
model from Hugging Face.

**Configure API keys:**

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Then open `.env` and fill in the keys you have. **Every key is optional** — the
app degrades gracefully, skipping an agent or falling back to template output
rather than failing:

| Variable | Powers | Without it |
|---|---|---|
| *An LLM provider (see below)* | Report synthesis, Q&A | Deterministic template output; `llm_used: false` |
| `BRAVE_API_KEY` | `WebResearchAgent` | Agent returns no evidence |
| `SERP_API_KEY` | `SerpSearchAgent` | Agent returns no evidence |
| `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` | Google Programmable Search | Falls back to other search agents |

### Choosing an LLM provider

Configure **one** of these. They are tried in order, so the first one
configured wins:

| Priority | Provider | Variables | Notes |
|---|---|---|---|
| 1 | **Ollama** | *(none)* — just run `ollama serve` | Local and free. Override with `OLLAMA_BASE_URL` / `OLLAMA_MODEL`. |
| 2 | **Groq** | `GROQ_API_KEY` | Free tier, no card. Default model `llama-3.3-70b-versatile`; override with `GROQ_MODEL`. |
| 3 | **Any OpenAI-compatible API** | `LLM_API_KEY` + `LLM_BASE_URL` + `LLM_MODEL` | OpenAI, OpenRouter, Together, self-hosted gateways. |
| 4 | **DeepSeek** | `DEEPSEEK_API_KEY` | Override model with `DEEPSEEK_MODEL`. |

All four speak the same interface, so switching providers is a `.env` change —
no code edits required.

**Free-tier token limits.** Synthesis sends a large evidence payload, which can
exceed a free tier's tokens-per-minute cap (Groq's is 8,000). The app detects
this and automatically retries with less evidence, so it degrades in detail
rather than falling back to a template. Set `SYNTHESIS_MAX_EVIDENCE` (default 5)
to control the starting amount.

**Groq model availability varies by account.** If you see a 404 saying the model
does not exist, list the models your key can reach:

```bash
curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
```

Then set `GROQ_MODEL` to one of them.

`.env` is gitignored and must never be committed.

## 3. Frontend setup

```bash
cd frontend
npm install
```

---

## Running the app

The backend and frontend run as two processes, so use two terminals.

**Terminal 1 — backend** (run from the **project root**, not `backend/api/`):

```bash
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

API on http://127.0.0.1:8000 — check http://127.0.0.1:8000/health, which should
return `{"status":"healthy"}`.

> **Why not `python backend/api/run_server.py`?** That entrypoint fails with
> `ModuleNotFoundError: No module named 'backend'`, because uvicorn's reloader
> puts `backend/api/` on `sys.path` instead of the project root. Launch via
> `python -m uvicorn` from the root as above. If you still hit the import error,
> set the path explicitly:
>
> ```bash
> # Windows (PowerShell)
> $env:PYTHONPATH = (Get-Location).Path
> # macOS / Linux / Git Bash
> export PYTHONPATH=$PWD
> ```
>
> Add `--reload` for auto-restart during development.

**Terminal 2 — frontend:**

```bash
cd frontend
npm run dev
```

UI on http://localhost:5173. Open that in your browser — it talks to the
backend on port 8000, which must already be running.

### Using it

1. Enter a startup idea; optionally add problem, audience, region and maturity.
2. Toggle AI synthesis (needs an LLM provider configured — see above).
3. Click **Start Research** and watch the agents report in live.
4. Read the **Report**, inspect **Agent Results**, or ask follow-ups in **Q&A**.

### CLI

```bash
python -m backend.cli --idea "Your startup idea" --problem "Problem it solves" --region "United States"
```

---

## Data files

`data/vector/` ships with the repo, so the Q&A index works out of the box.

`data/raw/` and `data/processed/` are **gitignored** — they hold multi-hundred-MB
Crunchbase and Growjo datasets. A fresh clone therefore won't have them, and
`FundingIntelligenceAgent` and `CompetitorGrowthAgent` will return no evidence
until you supply them. The rest of the pipeline runs fine without them.

To populate those directories, place the source files in `data/raw/`
(`objects.csv`, `Crunchbase.xlsx`, `Growjo-1k-list.csv`,
`simplify-company-data.csv`) and run the cleaning scripts in `scripts/`:

```bash
python scripts/clean_crunchbase_excel.py
python scripts/clean_growjo.py
python scripts/clean_simplify.py
python scripts/build_master_companies.py
```

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | API name and version |
| `GET` | `/health` | Health check |
| `POST` | `/api/research` | Run the pipeline synchronously; returns `session_id` |
| `WS` | `/ws/research` | Same pipeline with streaming progress events |
| `POST` | `/api/qa` | Ask a question; requires the `session_id` from a research run |

Sessions are held in memory, so restarting the backend invalidates any
`session_id` and its Q&A history.

---

## Project structure

```
Startup-agent/
├── backend/
│   ├── api/          # FastAPI app (main.py) and server entrypoint
│   ├── agents/       # Research agent implementations
│   ├── planning/     # Builds the task plan from an idea brief
│   ├── synthesis/    # Report generation
│   ├── qa/           # Question answering over results
│   ├── llm/          # LLM client and provider selection
│   ├── intake/       # Idea interpretation
│   ├── models/       # Pydantic data models
│   ├── cli/          # Command-line interface
│   ├── core/         # Shared primitives
│   ├── config/       # Configuration
│   ├── context/      # Run context
│   ├── interfaces/   # Shared interfaces
│   └── utils/        # Helpers
├── frontend/         # React + Vite + Tailwind UI
├── data/
│   ├── vector/       # FAISS Q&A index (committed)
│   ├── raw/          # Source datasets (gitignored)
│   └── processed/    # Cleaned datasets (gitignored)
├── scripts/          # Data preparation and evaluation scripts
├── docs/             # Methodology and write-ups
└── requirements.txt
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'backend'`** — you launched from the
wrong directory. See the backend run command above.

**`llm_used` is always `false`** — either no LLM provider is configured, or the
configured one is rejecting requests (out of credits, invalid key). The backend
log distinguishes these; look for a line starting `LLM Q&A unavailable:`.

**Agents return zero evidence** — usually a missing API key, a missing dataset
(see *Data files*), or upstream rate limiting. SEC filings in particular return
HTTP 429 under load. Agent failures are isolated: one failing agent doesn't stop
the run.

**`npm run dev` — command not found** — Node isn't on your `PATH`. Verify with
`node --version`.

---

## License

Released under the [MIT License](LICENSE) — © 2026 Shehla Mushtaq.

Note that this covers the code in this repository only. The Crunchbase, Growjo
and Simplify datasets referenced under `data/` are not included in the repo and
carry their own separate licensing terms.
