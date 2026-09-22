# Startup Research Assistant

A comprehensive startup research assistant application that helps entrepreneurs with business intelligence, market research, competitor analysis, and funding insights through a beautiful web interface.

## Features

- 🚀 **Multi-Agent Research System** - Automated research using specialized AI agents
- 📊 **Comprehensive Analysis** - Funding intelligence, competitor analysis, SEC filings, web research
- 🎨 **Beautiful Web UI** - Modern, responsive React frontend with real-time updates
- 🤖 **AI-Powered Synthesis** - LLM-powered report generation (Ollama or DeepSeek)
- 💬 **Interactive Q&A** - Ask questions about your research results
- ⚡ **Real-time Progress** - WebSocket-based live updates during research

## Project Structure

```
refined-startup-agent/
├── backend/
│   ├── api/              # FastAPI backend server
│   ├── agents/           # Research agent implementations
│   ├── planning/         # Research planning logic
│   ├── synthesis/        # Report synthesis
│   ├── qa/              # Q&A system
│   ├── models/          # Data models & schemas
│   └── utils/           # Utility functions
├── frontend/            # React frontend application
│   ├── src/
│   │   ├── components/  # UI components
│   │   └── types.ts     # TypeScript types
│   └── package.json
├── data/                # Data files and cache
├── scripts/             # Utility scripts
├── requirements.txt     # Python dependencies
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+ and npm
- (Optional) Ollama for local LLM or DeepSeek API key

### Backend Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv startup_assistant_env
   
   # Windows
   startup_assistant_env\Scripts\activate
   
   # macOS/Linux
   source startup_assistant_env/bin/activate
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file in the project root
   - Add your API keys (SerpAPI, Brave Search, DeepSeek, etc.)
   - See `.env.example` for required variables

4. **Run the backend server:**
   ```bash
   python backend/api/run_server.py
   ```
   
   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```
   
   The frontend will be available at `http://localhost:5173`

### Running the Full Application

1. **Terminal 1 - Start Backend:**
   ```bash
   python backend/api/run_server.py
   ```

2. **Terminal 2 - Start Frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Open your browser:**
   Navigate to `http://localhost:5173`

## Usage

### Web Interface

1. Enter your startup idea in the form
2. Optionally fill in problem, audience, region, maturity stage
3. Toggle AI-powered synthesis (requires Ollama or DeepSeek)
4. Click "Start Research" and watch real-time progress
5. View results in the Report, Agent Results, or Q&A tabs

### CLI Interface

You can also use the CLI interface:

```bash
python -m backend.cli --idea "Your startup idea" --problem "Problem it solves" --region "United States"
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /api/research` - Run research synchronously
- `WS /ws/research` - WebSocket endpoint for real-time research
- `POST /api/qa` - Ask questions about research results

## Development

### Backend Development

- API server auto-reloads on file changes
- Check `backend/api/main.py` for API routes
- Agent implementations in `backend/agents/`

### Frontend Development

- Hot module replacement enabled
- TypeScript for type safety
- Tailwind CSS for styling
- Components in `frontend/src/components/`

## License

[Add license information here]


