# Startup Research Assistant - Frontend

Beautiful, modern React frontend for the Startup Research Assistant application.

## Features

- 🎨 **Modern UI** - Beautiful, responsive design with Tailwind CSS
- ⚡ **Real-time Updates** - WebSocket integration for live research progress
- 📊 **Interactive Results** - View reports, agent responses, and Q&A
- 🚀 **Fast Development** - Built with Vite for instant hot module replacement

## Tech Stack

- **React 18** - Modern React with hooks
- **TypeScript** - Type-safe development
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icon library

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- Backend API server running on port 8000

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Project Structure

```
frontend/
├── src/
│   ├── components/      # React components
│   │   ├── IdeaForm.tsx
│   │   ├── ResearchProgress.tsx
│   │   └── ResearchResults.tsx
│   ├── types.ts         # TypeScript type definitions
│   ├── App.tsx          # Main application component
│   ├── main.tsx         # Application entry point
│   └── index.css        # Global styles
├── index.html
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── vite.config.ts
```

## Features Overview

### Idea Form
- Comprehensive input form for startup idea details
- Optional fields for problem, audience, region, maturity stage
- Toggle for AI-powered synthesis

### Research Progress
- Real-time progress tracking via WebSocket
- Task status indicators
- Visual progress bar

### Research Results
- **Report Tab**: View the synthesized research report
- **Agent Results Tab**: See individual agent responses with citations
- **Q&A Tab**: Ask questions about the research results

## API Integration

The frontend communicates with the backend API via:
- **REST API**: `/api/research` for synchronous research requests
- **WebSocket**: `/ws/research` for real-time progress updates

Make sure the backend server is running before starting the frontend.

