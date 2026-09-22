import React, { useState } from 'react'
import IdeaForm from './components/IdeaForm'
import ResearchProgress from './components/ResearchProgress'
import ResearchResults from './components/ResearchResults'
import { ResearchData, IdeaInput } from './types'

function App() {
  const [researchData, setResearchData] = useState<ResearchData | null>(null)
  const [isResearching, setIsResearching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentIdeaInput, setCurrentIdeaInput] = useState<IdeaInput | null>(null)
  const [currentUseLLM, setCurrentUseLLM] = useState(true)
  const [progress, setProgress] = useState(0)
  const [progressMessage, setProgressMessage] = useState('Initializing...')
  const [tasks, setTasks] = useState<Array<{ id: string; agent: string; status: 'pending' | 'running' | 'completed' | 'error' }>>([])
  const wsRef = React.useRef<WebSocket | null>(null)

  const handleStartResearch = async (ideaInput: IdeaInput, useLLM: boolean) => {
    setIsResearching(true)
    setError(null)
    setResearchData(null)
    setCurrentIdeaInput(ideaInput)
    setCurrentUseLLM(useLLM)
    setProgress(0)
    setProgressMessage('Connecting to server...')
    setTasks([])

    try {
      const ws = new WebSocket('ws://localhost:8000/ws/research')
      wsRef.current = ws
      
      ws.onopen = () => {
        ws.send(JSON.stringify({
          idea_input: ideaInput,
          use_llm: useLLM,
        }))
      }

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        
        if (data.type === 'complete') {
          setResearchData({
            session_id: data.session_id,
            idea_brief: data.idea_brief,
            tasks: data.tasks,
            responses: data.responses,
            report: data.report,
            llm_used: data.llm_used,
          })
          setIsResearching(false)
          setProgress(100)
          ws.close()
        } else if (data.type === 'error') {
          setError(data.message)
          setIsResearching(false)
          ws.close()
        } else if (data.type === 'progress') {
          setProgress(data.progress || 0)
          setProgressMessage(data.message || 'Processing...')
        } else if (data.type === 'tasks') {
          setTasks(data.tasks.map((t: any) => ({
            id: t.task_id,
            agent: t.agent,
            status: 'pending' as const,
          })))
        } else if (data.type === 'task_start') {
          setTasks((prev) => prev.map((t) =>
            t.id === data.task_id ? { ...t, status: 'running' as const } : t
          ))
          setProgress(data.progress || 0)
          setProgressMessage(data.message || 'Running task...')
        } else if (data.type === 'task_complete') {
          setTasks((prev) => prev.map((t) =>
            t.id === data.task_id ? { ...t, status: 'completed' as const } : t
          ))
          setProgress(data.progress || 0)
        } else if (data.type === 'status') {
          setProgressMessage(data.message || 'Processing...')
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setError('Connection error. Make sure the backend server is running on port 8000.')
        setIsResearching(false)
      }

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason)
        if (isResearching && !researchData) {
          if (event.code !== 1000) { // 1000 is normal closure
            setError(`Connection closed unexpectedly. Code: ${event.code}, Reason: ${event.reason || 'Unknown'}`)
          }
          setIsResearching(false)
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start research')
      setIsResearching(false)
    }
  }

  const handleReset = () => {
    setResearchData(null)
    setError(null)
    setIsResearching(false)
  }

  return (
    <div className="min-h-screen">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-4">
            Startup Research Assistant
          </h1>
          <p className="text-xl text-gray-600">
            Comprehensive business intelligence for your startup idea
          </p>
        </header>

        {/* Main Content */}
        {!researchData && !isResearching && (
          <IdeaForm onSubmit={handleStartResearch} />
        )}

        {isResearching && (
          <ResearchProgress 
            onCancel={handleReset} 
            progress={progress}
            progressMessage={progressMessage}
            tasks={tasks}
          />
        )}

        {error && (
          <div className="card max-w-2xl mx-auto mb-8">
            <div className="flex items-center gap-3 text-red-600">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <h3 className="font-semibold text-lg">Error</h3>
                <p className="text-sm">{error}</p>
              </div>
            </div>
            <button
              onClick={handleReset}
              className="btn-secondary mt-4"
            >
              Try Again
            </button>
          </div>
        )}

        {researchData && !isResearching && (
          <ResearchResults data={researchData} onReset={handleReset} />
        )}
      </div>
    </div>
  )
}

export default App

