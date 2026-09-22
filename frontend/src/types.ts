export interface IdeaInput {
  idea: string
  problem?: string
  audience?: string
  region?: string
  maturity?: string
  goals?: string
  assumptions?: string
}

export interface Task {
  task_id: string
  agent: string
  instruction: string
  inputs: Record<string, any>
  priority: number
}

export interface AgentResponse {
  agent: string
  task_id: string
  summary: string
  confidence: string
  citations: Citation[]
  evidence_count: number
}

export interface Citation {
  label: string
  url: string
  excerpt?: string
}

export interface ResearchData {
  session_id?: string
  idea_brief: {
    idea_id: string
    raw_input: string
    problem?: string
    audience?: string
    region?: string
    maturity_stage?: string
    goals?: string[]
    assumptions?: string[]
  }
  tasks: Task[]
  responses: AgentResponse[]
  report?: string
  llm_used: boolean
}

export interface ProgressUpdate {
  type: string
  step?: string
  message?: string
  progress?: number
  task_id?: string
  agent?: string
  response?: AgentResponse
}

