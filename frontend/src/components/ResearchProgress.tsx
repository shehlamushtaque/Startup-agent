import { useEffect, useState, useRef } from 'react'
import { Loader2, CheckCircle2, XCircle, AlertCircle } from 'lucide-react'
import { ProgressUpdate } from '../types'

interface ResearchProgressProps {
  onCancel: () => void
  progress: number
  progressMessage: string
  tasks: Array<{ id: string; agent: string; status: 'pending' | 'running' | 'completed' | 'error' }>
}

export default function ResearchProgress({ onCancel, progress, progressMessage, tasks }: ResearchProgressProps) {

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="w-5 h-5 text-green-500" />
      case 'running':
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
      case 'error':
        return <XCircle className="w-5 h-5 text-red-500" />
      default:
        return <div className="w-5 h-5 rounded-full border-2 border-gray-300" />
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="card">
        <div className="flex items-center gap-4 mb-6">
          <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl">
            <Loader2 className="w-6 h-6 text-white animate-spin" />
          </div>
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-gray-800">Research in Progress</h2>
            <p className="text-gray-600">{progressMessage}</p>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-semibold text-gray-700">Overall Progress</span>
            <span className="text-sm font-semibold text-blue-600">{progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Task List */}
        {tasks.length > 0 && (
          <div className="space-y-3 mb-6">
            <h3 className="font-semibold text-gray-700 mb-3">Research Tasks</h3>
            {tasks.map((task) => (
              <div
                key={task.id}
                className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200"
              >
                {getStatusIcon(task.status)}
                <span className="flex-1 text-sm text-gray-700">{task.agent}</span>
                <span className="text-xs text-gray-500 capitalize">{task.status}</span>
              </div>
            ))}
          </div>
        )}

        {/* Info Message */}
        <div className="flex items-start gap-3 p-4 bg-blue-50 rounded-xl border border-blue-200">
          <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
          <div className="text-sm text-gray-700">
            <p className="font-semibold mb-1">Research is running...</p>
            <p className="text-gray-600">
              Our AI agents are gathering comprehensive intelligence about your startup idea.
              This may take a few minutes.
            </p>
          </div>
        </div>

        <button
          onClick={onCancel}
          className="btn-secondary w-full mt-6"
        >
          Cancel Research
        </button>
      </div>
    </div>
  )
}

