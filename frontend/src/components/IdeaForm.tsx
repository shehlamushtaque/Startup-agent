import { useState } from 'react'
import { IdeaInput } from '../types'
import { Sparkles, Lightbulb, Target, Globe, TrendingUp, HelpCircle } from 'lucide-react'

interface IdeaFormProps {
  onSubmit: (ideaInput: IdeaInput, useLLM: boolean) => void
}

export default function IdeaForm({ onSubmit }: IdeaFormProps) {
  const [idea, setIdea] = useState('')
  const [problem, setProblem] = useState('')
  const [audience, setAudience] = useState('')
  const [region, setRegion] = useState('')
  const [maturity, setMaturity] = useState('')
  const [goals, setGoals] = useState('')
  const [assumptions, setAssumptions] = useState('')
  const [useLLM, setUseLLM] = useState(true)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!idea.trim()) return

    onSubmit({
      idea: idea.trim(),
      problem: problem.trim() || undefined,
      audience: audience.trim() || undefined,
      region: region.trim() || undefined,
      maturity: maturity || undefined,
      goals: goals.trim() || undefined,
      assumptions: assumptions.trim() || undefined,
    }, useLLM)
  }

  return (
    <div className="max-w-4xl mx-auto">
      <form onSubmit={handleSubmit} className="card space-y-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-800">Describe Your Startup Idea</h2>
            <p className="text-gray-600">Provide details to get comprehensive research insights</p>
          </div>
        </div>

        {/* Main Idea */}
        <div>
          <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
            <Lightbulb className="w-4 h-4 text-yellow-500" />
            Startup Idea <span className="text-red-500">*</span>
          </label>
          <textarea
            value={idea}
            onChange={(e) => setIdea(e.target.value)}
            placeholder="Describe your startup idea in detail..."
            className="input-field min-h-[120px] resize-none"
            required
          />
        </div>

        {/* Two Column Layout */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Problem */}
          <div>
            <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
              <HelpCircle className="w-4 h-4 text-blue-500" />
              Problem / Industry Focus
            </label>
            <input
              type="text"
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
              placeholder="What problem does it solve?"
              className="input-field"
            />
          </div>

          {/* Audience */}
          <div>
            <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
              <Target className="w-4 h-4 text-green-500" />
              Target Audience
            </label>
            <input
              type="text"
              value={audience}
              onChange={(e) => setAudience(e.target.value)}
              placeholder="Who is your target customer?"
              className="input-field"
            />
          </div>

          {/* Region */}
          <div>
            <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
              <Globe className="w-4 h-4 text-purple-500" />
              Region
            </label>
            <input
              type="text"
              value={region}
              onChange={(e) => setRegion(e.target.value)}
              placeholder="Geographic focus (e.g., United States)"
              className="input-field"
            />
          </div>

          {/* Maturity */}
          <div>
            <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
              <TrendingUp className="w-4 h-4 text-orange-500" />
              Maturity Stage
            </label>
            <select
              value={maturity}
              onChange={(e) => setMaturity(e.target.value)}
              className="input-field"
            >
              <option value="">Select stage...</option>
              <option value="concept">Concept</option>
              <option value="mvp">MVP</option>
              <option value="pre-seed">Pre-Seed</option>
              <option value="seed">Seed</option>
              <option value="growth">Growth</option>
            </select>
          </div>
        </div>

        {/* Goals */}
        <div>
          <label className="text-sm font-semibold text-gray-700 mb-2 block">
            Business Goals (comma-separated)
          </label>
          <input
            type="text"
            value={goals}
            onChange={(e) => setGoals(e.target.value)}
            placeholder="e.g., Validate market, Find investors, Understand competition"
            className="input-field"
          />
        </div>

        {/* Assumptions */}
        <div>
          <label className="text-sm font-semibold text-gray-700 mb-2 block">
            Assumptions / Hypotheses (comma-separated)
          </label>
          <input
            type="text"
            value={assumptions}
            onChange={(e) => setAssumptions(e.target.value)}
            placeholder="e.g., Market size is $X, Customers need Y, Competitor Z exists"
            className="input-field"
          />
        </div>

        {/* LLM Toggle */}
        <div className="flex items-center gap-3 p-4 bg-blue-50 rounded-xl border border-blue-200">
          <input
            type="checkbox"
            id="useLLM"
            checked={useLLM}
            onChange={(e) => setUseLLM(e.target.checked)}
            className="w-5 h-5 text-blue-600 rounded focus:ring-blue-500"
          />
          <label htmlFor="useLLM" className="text-sm text-gray-700 cursor-pointer">
            <span className="font-semibold">Use AI-powered synthesis</span>
            <span className="text-gray-500 block mt-1">
              Enable LLM for enhanced report generation (requires Ollama or DeepSeek API)
            </span>
          </label>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!idea.trim()}
          className="btn-primary w-full text-lg py-4"
        >
          <span className="flex items-center justify-center gap-2">
            <Sparkles className="w-5 h-5" />
            Start Research
          </span>
        </button>
      </form>
    </div>
  )
}

