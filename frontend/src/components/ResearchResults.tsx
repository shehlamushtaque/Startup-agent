import { useState } from 'react'
import { ResearchData } from '../types'
import ReactMarkdown from 'react-markdown'
import { 
  FileText, CheckCircle2, AlertCircle, TrendingUp, 
  ExternalLink, MessageSquare, RefreshCw, Download,
  Sparkles, Users, Globe, Target
} from 'lucide-react'
import { jsPDF } from 'jspdf'

interface ResearchResultsProps {
  data: ResearchData
  onReset: () => void
}

export default function ResearchResults({ data, onReset }: ResearchResultsProps) {
  const [activeTab, setActiveTab] = useState<'report' | 'agents' | 'qa'>('report')
  const [question, setQuestion] = useState('')
  const [qaAnswer, setQaAnswer] = useState<string | null>(null)
  const [qaSources, setQaSources] = useState<any[]>([])
  const [isAsking, setIsAsking] = useState(false)

  const handleAskQuestion = async () => {
    if (!question.trim()) return

    setIsAsking(true)
    try {
      const response = await fetch('/api/qa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question.trim(),
          session_id: data.session_id,
          use_llm: data.llm_used,
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to get answer')
      }

      const result = await response.json()
      setQaAnswer(result.answer || 'No answer available.')
      setQaSources(result.sources || [])
    } catch (error) {
      setQaAnswer(error instanceof Error ? error.message : 'Failed to get answer. Please try again.')
      setQaSources([])
    } finally {
      setIsAsking(false)
    }
  }

  const handleDownloadReport = () => {
    if (!data.report) return
    const doc = new jsPDF({ unit: 'pt', format: 'a4' })
    const margin = 40
    const maxWidth = 515
    const reportText = doc.splitTextToSize(data.report, maxWidth)
    doc.setFont('Helvetica', 'normal')
    doc.setFontSize(12)
    doc.text(reportText, margin, margin)
    const filename = `startup-report-${data.idea_brief.idea_id || 'idea'}.pdf`
    doc.save(filename)
  }

  const getConfidenceBadge = (confidence: string) => {
    const colors = {
      high: 'bg-green-100 text-green-800',
      medium: 'bg-yellow-100 text-yellow-800',
      low: 'bg-red-100 text-red-800',
    }
    return (
      <span className={`badge ${colors[confidence as keyof typeof colors] || colors.medium}`}>
        {confidence.toUpperCase()}
      </span>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg">
                <CheckCircle2 className="w-6 h-6 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-gray-800">Research Complete</h2>
              {data.llm_used && (
                <span className="badge bg-purple-100 text-purple-800">
                  <Sparkles className="w-3 h-3 mr-1" />
                  AI-Powered
                </span>
              )}
            </div>
            <p className="text-gray-600 mb-4">{data.idea_brief.raw_input}</p>
            
            {/* Idea Brief Details */}
            <div className="grid md:grid-cols-3 gap-4 mt-4">
              {data.idea_brief.problem && (
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <AlertCircle className="w-4 h-4 text-blue-500" />
                  <span className="font-semibold">Problem:</span>
                  <span>{data.idea_brief.problem}</span>
                </div>
              )}
              {data.idea_brief.audience && (
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <Target className="w-4 h-4 text-green-500" />
                  <span className="font-semibold">Audience:</span>
                  <span>{data.idea_brief.audience}</span>
                </div>
              )}
              {data.idea_brief.region && (
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <Globe className="w-4 h-4 text-purple-500" />
                  <span className="font-semibold">Region:</span>
                  <span>{data.idea_brief.region}</span>
                </div>
              )}
            </div>
          </div>
          <button onClick={onReset} className="btn-secondary">
            <RefreshCw className="w-4 h-4 mr-2" />
            New Research
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="card">
        <div className="flex gap-2 border-b border-gray-200 mb-6">
          <button
            onClick={() => setActiveTab('report')}
            className={`px-6 py-3 font-semibold transition-colors border-b-2 ${
              activeTab === 'report'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <FileText className="w-4 h-4 inline mr-2" />
            Report
          </button>
          <button
            onClick={() => setActiveTab('agents')}
            className={`px-6 py-3 font-semibold transition-colors border-b-2 ${
              activeTab === 'agents'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <TrendingUp className="w-4 h-4 inline mr-2" />
            Agent Results ({data.responses.length})
          </button>
          <button
            onClick={() => setActiveTab('qa')}
            className={`px-6 py-3 font-semibold transition-colors border-b-2 ${
              activeTab === 'qa'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <MessageSquare className="w-4 h-4 inline mr-2" />
            Q&A
          </button>
        </div>

        {/* Report Tab */}
        {activeTab === 'report' && (
          <div className="space-y-4">
            {data.report ? (
              <div className="prose max-w-none">
                <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
                  <div className="prose prose-sm max-w-none">
                    <ReactMarkdown
                      components={{
                        a: ({node, ...props}) => (
                          <a {...props} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800 hover:underline" />
                        ),
                        h2: ({node, ...props}) => (
                          <h2 {...props} className="text-xl font-bold mt-6 mb-3 text-gray-800" />
                        ),
                        h3: ({node, ...props}) => (
                          <h3 {...props} className="text-lg font-semibold mt-4 mb-2 text-gray-700" />
                        ),
                        p: ({node, ...props}) => (
                          <p {...props} className="mb-3 text-gray-700 leading-relaxed" />
                        ),
                        ul: ({node, ...props}) => (
                          <ul {...props} className="list-disc list-inside mb-3 space-y-1 text-gray-700" />
                        ),
                        li: ({node, ...props}) => (
                          <li {...props} className="ml-4" />
                        ),
                        strong: ({node, ...props}) => (
                          <strong {...props} className="font-semibold text-gray-800" />
                        ),
                        em: ({node, ...props}) => (
                          <em {...props} className="italic text-gray-600" />
                        ),
                      }}
                    >
                      {data.report}
                    </ReactMarkdown>
                  </div>
                </div>
                <button className="btn-secondary mt-4" onClick={handleDownloadReport}>
                  <Download className="w-4 h-4 mr-2" />
                  Download Report
                </button>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <FileText className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p>No report available</p>
              </div>
            )}
          </div>
        )}

        {/* Agents Tab */}
        {activeTab === 'agents' && (
          <div className="space-y-4">
            {data.responses.map((response, idx) => (
              <div key={response.task_id} className="border border-gray-200 rounded-xl p-6 hover:shadow-lg transition-shadow">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <TrendingUp className="w-5 h-5 text-blue-600" />
                    </div>
                    <div>
                      <h3 className="font-bold text-lg text-gray-800">{response.agent}</h3>
                      <p className="text-sm text-gray-500">Task {idx + 1} of {data.responses.length}</p>
                    </div>
                  </div>
                  {getConfidenceBadge(response.confidence)}
                </div>
                
                <p className="text-gray-700 mb-4">{response.summary}</p>
                
                {/* Always show citations section if citations exist */}
                {response.citations && Array.isArray(response.citations) && response.citations.length > 0 ? (() => {
                  // Filter out tool names - these are internal programming functions, not real sources
                  const toolNames = ['CompanyFundingTool', 'CompanyQAIndexTool', 'CompetitorGrowthTool', 
                                    'SerpApiTool', 'BraveSearchTool', 'SecFilingsTool'];
                  
                  const validCitations = response.citations.filter((citation: any) => {
                    if (typeof citation === 'string') {
                      return !toolNames.includes(citation);
                    }
                    const label = citation?.label || citation?.title || citation?.name || '';
                    return !toolNames.includes(label);
                  });
                  
                  if (validCitations.length === 0) {
                    return null; // Don't show citations section if all are tool names
                  }
                  
                  return (
                    <div className="mt-4 pt-4 border-t border-gray-200">
                      <h4 className="font-semibold text-sm text-gray-700 mb-3 flex items-center gap-2">
                        <ExternalLink className="w-4 h-4 text-blue-600" />
                        Sources & Citations ({validCitations.length})
                      </h4>
                      <div className="space-y-2">
                        {validCitations.map((citation: any, cIdx: number) => {
                          // Handle different citation formats
                          const citationObj = typeof citation === 'string' 
                            ? { label: citation, url: citation.startsWith('http') ? citation : '', excerpt: '' }
                            : citation;
                          
                          const label = citationObj?.label || citationObj?.title || citationObj?.name || 'Source';
                          const url = citationObj?.url || citationObj?.link || citationObj?.href || '';
                          const excerpt = citationObj?.excerpt || citationObj?.description || citationObj?.content || '';
                          const isValidUrl = url && (url.startsWith('http://') || url.startsWith('https://'));
                        
                        return (
                          <div key={cIdx} className="flex items-start gap-2 p-3 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors border border-gray-200">
                            {isValidUrl ? (
                              <a
                                href={url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-start gap-2 text-sm text-blue-600 hover:text-blue-800 hover:underline flex-1 group"
                              >
                                <ExternalLink className="w-4 h-4 mt-0.5 flex-shrink-0 group-hover:scale-110 transition-transform" />
                                <div className="flex-1 min-w-0">
                                  <span className="font-medium block">{label}</span>
                                  {excerpt && (
                                    <p className="text-xs text-gray-600 mt-1 line-clamp-2">{excerpt}</p>
                                  )}
                                  <p className="text-xs text-gray-400 mt-1 truncate">{url}</p>
                                </div>
                              </a>
                            ) : (
                              <div className="flex items-start gap-2 text-sm text-gray-700 flex-1">
                                <ExternalLink className="w-4 h-4 mt-0.5 flex-shrink-0 text-gray-400" />
                                <div className="flex-1">
                                  <span className="font-medium block">{label}</span>
                                  {excerpt && (
                                    <p className="text-xs text-gray-500 mt-1">{excerpt}</p>
                                  )}
                                  {url && !isValidUrl && (
                                    <p className="text-xs text-gray-400 mt-1">{url}</p>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                        })}
                      </div>
                    </div>
                  );
                })() : null}
                
                {response.evidence_count > 0 && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <h4 className="font-semibold text-sm text-gray-700 mb-3 flex items-center gap-2">
                      <FileText className="w-4 h-4 text-blue-600" />
                      Evidence Items ({response.evidence_count})
                    </h4>
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-2">
                      <p className="text-xs text-blue-800">
                        <strong>Evidence items</strong> are detailed data points collected by the agent (company data, metrics, research findings). 
                        <strong>Citations</strong> are the source links/URLs where this information came from.
                      </p>
                    </div>
                    <p className="text-sm text-gray-600">
                      This agent collected <strong>{response.evidence_count}</strong> evidence items from various sources. 
                      {response.citations && response.citations.length > 0 && (
                        <span> See <strong>Citations</strong> section above for source links.</span>
                      )}
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Q&A Tab */}
        {activeTab === 'qa' && (
          <div className="space-y-4">
            <div className="flex gap-3">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleAskQuestion()}
                placeholder="Ask a question about the research results..."
                className="input-field flex-1"
              />
              <button
                onClick={handleAskQuestion}
                disabled={!question.trim() || isAsking}
                className="btn-primary"
              >
                {isAsking ? 'Asking...' : 'Ask'}
              </button>
            </div>

            {qaAnswer && (
              <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
                <h4 className="font-semibold text-gray-800 mb-2">Answer</h4>
                <p className="text-gray-700 whitespace-pre-wrap">{qaAnswer}</p>
              </div>
            )}

            {qaAnswer && qaSources.length > 0 && (
              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
                  <ExternalLink className="w-4 h-4 text-blue-600" />
                  Supporting Sources
                </h4>
                <div className="space-y-3">
                  {qaSources.map((source, idx) => {
                    const label = source.title || source.source || `Source ${idx + 1}`
                    const url = source.source
                    const isUrl = url && (url.startsWith('http://') || url.startsWith('https://'))
                    const evidenceId = source.id || `E${idx + 1}`
                    return (
                      <div key={`${label}-${idx}`} className="p-3 rounded-lg border border-gray-100 bg-gray-50">
                        <p className="text-xs uppercase tracking-wide text-gray-500 mb-1">
                          Evidence {evidenceId}
                        </p>
                        {isUrl ? (
                          <a
                            href={url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-600 hover:text-blue-800 font-medium flex items-center gap-2"
                          >
                            <ExternalLink className="w-4 h-4" />
                            {label}
                          </a>
                        ) : (
                          <p className="text-gray-700 font-medium">{label}</p>
                        )}
                        {source.metadata && source.metadata.funding_total_usd && (
                          <p className="text-xs text-gray-500 mt-1">
                            Funding: ${Number(source.metadata.funding_total_usd).toLocaleString()}
                          </p>
                        )}
                        {source.metadata && source.metadata.growjo_growth_percent && (
                          <p className="text-xs text-gray-500 mt-1">
                            Growth: {source.metadata.growjo_growth_percent}%
                          </p>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {!qaAnswer && (
              <div className="text-center py-12 text-gray-500">
                <MessageSquare className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p>Ask a question to get insights about your research</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

