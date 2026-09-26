import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { AgentResearchResponse, RecentResearchItem } from '../types';
import {
  Bot,
  GitBranch,
  CheckCircle2,
  Clock,
  BookOpen,
  ArrowRight,
  Zap,
  RefreshCw,
  ChevronRight,
  Layers,
  FileCheck,
  AlertTriangle,
  Search,
  Boxes,
  ShieldCheck,
  Sparkles,
  HelpCircle,
} from 'lucide-react';
import { EmptyState } from './VisualAssets';
import { FlightRecorder } from './FlightRecorder';
import { WhyThisAnswerModal } from './WhyThisAnswerModal';

interface WorkflowNodeDef {
  key: string;
  label: string;
  stageName: string;
  icon: React.ComponentType<{ className?: string }>;
  nodeMatch: string[];
}

const WORKFLOW_NODES: WorkflowNodeDef[] = [
  { key: 'query', label: 'Query', stageName: 'Input Query', icon: Search, nodeMatch: ['query', 'input'] },
  { key: 'analyze', label: 'Analyze', stageName: 'Query Analysis', icon: Search, nodeMatch: ['analyze_query', 'analyze'] },
  { key: 'decompose', label: 'Decompose', stageName: 'Decomposition', icon: GitBranch, nodeMatch: ['plan_query', 'decompose'] },
  { key: 'retrieve', label: 'Retrieve', stageName: 'Hybrid Retrieval', icon: Layers, nodeMatch: ['retrieve', 'retrieve_subqueries'] },
  { key: 'rerank', label: 'Rerank', stageName: 'BGE Reranking', icon: Zap, nodeMatch: ['rerank', 'rerank_evidence'] },
  { key: 'aggregate', label: 'Aggregate', stageName: 'Aggregation', icon: Boxes, nodeMatch: ['assemble_evidence', 'aggregate_evidence'] },
  { key: 'sufficiency', label: 'Sufficiency', stageName: 'Sufficiency Check', icon: ShieldCheck, nodeMatch: ['assess_sufficiency', 'sufficiency'] },
  { key: 'refine', label: 'Refine', stageName: 'Iterative Refinement', icon: RefreshCw, nodeMatch: ['refine_queries', 'refine'] },
  { key: 'synthesize', label: 'Synthesize', stageName: 'Grounded Synthesis', icon: Sparkles, nodeMatch: ['generate', 'synthesize_answer'] },
  { key: 'verify', label: 'Verify', stageName: 'Citation Verification', icon: CheckCircle2, nodeMatch: ['verify_citations', 'verify'] },
];

interface AgenticResearchProps {
  initialQuery?: string;
  autoExecute?: boolean;
  onRecordRecent?: (item: RecentResearchItem) => void;
}

export const AgenticResearch: React.FC<AgenticResearchProps> = ({
  initialQuery = '',
  autoExecute = false,
  onRecordRecent,
}) => {
  const [query, setQuery] = useState(initialQuery);
  const [maxIterations, setMaxIterations] = useState(2);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<AgentResearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedNodeKey, setSelectedNodeKey] = useState<string | null>(null);
  const [isWhyModalOpen, setIsWhyModalOpen] = useState(false);

  const EXAMPLE_AGENT_PROMPT =
    'Compare the main objectives of the enterprise knowledge platform with how it retrieves information for complex questions.';

  const executeAgentResearch = async (targetQuery: string) => {
    const q = targetQuery.trim();
    if (!q) return;

    setLoading(true);
    setError(null);
    setSelectedNodeKey(null);

    const startTime = performance.now();

    try {
      const res = await api.agentResearch(q, 5, 10, true, maxIterations, true);
      setResponse(res);

      const endTime = performance.now();
      const measuredLatency =
        res.total_duration_ms ??
        res.timings_ms?.total_pipeline_ms ??
        res.timings_ms?.total_latency_ms ??
        (endTime - startTime);

      if (onRecordRecent) {
        onRecordRecent({
          id: `agt-${Date.now()}`,
          query: q,
          mode: 'agent',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          status: 'completed',
          latency_ms: measuredLatency,
          citation_count: res.citations?.length || 0,
        });
      }
    } catch (err: any) {
      setError(err.message || 'Agent research failed');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (initialQuery && initialQuery !== query) {
      setQuery(initialQuery);
      if (autoExecute) {
        executeAgentResearch(initialQuery);
      }
    }
  }, [initialQuery, autoExecute]);

  // Helper to find execution trace matching a workflow node
  const getTraceForNode = (nodeDef: WorkflowNodeDef) => {
    if (!response || !response.execution_trace) return null;
    return response.execution_trace.find((t) =>
      nodeDef.nodeMatch.some((m) =>
        (t.node || t.step || t.stage || '').toLowerCase().includes(m)
      )
    );
  };

  return (
    <div className="space-y-6">
      {/* Title Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-indigo-100 text-indigo-700">
              <Bot className="w-5 h-5" />
            </div>
            LangGraph Agentic Research State Machine
          </h2>
          <p className="text-xs text-slate-500 mt-1 max-w-3xl">
            Decomposes complex compound questions into targeted sub-queries, executes parallel retrieval, assesses evidence sufficiency, and performs bounded iterative refinement (max iterations &le; 2).
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <span className="px-2.5 py-1 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200/80 font-medium">
            LangGraph State Machine
          </span>
          <span className="px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200/80 font-medium">
            Bounded Iterations: &le; 2
          </span>
        </div>
      </div>

      {/* Query Input Box */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          executeAgentResearch(query);
        }}
        className="p-5 bg-white border border-slate-200/80 rounded-3xl shadow-xs space-y-4"
      >
        <div>
          <textarea
            rows={3}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a complex multi-document or multi-hop question (e.g. 'Compare the main objectives of the enterprise knowledge platform with how it retrieves information for complex questions')..."
            className="w-full px-4 py-3 bg-slate-50/70 border border-slate-200 rounded-2xl text-slate-900 placeholder-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/10 focus:bg-white text-sm transition"
          />
        </div>

        {/* Example Prompt Suggestion */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-slate-400 font-medium text-[11px]">Benchmark Inquiry:</span>
          <button
            type="button"
            onClick={() => {
              setQuery(EXAMPLE_AGENT_PROMPT);
              executeAgentResearch(EXAMPLE_AGENT_PROMPT);
            }}
            disabled={loading}
            className="px-3 py-1.5 rounded-xl bg-slate-100/80 hover:bg-indigo-50 hover:text-indigo-700 hover:border-indigo-200 text-slate-700 border border-slate-200/80 transition-all text-[11px] font-medium cursor-pointer hover:scale-[1.01]"
          >
            {EXAMPLE_AGENT_PROMPT}
          </button>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-100">
          <div className="flex flex-wrap items-center gap-3 text-xs text-slate-600">
            <div className="flex items-center gap-1.5 bg-slate-50 px-2.5 py-1 rounded-lg border border-slate-200">
              <span className="font-semibold text-slate-700">Max Iterations:</span>
              <select
                value={maxIterations}
                onChange={(e) => setMaxIterations(Number(e.target.value))}
                className="bg-white border border-slate-300 rounded px-1.5 py-0.5 text-slate-800 font-medium text-xs"
              >
                <option value={1}>1 (Single pass)</option>
                <option value={2}>2 (Iterative refinement)</option>
              </select>
            </div>

            <span className="inline-flex items-center gap-1 text-indigo-700 font-medium">
              <Zap className="w-3.5 h-3.5" />
              Sub-Query Decomposition Active
            </span>
            <span>•</span>
            <span className="inline-flex items-center gap-1 text-emerald-700 font-medium">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Bounded Abstention Active
            </span>
          </div>

          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="px-6 py-2.5 bg-gradient-to-r from-indigo-500 via-indigo-600 to-purple-600 hover:from-indigo-600 hover:to-purple-700 disabled:opacity-50 text-white rounded-xl text-xs font-semibold transition-all shadow-md shadow-indigo-500/20 flex items-center gap-2 cursor-pointer hover:scale-[1.02] active:scale-[0.99]"
          >
            {loading ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                Executing Graph State Machine...
              </>
            ) : (
              <>
                Run Agent Research
                <ArrowRight className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </form>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-2xl text-rose-700 text-sm shadow-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 flex-shrink-0 text-rose-600" />
          {error}
        </div>
      )}

      {/* Visual Workflow State Machine Diagram */}
      <div className="p-6 bg-white border border-slate-200/80 rounded-3xl shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-600" />
            LangGraph State Machine Pipeline (Click node for trace inspection)
          </h3>
          <span className="text-[11px] text-slate-400 font-mono">
            {loading ? 'Executing Pipeline...' : response ? 'Trace Available' : 'Graph Definition'}
          </span>
        </div>

        <div className="overflow-x-auto pb-2">
          <div className="flex items-center gap-1.5 min-w-[820px]">
            {WORKFLOW_NODES.map((nodeDef, idx) => {
              const NodeIcon = nodeDef.icon;
              const traceItem = getTraceForNode(nodeDef);
              const isExecuted = Boolean(traceItem) || (nodeDef.key === 'query' && (loading || response));
              const isRunning = loading && !response;
              const isSelected = selectedNodeKey === nodeDef.key;

              return (
                <React.Fragment key={nodeDef.key}>
                  <button
                    type="button"
                    onClick={() => setSelectedNodeKey(isSelected ? null : nodeDef.key)}
                    className={`flex-1 min-w-[76px] p-2.5 rounded-2xl text-center border transition-all cursor-pointer ${
                      isSelected
                        ? 'bg-indigo-600 text-white border-indigo-600 shadow-md ring-2 ring-indigo-200'
                        : isExecuted
                        ? 'bg-emerald-50/90 text-emerald-900 border-emerald-300 hover:bg-emerald-100/70 shadow-2xs'
                        : isRunning
                        ? 'bg-indigo-50 text-indigo-900 border-indigo-300 ring-2 ring-indigo-200 animate-pulse'
                        : 'bg-slate-50 text-slate-500 border-slate-200/70 hover:bg-slate-100'
                    }`}
                  >
                    <div className="flex items-center justify-center mb-1">
                      <NodeIcon
                        className={`w-3.5 h-3.5 ${
                          isSelected
                            ? 'text-white'
                            : isExecuted
                            ? 'text-emerald-700'
                            : isRunning
                            ? 'text-indigo-600'
                            : 'text-slate-400'
                        }`}
                      />
                    </div>
                    <div className="text-[9px] font-mono uppercase font-semibold tracking-wider">
                      {nodeDef.key === 'query' ? 'Start' : `N${idx}`}
                    </div>
                    <div className="text-xs font-bold truncate mt-0.5">
                      {nodeDef.label}
                    </div>
                    <div className="text-[9px] font-mono mt-1 opacity-85">
                      {traceItem?.duration_ms !== undefined
                        ? `${traceItem.duration_ms.toFixed(1)} ms`
                        : isExecuted
                        ? 'Done'
                        : isRunning
                        ? 'Running'
                        : 'Idle'}
                    </div>
                  </button>

                  {idx < WORKFLOW_NODES.length - 1 && (
                    <ChevronRight
                      className={`w-3.5 h-3.5 flex-shrink-0 transition-colors ${
                        loading ? 'text-indigo-500 animate-pulse' : 'text-slate-300'
                      }`}
                    />
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>

        {/* Selected Node Details Drawer */}
        {selectedNodeKey && (
          <div className="p-4 bg-slate-50 border border-slate-200 rounded-2xl text-xs space-y-2 mt-2">
            {(() => {
              const nodeDef = WORKFLOW_NODES.find((n) => n.key === selectedNodeKey)!;
              const traceItem = getTraceForNode(nodeDef);
              if (selectedNodeKey === 'query') {
                return (
                  <div>
                    <div className="font-bold text-slate-800">Query Input Node</div>
                    <div className="text-slate-600 mt-1 font-mono">{query || 'No query input yet'}</div>
                  </div>
                );
              }
              if (!traceItem) {
                return (
                  <div className="text-slate-500 italic">
                    Node <strong>{nodeDef.stageName}</strong> was skipped or has not executed in this run.
                  </div>
                );
              }
              return (
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between font-bold text-slate-800">
                    <span className="font-mono text-indigo-700">{traceItem.node || nodeDef.stageName}</span>
                    <span className="text-slate-500 font-mono text-[11px]">
                      Duration: <strong>{(traceItem.duration_ms ?? traceItem.latency_ms ?? 0).toFixed(2)} ms</strong>
                    </span>
                  </div>
                  {traceItem.input_summary && (
                    <div className="text-slate-600">
                      <span className="font-semibold text-slate-700">Input:</span> {traceItem.input_summary}
                    </div>
                  )}
                  {traceItem.output_summary && (
                    <div className="text-slate-700 font-medium">
                      <span className="font-semibold text-indigo-900">Output:</span> {traceItem.output_summary}
                    </div>
                  )}
                </div>
              );
            })()}
          </div>
        )}
      </div>

      {/* Agent Workflow Execution Response */}
      {response ? (
        <div className="space-y-6">
          {/* Status KPI Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-5 bg-white rounded-2xl border border-slate-200/80 shadow-xs text-center space-y-1">
              <div className="text-xs font-bold text-slate-400 uppercase tracking-wider">Status</div>
              <div className="text-xl font-extrabold text-emerald-600 capitalize">
                {response.status}
              </div>
              {response.provider === 'mock' && (
                <div className="text-[10px] text-sky-600 font-medium">Mock LLM (Dev Mode)</div>
              )}
            </div>

            <div className="p-5 bg-white rounded-2xl border border-slate-200/80 shadow-xs text-center space-y-1">
              <div className="text-xs font-bold text-slate-400 uppercase tracking-wider">Sub-Queries</div>
              <div className="text-xl font-extrabold text-slate-900">
                {response.sub_queries?.length || 1}
              </div>
              <div className="text-[10px] text-slate-400">Decomposed</div>
            </div>

            <div className="p-5 bg-white rounded-2xl border border-slate-200/80 shadow-xs text-center space-y-1">
              <div className="text-xs font-bold text-slate-400 uppercase tracking-wider">Iterations Used</div>
              <div className="text-xl font-extrabold text-indigo-600">
                {response.iterations_used || 1}
              </div>
              <div className="text-[10px] text-slate-400">of {maxIterations} Max</div>
            </div>

            <div className="p-5 bg-white rounded-2xl border border-slate-200/80 shadow-xs text-center space-y-1">
              <div className="text-xs font-bold text-slate-400 uppercase tracking-wider">Total Duration</div>
              <div className="text-xl font-extrabold text-sky-600">
                {(() => {
                  const total =
                    response.total_duration_ms ??
                    response.timings_ms?.total_pipeline_ms ??
                    response.timings_ms?.total_latency_ms ??
                    response.execution_trace?.reduce((acc, curr) => acc + (curr.duration_ms ?? curr.latency_ms ?? 0), 0) ??
                    0;
                  return total > 0 ? `${total.toFixed(1)} ms` : '—';
                })()}
              </div>
              <div className="text-[10px] text-slate-400">Measured Duration</div>
            </div>
          </div>

          {/* Sub-Queries Decomposition Cards */}
          {response.sub_queries && response.sub_queries.length > 0 && (
            <div className="p-6 bg-white border border-slate-200/80 rounded-3xl space-y-4 shadow-xs">
              <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
                <GitBranch className="w-4 h-4 text-indigo-600" />
                Query Decomposition Plan ({response.sub_queries.length} Sub-Queries)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {response.sub_queries.map((sq, idx) => {
                  const sqId = sq.id || sq.sub_query_id || `SQ-${idx + 1}`;
                  const sqText = sq.query || sq.sub_query_text || 'Decomposed Sub-Query';
                  const sqReason = sq.reasoning || sq.purpose;
                  return (
                    <div key={sqId || idx} className="p-4 bg-slate-50/80 rounded-2xl border border-slate-200/70 text-xs space-y-2">
                      <div className="flex items-center justify-between font-mono text-indigo-700 font-semibold">
                        <span>Sub-Query #{idx + 1} ({sqId})</span>
                        <span className="text-slate-500 font-medium capitalize text-[11px] bg-white px-2 py-0.5 rounded border border-slate-200">
                          {sq.status || 'retrieved'}
                        </span>
                      </div>
                      <div className="text-slate-900 font-semibold text-xs leading-snug">{sqText}</div>
                      {sqReason && (
                        <div className="text-slate-500 text-xs italic bg-white p-2.5 rounded-xl border border-slate-200/60 mt-1">
                          {sqReason}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* LangGraph Research Flight Recorder */}
          {response.execution_trace && response.execution_trace.length > 0 && (
            <FlightRecorder
              executionTrace={response.execution_trace}
              timings={response.timings_ms}
              totalDurationMs={response.total_duration_ms}
              status={response.status}
            />
          )}

          {/* Final Synthesized Answer */}
          <div className="p-6 bg-white border border-slate-200/80 rounded-3xl space-y-4 shadow-sm">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 flex-wrap gap-2">
              <div className="flex items-center gap-2 text-sm font-bold text-slate-900">
                <FileCheck className="w-4 h-4 text-emerald-600" />
                Final Grounded Multi-Hop Synthesis
              </div>
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => setIsWhyModalOpen(true)}
                  className="px-2.5 py-1 rounded-xl bg-slate-100 hover:bg-indigo-50 hover:text-indigo-700 text-slate-700 text-xs font-semibold border border-slate-200 hover:border-indigo-300 transition flex items-center gap-1.5 cursor-pointer shadow-2xs"
                  title="Explain multi-hop sub-query retrieval and verification"
                >
                  <HelpCircle className="w-3.5 h-3.5 text-indigo-600" />
                  <span>Why this answer?</span>
                </button>
                <div className="text-xs text-slate-500 font-mono">
                  Provenance Chunks: <strong className="text-slate-800">{response.citations?.length || 0}</strong>
                </div>
              </div>
            </div>

            <div className="text-slate-800 text-sm leading-relaxed whitespace-pre-line bg-slate-50/70 p-5 rounded-2xl border border-slate-200/60 font-sans">
              {response.final_answer}
            </div>
          </div>

          {/* Provenance Citations */}
          {response.citations && response.citations.length > 0 && (
            <div className="p-6 bg-white border border-slate-200/80 rounded-3xl space-y-4 shadow-xs">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-sky-600" />
                Cross-Document Citations & Provenance ({response.citations.length})
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {response.citations.map((cite, idx) => (
                  <div key={idx} className="p-4 bg-slate-50/80 rounded-2xl border border-slate-200/70 text-xs space-y-2">
                    <div className="flex items-center justify-between font-semibold text-slate-900">
                      <div className="flex items-center gap-2">
                        <span className="w-5 h-5 rounded-lg bg-indigo-100 text-indigo-700 font-mono text-[10px] font-bold flex items-center justify-center">
                          {String(idx + 1).padStart(2, '0')}
                        </span>
                        <span className="text-indigo-900 font-bold">{cite.document_name}</span>
                      </div>
                      <span className="text-slate-500 font-mono text-[11px] bg-white px-2 py-0.5 rounded border border-slate-200">
                        Page {cite.page_number || 1}
                      </span>
                    </div>
                    {cite.section_title && (
                      <div className="text-slate-600 font-medium truncate text-[11px]">
                        Section: {cite.section_title}
                      </div>
                    )}
                    {cite.source_snippet && (
                      <div className="text-slate-700 text-[11px] italic bg-white p-2.5 rounded-xl border border-slate-200/80 leading-relaxed">
                        "{cite.source_snippet}"
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Execution Trace Timeline */}
          {response.execution_trace && response.execution_trace.length > 0 && (
            <div className="p-6 bg-white border border-slate-200/80 rounded-3xl space-y-3 shadow-xs">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
                <Clock className="w-4 h-4 text-indigo-600" />
                LangGraph State Machine Execution Trace ({response.execution_trace.length} Spans)
              </h4>
              <div className="space-y-2">
                {response.execution_trace.map((item, idx) => {
                  const nodeName = item.node || item.step || item.stage || `Span ${idx + 1}`;
                  const duration = item.duration_ms ?? item.latency_ms ?? 0;
                  return (
                    <div
                      key={idx}
                      className="p-3.5 bg-slate-50/70 hover:bg-indigo-50/20 rounded-2xl border border-slate-200/60 flex flex-col md:flex-row md:items-center justify-between text-xs transition-all gap-2"
                    >
                      <div className="flex items-center gap-2.5">
                        <span className="w-6 h-6 rounded-full bg-indigo-50 text-indigo-700 font-bold flex items-center justify-center text-[11px] border border-indigo-200">
                          {idx + 1}
                        </span>
                        <span className="text-slate-900 font-semibold font-mono">{nodeName}</span>
                        {item.output_summary && (
                          <span className="text-slate-500 text-[11px] font-sans truncate max-w-md hidden md:inline">
                            — {item.output_summary}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-3 text-slate-500 font-mono text-[11px] self-end md:self-auto">
                        <span className="font-semibold text-slate-700">{duration.toFixed(2)} ms</span>
                        <ArrowRight className="w-3.5 h-3.5 text-slate-400" />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      ) : (
        <EmptyState
          type="no-research"
          title="Ready for Agentic Research"
          description="Decompose compound inquiries across document sections with LangGraph state machine execution."
          actionLabel="Run Comparative Benchmark Query"
          onAction={() => executeAgentResearch(EXAMPLE_AGENT_PROMPT)}
        />
      )}

      {/* Multi-Hop Explainability Modal */}
      <WhyThisAnswerModal
        isOpen={isWhyModalOpen}
        onClose={() => setIsWhyModalOpen(false)}
        response={response}
        query={query}
      />
    </div>
  );
};
