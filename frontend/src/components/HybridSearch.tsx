import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { SearchResponse, SearchResultChunk, RecentResearchItem } from '../types';
import { Search, Sliders, Layers, Sparkles, ArrowRight, FileText } from 'lucide-react';
import { EmptyState } from './VisualAssets';

interface HybridSearchProps {
  initialQuery?: string;
  autoExecute?: boolean;
  onRecordRecent?: (item: RecentResearchItem) => void;
}

export const HybridSearch: React.FC<HybridSearchProps> = ({
  initialQuery = '',
  autoExecute = false,
  onRecordRecent,
}) => {
  const [query, setQuery] = useState(initialQuery);
  const [mode, setMode] = useState<'hybrid' | 'dense' | 'bm25'>('hybrid');
  const [topK, setTopK] = useState(5);
  const [candidateTopK, setCandidateTopK] = useState(10);
  const [rerank, setRerank] = useState(true);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const executeSearch = async (targetQuery: string) => {
    const q = targetQuery.trim();
    if (!q) return;

    setLoading(true);
    setError(null);
    const startTime = performance.now();

    try {
      const res = await api.search(q, mode, topK, candidateTopK, rerank);
      setResponse(res);

      const endTime = performance.now();
      const measuredLatency = res.latency_breakdown
        ? Object.values(res.latency_breakdown).reduce((a, b) => a + b, 0)
        : (endTime - startTime);

      if (onRecordRecent) {
        onRecordRecent({
          id: `sch-${Date.now()}`,
          query: q,
          mode: 'search',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          status: 'retrieved',
          latency_ms: measuredLatency,
          citation_count: res.results?.length || 0,
        });
      }
    } catch (err: any) {
      setError(err.message || 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (initialQuery && initialQuery !== query) {
      setQuery(initialQuery);
      if (autoExecute) {
        executeSearch(initialQuery);
      }
    }
  }, [initialQuery, autoExecute]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    executeSearch(query);
  };

  return (
    <div className="space-y-6">
      {/* Title Header */}
      <div>
        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-sky-100 text-sky-700">
            <Search className="w-5 h-5" />
          </div>
          Two-Stage Hybrid Search & Reranking
        </h2>
        <p className="text-xs text-slate-500 mt-1 max-w-3xl">
          Combines dense vector embeddings (FastEmbed `bge-small-en-v1.5`) and lexical BM25 via Reciprocal Rank Fusion ($k=60$) with neural cross-encoder reranking.
        </p>
      </div>

      <form onSubmit={handleSearch} className="space-y-4">
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search enterprise corpus (e.g., 'What is the accuracy of Hybrid RAG in the benchmark?')..."
            className="w-full px-4 py-3.5 bg-white border border-slate-200 rounded-2xl text-slate-900 placeholder-slate-400 focus:outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10 text-sm shadow-xs transition pr-28"
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="absolute right-2 top-2 bottom-2 px-5 bg-sky-600 hover:bg-sky-700 disabled:opacity-50 text-white rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-xs cursor-pointer hover:scale-[1.02] active:scale-[0.99]"
          >
            {loading ? 'Searching...' : (
              <>
                Search
                <ArrowRight className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>

        {/* Search Controls */}
        <div className="flex flex-wrap items-center gap-4 text-xs text-slate-700 bg-white p-4 rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-600" />
            <span className="font-semibold text-slate-800">Mode:</span>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as any)}
              className="bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1 text-slate-900 font-medium text-xs focus:ring-2 focus:ring-sky-500/20"
            >
              <option value="hybrid">Hybrid (Dense + BM25 RRF)</option>
              <option value="dense">Dense Only (Qdrant)</option>
              <option value="bm25">BM25 Lexical Only</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-sky-600" />
            <span className="font-semibold text-slate-800">Top K:</span>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1 text-slate-900 font-medium text-xs"
            >
              <option value={3}>3</option>
              <option value={5}>5</option>
              <option value={10}>10</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-800">Candidate Pool:</span>
            <select
              value={candidateTopK}
              onChange={(e) => setCandidateTopK(Number(e.target.value))}
              className="bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1 text-slate-900 font-medium text-xs"
            >
              <option value={10}>10 candidates</option>
              <option value={20}>20 candidates</option>
            </select>
          </div>

          <label className="flex items-center gap-2 cursor-pointer ml-auto bg-sky-50 px-3 py-1 rounded-xl border border-sky-200/70">
            <input
              type="checkbox"
              checked={rerank}
              onChange={(e) => setRerank(e.target.checked)}
              className="rounded border-slate-300 text-sky-600 focus:ring-sky-500/30"
            />
            <span className="font-semibold flex items-center gap-1.5 text-sky-800 text-xs">
              <Sparkles className="w-3.5 h-3.5 text-sky-600" />
              BGE Cross-Encoder Reranker
            </span>
          </label>
        </div>
      </form>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-700 text-sm shadow-xs">
          {error}
        </div>
      )}

      {/* Results */}
      {response ? (
        response.results && response.results.length > 0 ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between text-xs text-slate-500 px-1">
              <span>
                Retrieved <strong className="text-slate-900">{response.results.length}</strong> chunks (Strategy: <span className="font-mono font-medium text-slate-800 capitalize">{response.retrieval_mode}</span>)
              </span>
              {response.latency_breakdown && (
                <span className="font-mono text-slate-600 font-medium">
                  Latency: {(Object.values(response.latency_breakdown).reduce((a, b) => a + b, 0)).toFixed(2)} ms
                </span>
              )}
            </div>

            <div className="space-y-3">
              {response.results.map((chunk: SearchResultChunk) => (
                <div
                  key={chunk.chunk_id}
                  className="p-5 bg-white border border-slate-200/80 rounded-2xl space-y-3 shadow-xs hover:border-slate-300 hover:shadow-sm transition-all duration-150"
                >
                  <div className="flex items-center justify-between text-xs pb-2 border-b border-slate-100 flex-wrap gap-2">
                    <div className="flex items-center gap-2">
                      <span className="px-2.5 py-0.5 bg-sky-50 text-sky-700 border border-sky-200/80 rounded-lg font-bold font-mono text-[11px]">
                        Rank #{chunk.rank}
                      </span>
                      <span className="text-slate-900 font-bold flex items-center gap-1.5">
                        <FileText className="w-3.5 h-3.5 text-slate-500" />
                        {chunk.document_name}
                      </span>
                      {chunk.page_number && (
                        <span className="text-slate-500 font-mono text-[11px] bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                          Page {chunk.page_number}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-slate-400 font-mono">Similarity:</span>
                      <span className="font-mono text-xs px-2.5 py-0.5 rounded-full font-bold bg-slate-100 text-slate-700 border border-slate-200">
                        {chunk.score.toFixed(4)}
                      </span>
                    </div>
                  </div>

                  {chunk.section_title && (
                    <div className="text-xs font-semibold text-slate-700 bg-slate-50 px-2.5 py-1 rounded-lg border border-slate-200/60 truncate">
                      Section: {chunk.section_title}
                    </div>
                  )}

                  <div className="text-slate-700 text-xs leading-relaxed bg-slate-50 p-4 rounded-xl border border-slate-200/60 font-mono">
                    {chunk.text}
                  </div>

                  <div className="text-[10px] text-slate-400 font-mono flex items-center justify-between">
                    <span>Chunk ID: {chunk.chunk_id?.slice(0, 12)}...</span>
                    <span>Document ID: {chunk.document_id?.slice(0, 12)}...</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <EmptyState
            type="no-results"
            title="No relevant chunks found."
            description="The current hybrid search query returned 0 matches in the active corpus."
          />
        )
      ) : (
        <EmptyState
          type="no-research"
          title="Ready for Hybrid Search"
          description="Enter keywords or a technical query to execute combined dense semantic retrieval and Okapi BM25 reciprocal rank fusion."
          actionLabel="Run Search"
          onAction={() => executeSearch('hybrid retrieval combining semantic vector search')}
        />
      )}
    </div>
  );
};
