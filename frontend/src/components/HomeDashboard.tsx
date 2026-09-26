import React, { useState } from 'react';
import {
  Sparkles,
  ShieldCheck,
  Workflow,
  GitMerge,
  FolderOpen,
  ArrowRight,
  Zap,
  CheckCircle2,
  Clock,
  Layers,
  Database,
  Boxes,
  Shield,
  Users,
  BookOpen,
  ClipboardCheck,
} from 'lucide-react';
import { RecentResearchItem } from '../types';
import { HeroAbstractIllustration, EmptyState } from './VisualAssets';

interface HomeDashboardProps {
  onExecuteQuery: (query: string, mode: 'research' | 'agent' | 'search') => void;
  recentResearch: RecentResearchItem[];
  documentCount?: number;
  isBackendHealthy: boolean | null;
}

export const HomeDashboard: React.FC<HomeDashboardProps> = ({
  onExecuteQuery,
  recentResearch,
  documentCount = 1,
  isBackendHealthy,
}) => {
  const [queryInput, setQueryInput] = useState('');
  const [selectedMode, setSelectedMode] = useState<'research' | 'agent' | 'search'>('research');

  const BENCHMARK_QUERIES = [
    {
      query: 'What retrieval methods does Company ABC use?',
      mode: 'research' as const,
      label: 'Retrieval Methods',
    },
    {
      query: 'How does Company ABC handle complex questions?',
      mode: 'research' as const,
      label: 'Complex Questions',
    },
    {
      query: 'Compare the main objectives of the enterprise knowledge platform with how it retrieves information for complex questions.',
      mode: 'agent' as const,
      label: 'Agentic Comparison',
    },
    {
      query: "What is Company ABC's employee vacation allowance?",
      mode: 'research' as const,
      label: 'Abstention Test',
    },
    {
      query: 'What is hybrid RAG and how does it improve enterprise document search?',
      mode: 'research' as const,
      label: 'Hybrid RAG Search',
    },
  ];

  const QUICK_TEMPLATES = [
    {
      title: 'Product Architecture & System Design',
      desc: 'Analyze core RAG components, reciprocal rank fusion, and cross-encoder reranker layers.',
      query: 'What retrieval methods does Company ABC use?',
      mode: 'research' as const,
      tag: 'Architecture',
      icon: Boxes,
      color: 'bg-sky-100 text-sky-700',
    },
    {
      title: 'Complex Multi-Hop Agentic Synthesis',
      desc: 'Decompose compound queries and retrieve evidence across multiple document sections.',
      query: 'Compare the main objectives of the enterprise knowledge platform with how it retrieves information for complex questions.',
      mode: 'agent' as const,
      tag: 'Agentic',
      icon: Workflow,
      color: 'bg-purple-100 text-purple-700',
    },
    {
      title: 'Security, Zero-Trust & Access Policy',
      desc: 'Verify enterprise document access policies and evidence boundary protections.',
      query: 'How does Company ABC handle complex questions?',
      mode: 'research' as const,
      tag: 'Security',
      icon: Shield,
      color: 'bg-emerald-100 text-emerald-700',
    },
    {
      title: 'Zero-Hallucination Bounded Abstention',
      desc: 'Test query intent isolation when corpus does not contain sufficient factual evidence.',
      query: "What is Company ABC's employee vacation allowance?",
      mode: 'research' as const,
      tag: 'HR & Policy',
      icon: Users,
      color: 'bg-amber-100 text-amber-700',
    },
    {
      title: 'Dense & Lexical Hybrid Search',
      desc: 'Inspect chunk-level similarity scores from Qdrant vector index and Okapi BM25.',
      query: 'hybrid retrieval combining semantic vector search and keyword search',
      mode: 'search' as const,
      tag: 'Data Index',
      icon: Database,
      color: 'bg-indigo-100 text-indigo-700',
    },
    {
      title: 'Compliance & Grounding Audit',
      desc: 'Examine sentence-level empirical claims, citations, and verified support coverage.',
      query: 'What are the main objectives of the internal enterprise knowledge platform?',
      mode: 'research' as const,
      tag: 'Audit',
      icon: ClipboardCheck,
      color: 'bg-teal-100 text-teal-700',
    },
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryInput.trim()) return;
    onExecuteQuery(queryInput.trim(), selectedMode);
  };

  return (
    <div className="space-y-8 pb-10">
      {/* Welcome Hero Banner with Subtle Grid & Abstract AI Illustration */}
      <div className="relative overflow-hidden p-6 sm:p-8 bg-gradient-to-br from-white via-sky-50/30 to-indigo-50/20 rounded-3xl border border-slate-200/80 shadow-xs">
        {/* Subtle geometric dot pattern background */}
        <div
          className="absolute inset-0 opacity-[0.035] pointer-events-none"
          style={{
            backgroundImage: 'radial-gradient(#0f172a 1px, transparent 1px)',
            backgroundSize: '18px 18px',
          }}
          aria-hidden="true"
        />

        <div className="relative z-10 flex flex-col lg:flex-row items-center justify-between gap-6">
          <div className="max-w-2xl space-y-3.5">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-100/80 text-sky-800 text-xs font-semibold border border-sky-200/60 shadow-2xs">
              <Sparkles className="w-3.5 h-3.5 text-sky-600 animate-spin" style={{ animationDuration: '4s' }} />
              Next-Gen Agentic RAG Platform
            </div>

            <h1 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold text-slate-900 tracking-tight leading-tight">
              Enterprise AI Knowledge & Research Copilot
            </h1>

            <p className="text-slate-600 text-sm sm:text-base leading-relaxed">
              Enterprise knowledge discovery, reciprocal rank fusion hybrid search, sentence-level claim verification, and LangGraph multi-agent synthesis.
            </p>

            {/* Quick System Capability Badges */}
            <div className="pt-2 flex flex-wrap items-center gap-2 text-xs">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white border border-slate-200 text-slate-700 font-medium shadow-2xs">
                <Database className="w-3.5 h-3.5 text-sky-600" />
                Qdrant + Okapi BM25 RRF
              </span>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white border border-slate-200 text-slate-700 font-medium shadow-2xs">
                <Zap className="w-3.5 h-3.5 text-indigo-600" />
                FastEmbed bge-small 384d
              </span>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white border border-slate-200 text-slate-700 font-medium shadow-2xs">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                XML Evidence Isolation
              </span>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white border border-slate-200 text-slate-700 font-medium shadow-2xs">
                <Workflow className="w-3.5 h-3.5 text-purple-600" />
                LangGraph Multi-Step Agent
              </span>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white border border-slate-200 text-slate-700 font-medium shadow-2xs font-mono">
                Corpus: {documentCount} Doc{documentCount !== 1 ? 's' : ''}
              </span>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white border border-slate-200 text-slate-700 font-medium shadow-2xs font-mono">
                <span className={`w-2 h-2 rounded-full ${isBackendHealthy ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
                {isBackendHealthy ? 'Engine Online' : 'Engine Offline'}
              </span>
            </div>
          </div>

          {/* Right Hero Illustration */}
          <div className="w-full lg:w-[400px] xl:w-[440px] flex-shrink-0 hidden md:block">
            <HeroAbstractIllustration />
          </div>
        </div>
      </div>

      {/* 4 Core Platform Capabilities (Reference B) */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-2">
            <Layers className="w-4 h-4 text-sky-600" />
            Core Enterprise Capabilities
          </h2>
          <span className="text-xs text-slate-400 font-mono">Architecture v1.0</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Card 1: Grounded Research (Light Blue Tint) */}
          <div
            onClick={() => onExecuteQuery('What retrieval methods does Company ABC use?', 'research')}
            className="p-5 bg-sky-50/40 hover:bg-sky-50/80 rounded-2xl border border-sky-100/90 hover:border-sky-300 shadow-2xs hover:shadow-md transition-all duration-200 cursor-pointer group flex flex-col justify-between"
          >
            <div className="space-y-2.5">
              <div className="w-10 h-10 rounded-xl bg-sky-100 text-sky-700 flex items-center justify-center group-hover:scale-105 transition-transform duration-200 shadow-2xs">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-slate-900 text-sm group-hover:text-sky-700 transition">
                Grounded Research
              </h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                Strict XML prompt evidence boundaries isolate retrieved chunks, followed by automated sentence-level claim audit and verification.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-sky-100/80 flex items-center justify-between text-xs text-sky-700 font-semibold">
              <span>Execute Research</span>
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

          {/* Card 2: Agentic Workflow (Light Purple Tint) */}
          <div
            onClick={() => onExecuteQuery('Compare the main objectives of the enterprise knowledge platform with how it retrieves information for complex questions.', 'agent')}
            className="p-5 bg-purple-50/40 hover:bg-purple-50/80 rounded-2xl border border-purple-100/90 hover:border-purple-300 shadow-2xs hover:shadow-md transition-all duration-200 cursor-pointer group flex flex-col justify-between"
          >
            <div className="space-y-2.5">
              <div className="w-10 h-10 rounded-xl bg-purple-100 text-purple-700 flex items-center justify-center group-hover:scale-105 transition-transform duration-200 shadow-2xs">
                <Workflow className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-slate-900 text-sm group-hover:text-purple-700 transition">
                Agentic Workflow
              </h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                Autonomous query decomposition into sub-queries, parallel multi-hop retrieval, evidence sufficiency evaluation, and bounded loop refinement.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-purple-100/80 flex items-center justify-between text-xs text-purple-700 font-semibold">
              <span>Launch Workflow</span>
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

          {/* Card 3: Hybrid Search (Light Emerald Tint) */}
          <div
            onClick={() => onExecuteQuery('hybrid retrieval combining semantic vector search', 'search')}
            className="p-5 bg-emerald-50/40 hover:bg-emerald-50/80 rounded-2xl border border-emerald-100/90 hover:border-emerald-300 shadow-2xs hover:shadow-md transition-all duration-200 cursor-pointer group flex flex-col justify-between"
          >
            <div className="space-y-2.5">
              <div className="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-700 flex items-center justify-center group-hover:scale-105 transition-transform duration-200 shadow-2xs">
                <GitMerge className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-slate-900 text-sm group-hover:text-emerald-700 transition">
                Hybrid Search
              </h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                Dense FastEmbed semantic search paired with Okapi BM25 via Reciprocal Rank Fusion ($k=60$) & BGE Cross-Encoder reranking.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-emerald-100/80 flex items-center justify-between text-xs text-emerald-700 font-semibold">
              <span>Inspect Retrieval</span>
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

          {/* Card 4: Document Library (Light Amber/Orange Tint) */}
          <div
            onClick={() => onExecuteQuery('How does Company ABC handle complex questions?', 'research')}
            className="p-5 bg-amber-50/40 hover:bg-amber-50/80 rounded-2xl border border-amber-100/90 hover:border-amber-300 shadow-2xs hover:shadow-md transition-all duration-200 cursor-pointer group flex flex-col justify-between"
          >
            <div className="space-y-2.5">
              <div className="w-10 h-10 rounded-xl bg-amber-100 text-amber-700 flex items-center justify-center group-hover:scale-105 transition-transform duration-200 shadow-2xs">
                <FolderOpen className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-slate-900 text-sm group-hover:text-amber-800 transition">
                Document Library
              </h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                Structure-aware ingestion for PDF, DOCX, TXT, MD, HTML with chunk inspection and transparent cross-document attribution.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-amber-100/80 flex items-center justify-between text-xs text-amber-800 font-semibold">
              <span>Browse Documents</span>
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </div>
      </div>

      {/* Primary Research Input Box (Console) */}
      <div className="p-6 bg-white rounded-3xl border border-slate-200/80 shadow-xs space-y-4">
        {/* Mode Selector Tabs */}
        <div className="flex items-center justify-between flex-wrap gap-2 pb-2 border-b border-slate-100">
          <div className="flex items-center gap-1.5 p-1 bg-slate-100/80 rounded-xl">
            <button
              type="button"
              onClick={() => setSelectedMode('research')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer flex items-center gap-1.5 ${
                selectedMode === 'research'
                  ? 'bg-white text-sky-800 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5 text-sky-600" />
              Grounded Research
            </button>
            <button
              type="button"
              onClick={() => setSelectedMode('agent')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer flex items-center gap-1.5 ${
                selectedMode === 'agent'
                  ? 'bg-white text-indigo-800 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Workflow className="w-3.5 h-3.5 text-indigo-600" />
              Agentic Workflow
            </button>
            <button
              type="button"
              onClick={() => setSelectedMode('search')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer flex items-center gap-1.5 ${
                selectedMode === 'search'
                  ? 'bg-white text-slate-800 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <GitMerge className="w-3.5 h-3.5 text-slate-500" />
              Hybrid Search
            </button>
          </div>

          <div className="text-xs text-slate-400 font-mono hidden sm:block">
            {selectedMode === 'research' && 'Mode: Single-Turn Grounded Generation & Claim Audit'}
            {selectedMode === 'agent' && 'Mode: LangGraph Multi-Hop Decomposition Pipeline'}
            {selectedMode === 'search' && 'Mode: Chunk Retrieval with Reciprocal Rank Fusion'}
          </div>
        </div>

        {/* Input Textarea Form */}
        <form onSubmit={handleSubmit} className="space-y-3">
          <div className="relative">
            <textarea
              rows={3}
              value={queryInput}
              onChange={(e) => setQueryInput(e.target.value)}
              placeholder={
                selectedMode === 'agent'
                  ? 'Ask a complex multi-document or comparative question (e.g. "Compare the main objectives of the enterprise knowledge platform with how it retrieves information for complex questions")...'
                  : 'Ask a grounded technical research question (e.g. "What retrieval methods does Company ABC use?")...'
              }
              className="w-full px-4 py-3.5 bg-slate-50/70 border border-slate-200 rounded-2xl text-slate-900 placeholder-slate-400 focus:outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10 focus:bg-white text-sm transition"
            />
          </div>

          {/* Quick Benchmark Prompt Chips */}
          <div className="space-y-1.5">
            <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
              Recommended Benchmark Inquiries:
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {BENCHMARK_QUERIES.map((item, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => {
                    setQueryInput(item.query);
                    setSelectedMode(item.mode);
                    onExecuteQuery(item.query, item.mode);
                  }}
                  className="px-3 py-1.5 rounded-xl bg-slate-100/90 hover:bg-sky-50 hover:text-sky-700 hover:border-sky-200 text-slate-700 border border-slate-200/80 transition-all text-xs font-medium cursor-pointer flex items-center gap-1.5 shadow-2xs hover:scale-[1.02]"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-sky-500" />
                  <span className="font-semibold">{item.label}:</span>
                  <span className="truncate max-w-xs">{item.query}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Bottom Actions */}
          <div className="pt-2 flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <span className="inline-flex items-center gap-1 text-emerald-700 font-medium">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                Zero Hallucinations
              </span>
              <span>•</span>
              <span className="inline-flex items-center gap-1 text-sky-700 font-medium">
                <Database className="w-3.5 h-3.5 text-sky-600" />
                Evidence Grounded
              </span>
            </div>

            <div className="flex items-center gap-2">
              {queryInput && (
                <button
                  type="button"
                  onClick={() => setQueryInput('')}
                  className="px-3 py-2 text-xs text-slate-500 hover:text-slate-800 transition font-medium cursor-pointer"
                >
                  Clear
                </button>
              )}
              <button
                type="submit"
                disabled={!queryInput.trim()}
                className={`px-6 py-2.5 rounded-xl text-xs font-bold text-white transition-all flex items-center gap-2 shadow-md cursor-pointer disabled:opacity-50 hover:scale-[1.02] active:scale-[0.99] ${
                  selectedMode === 'agent'
                    ? 'bg-gradient-to-r from-indigo-500 via-indigo-600 to-purple-600 hover:from-indigo-600 hover:to-purple-700 shadow-indigo-500/20'
                    : 'bg-gradient-to-r from-sky-500 via-blue-600 to-indigo-600 hover:from-sky-600 hover:to-indigo-700 shadow-sky-500/20'
                }`}
              >
                <span>
                  {selectedMode === 'agent'
                    ? 'Execute Agentic Workflow'
                    : selectedMode === 'search'
                    ? 'Run Hybrid Search'
                    : 'Execute Grounded Research'}
                </span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </form>
      </div>

      {/* Quick Research Templates (6 tiles with meaningful icons) */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-indigo-600" />
            Quick Enterprise Research Templates
          </h2>
          <span className="text-xs text-slate-400 font-mono">6 Curated Workflows</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {QUICK_TEMPLATES.map((tmpl, idx) => {
            const Icon = tmpl.icon;
            return (
              <div
                key={idx}
                onClick={() => onExecuteQuery(tmpl.query, tmpl.mode)}
                className="p-5 bg-white rounded-2xl border border-slate-200/80 shadow-xs hover:shadow-md hover:border-slate-300 transition-all duration-200 cursor-pointer group flex flex-col justify-between hover:scale-[1.01]"
              >
                <div className="space-y-2.5">
                  <div className="flex items-center justify-between">
                    <div className={`p-2 rounded-xl ${tmpl.color} shadow-2xs`}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <span className="px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 font-mono text-[10px] font-semibold uppercase">
                      {tmpl.tag}
                    </span>
                  </div>

                  <h3 className="font-bold text-slate-900 text-sm group-hover:text-sky-700 transition">
                    {tmpl.title}
                  </h3>

                  <p className="text-xs text-slate-500 leading-relaxed">
                    {tmpl.desc}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-600 font-medium group-hover:text-sky-600">
                  <span className="truncate max-w-[200px] italic text-[11px] text-slate-400">
                    "{tmpl.query}"
                  </span>
                  <ArrowRight className="w-3.5 h-3.5 flex-shrink-0 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Recent Research Activity Table */}
      <div className="bg-white rounded-3xl border border-slate-200/80 shadow-xs overflow-hidden">
        <div className="p-5 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-sky-600" />
            <h3 className="font-bold text-slate-800 text-xs uppercase tracking-wider">
              Recent Research Activity & Verification Log
            </h3>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            {recentResearch.length} Logged Executions
          </span>
        </div>

        {recentResearch.length === 0 ? (
          <EmptyState
            type="no-research"
            onAction={() => onExecuteQuery(BENCHMARK_QUERIES[0].query, 'research')}
            actionLabel="Run Benchmark Query"
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50/70 border-b border-slate-100 text-slate-500 font-semibold uppercase tracking-wider text-[10px]">
                <tr>
                  <th className="px-5 py-3">Inquiry Query</th>
                  <th className="px-4 py-3">Mode</th>
                  <th className="px-4 py-3">Verification Status</th>
                  <th className="px-4 py-3">Latency</th>
                  <th className="px-4 py-3">Timestamp</th>
                  <th className="px-5 py-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {recentResearch.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50/80 transition">
                    <td className="px-5 py-3.5 font-medium text-slate-800 max-w-sm truncate">
                      {item.query}
                    </td>
                    <td className="px-4 py-3.5">
                      <span className="px-2 py-0.5 rounded-full font-mono text-[10px] font-semibold bg-slate-100 text-slate-700 capitalize">
                        {item.mode}
                      </span>
                    </td>
                    <td className="px-4 py-3.5">
                      <span
                        className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-medium text-[11px] ${
                          item.status === 'grounded' || item.status === 'completed'
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                            : item.status === 'abstained'
                            ? 'bg-amber-50 text-amber-700 border border-amber-200'
                            : 'bg-sky-50 text-sky-700 border border-sky-200'
                        }`}
                      >
                        {item.status === 'grounded' ? (
                          <>
                            <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                            100% Grounded
                          </>
                        ) : item.status === 'abstained' ? (
                          'Abstained (Insufficient Evidence)'
                        ) : (
                          item.status
                        )}
                      </span>
                    </td>
                    <td className="px-4 py-3.5 font-mono text-slate-600">
                      {item.latency_ms !== undefined && !isNaN(item.latency_ms)
                        ? `${item.latency_ms.toFixed(1)} ms`
                        : '—'}
                    </td>
                    <td className="px-4 py-3.5 font-mono text-slate-400 text-[11px]">
                      {item.timestamp}
                    </td>
                    <td className="px-5 py-3.5 text-right">
                      <button
                        type="button"
                        onClick={() => onExecuteQuery(item.query, item.mode)}
                        className="text-sky-700 hover:text-sky-900 font-semibold text-xs flex items-center gap-1 ml-auto cursor-pointer"
                      >
                        Re-run
                        <ArrowRight className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
