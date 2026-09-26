import React from 'react';
import {
  LayoutDashboard,
  Sparkles,
  Workflow,
  GitMerge,
  FileText,
  Network,
  BarChart3,
  Activity,
  Clock,
  BookOpen,
  ChevronRight,
  X,
  Database,
  CheckCircle2,
} from 'lucide-react';
import { TabType, RecentResearchItem } from '../types';

interface SidebarProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  recentResearch: RecentResearchItem[];
  onSelectRecentQuery: (item: RecentResearchItem) => void;
  isBackendHealthy: boolean | null;
  documentCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  isOpen,
  setIsOpen,
  recentResearch,
  onSelectRecentQuery,
  isBackendHealthy,
  documentCount = 1,
}) => {
  const NAV_ITEMS: { id: TabType; label: string; icon: React.ComponentType<{ className?: string }>; badge?: string }[] = [
    { id: 'dashboard', label: 'Home Dashboard', icon: LayoutDashboard },
    { id: 'research', label: 'Grounded Research', icon: Sparkles, badge: 'Zero-Hallucination' },
    { id: 'agent', label: 'Agentic Workflow', icon: Workflow, badge: 'LangGraph' },
    { id: 'search', label: 'Hybrid Search', icon: GitMerge, badge: 'Dense+BM25' },
    { id: 'documents', label: 'Knowledge Base', icon: FileText, badge: `${documentCount} docs` },
    { id: 'graph', label: 'Knowledge Graph', icon: Network },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'health', label: 'Health & Metrics', icon: Activity },
  ];

  const KNOWLEDGE_CATEGORIES = [
    { name: 'All Documents', count: documentCount, tag: 'all' },
    { name: 'Company Policies', count: 1, tag: 'policy' },
    { name: 'Engineering Specs', count: 0, tag: 'engineering' },
    { name: 'Research Papers', count: 0, tag: 'research' },
    { name: 'HR & Operations', count: 0, tag: 'hr' },
    { name: 'Security & Compliance', count: 0, tag: 'security' },
  ];

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-40 lg:hidden transition-opacity"
          onClick={() => setIsOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 w-72 bg-white/95 backdrop-blur-md border-r border-slate-200/80 flex flex-col transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          isOpen ? 'translate-x-0 shadow-2xl' : '-translate-x-full'
        } lg:static lg:z-auto lg:shadow-none`}
      >
        {/* Brand Header */}
        <div className="p-5 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 via-indigo-600 to-purple-600 flex items-center justify-center text-white shadow-md shadow-sky-500/20 flex-shrink-0">
              <Sparkles className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <div className="font-extrabold text-slate-900 text-sm tracking-tight leading-snug">
                Enterprise AI Copilot
              </div>
              <div className="text-[11px] font-medium text-slate-400">
                Agentic RAG & Research
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setIsOpen(false)}
            aria-label="Close navigation menu"
            className="lg:hidden p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Workspace Pill */}
        <div className="px-5 py-2.5 bg-slate-50/70 border-b border-slate-100 flex items-center justify-between text-xs">
          <span className="text-slate-500 font-medium text-[11px]">Workspace:</span>
          <span className="font-mono text-[11px] font-semibold px-2 py-0.5 rounded-md bg-white border border-slate-200 text-slate-700 shadow-2xs">
            Production v1.0
          </span>
        </div>

        {/* Scrollable Navigation Body */}
        <div className="flex-1 overflow-y-auto px-3.5 py-4 space-y-6">
          {/* Main Navigation */}
          <div>
            <div className="px-3 pb-2 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
              Platform Modules
            </div>
            <nav className="space-y-1">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => {
                      setActiveTab(item.id);
                      setIsOpen(false);
                    }}
                    className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold transition-all group cursor-pointer ${
                      isActive
                        ? 'bg-sky-50/80 text-sky-900 border border-sky-200/80 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70 border border-transparent'
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <div
                        className={`p-1 rounded-lg transition-colors ${
                          isActive
                            ? 'bg-sky-100 text-sky-700'
                            : 'bg-slate-100/70 text-slate-400 group-hover:text-slate-600 group-hover:bg-slate-200/60'
                        }`}
                      >
                        <Icon className="w-4 h-4" />
                      </div>
                      <span>{item.label}</span>
                    </div>
                    {item.badge && (
                      <span
                        className={`text-[10px] font-mono px-2 py-0.5 rounded-full font-medium ${
                          isActive
                            ? 'bg-sky-100/80 text-sky-800'
                            : 'bg-slate-100 text-slate-500 group-hover:bg-slate-200/70'
                        }`}
                      >
                        {item.badge}
                      </span>
                    )}
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Recent Research Queries */}
          <div>
            <div className="px-3 pb-2 flex items-center justify-between">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <Clock className="w-3 h-3 text-slate-400" />
                Recent Research
              </span>
              <span className="text-[10px] text-slate-400 font-mono">
                {recentResearch.length}
              </span>
            </div>

            <div className="space-y-1">
              {recentResearch.length === 0 ? (
                <div className="px-3 py-2 text-[11px] text-slate-400 italic">
                  No previous research queries
                </div>
              ) : (
                recentResearch.slice(0, 5).map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    title={item.query}
                    onClick={() => {
                      onSelectRecentQuery(item);
                      setIsOpen(false);
                    }}
                    className="w-full text-left p-2.5 rounded-xl hover:bg-slate-100/80 text-xs transition group cursor-pointer block border border-transparent hover:border-slate-200/60"
                  >
                    <div className="text-slate-700 font-medium text-[11px] leading-snug line-clamp-2 group-hover:text-sky-700 transition-colors">
                      {item.query}
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-slate-400 mt-1.5 font-mono pt-1 border-t border-slate-100/70">
                      <span className="capitalize px-1.5 py-0.2 rounded bg-slate-100 text-slate-600 font-medium">
                        {item.mode}
                      </span>
                      <span className="tabular-nums font-medium">{item.timestamp}</span>
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>

          {/* Knowledge Library Categories */}
          <div>
            <div className="px-3 pb-2 text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <BookOpen className="w-3 h-3 text-slate-400" />
              Knowledge Library
            </div>
            <div className="space-y-0.5">
              {KNOWLEDGE_CATEGORIES.map((cat, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => {
                    setActiveTab('documents');
                    setIsOpen(false);
                  }}
                  className="w-full flex items-center justify-between px-3 py-1.5 rounded-lg text-[11px] text-slate-600 hover:text-slate-900 hover:bg-slate-100/60 transition cursor-pointer"
                >
                  <span className="truncate">{cat.name}</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-500 font-medium">
                    {cat.count}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom System / Storage Status Card */}
        <div className="p-4 border-t border-slate-100 bg-slate-50/60">
          <div className="p-3 bg-white border border-slate-200/80 rounded-2xl shadow-2xs space-y-2">
            <div className="flex items-center justify-between text-[11px]">
              <div className="flex items-center gap-1.5 font-bold text-slate-800">
                <Database className="w-3.5 h-3.5 text-indigo-600" />
                <span>Vector & Lexical Engine</span>
              </div>
              <span
                className={`w-2 h-2 rounded-full ${
                  isBackendHealthy === true
                    ? 'bg-emerald-500 animate-pulse'
                    : isBackendHealthy === false
                    ? 'bg-rose-500'
                    : 'bg-amber-400 animate-ping'
                }`}
              />
            </div>

            <div className="space-y-1 text-[10px] text-slate-500 font-mono">
              <div className="flex justify-between">
                <span>Embeddings:</span>
                <span className="font-semibold text-slate-700">bge-small 384d</span>
              </div>
              <div className="flex justify-between">
                <span>Retrieval:</span>
                <span className="font-semibold text-slate-700">Hybrid + RRF</span>
              </div>
              <div className="flex justify-between">
                <span>Reranker:</span>
                <span className="font-semibold text-slate-700">BGE Cross-Enc</span>
              </div>
            </div>

            <div className="pt-1.5 border-t border-slate-100 flex items-center justify-between text-[10px]">
              <span className="text-emerald-700 font-semibold flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                Operational
              </span>
              <button
                type="button"
                onClick={() => {
                  setActiveTab('health');
                  setIsOpen(false);
                }}
                className="text-sky-700 hover:text-sky-900 font-medium flex items-center gap-0.5 cursor-pointer"
              >
                Inspect
                <ChevronRight className="w-3 h-3" />
              </button>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};
