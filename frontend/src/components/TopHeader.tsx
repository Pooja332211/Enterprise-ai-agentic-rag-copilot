import React, { useState } from 'react';
import {
  Menu,
  Search,
  Bot,
  Sparkles,
  Activity,
  Bell,
  Settings,
} from 'lucide-react';
import { TabType } from '../types';
import { BACKEND_URL } from '../services/api';

interface TopHeaderProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
  onOpenSidebar: () => void;
  isBackendHealthy: boolean | null;
  onQuickSearch: (query: string, mode: 'research' | 'agent' | 'search') => void;
  onRefreshHealth: () => void;
  onOpenCommandPalette?: () => void;
}

export const TopHeader: React.FC<TopHeaderProps> = ({
  activeTab,
  setActiveTab,
  onOpenSidebar,
  isBackendHealthy,
  onQuickSearch,
  onRefreshHealth,
  onOpenCommandPalette,
}) => {
  const [quickInput, setQuickInput] = useState('');

  const getTabDetails = (tab: TabType) => {
    switch (tab) {
      case 'dashboard':
        return { title: 'Enterprise Research Dashboard', subtitle: 'Overview, quick templates & multi-agent execution' };
      case 'research':
        return { title: 'Grounded Research Workspace', subtitle: 'Strict XML evidence isolation & sentence-level claim audit' };
      case 'agent':
        return { title: 'Agentic Workflow Pipeline', subtitle: 'LangGraph multi-step decomposition & iterative retrieval' };
      case 'search':
        return { title: 'Hybrid Retrieval Engine', subtitle: 'Dense vector embeddings + Okapi BM25 reciprocal rank fusion' };
      case 'documents':
        return { title: 'Knowledge Base Management', subtitle: 'Multi-format ingestion, chunk inspection & vector indexing' };
      case 'graph':
        return { title: 'Knowledge & Entity Graph', subtitle: 'Semantic relationships, document topology & cross-references' };
      case 'analytics':
        return { title: 'Observability & Analytics', subtitle: 'Retrieval accuracy, latency distribution & claim verification benchmarks' };
      case 'health':
        return { title: 'System Health & Telemetry', subtitle: 'Kubernetes-style liveness/readiness probes & Prometheus metrics' };
      default:
        return { title: 'Enterprise AI Copilot', subtitle: 'Agentic RAG & Knowledge Platform' };
    }
  };

  const currentInfo = getTabDetails(activeTab);

  const handleQuickSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!quickInput.trim()) return;
    onQuickSearch(quickInput.trim(), 'research');
    setQuickInput('');
  };

  return (
    <header className="sticky top-0 z-30 bg-white/95 backdrop-blur-md border-b border-slate-200/80 px-4 sm:px-6 py-2.5">
      <div className="flex items-center justify-between gap-3 sm:gap-4 min-h-[38px]">
        {/* Left: Mobile hamburger & breadcrumb */}
        <div className="flex items-center gap-2.5 min-w-0 flex-shrink-0">
          <button
            type="button"
            onClick={onOpenSidebar}
            className="lg:hidden p-1.5 rounded-xl text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition cursor-pointer"
            aria-label="Toggle navigation menu"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div className="min-w-0">
            <h1 className="text-sm sm:text-base font-bold text-slate-900 tracking-tight truncate leading-tight">
              {currentInfo.title}
            </h1>
            <p className="text-[11px] text-slate-500 truncate hidden xl:block leading-tight mt-0.5 max-w-[260px]">
              {currentInfo.subtitle}
            </p>
          </div>
        </div>

        {/* Center: Global Quick Query Bar */}
        <div className="hidden md:flex flex-1 max-w-lg mx-2 lg:mx-4 min-w-[240px]">
          <form onSubmit={handleQuickSubmit} className="w-full relative flex items-center">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="text"
              value={quickInput}
              onChange={(e) => setQuickInput(e.target.value)}
              placeholder="Ask or search knowledge base..."
              title="Search or execute query (Ctrl+K for palette)"
              className="w-full h-9 pl-10 pr-20 bg-slate-100/80 hover:bg-slate-100 focus:bg-white border border-slate-200/80 rounded-xl text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-500/10 transition shadow-2xs font-medium"
            />
            <div className="absolute right-1.5 top-1/2 -translate-y-1/2 flex items-center gap-1">
              {onOpenCommandPalette && (
                <button
                  type="button"
                  onClick={onOpenCommandPalette}
                  className="px-1.5 py-0.5 text-[10px] font-mono text-slate-500 hover:text-slate-800 bg-white border border-slate-200 rounded shadow-2xs cursor-pointer transition flex items-center gap-0.5"
                  title="Open Command Palette (Ctrl+K / ⌘K)"
                >
                  <kbd>Ctrl</kbd>+<kbd>K</kbd>
                </button>
              )}
              <span className="text-[10px] font-mono text-slate-400 bg-white px-1.5 py-0.5 rounded border border-slate-200 shadow-2xs">
                ↵
              </span>
            </div>
          </form>
        </div>

        {/* Right: Actions, Health Badge & User Profile */}
        <div className="flex items-center gap-2 sm:gap-2.5 flex-shrink-0">
          {/* Quick Action Buttons */}
          <div className="hidden xl:flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setActiveTab('research')}
              className="h-8.5 px-3 rounded-xl bg-sky-50 hover:bg-sky-100 text-sky-700 text-xs font-semibold border border-sky-200/80 transition flex items-center gap-1.5 cursor-pointer hover:shadow-2xs"
            >
              <Sparkles className="w-3.5 h-3.5 text-sky-600" />
              Grounded
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('agent')}
              className="h-8.5 px-3 rounded-xl bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-xs font-semibold border border-indigo-200/80 transition flex items-center gap-1.5 cursor-pointer hover:shadow-2xs"
            >
              <Bot className="w-3.5 h-3.5 text-indigo-600" />
              Agentic
            </button>
          </div>

          {/* Backend Connection Status Badge */}
          <div
            onClick={onRefreshHealth}
            title={`Backend: ${BACKEND_URL} (Click to refresh probe)`}
            className={`cursor-pointer h-8.5 px-2.5 rounded-xl border flex items-center gap-2 text-xs font-medium transition hover:shadow-2xs ${
              isBackendHealthy === true
                ? 'bg-emerald-50/80 text-emerald-800 border-emerald-200/80 hover:bg-emerald-100/60'
                : isBackendHealthy === false
                ? 'bg-rose-50 text-rose-800 border-rose-200/80 hover:bg-rose-100'
                : 'bg-amber-50 text-amber-800 border-amber-200/80'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isBackendHealthy === true
                  ? 'bg-emerald-500 animate-pulse'
                  : isBackendHealthy === false
                  ? 'bg-rose-500'
                  : 'bg-amber-500 animate-ping'
              }`}
            />
            <Activity className="w-3.5 h-3.5 text-slate-500" />
            <span className="hidden sm:inline font-mono text-[11px]">
              {isBackendHealthy === true
                ? 'Backend 8000'
                : isBackendHealthy === false
                ? 'Offline'
                : 'Connecting...'}
            </span>
          </div>

          {/* Notifications Button */}
          <button
            type="button"
            aria-label="View notifications"
            onClick={() => setActiveTab('health')}
            className="relative p-2 rounded-xl text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition cursor-pointer"
          >
            <Bell className="w-4 h-4" />
            <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-sky-500" />
          </button>

          {/* Settings Button */}
          <button
            type="button"
            aria-label="Platform settings"
            onClick={() => setActiveTab('health')}
            className="p-2 rounded-xl text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition cursor-pointer"
          >
            <Settings className="w-4 h-4" />
          </button>

          {/* Tenant Avatar */}
          <div className="flex items-center gap-2 pl-2 border-l border-slate-200/80">
            <div
              className="w-8 h-8 rounded-full bg-gradient-to-tr from-slate-700 via-indigo-900 to-slate-900 text-white font-bold text-xs flex items-center justify-center shadow-xs select-none"
              title="Acme Corp Enterprise Tenant"
            >
              EA
            </div>
            <div className="hidden xl:block text-left">
              <div className="text-xs font-bold text-slate-800 leading-tight">
                Enterprise AI
              </div>
              <div className="text-[10px] text-slate-400">
                Acme Tenant
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
