import React from 'react';
import { ShieldCheck, Activity, Search, Bot, Database, Sparkles } from 'lucide-react';
import { BACKEND_URL } from '../services/api';

interface HeaderProps {
  activeTab: 'research' | 'agent' | 'search' | 'documents' | 'health';
  setActiveTab: (tab: 'research' | 'agent' | 'search' | 'documents' | 'health') => void;
  isBackendHealthy: boolean | null;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab, isBackendHealthy }) => {
  return (
    <header className="border-b border-slate-200/80 bg-white/85 backdrop-blur-md sticky top-0 z-50 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 via-blue-600 to-indigo-600 flex items-center justify-center shadow-md shadow-sky-500/25 ring-2 ring-white">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-base font-bold bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 bg-clip-text text-transparent">
                Enterprise AI Copilot
              </h1>
              <p className="text-xs text-slate-500 font-medium">Knowledge & Research Platform</p>
            </div>
          </div>

          {/* Navigation Pill Menu */}
          <nav className="flex items-center gap-1 bg-slate-100/90 p-1 rounded-xl border border-slate-200/70 shadow-inner">
            <button
              onClick={() => setActiveTab('research')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'research'
                  ? 'bg-white text-sky-700 shadow-xs border border-slate-200/80'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-white/60'
              }`}
            >
              <ShieldCheck className={`w-4 h-4 ${activeTab === 'research' ? 'text-sky-600' : 'text-slate-400'}`} />
              Grounded Research
            </button>
            <button
              onClick={() => setActiveTab('agent')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'agent'
                  ? 'bg-white text-indigo-700 shadow-xs border border-slate-200/80'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-white/60'
              }`}
            >
              <Bot className={`w-4 h-4 ${activeTab === 'agent' ? 'text-indigo-600' : 'text-slate-400'}`} />
              Agentic Workflow
            </button>
            <button
              onClick={() => setActiveTab('search')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'search'
                  ? 'bg-white text-blue-700 shadow-xs border border-slate-200/80'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-white/60'
              }`}
            >
              <Search className={`w-4 h-4 ${activeTab === 'search' ? 'text-blue-600' : 'text-slate-400'}`} />
              Hybrid Search
            </button>
            <button
              onClick={() => setActiveTab('documents')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'documents'
                  ? 'bg-white text-emerald-700 shadow-xs border border-slate-200/80'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-white/60'
              }`}
            >
              <Database className={`w-4 h-4 ${activeTab === 'documents' ? 'text-emerald-600' : 'text-slate-400'}`} />
              Documents
            </button>
            <button
              onClick={() => setActiveTab('health')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'health'
                  ? 'bg-white text-violet-700 shadow-xs border border-slate-200/80'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-white/60'
              }`}
            >
              <Activity className={`w-4 h-4 ${activeTab === 'health' ? 'text-violet-600' : 'text-slate-400'}`} />
              Health & Metrics
            </button>
          </nav>

          {/* Connection Badge */}
          <div className="flex items-center gap-2 text-xs">
            <div
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
                isBackendHealthy === true
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200/80'
                  : isBackendHealthy === false
                  ? 'bg-rose-50 text-rose-700 border-rose-200/80'
                  : 'bg-amber-50 text-amber-700 border-amber-200/80'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isBackendHealthy === true
                    ? 'bg-emerald-500'
                    : isBackendHealthy === false
                    ? 'bg-rose-500 animate-pulse'
                    : 'bg-amber-500 animate-pulse'
                }`}
              />
              <span className="hidden sm:inline">
                {isBackendHealthy === true ? 'Live API Connected' : isBackendHealthy === false ? 'API Offline' : 'Connecting...'}
              </span>
            </div>
            <span className="text-slate-400 font-mono text-[11px] hidden lg:inline">
              {BACKEND_URL}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
