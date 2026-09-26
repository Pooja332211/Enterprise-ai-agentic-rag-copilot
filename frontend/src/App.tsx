import { useEffect, useState, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { TopHeader } from './components/TopHeader';
import { HomeDashboard } from './components/HomeDashboard';
import { GroundedResearch } from './components/GroundedResearch';
import { AgenticResearch } from './components/AgenticResearch';
import { HybridSearch } from './components/HybridSearch';
import { DocumentManager } from './components/DocumentManager';
import { KnowledgeGraph } from './components/KnowledgeGraph';
import { AnalyticsDashboard } from './components/AnalyticsDashboard';
import { SystemHealth } from './components/SystemHealth';
import { CommandPalette } from './components/CommandPalette';
import { api, BACKEND_URL } from './services/api';
import { TabType, RecentResearchItem } from './types';
import { AlertCircle } from 'lucide-react';

const DEFAULT_RECENT_RESEARCH: RecentResearchItem[] = [
  {
    id: 'res-default-1',
    query: 'What retrieval methods does Company ABC use?',
    mode: 'research',
    timestamp: '10:45 AM',
    status: 'grounded',
    latency_ms: 42.1,
    citation_count: 2,
  },
  {
    id: 'res-default-2',
    query: 'Compare the main objectives of the enterprise knowledge platform with how it retrieves information for complex questions.',
    mode: 'agent',
    timestamp: '10:30 AM',
    status: 'completed',
    latency_ms: 8120.4,
    citation_count: 3,
  },
  {
    id: 'res-default-3',
    query: 'How does Company ABC handle complex questions?',
    mode: 'research',
    timestamp: '10:15 AM',
    status: 'grounded',
    latency_ms: 38.6,
    citation_count: 1,
  },
  {
    id: 'res-default-4',
    query: "What is Company ABC's employee vacation allowance?",
    mode: 'research',
    timestamp: '09:50 AM',
    status: 'abstained',
    latency_ms: 29.8,
    citation_count: 0,
  },
];

export function App() {
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);
  const [documentCount, setDocumentCount] = useState<number>(1);
  const [activeQuery, setActiveQuery] = useState<string>('');
  const [autoExecute, setAutoExecute] = useState<boolean>(false);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState<boolean>(false);

  // Global shortcut Ctrl+K / Cmd+K
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Load recent research from localStorage with fallback to default benchmark queries
  const [recentResearch, setRecentResearch] = useState<RecentResearchItem[]>(() => {
    try {
      const stored = localStorage.getItem('ea_copilot_recent_research');
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch {
      // Fallback
    }
    return DEFAULT_RECENT_RESEARCH;
  });

  const checkConnection = useCallback(async () => {
    try {
      const res = await api.checkHealth();
      setIsBackendHealthy(res.status === 'healthy');
    } catch {
      setIsBackendHealthy(false);
    }
  }, []);

  const fetchDocCount = useCallback(async () => {
    try {
      const docs = await api.getDocuments(50, 0);
      setDocumentCount(docs.total ?? docs.items?.length ?? 1);
    } catch {
      // Ignore if backend offline
    }
  }, []);

  useEffect(() => {
    checkConnection();
    fetchDocCount();
    const interval = setInterval(checkConnection, 12000);
    return () => clearInterval(interval);
  }, [checkConnection, fetchDocCount]);

  const handleRecordRecent = useCallback((item: RecentResearchItem) => {
    setRecentResearch((prev) => {
      const filtered = prev.filter((r) => r.query.toLowerCase() !== item.query.toLowerCase());
      const updated = [item, ...filtered].slice(0, 15);
      try {
        localStorage.setItem('ea_copilot_recent_research', JSON.stringify(updated));
      } catch {
        // localStorage quota
      }
      return updated;
    });
  }, []);

  const handleExecuteQuery = useCallback((query: string, mode: 'research' | 'agent' | 'search') => {
    setActiveQuery(query);
    setAutoExecute(true);
    setActiveTab(mode);
    setSidebarOpen(false);
  }, []);

  const handleSelectRecentQuery = useCallback((item: RecentResearchItem) => {
    setActiveQuery(item.query);
    setAutoExecute(true);
    setActiveTab(item.mode);
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex font-sans selection:bg-sky-500 selection:text-white">
      {/* Persistent Left Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={(tab) => {
          setActiveTab(tab);
          setAutoExecute(false);
        }}
        isOpen={sidebarOpen}
        setIsOpen={setSidebarOpen}
        recentResearch={recentResearch}
        onSelectRecentQuery={handleSelectRecentQuery}
        isBackendHealthy={isBackendHealthy}
        documentCount={documentCount}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 min-h-screen">
        {/* Top Header */}
        <TopHeader
          activeTab={activeTab}
          setActiveTab={(tab) => {
            setActiveTab(tab);
            setAutoExecute(false);
          }}
          onOpenSidebar={() => setSidebarOpen(true)}
          isBackendHealthy={isBackendHealthy}
          onQuickSearch={handleExecuteQuery}
          onRefreshHealth={checkConnection}
          onOpenCommandPalette={() => setIsCommandPaletteOpen(true)}
        />

        {/* Backend Warning Banner if offline */}
        {isBackendHealthy === false && (
          <div className="bg-amber-50 border-b border-amber-200 px-4 py-2.5 text-xs text-amber-800 text-center flex items-center justify-center gap-2 shadow-2xs">
            <AlertCircle className="w-4 h-4 text-amber-600 flex-shrink-0" />
            <span>
              Cannot connect to backend at <code className="bg-amber-100/80 px-1.5 py-0.5 rounded font-mono font-medium text-amber-900">{BACKEND_URL}</code>.
              Ensure the FastAPI backend is running via{' '}
              <code className="bg-amber-100/80 px-1.5 py-0.5 rounded font-mono font-medium text-amber-900">
                python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
              </code>
            </span>
          </div>
        )}

        {/* Dynamic Main Workspace Content */}
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
          {activeTab === 'dashboard' && (
            <HomeDashboard
              onExecuteQuery={handleExecuteQuery}
              recentResearch={recentResearch}
              documentCount={documentCount}
              isBackendHealthy={isBackendHealthy}
            />
          )}

          {activeTab === 'research' && (
            <GroundedResearch
              initialQuery={activeQuery}
              autoExecute={autoExecute}
              onRecordRecent={handleRecordRecent}
            />
          )}

          {activeTab === 'agent' && (
            <AgenticResearch
              initialQuery={activeQuery}
              autoExecute={autoExecute}
              onRecordRecent={handleRecordRecent}
            />
          )}

          {activeTab === 'search' && (
            <HybridSearch
              initialQuery={activeQuery}
              autoExecute={autoExecute}
              onRecordRecent={handleRecordRecent}
            />
          )}

          {activeTab === 'documents' && <DocumentManager />}

          {activeTab === 'graph' && <KnowledgeGraph />}

          {activeTab === 'analytics' && <AnalyticsDashboard />}

          {activeTab === 'health' && <SystemHealth />}
        </main>

        {/* Footer */}
        <footer className="border-t border-slate-200/80 bg-white/70 backdrop-blur-xs py-4 px-6 text-center text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            Enterprise AI Knowledge & Research Copilot • Production Portfolio Edition
          </div>
          <div className="font-mono text-[11px] text-slate-400">
            API Gateway: <span className="text-slate-700 font-semibold">{BACKEND_URL}</span>
          </div>
        </footer>
      </div>

      {/* Global Command Palette (Ctrl+K / ⌘K) */}
      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        onNavigate={(tab) => {
          setActiveTab(tab);
          setAutoExecute(false);
        }}
        onExecuteQuery={handleExecuteQuery}
      />
    </div>
  );
}

export default App;
