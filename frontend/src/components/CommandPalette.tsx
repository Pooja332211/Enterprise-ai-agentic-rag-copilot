import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Search,
  LayoutDashboard,
  ShieldCheck,
  Bot,
  Layers,
  FileText,
  Network,
  BarChart3,
  Activity,
  ArrowRight,
  CornerDownLeft,
  X,
  Sparkles,
  Command,
} from 'lucide-react';
import { TabType } from '../types';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigate: (tab: TabType) => void;
  onExecuteQuery: (query: string, mode: 'research' | 'agent' | 'search') => void;
}

interface CommandItem {
  id: string;
  category: 'Navigation' | 'Research Benchmark' | 'Quick Actions';
  title: string;
  subtitle?: string;
  icon: React.ComponentType<{ className?: string }>;
  action: () => void;
  badge?: string;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  onNavigate,
  onExecuteQuery,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const commands: CommandItem[] = useMemo(
    () => [
      // Navigation
      {
        id: 'nav-dashboard',
        category: 'Navigation',
        title: 'Enterprise Research Dashboard',
        subtitle: 'Overview, quick templates & platform telemetry',
        icon: LayoutDashboard,
        action: () => {
          onNavigate('dashboard');
          onClose();
        },
      },
      {
        id: 'nav-grounded',
        category: 'Navigation',
        title: 'Grounded Research Workspace',
        subtitle: 'Strict XML evidence isolation & sentence claim audit',
        icon: ShieldCheck,
        badge: 'Zero Hallucinations',
        action: () => {
          onNavigate('research');
          onClose();
        },
      },
      {
        id: 'nav-agentic',
        category: 'Navigation',
        title: 'Agentic Workflow Pipeline',
        subtitle: 'LangGraph multi-step decomposition & iterative retrieval',
        icon: Bot,
        badge: 'StateGraph',
        action: () => {
          onNavigate('agent');
          onClose();
        },
      },
      {
        id: 'nav-hybrid',
        category: 'Navigation',
        title: 'Hybrid Retrieval Engine',
        subtitle: 'Dense BGE-384d vector embeddings + Okapi BM25 RRF',
        icon: Layers,
        badge: 'RRF k=60',
        action: () => {
          onNavigate('search');
          onClose();
        },
      },
      {
        id: 'nav-documents',
        category: 'Navigation',
        title: 'Knowledge Base Documents',
        subtitle: 'Ingest PDF, DOCX, TXT with recursive text chunking',
        icon: FileText,
        action: () => {
          onNavigate('documents');
          onClose();
        },
      },
      {
        id: 'nav-graph',
        category: 'Navigation',
        title: 'Knowledge & Entity Graph',
        subtitle: 'Semantic node links, citations & cross-references',
        icon: Network,
        action: () => {
          onNavigate('graph');
          onClose();
        },
      },
      {
        id: 'nav-analytics',
        category: 'Navigation',
        title: 'Observability & Analytics',
        subtitle: 'MRR, latency distributions & verification audits',
        icon: BarChart3,
        action: () => {
          onNavigate('analytics');
          onClose();
        },
      },
      {
        id: 'nav-health',
        category: 'Navigation',
        title: 'System Health & Telemetry',
        subtitle: 'FastAPI, Qdrant & Prometheus operational status',
        icon: Activity,
        badge: 'Live',
        action: () => {
          onNavigate('health');
          onClose();
        },
      },

      // Benchmarks
      {
        id: 'bench-retrieval',
        category: 'Research Benchmark',
        title: 'What retrieval methods does Company ABC use?',
        subtitle: 'Grounded query verifying dense semantic vector + BM25 keyword search',
        icon: Sparkles,
        badge: 'Core Benchmark',
        action: () => {
          onExecuteQuery('What retrieval methods does Company ABC use?', 'research');
          onClose();
        },
      },
      {
        id: 'bench-complex',
        category: 'Research Benchmark',
        title: 'Compare the main objectives of the enterprise knowledge platform with how it retrieves information for complex questions.',
        subtitle: 'Agentic multi-hop query triggering LangGraph decomposition',
        icon: Bot,
        badge: 'Compound',
        action: () => {
          onExecuteQuery(
            'Compare the main objectives of the enterprise knowledge platform with how it retrieves information for complex questions.',
            'agent'
          );
          onClose();
        },
      },
      {
        id: 'bench-handling',
        category: 'Research Benchmark',
        title: 'How does Company ABC handle complex questions?',
        subtitle: 'Targeted single-turn question on query decomposition',
        icon: Sparkles,
        action: () => {
          onExecuteQuery('How does Company ABC handle complex questions?', 'research');
          onClose();
        },
      },
      {
        id: 'bench-abstention',
        category: 'Research Benchmark',
        title: "What is Company ABC's employee vacation allowance?",
        subtitle: 'Verification abstention test (expected 0 unsupported claims)',
        icon: ShieldCheck,
        badge: 'Abstention Test',
        action: () => {
          onExecuteQuery("What is Company ABC's employee vacation allowance?", 'research');
          onClose();
        },
      },
    ],
    [onNavigate, onExecuteQuery, onClose]
  );

  const filteredCommands = useMemo(() => {
    if (!searchTerm.trim()) return commands;
    const term = searchTerm.toLowerCase();
    return commands.filter(
      (c) =>
        c.title.toLowerCase().includes(term) ||
        c.subtitle?.toLowerCase().includes(term) ||
        c.category.toLowerCase().includes(term) ||
        c.badge?.toLowerCase().includes(term)
    );
  }, [commands, searchTerm]);

  // Reset selectedIndex when filter changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [searchTerm]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
      setSearchTerm('');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  // Global key listener for Ctrl+K or Cmd+K
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        if (isOpen) {
          onClose();
        } else {
          // Can be triggered externally via App state
        }
      } else if (e.key === 'Escape' && isOpen) {
        e.preventDefault();
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Arrow key navigation inside palette
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1) % (filteredCommands.length || 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev - 1 + filteredCommands.length) % (filteredCommands.length || 1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (filteredCommands[selectedIndex]) {
        filteredCommands[selectedIndex].action();
      } else if (searchTerm.trim()) {
        // Direct execution of custom query in Grounded mode
        onExecuteQuery(searchTerm.trim(), 'research');
        onClose();
      }
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-16 sm:pt-24 px-4 bg-slate-900/40 backdrop-blur-xs"
      onClick={onClose}
      aria-modal="true"
      role="dialog"
    >
      <div
        className="w-full max-w-2xl bg-white rounded-3xl border border-slate-200/90 shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150 flex flex-col max-h-[80vh]"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        {/* Command Search Bar */}
        <div className="p-4 border-b border-slate-100 flex items-center gap-3 bg-slate-50/50">
          <Search className="w-5 h-5 text-slate-400 flex-shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Type a command, query, or navigate workspace..."
            className="flex-1 bg-transparent text-sm text-slate-900 placeholder-slate-400 focus:outline-none font-medium"
          />
          <div className="flex items-center gap-1.5 flex-shrink-0">
            <kbd className="px-2 py-0.5 text-[10px] font-mono text-slate-500 bg-white border border-slate-200 rounded shadow-2xs">
              ESC
            </kbd>
            <button
              type="button"
              onClick={onClose}
              className="p-1 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200/60 transition cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Results List */}
        <div
          ref={listRef}
          className="flex-1 overflow-y-auto p-3 space-y-1 divide-y divide-slate-100/60"
        >
          {filteredCommands.length === 0 ? (
            <div className="py-10 text-center space-y-2">
              <Command className="w-8 h-8 text-slate-300 mx-auto" />
              <p className="text-xs text-slate-600 font-medium">
                No matching commands found for &ldquo;{searchTerm}&rdquo;
              </p>
              <button
                type="button"
                onClick={() => {
                  onExecuteQuery(searchTerm.trim(), 'research');
                  onClose();
                }}
                className="mt-2 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-sky-50 text-sky-700 hover:bg-sky-100 text-xs font-semibold border border-sky-200 cursor-pointer transition"
              >
                Execute &ldquo;{searchTerm}&rdquo; in Grounded Research
                <ArrowRight className="w-3 h-3" />
              </button>
            </div>
          ) : (
            filteredCommands.map((cmd, idx) => {
              const Icon = cmd.icon;
              const isSelected = idx === selectedIndex;
              return (
                <div
                  key={cmd.id}
                  onClick={cmd.action}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={`pt-1.5 first:pt-0 flex items-center justify-between gap-3 p-3 rounded-2xl cursor-pointer transition-all duration-100 ${
                    isSelected
                      ? 'bg-sky-50/80 text-sky-950 border border-sky-200/70 shadow-2xs'
                      : 'hover:bg-slate-50 text-slate-700 border border-transparent'
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div
                      className={`p-2 rounded-xl flex-shrink-0 transition ${
                        isSelected
                          ? 'bg-sky-500 text-white shadow-xs'
                          : 'bg-slate-100 text-slate-600'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-slate-900 truncate">
                          {cmd.title}
                        </span>
                        {cmd.badge && (
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200/80 font-semibold flex-shrink-0">
                            {cmd.badge}
                          </span>
                        )}
                      </div>
                      {cmd.subtitle && (
                        <div className="text-[11px] text-slate-500 truncate mt-0.5">
                          {cmd.subtitle}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider hidden sm:inline">
                      {cmd.category}
                    </span>
                    {isSelected && (
                      <span className="flex items-center gap-1 text-[11px] font-mono text-sky-700 bg-sky-100/70 px-2 py-0.5 rounded-md">
                        <CornerDownLeft className="w-3 h-3" /> Select
                      </span>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Footer shortcuts */}
        <div className="p-3 bg-slate-50/80 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500 px-4">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 text-[10px] font-mono bg-white border border-slate-200 rounded">
                ↑
              </kbd>
              <kbd className="px-1.5 py-0.5 text-[10px] font-mono bg-white border border-slate-200 rounded">
                ↓
              </kbd>{' '}
              Navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 text-[10px] font-mono bg-white border border-slate-200 rounded">
                ↵
              </kbd>{' '}
              Execute
            </span>
          </div>
          <div className="font-mono text-[10px] text-slate-400">
            Enterprise AI Copilot Palette
          </div>
        </div>
      </div>
    </div>
  );
};
