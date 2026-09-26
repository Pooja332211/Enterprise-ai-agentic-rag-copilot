import React, { useState } from 'react';
import {
  FileText,
  FileCode,
  File,
  Search,
  Sparkles,
  UploadCloud,
  Layers,
  ArrowRight,
  X,
  CheckCircle2,
  Database,
  Cpu,
  Workflow,
  ShieldCheck,
  Zap,
} from 'lucide-react';

export type PipelineStage = 'idle' | 'input' | 'retrieval' | 'rerank' | 'agent' | 'verify' | 'completed';

interface NodeTechSpec {
  key: string;
  title: string;
  modelTech: string;
  category: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  specs: string[];
  description: string;
}

const NODE_SPECS: Record<string, NodeTechSpec> = {
  documents: {
    key: 'documents',
    title: 'Document Ingestion & Storage',
    modelTech: 'SQLite + Structure-Aware Chunking',
    category: 'Corpus Storage',
    icon: FileText,
    color: 'text-sky-600',
    specs: [
      'Multi-format support: PDF, DOCX, TXT, MD, HTML',
      'Hierarchical section & page header extraction',
      'Deterministic document hashing & deduplication',
    ],
    description: 'Enterprise documents are parsed into structured semantic chunks with page numbers and section paths, then synchronized across vector and lexical stores.',
  },
  dense: {
    key: 'dense',
    title: 'Dense Semantic Vector Search',
    modelTech: 'BAAI/bge-small-en-v1.5 (384-d)',
    category: 'Embedding & Vector Engine',
    icon: Cpu,
    color: 'text-indigo-600',
    specs: [
      'FastEmbed local ONNX inference runtime',
      '384-dimensional dense vector embeddings',
      'Qdrant vector engine with cosine similarity search',
    ],
    description: 'Captures deep conceptual and semantic intent beyond exact keyword matches, projecting questions and document chunks into a unified metric space.',
  },
  bm25: {
    key: 'bm25',
    title: 'Lexical Sparse Search (BM25)',
    modelTech: 'Okapi BM25Plus Algorithm',
    category: 'Lexical Token Matching',
    icon: Database,
    color: 'text-amber-600',
    specs: [
      'Inverted index with term-frequency & IDF weighting',
      'Strict keyword & acronym exact match sensitivity',
      'Length-normalized document ranking',
    ],
    description: 'Guarantees precise token and identifier recall (e.g. policy codes, product names, specific parameters) complementing semantic similarity.',
  },
  rerank: {
    key: 'rerank',
    title: 'RRF & Neural Cross-Encoder',
    modelTech: 'Reciprocal Rank Fusion (k=60) + BAAI/bge-reranker-base',
    category: 'Neural Reranking & Fusion',
    icon: Zap,
    color: 'text-purple-600',
    specs: [
      'Reciprocal Rank Fusion with smoothing parameter k=60',
      'ONNX-accelerated BGE cross-attention neural reranker',
      'Dynamic distractor filtering with relative score cutoffs',
    ],
    description: 'Aggregates ranked lists from dense vector search and BM25 lexical search, then applies full cross-attention scoring between query and chunks.',
  },
  langgraph: {
    key: 'langgraph',
    title: 'LangGraph Multi-Agent State Machine',
    modelTech: 'Cyclic StateGraph Orchestrator',
    category: 'Agentic Workflow',
    icon: Workflow,
    color: 'text-purple-700',
    specs: [
      '10-node cyclic execution graph with conditional edges',
      'Sub-query decomposition for compound queries',
      'Evidence sufficiency checks & bounded iterative refinement',
    ],
    description: 'Decomposes complex multi-hop questions into targeted sub-queries, queries evidence pools in parallel, and assesses sufficiency before synthesis.',
  },
  verified: {
    key: 'verified',
    title: 'Sentence-Level Claim Verification',
    modelTech: 'XML Evidence Isolation & Grounding Audit',
    category: 'Safety & Hallucination Resistance',
    icon: ShieldCheck,
    color: 'text-emerald-600',
    specs: [
      'Strict prompt evidence boundary XML isolation',
      'Empirical sentence-to-chunk token overlap scoring',
      'Deterministic bounded abstention on ungrounded queries',
    ],
    description: 'Ensures zero hallucinations by isolating LLM synthesis strictly to provided evidence chunks, auditing every generated sentence against provenance sources.',
  },
};

/**
 * Interactive & Live Reactive Architecture Pipeline Illustration
 */
export const HeroAbstractIllustration: React.FC<{
  className?: string;
  activeStage?: PipelineStage;
  isExecuting?: boolean;
  interactive?: boolean;
}> = ({ className = '', activeStage = 'idle', isExecuting = false, interactive = true }) => {
  const [activeModalNode, setActiveModalNode] = useState<NodeTechSpec | null>(null);

  const handleNodeClick = (nodeKey: string) => {
    if (!interactive) return;
    const spec = NODE_SPECS[nodeKey];
    if (spec) {
      setActiveModalNode(spec);
    }
  };

  // Determine stage active states
  const isInputActive = isExecuting && (activeStage === 'input' || activeStage === 'idle');
  const isRetrievalActive = isExecuting && activeStage === 'retrieval';
  const isRerankActive = isExecuting && activeStage === 'rerank';
  const isAgentActive = isExecuting && activeStage === 'agent';
  const isVerifyActive = isExecuting && activeStage === 'verify';
  const isDone = activeStage === 'completed';

  return (
    <div className={`relative select-none ${className}`}>
      {/* Interactive Architecture SVG Canvas */}
      <svg
        viewBox="0 0 460 260"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full drop-shadow-sm transition-all"
      >
        <defs>
          <linearGradient id="heroGradBg" x1="0" y1="0" x2="460" y2="260" gradientUnits="userSpaceOnUse">
            <stop stopColor="#0284c7" stopOpacity="0.08" />
            <stop offset="0.5" stopColor="#6366f1" stopOpacity="0.06" />
            <stop offset="1" stopColor="#8b5cf6" stopOpacity="0.04" />
          </linearGradient>

          <linearGradient id="flowLineGrad" x1="40" y1="130" x2="420" y2="130" gradientUnits="userSpaceOnUse">
            <stop stopColor="#38bdf8" />
            <stop offset="0.3" stopColor="#818cf8" />
            <stop offset="0.7" stopColor="#c084fc" />
            <stop offset="1" stopColor="#34d399" />
          </linearGradient>

          <linearGradient id="nodeBlue" x1="0" y1="0" x2="1" y2="1">
            <stop stopColor="#0284c7" />
            <stop offset="1" stopColor="#0369a1" />
          </linearGradient>

          <linearGradient id="nodeIndigo" x1="0" y1="0" x2="1" y2="1">
            <stop stopColor="#6366f1" />
            <stop offset="1" stopColor="#4f46e5" />
          </linearGradient>

          <linearGradient id="nodeEmerald" x1="0" y1="0" x2="1" y2="1">
            <stop stopColor="#10b981" />
            <stop offset="1" stopColor="#059669" />
          </linearGradient>

          <filter id="glowSubtle" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>

          <filter id="glowActive" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Ambient background container */}
        <rect
          width="460"
          height="260"
          rx="20"
          fill="url(#heroGradBg)"
          stroke="#e2e8f0"
          strokeWidth="1"
          strokeDasharray="4 4"
        />

        {/* Connecting circuit lines */}
        <path
          d="M 65 130 C 110 130, 115 85, 160 85 S 205 130, 250 130 S 295 85, 340 85 S 385 130, 415 130"
          stroke="url(#flowLineGrad)"
          strokeWidth={isExecuting ? '3.5' : '2.5'}
          strokeLinecap="round"
          fill="none"
          opacity={isExecuting ? '1' : '0.8'}
          strokeDasharray={isExecuting ? '8 4' : 'none'}
          className={isExecuting ? 'animate-pulse' : ''}
        />
        <path
          d="M 65 130 C 110 130, 115 175, 160 175 S 205 130, 250 130 S 295 175, 340 175 S 385 130, 415 130"
          stroke="url(#flowLineGrad)"
          strokeWidth={isExecuting ? '3' : '2'}
          strokeLinecap="round"
          fill="none"
          opacity={isExecuting ? '0.8' : '0.5'}
          strokeDasharray={isExecuting ? '6 4' : '4 3'}
        />

        {/* Node 1: Documents Layer */}
        <g
          transform="translate(45, 110)"
          onClick={() => handleNodeClick('documents')}
          className={interactive ? 'cursor-pointer' : ''}
        >
          <rect
            x="-24"
            y="-24"
            width="48"
            height="48"
            rx="12"
            fill="#ffffff"
            stroke={isInputActive ? '#38bdf8' : '#cbd5e1'}
            strokeWidth={isInputActive ? '2.5' : '1.5'}
            filter={isInputActive ? 'url(#glowActive)' : 'none'}
          />
          <rect x="-14" y="-14" width="28" height="28" rx="6" fill="#f0f9ff" stroke="#bae6fd" strokeWidth="1" />
          <path d="M -6 -7 L 2 -7 L 6 -3 L 6 7 L -6 7 Z" fill="#0284c7" opacity="0.85" />
          <path d="M 2 -7 L 2 -3 L 6 -3" fill="#38bdf8" />
          <line x1="-3" y1="0" x2="3" y2="0" stroke="#ffffff" strokeWidth="1" strokeLinecap="round" />
          <line x1="-3" y1="3" x2="3" y2="3" stroke="#ffffff" strokeWidth="1" strokeLinecap="round" />
          <text x="0" y="36" textAnchor="middle" fill="#64748b" fontSize="9" fontWeight="600" fontFamily="sans-serif">
            DOCUMENTS
          </text>
        </g>

        {/* Node 2A: Dense Vector Embedding */}
        <g
          transform="translate(155, 65)"
          onClick={() => handleNodeClick('dense')}
          className={interactive ? 'cursor-pointer' : ''}
        >
          <rect
            x="-22"
            y="-22"
            width="44"
            height="44"
            rx="12"
            fill="#ffffff"
            stroke={isRetrievalActive ? '#818cf8' : '#e2e8f0'}
            strokeWidth={isRetrievalActive ? '2.5' : '1.5'}
            filter={isRetrievalActive ? 'url(#glowActive)' : 'none'}
          />
          <circle cx="0" cy="0" r="14" fill="#e0e7ff" />
          <circle cx="-5" cy="-4" r="2.5" fill="#6366f1" />
          <circle cx="5" cy="-2" r="2.5" fill="#818cf8" />
          <circle cx="0" cy="5" r="2.5" fill="#4f46e5" />
          <line x1="-5" y1="-4" x2="5" y2="-2" stroke="#6366f1" strokeWidth="1" />
          <line x1="5" y1="-2" x2="0" y2="5" stroke="#6366f1" strokeWidth="1" />
          <line x1="-5" y1="-4" x2="0" y2="5" stroke="#6366f1" strokeWidth="1" />
          <text x="0" y="32" textAnchor="middle" fill="#64748b" fontSize="8" fontWeight="600" fontFamily="sans-serif">
            DENSE (384d)
          </text>
        </g>

        {/* Node 2B: Lexical BM25 Inverted Index */}
        <g
          transform="translate(155, 175)"
          onClick={() => handleNodeClick('bm25')}
          className={interactive ? 'cursor-pointer' : ''}
        >
          <rect
            x="-22"
            y="-22"
            width="44"
            height="44"
            rx="12"
            fill="#ffffff"
            stroke={isRetrievalActive ? '#f59e0b' : '#e2e8f0'}
            strokeWidth={isRetrievalActive ? '2.5' : '1.5'}
            filter={isRetrievalActive ? 'url(#glowActive)' : 'none'}
          />
          <circle cx="0" cy="0" r="14" fill="#fef3c7" />
          <rect x="-7" y="-7" width="14" height="4" rx="1.5" fill="#d97706" />
          <rect x="-7" y="-1" width="14" height="4" rx="1.5" fill="#f59e0b" />
          <rect x="-7" y="5" width="10" height="4" rx="1.5" fill="#fbbf24" />
          <text x="0" y="32" textAnchor="middle" fill="#64748b" fontSize="8" fontWeight="600" fontFamily="sans-serif">
            OKAPI BM25
          </text>
        </g>

        {/* Node 3: Hybrid RRF + Cross-Encoder */}
        <g
          transform="translate(250, 110)"
          onClick={() => handleNodeClick('rerank')}
          className={interactive ? 'cursor-pointer' : ''}
        >
          <rect
            x="-24"
            y="-24"
            width="48"
            height="48"
            rx="14"
            fill="#ffffff"
            stroke={isRerankActive ? '#c084fc' : '#c7d2fe'}
            strokeWidth={isRerankActive ? '3' : '1.5'}
            filter={isRerankActive ? 'url(#glowActive)' : 'url(#glowSubtle)'}
          />
          <rect x="-16" y="-16" width="32" height="32" rx="9" fill="url(#nodeIndigo)" />
          <path
            d="M 0 -8 L 2.5 -2.5 L 8 0 L 2.5 2.5 L 0 8 L -2.5 2.5 L -8 0 L -2.5 -2.5 Z"
            fill="#ffffff"
          />
          <text x="0" y="36" textAnchor="middle" fill="#4338ca" fontSize="9" fontWeight="700" fontFamily="sans-serif">
            RRF + RERANK
          </text>
        </g>

        {/* Node 4: LangGraph Multi-Agent Workflow */}
        <g
          transform="translate(340, 65)"
          onClick={() => handleNodeClick('langgraph')}
          className={interactive ? 'cursor-pointer' : ''}
        >
          <rect
            x="-22"
            y="-22"
            width="44"
            height="44"
            rx="12"
            fill="#ffffff"
            stroke={isAgentActive ? '#a855f7' : '#e2e8f0'}
            strokeWidth={isAgentActive ? '2.5' : '1.5'}
            filter={isAgentActive ? 'url(#glowActive)' : 'none'}
          />
          <circle cx="0" cy="0" r="14" fill="#f3e8ff" />
          <path
            d="M -6 -4 A 7 7 0 1 1 6 4"
            stroke="#9333ea"
            strokeWidth="1.75"
            strokeLinecap="round"
            fill="none"
          />
          <polygon points="6,1 8,6 3,5" fill="#9333ea" />
          <text x="0" y="32" textAnchor="middle" fill="#64748b" fontSize="8" fontWeight="600" fontFamily="sans-serif">
            LANGGRAPH
          </text>
        </g>

        {/* Node 5: Grounded Answer & Claim Verification */}
        <g
          transform="translate(415, 110)"
          onClick={() => handleNodeClick('verified')}
          className={interactive ? 'cursor-pointer' : ''}
        >
          <rect
            x="-26"
            y="-26"
            width="52"
            height="52"
            rx="14"
            fill="#ffffff"
            stroke={isVerifyActive || isDone ? '#10b981' : '#a7f3d0'}
            strokeWidth={isVerifyActive || isDone ? '3' : '2'}
            filter={isVerifyActive || isDone ? 'url(#glowActive)' : 'url(#glowSubtle)'}
          />
          <rect x="-18" y="-18" width="36" height="36" rx="10" fill="url(#nodeEmerald)" />
          <path
            d="M 0 -8 L 8 -5 C 8 2 4 8 0 10 C -4 8 -8 2 -8 -5 Z"
            fill="#ffffff"
            opacity="0.95"
          />
          <path
            d="M -3 0 L -1 3 L 4 -2"
            stroke="#059669"
            strokeWidth="1.75"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
          />
          <text x="0" y="38" textAnchor="middle" fill="#065f46" fontSize="9" fontWeight="700" fontFamily="sans-serif">
            VERIFIED
          </text>
        </g>

        {/* Small floating pulse dots */}
        <circle cx="108" cy="98" r="2.5" fill="#38bdf8" opacity="0.8" />
        <circle cx="202" cy="148" r="2" fill="#818cf8" opacity="0.7" />
        <circle cx="295" cy="112" r="3" fill="#c084fc" opacity="0.8" />
        <circle cx="378" cy="142" r="2.5" fill="#34d399" opacity="0.8" />
      </svg>

      {/* Interactive Node Technical Inspector Modal */}
      {activeModalNode && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs"
          onClick={() => setActiveModalNode(null)}
        >
          <div
            className="w-full max-w-md bg-white rounded-3xl p-6 border border-slate-200 shadow-2xl space-y-4 animate-in fade-in zoom-in-95 duration-150"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-xl bg-slate-100 text-slate-700">
                  <activeModalNode.icon className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-slate-900 text-sm">{activeModalNode.title}</h3>
                  <div className="text-[11px] font-mono text-slate-400">{activeModalNode.category}</div>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setActiveModalNode(null)}
                aria-label="Close tech inspector"
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div className="p-3 bg-slate-50 rounded-2xl border border-slate-200/80">
                <div className="font-mono text-[10px] uppercase font-bold text-slate-400">Underlying Technology</div>
                <div className="font-mono font-bold text-indigo-700 mt-0.5">{activeModalNode.modelTech}</div>
              </div>

              <p className="text-slate-600 leading-relaxed">{activeModalNode.description}</p>

              <div className="space-y-1.5 pt-1">
                <div className="font-mono text-[10px] uppercase font-bold text-slate-400">Architectural Specifications</div>
                {activeModalNode.specs.map((spec, i) => (
                  <div key={i} className="flex items-start gap-2 text-slate-700">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 flex-shrink-0 mt-0.5" />
                    <span>{spec}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="pt-2 border-t border-slate-100 flex justify-end">
              <button
                type="button"
                onClick={() => setActiveModalNode(null)}
                className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-semibold transition"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * Stylized generic file-type icon with consistent brand treatment
 */
export const FileTypeIcon: React.FC<{
  extension: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}> = ({ extension, size = 'md', className = '' }) => {
  const ext = (extension || '').toLowerCase().replace(/^\./, '');

  let bgClass = 'bg-slate-100 text-slate-700 border-slate-200';
  let badgeClass = 'bg-slate-200 text-slate-800';
  let label = ext.toUpperCase() || 'FILE';
  let IconComponent = FileText;

  switch (ext) {
    case 'pdf':
      bgClass = 'bg-rose-50 text-rose-700 border-rose-200/80';
      badgeClass = 'bg-rose-100 text-rose-800 border-rose-200';
      label = 'PDF';
      IconComponent = FileText;
      break;
    case 'docx':
    case 'doc':
      bgClass = 'bg-sky-50 text-sky-700 border-sky-200/80';
      badgeClass = 'bg-sky-100 text-sky-800 border-sky-200';
      label = 'DOCX';
      IconComponent = FileText;
      break;
    case 'md':
    case 'markdown':
      bgClass = 'bg-purple-50 text-purple-700 border-purple-200/80';
      badgeClass = 'bg-purple-100 text-purple-800 border-purple-200';
      label = 'MD';
      IconComponent = FileCode;
      break;
    case 'html':
    case 'htm':
      bgClass = 'bg-amber-50 text-amber-700 border-amber-200/80';
      badgeClass = 'bg-amber-100 text-amber-800 border-amber-200';
      label = 'HTML';
      IconComponent = FileCode;
      break;
    case 'txt':
      bgClass = 'bg-slate-100 text-slate-700 border-slate-200/80';
      badgeClass = 'bg-slate-200 text-slate-700 border-slate-300';
      label = 'TXT';
      IconComponent = File;
      break;
    default:
      bgClass = 'bg-slate-50 text-slate-700 border-slate-200';
      badgeClass = 'bg-slate-100 text-slate-600 border-slate-200';
      label = ext ? ext.slice(0, 4).toUpperCase() : 'DOC';
      IconComponent = File;
      break;
  }

  const dim = size === 'sm' ? 'w-8 h-8' : size === 'lg' ? 'w-12 h-12' : 'w-10 h-10';
  const iconDim = size === 'sm' ? 'w-4 h-4' : size === 'lg' ? 'w-6 h-6' : 'w-5 h-5';

  return (
    <div
      className={`relative rounded-xl border flex items-center justify-center flex-shrink-0 transition-transform hover:scale-105 ${dim} ${bgClass} ${className}`}
      title={`${label} Document`}
    >
      <IconComponent className={iconDim} />
      <span
        className={`absolute -bottom-1 -right-1 text-[8px] font-mono font-bold px-1 rounded border leading-none py-0.5 shadow-2xs ${badgeClass}`}
      >
        {label}
      </span>
    </div>
  );
};

/**
 * Polished empty-state component with dedicated SVG illustrations
 */
export const EmptyState: React.FC<{
  type: 'no-documents' | 'no-research' | 'no-results' | 'no-citations';
  title?: string;
  description?: string;
  actionLabel?: string;
  onAction?: () => void;
  className?: string;
}> = ({ type, title, description, actionLabel, onAction, className = '' }) => {
  if (type === 'no-documents') {
    return (
      <div className={`p-10 text-center flex flex-col items-center justify-center space-y-3 ${className}`}>
        <div className="w-16 h-16 rounded-2xl bg-amber-50 border border-amber-200/80 flex items-center justify-center text-amber-600 shadow-2xs">
          <UploadCloud className="w-8 h-8" />
        </div>
        <h4 className="font-bold text-slate-800 text-sm">
          {title || 'Your knowledge base is empty.'}
        </h4>
        <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
          {description || 'Upload your first PDF, DOCX, TXT, MD, or HTML document to begin vector indexing and enterprise research.'}
        </p>
        {actionLabel && onAction && (
          <button
            type="button"
            onClick={onAction}
            className="mt-2 px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-xs cursor-pointer"
          >
            {actionLabel}
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    );
  }

  if (type === 'no-research') {
    return (
      <div className={`p-10 text-center flex flex-col items-center justify-center space-y-3 ${className}`}>
        <div className="w-16 h-16 rounded-2xl bg-sky-50 border border-sky-200/80 flex items-center justify-center text-sky-600 shadow-2xs">
          <Sparkles className="w-8 h-8" />
        </div>
        <h4 className="font-bold text-slate-800 text-sm">
          {title || 'No research queries executed yet.'}
        </h4>
        <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
          {description || 'Ask your first technical inquiry or select a benchmark template to generate verified, grounded answers.'}
        </p>
        {actionLabel && onAction && (
          <button
            type="button"
            onClick={onAction}
            className="mt-2 px-4 py-2 bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-600 hover:to-indigo-700 text-white rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-xs cursor-pointer"
          >
            {actionLabel}
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    );
  }

  if (type === 'no-results') {
    return (
      <div className={`p-10 text-center flex flex-col items-center justify-center space-y-3 ${className}`}>
        <div className="w-16 h-16 rounded-2xl bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-500 shadow-2xs">
          <Search className="w-8 h-8" />
        </div>
        <h4 className="font-bold text-slate-800 text-sm">
          {title || 'No matching evidence found.'}
        </h4>
        <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
          {description || 'Try broadening your search terms or verify that the corresponding documents are ingested.'}
        </p>
        {actionLabel && onAction && (
          <button
            type="button"
            onClick={onAction}
            className="mt-2 px-4 py-2 bg-slate-800 hover:bg-slate-900 text-white rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-xs cursor-pointer"
          >
            {actionLabel}
          </button>
        )}
      </div>
    );
  }

  return (
    <div className={`p-6 text-center flex flex-col items-center justify-center space-y-2 text-slate-400 ${className}`}>
      <Layers className="w-6 h-6 text-slate-300" />
      <span className="text-xs italic">
        {description || 'No citations available (insufficient context or abstained query).'}
      </span>
    </div>
  );
};
