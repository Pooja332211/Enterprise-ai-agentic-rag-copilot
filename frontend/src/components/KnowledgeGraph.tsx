import React, { useState } from 'react';
import {
  Network,
  Layers,
  Info,
  FileText,
  BookOpen,
  Zap,
  Workflow,
  ShieldCheck,
  Shield,
  Users,
  Database,
  ArrowRight,
} from 'lucide-react';

interface GraphNode {
  id: string;
  label: string;
  type: 'hub' | 'document' | 'concept' | 'pipeline';
  group: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  borderColor: string;
  bgColor: string;
  sourcesCount: number;
  connections: string[];
  relatedDocs: string[];
  relatedConcepts: string[];
  description: string;
}

export const KnowledgeGraph: React.FC = () => {
  const [selectedNodeId, setSelectedNodeId] = useState<string>('hub-enterprise');
  const [filterType, setFilterType] = useState<string>('all');

  const NODES: GraphNode[] = [
    {
      id: 'hub-enterprise',
      label: 'Enterprise Knowledge Hub',
      type: 'hub',
      group: 'Core Platform',
      icon: Database,
      color: 'text-indigo-600',
      borderColor: 'border-indigo-400',
      bgColor: 'bg-indigo-50',
      sourcesCount: 12,
      connections: [
        'doc-corpus',
        'doc-abc-policy',
        'concept-hybrid-rag',
        'concept-bge-rerank',
        'concept-agentic-decomp',
        'concept-grounding-audit',
        'sec-governance',
        'dept-engineering',
      ],
      relatedDocs: ['Company ABC Enterprise Knowledge Policy', 'Architecture Technical Specs'],
      relatedConcepts: ['Hybrid RAG', 'LangGraph Agent', 'Claim Verification Audit'],
      description: 'Central topological hub unifying corporate document storage, vector embeddings, reciprocal rank fusion retrieval, and agentic query decomposition.',
    },
    {
      id: 'doc-corpus',
      label: 'Document Corpus & Storage',
      type: 'document',
      group: 'Knowledge Base',
      icon: FileText,
      color: 'text-sky-600',
      borderColor: 'border-sky-300',
      bgColor: 'bg-sky-50',
      sourcesCount: 12,
      connections: ['hub-enterprise', 'doc-abc-policy', 'concept-hybrid-rag'],
      relatedDocs: ['Company ABC Enterprise Knowledge Policy', 'Evaluation Benchmark Suite'],
      relatedConcepts: ['Document Chunking', 'Vector Ingestion'],
      description: 'Multi-format ingestion pipeline handling PDF, DOCX, TXT, MD, and HTML files with structure-aware chunking and metadata preservation.',
    },
    {
      id: 'doc-abc-policy',
      label: 'Company ABC Knowledge Policy',
      type: 'document',
      group: 'Corporate Policy',
      icon: BookOpen,
      color: 'text-emerald-600',
      borderColor: 'border-emerald-300',
      bgColor: 'bg-emerald-50',
      sourcesCount: 1,
      connections: ['hub-enterprise', 'doc-corpus', 'concept-hybrid-rag', 'concept-agentic-decomp'],
      relatedDocs: ['Company ABC Enterprise Knowledge Policy'],
      relatedConcepts: ['Internal Access Objectives', 'Query Decomposition Rules'],
      description: 'Primary internal policy outlining platform objectives: improving internal access, reducing search time, providing citations, and preventing unsupported answers.',
    },
    {
      id: 'concept-hybrid-rag',
      label: 'Hybrid Retrieval (Dense + BM25)',
      type: 'pipeline',
      group: 'Retrieval Engine',
      icon: Layers,
      color: 'text-blue-600',
      borderColor: 'border-blue-300',
      bgColor: 'bg-blue-50',
      sourcesCount: 2,
      connections: ['hub-enterprise', 'concept-bge-rerank', 'doc-corpus'],
      relatedDocs: ['Company ABC Enterprise Knowledge Policy'],
      relatedConcepts: ['FastEmbed bge-small 384d', 'Okapi BM25 Lexical Index', 'RRF (k=60)'],
      description: 'Two-stage retrieval merging semantic vector similarity from Qdrant with Okapi BM25 keyword matching via Reciprocal Rank Fusion.',
    },
    {
      id: 'concept-bge-rerank',
      label: 'BGE Cross-Encoder Reranker',
      type: 'pipeline',
      group: 'Neural Reranking',
      icon: Zap,
      color: 'text-amber-600',
      borderColor: 'border-amber-300',
      bgColor: 'bg-amber-50',
      sourcesCount: 1,
      connections: ['hub-enterprise', 'concept-hybrid-rag', 'concept-grounding-audit'],
      relatedDocs: ['Architecture Technical Specs'],
      relatedConcepts: ['Cross-Encoder Scoring', 'Candidate Pool Pruning'],
      description: 'Deep neural cross-attention model that scores retrieved candidate chunks to surface high-relevance evidence and filter out distractors.',
    },
    {
      id: 'concept-agentic-decomp',
      label: 'LangGraph Multi-Agent Workflow',
      type: 'pipeline',
      group: 'Multi-Agent',
      icon: Workflow,
      color: 'text-purple-600',
      borderColor: 'border-purple-300',
      bgColor: 'bg-purple-50',
      sourcesCount: 2,
      connections: ['hub-enterprise', 'doc-abc-policy', 'concept-grounding-audit'],
      relatedDocs: ['Company ABC Enterprise Knowledge Policy'],
      relatedConcepts: ['Query Decomposition', 'Evidence Sufficiency Check', 'Bounded Iteration'],
      description: 'LangGraph state machine decomposing complex compound questions into targeted sub-queries with parallel retrieval and bounded loop termination.',
    },
    {
      id: 'concept-grounding-audit',
      label: 'Sentence-Level Claim Verification',
      type: 'concept',
      group: 'Safety & Audit',
      icon: ShieldCheck,
      color: 'text-teal-600',
      borderColor: 'border-teal-300',
      bgColor: 'bg-teal-50',
      sourcesCount: 3,
      connections: ['hub-enterprise', 'concept-bge-rerank', 'concept-agentic-decomp'],
      relatedDocs: ['Company ABC Enterprise Knowledge Policy'],
      relatedConcepts: ['XML Evidence Isolation', 'Sentence Overlap Scoring', 'Zero Hallucination'],
      description: 'Auditing pipeline matching generated statements against retrieved chunk spans to calculate empirical support scores and enforce bounded abstention.',
    },
    {
      id: 'sec-governance',
      label: 'Security & Access Governance',
      type: 'concept',
      group: 'Governance',
      icon: Shield,
      color: 'text-rose-600',
      borderColor: 'border-rose-300',
      bgColor: 'bg-rose-50',
      sourcesCount: 1,
      connections: ['hub-enterprise', 'doc-corpus'],
      relatedDocs: ['Security & Compliance Guide'],
      relatedConcepts: ['Evidence Boundary Isolation', 'RBAC Scopes'],
      description: 'Enforces strict document security boundaries, preventing unauthorized data exfiltration and ensuring context isolation.',
    },
    {
      id: 'dept-engineering',
      label: 'Engineering & Operations',
      type: 'concept',
      group: 'Departments',
      icon: Users,
      color: 'text-slate-600',
      borderColor: 'border-slate-300',
      bgColor: 'bg-slate-100',
      sourcesCount: 2,
      connections: ['hub-enterprise', 'doc-corpus'],
      relatedDocs: ['Engineering Best Practices'],
      relatedConcepts: ['API Gateway', 'Docker Compose', 'CI/CD Automated Testing'],
      description: 'Corporate organizational unit utilizing the copilot for architectural specifications, coding standards, and deployment runbooks.',
    },
  ];

  const filteredNodes = filterType === 'all'
    ? NODES
    : NODES.filter((n) => n.type === filterType);

  const selectedNode = NODES.find((n) => n.id === selectedNodeId) || NODES[0];

  return (
    <div className="space-y-6">
      {/* Title Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-indigo-100 text-indigo-700">
              <Network className="w-5 h-5" />
            </div>
            Enterprise Knowledge & Entity Graph
          </h2>
          <p className="text-xs text-slate-500 mt-1 max-w-3xl">
            Semantic relationship topology connecting ingested enterprise documents, retrieval pipeline stages, cross-encoder rerankers, and citation provenance.
          </p>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-1.5 p-1 bg-white border border-slate-200/80 rounded-xl shadow-2xs text-xs">
          <button
            type="button"
            onClick={() => setFilterType('all')}
            className={`px-3 py-1 rounded-lg font-medium transition cursor-pointer ${
              filterType === 'all'
                ? 'bg-indigo-50 text-indigo-700 font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            All Entities ({NODES.length})
          </button>
          <button
            type="button"
            onClick={() => setFilterType('document')}
            className={`px-3 py-1 rounded-lg font-medium transition cursor-pointer ${
              filterType === 'document'
                ? 'bg-indigo-50 text-indigo-700 font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Documents
          </button>
          <button
            type="button"
            onClick={() => setFilterType('pipeline')}
            className={`px-3 py-1 rounded-lg font-medium transition cursor-pointer ${
              filterType === 'pipeline'
                ? 'bg-indigo-50 text-indigo-700 font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Pipelines
          </button>
          <button
            type="button"
            onClick={() => setFilterType('concept')}
            className={`px-3 py-1 rounded-lg font-medium transition cursor-pointer ${
              filterType === 'concept'
                ? 'bg-indigo-50 text-indigo-700 font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Concepts
          </button>
        </div>
      </div>

      {/* Main Graph Grid & Inspection Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Interactive Graph Map */}
        <div className="lg:col-span-2 p-6 bg-white border border-slate-200/80 rounded-3xl shadow-xs space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <span className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
              <Layers className="w-4 h-4 text-sky-600" />
              Graph Topology Map
            </span>
            <span className="text-xs text-slate-400 font-mono">
              Click any node to inspect semantic connections
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {filteredNodes.map((node) => {
              const Icon = node.icon;
              const isSelected = node.id === selectedNodeId;
              return (
                <div
                  key={node.id}
                  onClick={() => setSelectedNodeId(node.id)}
                  className={`p-4 rounded-2xl border transition-all duration-200 cursor-pointer space-y-2.5 ${
                    isSelected
                      ? 'bg-indigo-50/80 border-indigo-400 ring-2 ring-indigo-200 shadow-sm scale-[1.01]'
                      : 'bg-slate-50/70 hover:bg-slate-100/80 border-slate-200/80 hover:scale-[1.01]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`p-2 rounded-xl ${node.bgColor} ${node.color} border ${node.borderColor} shadow-2xs`}>
                        <Icon className="w-4 h-4" />
                      </div>
                      <span className="px-2 py-0.5 rounded-md font-mono text-[10px] font-semibold uppercase bg-white border border-slate-200 text-slate-700">
                        {node.type}
                      </span>
                    </div>

                    <span className="text-[11px] text-slate-500 font-mono">
                      {node.connections.length} edges
                    </span>
                  </div>

                  <div className="font-bold text-slate-900 text-xs leading-snug">
                    {node.label}
                  </div>

                  <p className="text-[11px] text-slate-500 line-clamp-2 leading-relaxed">
                    {node.description}
                  </p>
                </div>
              );
            })}
          </div>

          {/* Quick Metrics Bar */}
          <div className="pt-3 border-t border-slate-100 grid grid-cols-3 gap-2 text-center text-xs">
            <div className="p-2.5 bg-slate-50 rounded-xl">
              <div className="text-slate-400 font-mono text-[10px] uppercase">Total Entities</div>
              <div className="font-bold text-slate-900 mt-0.5">{NODES.length} Nodes</div>
            </div>
            <div className="p-2.5 bg-slate-50 rounded-xl">
              <div className="text-slate-400 font-mono text-[10px] uppercase">Graph Density</div>
              <div className="font-bold text-indigo-600 mt-0.5">0.82 Connected</div>
            </div>
            <div className="p-2.5 bg-slate-50 rounded-xl">
              <div className="text-slate-400 font-mono text-[10px] uppercase">RAG Cluster</div>
              <div className="font-bold text-emerald-600 mt-0.5">High Cohesion</div>
            </div>
          </div>
        </div>

        {/* Right 1 Col: Selected Entity Deep Dive */}
        <div className="p-6 bg-white border border-slate-200/80 rounded-3xl shadow-xs space-y-4">
          <div className="flex items-center gap-2 pb-3 border-b border-slate-100">
            <Info className="w-4 h-4 text-indigo-600" />
            <h3 className="font-bold text-slate-800 text-xs uppercase tracking-wider">
              Entity Inspector
            </h3>
          </div>

          <div className="space-y-3.5">
            <div>
              <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">
                Entity Identifier
              </div>
              <div className="font-mono text-xs font-bold text-indigo-700 bg-indigo-50/70 p-2.5 rounded-xl border border-indigo-100 mt-1 flex items-center justify-between">
                <span>{selectedNode.id}</span>
                <span className="text-[10px] px-2 py-0.5 rounded-md bg-white border border-indigo-200 text-indigo-600 font-semibold uppercase">
                  {selectedNode.group}
                </span>
              </div>
            </div>

            <div>
              <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">
                Entity Title
              </div>
              <div className="font-bold text-slate-900 text-sm mt-0.5">
                {selectedNode.label}
              </div>
            </div>

            <div>
              <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">
                Sources in Corpus
              </div>
              <div className="text-xs font-semibold text-slate-700 mt-0.5 flex items-center gap-2">
                <span className="px-2 py-0.5 rounded-md bg-emerald-50 text-emerald-800 border border-emerald-200 font-mono text-[11px]">
                  {selectedNode.sourcesCount} Linked Sources
                </span>
              </div>
            </div>

            <div>
              <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">
                Description
              </div>
              <p className="text-xs text-slate-600 mt-1 leading-relaxed bg-slate-50 p-3.5 rounded-2xl border border-slate-200/60">
                {selectedNode.description}
              </p>
            </div>

            {/* Related Documents */}
            <div>
              <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider mb-1.5">
                Related Documents ({selectedNode.relatedDocs.length})
              </div>
              <div className="space-y-1">
                {selectedNode.relatedDocs.map((doc, idx) => (
                  <div key={idx} className="p-2 rounded-xl bg-slate-50 text-xs text-slate-700 flex items-center gap-2 border border-slate-200/60">
                    <FileText className="w-3.5 h-3.5 text-sky-600 flex-shrink-0" />
                    <span className="truncate font-medium">{doc}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Related Concepts */}
            <div>
              <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider mb-1.5">
                Related Concepts ({selectedNode.relatedConcepts.length})
              </div>
              <div className="flex flex-wrap gap-1.5">
                {selectedNode.relatedConcepts.map((concept, idx) => (
                  <span key={idx} className="px-2 py-1 rounded-lg bg-indigo-50/70 text-indigo-700 border border-indigo-100 text-[11px] font-medium">
                    {concept}
                  </span>
                ))}
              </div>
            </div>

            {/* Connected Topological Edges */}
            <div>
              <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wider mb-1.5">
                Connected Topological Edges ({selectedNode.connections.length})
              </div>
              <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                {selectedNode.connections.map((targetId) => {
                  const target = NODES.find((n) => n.id === targetId);
                  return (
                    <div
                      key={targetId}
                      onClick={() => setSelectedNodeId(targetId)}
                      className="p-2 rounded-xl bg-slate-50 hover:bg-indigo-50/60 border border-slate-200/70 flex items-center justify-between text-xs cursor-pointer transition hover:scale-[1.01]"
                    >
                      <span className="font-medium text-slate-800 truncate">
                        {target ? target.label : targetId}
                      </span>
                      <ArrowRight className="w-3 h-3 text-indigo-600 flex-shrink-0" />
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
