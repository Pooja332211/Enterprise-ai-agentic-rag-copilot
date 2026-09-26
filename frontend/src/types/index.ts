export interface HealthStatus {
  status: string;
  version?: string;
  timestamp?: string;
  environment?: string;
  services?: Record<string, { status: string; latency_ms?: number; details?: Record<string, unknown> }>;
}

export interface DocumentItem {
  id: string;
  filename: string;
  normalized_filename?: string;
  file_extension: string;
  file_size_bytes: number;
  status: string;
  chunk_count: number;
  page_count: number;
  created_at: string;
  updated_at?: string;
}

export interface DocumentListResponse {
  items: DocumentItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface SearchResultChunk {
  chunk_id: string;
  document_id: string;
  document_name: string;
  text: string;
  score: number;
  rank: number;
  page_number?: number;
  section_title?: string;
}

export interface SearchResponse {
  query: string;
  total_results: number;
  retrieval_mode: string;
  results: SearchResultChunk[];
  latency_breakdown?: Record<string, number>;
}

export interface CitationItem {
  citation_id: string;
  chunk_id: string;
  document_id: string;
  document_name: string;
  section_title?: string;
  page_number?: number;
  source_snippet?: string;
}

export interface ClaimItem {
  claim_id: string;
  text: string;
  evidence_chunk_ids: string[];
}

export interface ClaimVerificationItem {
  claim_id: string;
  text: string;
  status: 'supported' | 'partially_supported' | 'unsupported';
  support_score: number;
  cited_chunk_ids: string[];
  verified_chunk_ids: string[];
  missing_chunk_ids?: string[];
  rationale?: string;
  // Fallbacks for compatibility
  claim?: string;
  score?: number;
}

export interface ClaimVerification {
  claim: string;
  status: 'supported' | 'partially_supported' | 'unsupported';
  score: number;
  citation_ids: string[];
}

export interface VerificationSummary {
  status: 'verified' | 'partially_verified' | 'unverified';
  supported_claims: number;
  partially_supported_claims: number;
  unsupported_claims: number;
  total_claims: number;
  coverage_rate?: number;
  details: ClaimVerificationItem[];
}

export interface ResearchResponse {
  query: string;
  answer: string;
  citations: CitationItem[];
  claims?: ClaimItem[] | ClaimVerification[];
  verification?: VerificationSummary;
  evidence?: SearchResultChunk[];
  grounded: boolean;
  insufficient_evidence: boolean;
  model?: string;
  provider?: string;
  timings_ms?: Record<string, number>;
  total_latency_ms?: number;
  latency_breakdown?: Record<string, number>;
}

export interface SubQueryItem {
  id?: string;
  sub_query_id?: string;
  query?: string;
  sub_query_text?: string;
  reasoning?: string;
  purpose?: string;
  status?: string;
}

export interface ExecutionTraceItem {
  node?: string;
  step?: string;
  stage?: string;
  duration_ms?: number;
  latency_ms?: number;
  iteration?: number;
  input_summary?: string;
  output_summary?: string;
  timestamp: string;
  details?: Record<string, unknown>;
}

export interface AgentResearchResponse {
  query?: string;
  original_query?: string;
  sub_queries: SubQueryItem[];
  final_answer: string;
  citations: CitationItem[];
  iterations_used: number;
  status: string;
  execution_trace: ExecutionTraceItem[];
  timings_ms?: Record<string, number>;
  total_duration_ms?: number;
  insufficient_evidence?: boolean;
  provider?: string;
  model?: string;
  grounded?: boolean;
}

export interface RecentResearchItem {
  id: string;
  query: string;
  mode: 'research' | 'agent' | 'search';
  timestamp: string;
  status: 'grounded' | 'abstained' | 'completed' | 'retrieved';
  latency_ms?: number;
  citation_count?: number;
}

export type TabType = 'dashboard' | 'research' | 'agent' | 'search' | 'documents' | 'graph' | 'analytics' | 'health';
