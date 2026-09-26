import {
  HealthStatus,
  DocumentListResponse,
  SearchResponse,
  ResearchResponse,
  AgentResearchResponse,
} from '../types';

export const BACKEND_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = BACKEND_URL) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  public getBaseUrl(): string {
    return this.baseUrl;
  }

  public async checkHealth(): Promise<HealthStatus> {
    const res = await fetch(`${this.baseUrl}/health`);
    if (!res.ok) {
      throw new Error(`Health check failed: HTTP ${res.status}`);
    }
    return res.json();
  }

  public async checkReadiness(): Promise<HealthStatus> {
    const res = await fetch(`${this.baseUrl}/health/ready`);
    return res.json();
  }

  public async getMetrics(): Promise<string> {
    const res = await fetch(`${this.baseUrl}/metrics`);
    if (!res.ok) {
      throw new Error(`Metrics fetch failed: HTTP ${res.status}`);
    }
    return res.text();
  }

  public async getDocuments(limit = 20, offset = 0): Promise<DocumentListResponse> {
    const res = await fetch(`${this.baseUrl}/api/v1/documents?limit=${limit}&offset=${offset}`);
    if (!res.ok) {
      throw new Error(`Failed to list documents: HTTP ${res.status}`);
    }
    return res.json();
  }

  public async uploadDocument(file: File, force = false): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    const url = `${this.baseUrl}/api/v1/documents${force ? '?force=true' : ''}`;
    const res = await fetch(url, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok && res.status !== 409) {
      const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(err.detail || `Upload failed: HTTP ${res.status}`);
    }
    return res.json();
  }

  public async deleteDocument(documentId: string): Promise<void> {
    const res = await fetch(`${this.baseUrl}/api/v1/documents/${documentId}`, {
      method: 'DELETE',
    });
    if (!res.ok) {
      throw new Error(`Delete failed: HTTP ${res.status}`);
    }
  }

  public async search(
    query: string,
    mode: 'hybrid' | 'dense' | 'bm25' = 'hybrid',
    topK = 5,
    candidateTopK = 10,
    rerank = true
  ): Promise<SearchResponse> {
    const res = await fetch(`${this.baseUrl}/api/v1/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        mode,
        top_k: topK,
        candidate_top_k: candidateTopK,
        rerank,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Search failed' }));
      throw new Error(err.detail || `Search failed: HTTP ${res.status}`);
    }
    return res.json();
  }

  public async research(
    query: string,
    topK = 5,
    candidateTopK = 10,
    rerank = true
  ): Promise<ResearchResponse> {
    const res = await fetch(`${this.baseUrl}/api/v1/research`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k: topK,
        candidate_top_k: candidateTopK,
        rerank,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Research failed' }));
      throw new Error(err.detail || `Research failed: HTTP ${res.status}`);
    }
    return res.json();
  }

  public async agentResearch(
    query: string,
    topK = 5,
    candidateTopK = 10,
    rerank = true,
    maxIterations = 2,
    enableDecomposition = true
  ): Promise<AgentResearchResponse> {
    const res = await fetch(`${this.baseUrl}/api/v1/research/agent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k: topK,
        candidate_top_k: candidateTopK,
        rerank,
        max_iterations: maxIterations,
        enable_decomposition: enableDecomposition,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Agent research failed' }));
      throw new Error(err.detail || `Agent research failed: HTTP ${res.status}`);
    }
    return res.json();
  }
}

export const api = new ApiService();
