# Enterprise AI Knowledge & Research Copilot

A production-grade, zero-hallucination Agentic RAG platform engineered with **FastAPI**, **LangGraph**, **Qdrant**, and **PostgreSQL**. Features two-stage hybrid retrieval with Reciprocal Rank Fusion (RRF), CPU-optimized cross-encoder neural reranking, multi-hop query planning state machines, sentence-level claim verification, multi-layer LRU caching, native Prometheus observability, and defense-in-depth security.

---

## Portfolio & Engineering Highlights

```mermaid
flowchart TD
    subgraph Ingestion ["1. Multi-Format Ingestion & Dual Indexing"]
        Doc[Enterprise Document: PDF, DOCX, TXT, MD, HTML] --> Validate[MIME & SHA-256 Check]
        Validate --> Parse[Hierarchical Heading & Table Extraction]
        Parse --> Chunk[Structure-Aware Chunking: 300-500 Tokens]
        Chunk --> DenseIndex["Dense Vectors: FastEmbed BGE-small -> Qdrant"]
        Chunk --> LexicalIndex["Lexical Inverted Index: Okapi BM25"]
    end

    subgraph Retrieval ["2. Two-Stage Hybrid Retrieval & Reranking"]
        UserQ[User Query] --> Analysis[Complexity Classification & Sub-Query Planning]
        Analysis --> ParallelSearch[Parallel Dense + Sparse Retrieval]
        DenseIndex -.-> ParallelSearch
        LexicalIndex -.-> ParallelSearch
        ParallelSearch --> RRF["Reciprocal Rank Fusion (k=60)"]
        RRF --> Reranker["BGE Cross-Encoder Reranking (Dynamic Padding)"]
    end

    subgraph AgenticStateMachine ["3. LangGraph Agent State Machine"]
        Reranker --> EvidenceAgg[Evidence Aggregation & Provenance Trace]
        EvidenceAgg --> Sufficiency{Sufficiency Check?}
        Sufficiency -- "Insufficient (Iter < 2)" --> Refinement[Targeted Sub-Query Refinement]
        Refinement --> ParallelSearch
        Sufficiency -- "Insufficient (Iter >= 2)" --> Abstain[Deterministic Abstention: Zero Hallucination]
        Sufficiency -- "Sufficient" --> ContextPack[Token-Budgeted Context Assembly]
    end

    subgraph SynthesisAndVerification ["4. Grounded Synthesis & Verification"]
        ContextPack --> LLMGen["Grounded Generation: XML <context> Isolation"]
        LLMGen --> ClaimExtract[Sentence-Level Claim Extraction]
        ClaimExtract --> CitationAudit[Citation Verification & Provenance Mapping]
        CitationAudit --> FinalResponse[Grounded Research Response + Audit Trail]
        Abstain --> FinalResponse
        FinalResponse --> Metrics["Native Prometheus Exposition (/metrics) + X-Request-ID"]
    end
```

---

## 1. Problem Statement & Solution

### The Enterprise Problem
Corporate knowledge workers lose hours searching across fragmented, unstructured documents (PDFs, Word specs, architecture guidelines, benchmark reports). Generic LLM chat interfaces fail in production because:
1. **Hallucination & Lack of Verifiable Citations**: Generating plausible-sounding facts without traceable provenance to authoritative document pages or sections.
2. **Dense Retrieval Blind Spots**: Pure vector search misses exact technical keywords, IDs, and quantitative metrics.
3. **Compound Multi-Hop Questions**: Single-turn RAG fails when answers require comparing disparate sections across multiple documents.
4. **Latency Bottlenecks & Resource Inefficiency**: High-dimensional cross-encoders and un-cached LLM queries incur prohibitive latencies ($>2\text{--}5$s).
5. **Security Vulnerabilities**: Indirect prompt injection, path traversal in file uploads, and unredacted secrets in operational logs.

### The Copilot Solution
The **Enterprise AI Knowledge & Research Copilot** solves these challenges through:
- **Two-Stage Hybrid Search**: Dense embeddings (`BAAI/bge-small-en-v1.5`) and lexical BM25 fused via Reciprocal Rank Fusion ($k=60$), achieving **1.0000 Recall@3+** and **0.9160 NDCG@5**.
- **Dynamic Sequence Padding**: Optimized CPU cross-encoder inference latency from **2,302.50 ms** to **81.20 ms** (**28.37x speedup**).
- **LangGraph State Machines**: Multi-agent graph orchestrating query decomposition, parallel retrieval, and bounded iterative refinement ($max=2$).
- **Zero-Hallucination Grounding**: Sentence-level claim extraction and verification achieving **100% citation validity** and **100% negative abstention accuracy**.
- **Multi-Layer LRU Caching**: Transparent vector and pair-score caching achieving a **77.89% reranker hit rate**, lowering warm query latency to **7.31 ms** (Single) and **25.33 ms** (Agent).
- **Production-Grade Reliability**: Docker Compose, PostgreSQL pooling (`AsyncAdaptedQueuePool`), non-root container isolation, Prometheus `/metrics`, and 100% passing test coverage (173 tests).

---

## 2. Technology Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **API Framework** | FastAPI (Python 3.11) | High-performance async ASGI framework with automatic OpenAPI documentation |
| **Agent Orchestration** | LangGraph / LangChain Core | Cyclic state machines with explicit state contracts, conditional routing, and bounded loops |
| **Vector Database** | Qdrant | Vector engine with HNSW indexing, filtering, and payload indexes |
| **Relational Database** | PostgreSQL 16 / SQLAlchemy 2.0 | Async connection pooling (`AsyncAdaptedQueuePool`), ACID transactions, Alembic migrations |
| **Dense Embeddings** | FastEmbed (`bge-small-en-v1.5`) | Local 384-dimensional ONNX embeddings running on CPU without external API dependencies |
| **Lexical Search** | Okapi BM25 (`rank-bm25`) | Inverted index tokenization preserving technical terminology and numbers |
| **Neural Reranker** | BGE Reranker (`bge-reranker-base`) | Cross-encoder cross-attention scoring with optimized dynamic batch sequence padding |
| **LLM Engine** | Multi-Provider (Mock, OpenAI, Anthropic) | Pluggable provider abstraction with structured JSON output and prompt injection defense |
| **Caching Layer** | Custom Thread-Safe LRU | Deterministic SHA-256 key hashing with automated document invalidation hooks |
| **Observability** | Native Prometheus + Logging | Dependency-free `/metrics` exposition, `X-Request-ID` correlation, secret redaction |
| **Containerization** | Docker Compose | Multi-container stack with health checks, persistent volumes, and non-root users |
| **Testing & Quality** | Pytest (173 tests) | Comprehensive unit, integration, security, and automated evaluation suites |

---

## 3. Verified Benchmark Results & Key Metrics

All metrics reported below were **empirically measured** during controlled benchmarks with zero fabricated numbers.

### 3.1 Retrieval Ablation Study ($K=5$)

| Retrieval Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | NDCG@5 | Latency (Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Only** (`bge-small-en-v1.5`) | 0.6786 | 0.8929 | 0.9286 | **1.0000** | 0.8571 | 0.8501 | 14.39 ms |
| **BM25 Only** (Okapi BM25) | **0.7857** | **1.0000** | **1.0000** | **1.0000** | **0.9286** | **0.9460** | **0.60 ms** |
| **Hybrid RRF** ($k=60$) | 0.6786 | **1.0000** | **1.0000** | **1.0000** | 0.8929 | 0.9094 | 14.96 ms |
| **Hybrid + BGE Reranker** (Optimized) | **0.7143** | **1.0000** | **1.0000** | **1.0000** | 0.8810 | **0.9160** | **24.30 ms** (warm) |

*Key Takeaway*: Neural cross-encoder reranking over Hybrid RRF provides a **+5.3% relative lift in Recall@1** and improves NDCG@5 to 0.9160.

### 3.2 Grounded Generation & Citation Verification

| Metric | Target Policy | Measured Value | Verification Result |
| :--- | :---: | :---: | :---: |
| **Answer Grounding Rate** | $\ge 90.00\%$ | **1.0000 (100.0%)** | 28 of 28 claims strictly supported |
| **Citation Validity Rate** | $\ge 95.00\%$ | **100.00%** | 28 of 28 chunk IDs verified in evidence |
| **Citation Coverage Rate** | $\ge 95.00\%$ | **100.00%** | 100% of factual assertions cite sources |
| **Unsupported Claim Rate** | $0.00\%$ | **0.00%** | Zero ungrounded or hallucinated assertions |
| **Negative Query Abstention**| $\ge 90.00\%$ | **100.00%** | 4 of 4 unanswerable queries cleanly refused |

### 3.3 LangGraph Agentic Research Benchmark

| Agent Metric | Measured Score | Evaluation Notes |
| :--- | :---: | :--- |
| **Decomposition Accuracy** | **100.00%** (10 / 10) | Correctly split multi-part questions into sub-queries |
| **Evidence Sufficiency Accuracy** | **100.00%** (10 / 10) | Accurate assessment of evidence completeness |
| **Abstention Accuracy** | **100.00%** (3 / 3) | Clean refusal when evidence is absent |
| **Answer Grounding Rate** | **100.00%** (7 / 7) | All generated claims verified against evidence |
| **Citation Validity Rate** | **100.00%** (17 / 17) | 100% of citation chunk IDs verified in evidence |
| **Average Sub-Queries / Query** | **1.80** | 1.0 for simple queries, 2.0–3.0 for complex queries |
| **Loop Bounding Enforcement** | **100.00%** | Maximum 2 iterations strictly enforced |

### 3.4 Latency Profile: Cold vs. Warm Performance

Multi-layer LRU caching (71.88% embedding, 77.89% reranker hit rates) delivers order-of-magnitude warm speedups:

| Execution Pipeline | Cold Mean | Warm Mean | Speedup Factor | Cold p95 | Warm p95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single Research (`/api/v1/research`)** | 982.94 ms | **7.31 ms** | **134.46x** | 1,411.03 ms | **8.55 ms** |
| **Agentic Workflow (`/api/v1/research/agent`)**| 783.00 ms | **25.33 ms** | **30.91x** | 3,084.01 ms | **34.15 ms** |

---

## 4. Security Architecture & Threat Defense

Defense-in-depth implemented across all operational layers:
1. **Prompt-Injection XML Containment**: Retrieved context chunks are isolated inside `<context>...</context>` XML tags with explicit system prompt instructions informing the LLM that context is untrusted third-party data that must never override system instructions.
2. **Storage Path Traversal Protection**: `LocalStorageService` verifies canonical path resolution, strictly rejecting directory traversal attempts (`../`).
3. **Sensitive Data Redaction**: Automatic masking of passwords, bearer tokens, API keys, and authorization headers in logs and traces.
4. **Production Configuration Safeguards**: Pydantic `@model_validator` strictly halts startup in `ENVIRONMENT=production` if default `SECRET_KEY`, `DEBUG=true`, or in-memory/SQLite fallbacks are detected.
5. **Input Boundary Enforcement**: Request queries bounded to $\le 2,000$ characters and retrieval limits bounded to $\le 50$.
6. **Ingestion Reliability**: Non-silent failure tracking (`indexing_failed` / `failed`) with atomic rollback on vector or lexical errors.
7. **Readiness Health Probing**: `/health/ready` returns `HTTP 503 Service Unavailable` if PostgreSQL or Qdrant fails.

---

## 5. Repository Documentation Index

| Document | Description |
| :--- | :--- |
| [`docs/architecture/00-end-to-end-architecture.md`](docs/architecture/00-end-to-end-architecture.md) | **2-Minute Executive Overview** of the 10-stage architecture for interviews |
| [`docs/evaluation/final-evaluation.md`](docs/evaluation/final-evaluation.md) | **Comprehensive Evaluation Report** covering retrieval ablation, grounding, and latency |
| [`docs/portfolio-demo.md`](docs/portfolio-demo.md) | **3–5 Minute Live Demo Script** with reproducible curl commands |
| [`docs/portfolio-metrics.md`](docs/portfolio-metrics.md) | **Resume-Ready Bullet Points & KPIs** backed by empirical data |
| [`docs/security/01-security-and-reliability.md`](docs/security/01-security-and-reliability.md) | **Security Architecture Guide** covering prompt isolation, redaction, and validation |
| [`docs/deployment/10-production-deployment.md`](docs/deployment/10-production-deployment.md) | **Production Operations Manual** with Docker, scaling, backups, and health checks |
| [`docs/experiments/07-qlora-domain-adaptation.md`](docs/experiments/07-qlora-domain-adaptation.md) | **QLoRA Domain Adaptation Report** documenting methodology, leakage guards, and GPU path |
| [`docs/experiments/08-agentic-rag-evaluation.md`](docs/experiments/08-agentic-rag-evaluation.md) | **LangGraph Agentic Evaluation Report** on 10 multi-part queries |
| [`docs/experiments/09-production-performance-report.md`](docs/experiments/09-production-performance-report.md) | **Milestone 9 Performance Report** with dynamic padding and caching results |

---

## 6. Quickstart & Verification

### 6.1 Prerequisites
- Python 3.11+
- Git
- Optional: Docker & Docker Compose

### 6.2 Local Development Setup
```bash
# 1. Clone repository
git clone https://github.com/your-username/enterprise-ai-copilot.git
cd enterprise-ai-copilot

# 2. Create virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install backend dependencies
pip install -r backend/requirements.txt

# 4. Copy configuration template
cp .env.example .env

# 5. Run full test suite (173 tests)
python -m pytest backend/tests -q
```

### 6.3 Run the Production Smoke Test
Verifies all 9 core capabilities end-to-end:
```bash
python scripts/production_smoke_test.py
```

### 6.4 Reproduce Evaluation Benchmarks
```bash
# 1. Run 4-way retrieval ablation benchmark
python scripts/evaluate_retrieval.py

# 2. Run LangGraph agent state machine benchmark
python scripts/evaluate_agent.py

# 3. Run production latency benchmark & quality gates
python scripts/benchmark_performance.py
```

### 6.5 Docker Production Stack
```bash
# Validate Docker Compose configuration
docker compose config
docker compose --profile monitoring config

# Launch services with persistent storage & automated migrations
docker compose up -d --build
```

---

## 7. Known Limitations

1. **Host Docker Daemon**: While `docker compose config` syntax is fully validated with zero errors, local container execution requires a running Docker daemon.
2. **QLoRA Fine-Tuning Hardware Dependency**: 4-bit NF4 quantization requires an NVIDIA GPU with CUDA tensor cores ($\ge 6$ GB VRAM). On CPU or integrated Intel graphics, the experiment framework validates data leakage and executes diagnostics, but halts training safely without fabricating results.

---

## 8. Milestone Completion Log

- [x] **Milestone 1**: System Scaffolding, Configuration, DB Models & Vector Abstraction
- [x] **Milestone 2**: Multi-Format Document Ingestion, Hierarchy-Aware Parsing & Chunking
- [x] **Milestone 3**: Dense Vector (FastEmbed) + Lexical (BM25) Hybrid Retrieval via RRF
- [x] **Milestone 4**: Neural Cross-Encoder Reranking & Context Assembly Service
- [x] **Milestone 5**: Grounded RAG Generation, Multi-Provider LLMs & Citation Verification
- [x] **Milestone 6**: Systematic Evaluation Harness, Relevancy Masking & Retrieval Ablation
- [x] **Milestone 7**: Controlled QLoRA Domain Adaptation Subsystem & Leakage Auditing
- [x] **Milestone 8**: Query Planning & LangGraph Agent State Machines
- [x] **Milestone 9**: Production Observability (`/metrics`), Bounded Caching & Reranker Optimization
- [x] **Milestone 10**: Productionization, Deployment, Connection Pooling & Reliability
- [x] **Milestone 11**: Final Portfolio Hardening, Architecture Documentation & Release Preparation
