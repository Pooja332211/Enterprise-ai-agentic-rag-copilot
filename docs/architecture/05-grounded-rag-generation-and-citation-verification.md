# Architecture Decision Record (ADR) 05: Grounded RAG Generation, Research Agent & Citation Verification

## Status
Accepted

## Context
Milestones 1–4 established an enterprise document processing and retrieval architecture:
1. **Milestone 1–2**: Document ingestion across 5 formats (PDF, DOCX, Markdown, HTML, TXT), structural hierarchy extraction, deduplication, and persistence in PostgreSQL.
2. **Milestone 3**: FastEmbed dense embeddings (`BAAI/bge-small-en-v1.5`), Qdrant vector indexing, BM25 lexical keyword search, and Reciprocal Rank Fusion (RRF, $k=60$).
3. **Milestone 4**: Cross-encoder neural reranking (`BAAI/bge-reranker-base` on ONNX CPU) and context assembly with duplicate elimination and token budgeting.

Milestone 5 completes the end-to-end question answering pipeline by synthesizing grounded research answers from the assembled context and verifying every factual claim with deterministic citation validation:
```text
User Query
    │
    ▼
Hybrid Retrieval (Dense BGE + Lexical BM25 + RRF)
    │
    ▼
Cross-Encoder Reranking (BAAI/bge-reranker-base via ONNX CPU)
    │
    ▼
Context Assembly (Deduplication, Token & Character Budgeting)
    │
    ▼
Grounded LLM Generation (Strict Evidence Isolation & JSON Schema)
    │
    ▼
Citation Verification (Chunk Resolution, Lexical & Bigram Support Scoring)
    │
    ▼
Grounded Research Response (Answer, Claims, Verification Breakdown, Timing)
```

Key engineering challenges addressed in this milestone:
1. **Preventing LLM Hallucinations & Source Invention**: Ensuring the LLM only utilizes factual statements present in the assembled context blocks.
2. **Deterministic Citation Verification**: Verifying whether cited chunk identifiers actually exist in the retrieved evidence pool and whether the cited text supports the claimed statement.
3. **Zero-Dependency Local Execution**: Eliminating external API key requirements or quota limitations for local testing, CI, and evaluation.
4. **End-to-End Latency Profiling**: Capturing millisecond-level telemetry across every stage of the research pipeline.

## Decisions

### 1. Strict Evidence Isolation & Grounded System Prompt
- The LLM generation prompt strictly isolates the model to the retrieved evidence context and the user query.
- The prompt instructs the model to produce structured JSON conforming to a strict schema containing:
  - `summary`: Comprehensive, professional synthesized answer.
  - `claims`: Array of individual factual statements, each bound to specific `citation_chunk_ids`.
  - `insufficient_evidence`: Boolean flag indicating whether the provided context contains adequate information to address the query.
- If the assembled context lacks relevant facts, the generator triggers `insufficient_evidence = True` and emits an explicit fallback message without hallucinating answers.

### 2. LLM Provider Abstraction (`BaseLLMService`)
To provide modularity and decouple generation logic from specific vendors, the system introduces `BaseLLMService`:
- **`MockLLMService` (Default / CI / Local)**:
  - Deterministic generator operating entirely without external network requests or API keys.
  - Extracts key factual sentences from the assembled context blocks and matches them to query terms.
  - Produces compliant JSON outputs with valid chunk citations and real text overlap.
  - Handles empty contexts and supports configurable fault injection (`simulate_error`, `simulate_malformed`) for test coverage.
- **`OpenAILLMService`**: Native HTTP client using `requests` targeting OpenAI `/chat/completions` with JSON mode (`response_format={"type": "json_object"}`).
- **`AnthropicLLMService`**: Native client targeting Anthropic `/v1/messages`.
- **`OpenAICompatibleLLMService`**: Universal client for local open-source LLMs running via vLLM, Ollama, or LiteLLM.
- **Resilient Factory (`get_llm_service`)**: Selects provider from `LLM_PROVIDER` environment variable, falling back gracefully to `MockLLMService` if credentials are unconfigured.

### 3. Context Block Provenance Headers
The context assembly service was enhanced to include unique chunk identifiers in every evidence header:
```text
[Source: {document_id} | Chunk: {chunk_id} | Page: {page_number} | Section: {section_path} | Score: {score:.4f}]
{chunk_text}
```
This enables the LLM to directly reference specific chunk IDs for granular citation attribution.

### 4. Deterministic Citation Verification Service (`CitationVerificationService`)
Claims and citations emitted by the generator undergo rigorous post-generation verification:
1. **Identifier Resolution**:
   - Validates each cited ID against the set of retrieved evidence chunks.
   - Resolves both `chunk_id` and `document_id` references.
   - Any citation referencing an identifier not present in the assembled context is classified as `unsupported` (`chunk_not_found`).
2. **Lexical and N-Gram Support Scoring**:
   - For valid chunks, computes lexical token overlap (Jaccard similarity) and bigram phrase alignment between the claim text and the chunk text.
   - Support Score $S = 0.5 \times \text{token\_overlap} + 0.5 \times \text{bigram\_overlap}$.
   - Calibration Thresholds:
     - $S \ge 0.40 \implies$ `supported`
     - $0.20 \le S < 0.40 \implies$ `partially_supported`
     - $S < 0.20 \implies$ `unsupported` (`insufficient_text_overlap`)
3. **Citation Formatting**:
   - Synthesizes clean human-readable citations: `[Doc: <document_id> | Page: <page_number> | Section: <section_path>]`.
   - Aggregates overall verification summary (`verification_score`, `total_claims`, `supported_claims`, `partially_supported_claims`, `unsupported_claims`).

### 5. Research Orchestrator Service (`ResearchService`)
The orchestrator coordinates the multi-stage pipeline:
1. Validates the query and sets defaults.
2. Invokes `HybridRetrievalService` for candidate retrieval (dense + lexical + RRF).
3. If candidates exist, invokes `RerankerService` for cross-encoder reranking.
4. Invokes `ContextAssemblyService` to pack context within budget.
5. Invokes `GroundedGenerationService` for synthesis.
6. Invokes `CitationVerificationService` to validate all claims and assemble citations.
7. Computes and attaches timing breakdowns across all 8 sub-stages.

### 6. Research API Endpoint (`POST /api/v1/research`)
- Implemented and registered under `/api/v1/research`.
- Accepts `ResearchRequest` (query, filters, top_k, candidate_top_k, rerank, context limits, min_support_threshold).
- Returns `ResearchResponse` containing query, answer, claims with verification status, citations, retrieved chunks, verification summary, and latency breakdown.

## Consequences
- The research pipeline is fully operational end-to-end, grounded in factual documents, and protected against ungrounded generation.
- Full test suite passes: 103 tests (100% pass rate) with zero regressions across Milestones 1–4.
- Deterministic mock services allow full offline testing and CI/CD validation without API costs or rate-limiting risks.
- Granular latency profiling provides immediate visibility into performance bottlenecks.
- Architectural foundation is established for Milestone 6 (Multi-Hop Query Planning, Agentic Reasoning & Evaluation).
