# Architecture Decision Record (ADR) 08: Agentic Research Workflow via LangGraph State Machines

## Status
Accepted

## Context
Milestones 1–5 built a robust, verified single-turn grounded RAG pipeline:
```text
User Query → Hybrid Retrieval (Dense BGE + Lexical BM25) → Cross-Encoder Reranking → Context Assembly → Grounded LLM Generation → Citation Verification
```
While this linear pipeline achieves high precision for direct, single-step factual questions, enterprise research queries frequently exhibit:
1. **Multi-Document Synthesis**: Questions requiring facts scattered across disparate document formats (e.g., comparing specifications in Markdown with quantitative benchmark tables in HTML and methodology in DOCX).
2. **Explicit Comparisons**: Questions contrasting two architectures, policies, or operational standards that need distinct candidate pools rather than a single blended query.
3. **Compound Requirements**: Queries containing multiple sub-questions that dilute retrieval quality if issued as a single sentence.
4. **Iterative Refinement Needs**: Queries where initial retrieval yields insufficient or ambiguous evidence, requiring targeted second-pass retrieval.
5. **Strict Refusal Requirements**: Out-of-domain or ungrounded questions requiring verified evidence sufficiency assessment and clean abstention without hallucination.

Milestone 8 upgrades this system into a **controlled agentic research workflow** using **LangGraph state machines** while preserving complete backward compatibility with the existing single-turn RAG endpoint (`POST /api/v1/research`).

---

## Architecture

```text
                              ┌────────────────────────┐
                              │       User Query       │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │     analyze_query      │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │       plan_query       │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐◄───────────────┐
                              │        retrieve        │                │
                              └───────────┬────────────┘                │
                                          │                             │
                                          ▼                             │
                              ┌────────────────────────┐                │
                              │         rerank         │                │
                              └───────────┬────────────┘                │
                                          │                             │
                                          ▼                             │
                              ┌────────────────────────┐                │
                              │   assemble_evidence    │                │
                              └───────────┬────────────┘                │
                                          │                             │
                                          ▼                             │
                              ┌────────────────────────┐                │
                              │   assess_sufficiency   │                │
                              └───────────┬────────────┘                │
                                          │                             │
                    ┌─────────────────────┴─────────────────────┐       │
                    │ route_after_assessment:                   │       │
                    │ - Sufficient evidence                     │       │
                    │ - Insufficient & iteration < max_iter     │       │
                    │ - Insufficient & iteration >= max_iter    │       │
                    └──────┬──────────────────────┬─────────────┴───────┘
                           │                      │                 │
       [Sufficient]        ▼        [Iteration < max_iter]          ▼ [Iteration >= max_iter]
             ┌──────────────────┐   ┌────────────────────┐   ┌─────────────┐
             │     generate     │   │   refine_queries   │───┘             │
             └─────────┬────────┘   └────────────────────┘      │  abstain    │
                       │                                        └──────┬──────┘
                       ▼                                               │
             ┌─────────────────────────────────────────────────────────┴──────┐
             │                        verify_citations                        │
             └────────────────────────────────┬───────────────────────────────┘
                                              │
                                              ▼
                                   ┌──────────────────────┐
                                   │ AgentResearchResponse│
                                   └──────────────────────┘
```

---

## Decisions

### 1. State Definition & Strongly-Typed Schema
All workflow state is managed within a strongly-typed `AgentState` TypedDict:
- `original_query` / `normalized_query`: Sanitized input question.
- `query_type` / `complexity` / `needs_decomposition`: Classification metadata from `QueryAnalyzer`.
- `sub_queries`: List of `SubQueryPlan` models (`sub_query_id`, `sub_query_text`, `purpose`, `parent_query`, `priority`).
- `retrieval_results`: Candidate pools mapped by sub-query ID.
- `aggregated_evidence`: List of `AgentEvidenceItem` models with provenance tracking.
- `iteration_count` / `max_iterations`: Bounded loop counters.
- `evidence_sufficient` / `sufficiency_reason`: Audit trail of sufficiency decisions.
- `final_answer` / `claims` / `citations`: Grounded synthesis and verification outputs.
- `execution_trace`: List of `ExecutionTraceItem` auditing every node execution, iteration, and latency in milliseconds.

### 2. Query Analysis & Deterministic Decomposition
The `QueryAnalyzer` classifies queries into 8 categories:
- `simple_factual`, `semantic_lookup`, `technical_keyword`, `section_lookup` $\rightarrow$ `needs_decomposition = False`.
- `multi_document`, `comparison`, `multi_hop` $\rightarrow$ `needs_decomposition = True`.
- `no_answer` $\rightarrow$ ungrounded/out-of-domain detection.

When `needs_decomposition = True`, `QueryDecomposer` decomposes the query into 2–3 targeted sub-queries with distinct IDs (`SQ-1`, `SQ-2`), explicit purposes, and parent-child tracking.

### 3. Sub-Query Retrieval & Cross-Encoder Reranking
- **Sub-Query Hybrid Retrieval**: Each sub-query executes independent hybrid candidate retrieval (Dense BGE + Lexical BM25 via Reciprocal Rank Fusion).
- **Reranking**: Candidate pools are scored jointly against each sub-query using `BAAI/bge-reranker-base` (ONNX CPU runtime). Completed sub-queries are tracked in `completed_sub_queries` to avoid redundant search.

### 4. Evidence Aggregation & Provenance Tracking
`EvidenceAggregator` combines candidates across all sub-queries:
- **Deduplication**: Chunks discovered by multiple sub-queries are retained only once. The chunk preserves the highest observed rerank score.
- **Discovery Provenance**: The aggregator records all discovering sub-queries in `discovered_by_sub_queries` (e.g., `["SQ-1", "SQ-2"]`) and constructs an end-to-end trace:
  `Query -> SubQuery [SQ-1, SQ-2] -> Chunk chunk-001 -> Doc sample.md (Page 1 | Section: Core Capabilities)`.
- **Token Budget Packing**: Chunks are packed up to `context_max_tokens`, ensuring top-ranked evidence is preserved without overflow.

### 5. Sufficiency Assessment & Bounded Iterative Loops
`SufficiencyEvaluator` analyzes whether aggregated evidence provides complete coverage of all planned sub-queries and topical keywords.
- If coverage is sufficient $\rightarrow$ routes to `generate`.
- If coverage is insufficient and `iteration < max_iterations` (default: 2) $\rightarrow$ routes to `refine_queries`, which formulates targeted queries for missing aspects and loops back to `retrieve`.
- If coverage is insufficient and `iteration >= max_iterations` $\rightarrow$ routes to `abstain`, guaranteeing finite termination without infinite loops.

### 6. Evidence Isolation
The prompt provided to the LLM during grounded synthesis contains strictly the retrieved chunk content, citation IDs, and metadata headers. Internal agent trace logs, decomposition reasoning, and classification metadata are strictly isolated from the generation context, eliminating prompt leakage.

### 7. Dual API Architecture
- `POST /api/v1/research`: Preserved Milestone 5 single-pass research endpoint (100% backward compatible).
- `POST /api/v1/research/agent`: Milestone 8 LangGraph agent endpoint accepting `AgentResearchRequest` and returning `AgentResearchResponse`.

---

## Consequences

### Positive
- **High-Precision Multi-Document Synthesis**: Multi-part queries retrieve evidence across distinct files and sections with individual targeted queries.
- **Traceability & Provenance**: Every piece of evidence explicitly tracks which sub-queries discovered it and where in the source document it originated.
- **Guaranteed Loop Termination**: Hard loop bounds prevent non-terminating search loops.
- **100% Offline Testability**: Deterministic heuristic fallback allows running unit, integration, and benchmark tests on local developer workstations without cloud LLM API keys.

### Negative / Trade-Offs
- **CPU Latency Overhead**: Running cross-encoder neural reranking across multiple sub-queries increases overall query latency on CPU environments (~2.5s per sub-query candidate pool).
