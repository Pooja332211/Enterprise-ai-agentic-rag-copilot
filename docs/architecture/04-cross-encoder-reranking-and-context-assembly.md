# Architecture Decision Record (ADR) 04: Cross-Encoder Reranking & Context Assembly

## Status
Accepted

## Context
Milestone 3 provided a high-recall hybrid retrieval layer combining dense semantic vector search (Qdrant) and lexical keyword matching (BM25) fused via Reciprocal Rank Fusion ($k=60$). While hybrid retrieval successfully maximizes candidate recall across wide document corpora, bi-encoder embedding representations decouple query and document token interactions during vector calculation. This can allow false positives to surface in the top-ranked slots.

To prepare grounded, high-precision context for downstream agentic RAG and LLM synthesis (Milestone 5), the system requires:
1. **Full Query-Document Cross-Attention**: Jointly scoring $(query, candidate\_text)$ pairs using a cross-encoder model to capture deep contextual semantic nuance, negations, and exact relationship constraints.
2. **Context Budget Management**: Enforcing token and character limits to prevent context window overflow in downstream LLM generation.
3. **Redundancy Elimination**: Detecting and pruning near-duplicate chunks that consume budget without adding information.
4. **Structured Context Formatting**: Injecting consistent document provenance, page numbers, breadcrumbs, and cross-encoder relevance scores into formatted context blocks.

## Decisions

1. **Two-Stage Retrieval Architecture**:
   - **First-Stage Retrieval (Candidate Retrieval)**: Rapidly filters large document corpora down to a manageable candidate pool of size $K_{\text{candidate}}$ (default `candidate_top_k = 20` or $\max(top\_k \times 3, 20)$) via Dense + BM25 + RRF.
   - **Second-Stage Neural Reranking**: Evaluates full cross-attention on candidate chunks using a cross-encoder, re-ordering the candidate pool and assigning calibrated relevance scores.
   - **Filtering & Final Slicing**: Applies optional `min_relevance_threshold` filtering and selects the top $K$ items (`RETRIEVAL_FINAL_TOP_K = 5` by default).

2. **Reranker Service Abstraction (`BaseReranker`)**:
   - Decoupled via `BaseReranker` abstract base class defining `rerank(query, candidates, top_k)`.
   - **BGE Cross-Encoder (`BGERerankerService`)**:
     - Model: `BAAI/bge-reranker-base`.
     - Runtime: Local CPU inference using **ONNX Runtime** and Hugging Face `tokenizers`.
     - Rationale: Eliminates heavy PyTorch GPU dependencies and Windows DLL import constraints (`shm.dll`), executing cross-encoder inference entirely on CPU with multi-threaded intra-op parallelism (`intra_op_num_threads = 4`).
     - Normalization: Projects raw model output logits through a sigmoid activation function $\sigma(z) = \frac{1}{1 + e^{-z}}$ to yield calibrated relevance scores in $[0.0, 1.0]$.
     - Batch Scoring: Configurable batch execution (`RERANKER_BATCH_SIZE = 32`) with automatic sequence truncation (`max_length = 512`).
   - **Deterministic Mock Reranker (`MockRerankerService`)**:
     - Fast, zero-weight simulator for test environments and offline CI execution.
     - Deterministically scores candidates using term co-occurrence (Jaccard similarity), exact phrase match bonus ($+0.35$), section title keyword alignment ($+0.15$), and normalized first-stage retrieval score ($+0.15$).
   - **Resilient Factory (`get_reranker_service`)**:
     - Selects provider based on `RERANKER_PROVIDER` (`bge` or `mock`).
     - Gracefully falls back to `MockRerankerService` if ONNX initialization or weight caching encounters exceptions.

3. **Context Assembly Service (`ContextAssemblyService`)**:
   - **Token Estimation**: Employs a fast 4-character-per-token heuristic ($tokens \approx \lceil \text{characters} / 4.0 \rceil$) for token budgeting without external tokenizer latency.
   - **Structured Metadata Headers**: Formats every evidence chunk with rich provenance metadata:
     ```text
     [Source: {document_id} | Page: {page_number} | Section: {section_path} | Score: {score:.4f}]
     {chunk_text}
     ```
   - **Budget Packing**: Greedily packs sorted chunks in order of cross-encoder rank until either `CONTEXT_MAX_TOKENS` (default 2000) or `CONTEXT_MAX_CHARACTERS` (default 8000) is reached.
   - **Near-Duplicate Pruning**: Removes redundant candidate chunks based on token-level Jaccard similarity ($\ge 0.85$) against previously accepted chunks.
   - **Oversized Chunk Handling**: Rejects single chunks exceeding the overall budget limit, flags `is_truncated = True`, and packs only chunks that fit cleanly.

4. **API Integration & Backward Compatibility**:
   - Updated `POST /api/v1/search` with new optional fields:
     - `candidate_top_k: Optional[int] = None`
     - `rerank: bool = True`
     - `min_relevance_threshold: Optional[float] = None`
     - `context_assembly: bool = False`
     - `context_max_tokens: Optional[int] = None`
     - `context_max_characters: Optional[int] = None`
   - Response includes:
     - `reranked: bool`
     - `retrieval_score: float` (original first-stage score)
     - `rerank_score: float` (cross-encoder score)
     - `final_rank: int` (1-indexed final rank)
     - `assembled_context: Optional[AssembledContextResponse]`
   - Full backward compatibility: Legacy Milestone 3 search payloads execute without error, returning all previous fields plus new optional metadata.

5. **Timing Instrumentation**:
   - Granular profiling recorded in `breakdown_ms`:
     - `embedding_generation_ms`
     - `dense_search_ms`
     - `bm25_search_ms`
     - `fusion_ms`
     - `reranking_ms`
     - `context_assembly_ms`
     - `latency_ms` (total internal execution time)

## Consequences
- The retrieval pipeline now provides cross-encoder neural reranking and context budgeting ready for LLM consumption.
- Eliminates context window overflow risks and duplicate evidence in downstream agents.
- All 81 project tests pass with 100% success.
- Ready for Milestone 5 (Research Agent, LLM Synthesis, and Multi-Document Reasoning).
