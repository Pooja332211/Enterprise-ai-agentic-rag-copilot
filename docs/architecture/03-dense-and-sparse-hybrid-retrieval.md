# Architecture Decision Record (ADR) 03: Dense Embeddings, BM25 Lexical Indexing & Hybrid Retrieval

## Status
Accepted

## Context
Milestone 2 established the document ingestion and structure-aware chunking pipeline, creating normalized `DocumentChunk` records stored in relational persistence (PostgreSQL / SQLite). To empower enterprise search and downstream agentic RAG workflows, chunks must be retrievable with high precision across two complementary query patterns:

1. **Semantic Conceptual Queries**: User queries expressing concepts, intent, or paraphrased questions where exact wording differs from the document text (e.g. *"What methodologies did the research studies utilize?"*).
2. **Exact Lexical / Technical Queries**: User queries containing specific domain identifiers, acronyms, product names, code tokens, or exact numbers (e.g. *"Benchmark Results"*, *"Table 2"*, *"POST /api/v1/search"*).

Relying solely on dense embeddings suffers from the "lexical gap" (imprecision on exact technical terms and identifiers). Relying solely on lexical BM25 misses semantic synonyms and paraphrased questions. Combining both into a deterministic hybrid retrieval layer provides the highest recall and precision.

## Decisions

1. **Provider-Agnostic Embedding Layer**:
   - Implemented `BaseEmbeddingService` interface decoupling the application from specific embedding providers:
     - **FastEmbed (`FastEmbedEmbeddingService`)**: Primary local provider using ONNX Runtime with `BAAI/bge-small-en-v1.5` (384 dimensions). Executes on CPU in milliseconds without heavy PyTorch GPU dependencies or external API keys.
     - **OpenAI-Compatible (`OpenAIEmbeddingService`)**: Remote HTTP provider supporting standard `/v1/embeddings` endpoints (OpenAI, vLLM, Ollama, LiteLLM) with configurable batching.
     - **Mock Provider (`MockEmbeddingService`)**: Deterministic hash-based feature projector with L2 normalization producing reproducible 384-dimensional unit vectors for zero-download, offline test execution.
   - Resilient factory (`get_embedding_service`): Automatically falls back to mock provider if FastEmbed cannot initialize or network weights are unreachable.

2. **Deterministic Qdrant Vector Indexing**:
   - `VectorService` manages collection lifecycle, point upsert, vector query, and deletion.
   - **Deterministic Point IDs**: Point IDs in Qdrant directly reuse the UUID4 primary key (`DocumentChunk.id`). This guarantees 1:1 mapping, enforces idempotency, and eliminates duplicate vectors upon re-indexing.
   - **Rich Payload Metadata**: Each vector point stores `document_id`, `chunk_id`, `chunk_index`, `page_number`, `section_title`, `section_path`, and `text`.
   - **Filtered Search**: Supports Qdrant `FieldCondition` matching on `document_id` for document-scoped queries.

3. **BM25 Lexical Indexing**:
   - `BM25IndexService` built on `rank-bm25` using the `BM25Plus` algorithm.
   - **Positive IDF Guarantee**: `BM25Plus` prevents zero or negative IDF scores on small document collections where terms appear in high proportions of chunks.
   - **Tokenization**: Preserves hyphenated terms, alphanumeric technical identifiers, and code names.
   - **Corpus Augmentation**: Combines section titles, breadcrumb paths, and body text (`f"{section_title} {section_path} {text}"`) for natural keyword boost.
   - **Persistence & Rebuildability**: In-memory index for sub-millisecond retrieval, serialized to `data/indexes/bm25_index.pkl`, and 100% rebuildable from relational database records on demand.

4. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**:
   - Queries dense and lexical indexes concurrently.
   - Standard scale-invariant **RRF** ($k = 60$):
     $$RRF\_score(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
     where $w_{dense} = \alpha$ and $w_{bm25} = 1 - \alpha$ (default $\alpha = 0.5$).
   - Alternative **Weighted Score Blend** ($\alpha$ blend) with min-max normalization.
   - Formatted results preserve all Milestone 2 metadata (`page_number`, `section_title`, `section_path`) and explicitly tag evidence sources (`retrieval_sources: ["dense", "bm25"]`).

5. **Index Lifecycle Management**:
   - **Ingestion Hook**: Ingestion pipeline automatically advances to stage `INDEXING` (progress `0.9`) after chunk persistence, indexing vectors into Qdrant and chunks into BM25.
   - **Cascade Deletion Hook**: Calling `DELETE /api/v1/documents/{document_id}` purges vector points from Qdrant and chunks from BM25.
   - **Administrative Management Endpoints**:
     - `POST /api/v1/index/rebuild`: Full reconstruction of dense and lexical indexes from PostgreSQL / SQLite chunks.
     - `POST /api/v1/index/documents/{id}`: Re-indexes individual documents.
     - `DELETE /api/v1/index/documents/{id}`: Purges index entries without deleting relational documents.
     - `GET /api/v1/index/stats`: Reports index status, vector counts, and BM25 vocabulary metrics.

## Consequences
- High-recall, high-precision hybrid retrieval layer is fully operational without external model generation dependencies.
- Grounded citations and section paths from Milestone 2 are preserved throughout the retrieval pipeline.
- Ready for Milestone 4 (BGE Reranker and cross-encoder scoring).
