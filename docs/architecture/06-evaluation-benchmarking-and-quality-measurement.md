# Architecture Decision Record (ADR) 06: Evaluation, Benchmarking & Retrieval/Generation Quality Measurement

## Status
Accepted

## Context
Milestones 1–5 established an enterprise-grade agentic RAG and document research architecture:
1. **Milestone 1–2**: Ingestion across 5 document formats (PDF, DOCX, Markdown, HTML, TXT), structural hierarchy extraction, deduplication, and persistence in PostgreSQL / SQLite.
2. **Milestone 3**: Dense embeddings with FastEmbed (`BAAI/bge-small-en-v1.5`, 384-d), Qdrant vector indexing, BM25Plus lexical indexing, and Reciprocal Rank Fusion (RRF, $k=60$).
3. **Milestone 4**: Cross-encoder neural reranking (`BAAI/bge-reranker-base` on ONNX CPU) and context assembly with duplicate elimination and token budgeting.
4. **Milestone 5**: Core grounded generation with evidence isolation prompts, multi-provider LLM abstraction, deterministic citation verification with lexical/bigram support scoring, and research orchestration.

Milestone 6 transitions the platform from component implementation to empirical quality measurement, ablation benchmarking, and latency profiling. To evaluate enterprise readiness without fabrication or external API dependencies, the system requires:
1. **Controlled Evaluation Dataset**: Versioned benchmark queries spanning distinct enterprise retrieval patterns.
2. **4-Way Retrieval Ablation**: Direct empirical comparison of Dense-only, BM25-only, Hybrid RRF, and Hybrid + Cross-Encoder Reranking across standard IR metrics.
3. **Multi-Stage Latency Profiling**: Statistical distribution (mean, median, p95) across all 8 pipeline stages over multiple query executions.
4. **Grounded Generation & Citation Auditing**: Auditing citation validity against evidence, support rates, coverage, and claim verification.
5. **Abstention & Hallucination Resistance**: Measuring whether the system reliably abstains on ungrounded queries.
6. **External Dataset Trade-Off Analysis**: Rigorous evaluation of public/Kaggle datasets vs. controlled enterprise benchmarks.

---

## Decisions

### 1. External & Public Dataset Trade-Off Analysis
We investigated public IR and QA benchmarks (MS MARCO, SQuAD, HotpotQA, BEIR, and Kaggle enterprise QA collections):
- **Web Passage Collections (MS MARCO, SQuAD)**: Composed of short, single-paragraph web scrapes or Wikipedia snippets without document provenance. They lack multi-page document structures, heading breadcrumbs, physical page boundaries, tabular formatting, or enterprise file formats (PDF, DOCX).
- **Multi-Hop Benchmarks (HotpotQA)**: Emphasize Wikipedia entity links rather than hierarchical document sections or citation boundaries.
- **Kaggle Datasets**: Most public Kaggle enterprise datasets are either unstructured text dumps without ground-truth section paths, or require large external downloads (>5 GB) of unverified quality.
- **Decision**: Rather than blindly downloading an ill-fitting external dataset, we constructed a version-controlled, reproducible enterprise benchmark:
  `data/evaluation/benchmark_v1.json`
  grounded in a multi-format document corpus (PDF, DOCX, Markdown, HTML, TXT).
  The benchmark schema is strictly defined (`BenchmarkQuery`) so enterprise benchmarks such as **FinanceBench** or **DocVQA** can be converted into the internal evaluation format in future milestones without code changes.

### 2. Multi-Category Benchmark Query Formulation
The benchmark defines 18 manually authored, verified queries across 6 distinct categories:
1. **Exact Factual Lookup**: Targeting specific tabular data or numerical values (e.g. accuracy percentages, NDCG scores).
2. **Section-Specific Lookup**: Targeting explicit section paths and physical page numbers (e.g. PDF page 2 Methodology, Executive Summary).
3. **Technical Keyword Query**: Targeting code snippets, acronyms, and architectural formulas (e.g. `def query_copilot`, `NDCG@10 0.89`).
4. **Semantic / Paraphrased Query**: Abstract conceptual queries where query tokens differ significantly from document text.
5. **Multi-Document Synthesis**: Queries requiring cross-referencing information across multiple heterogeneous documents.
6. **No-Answer / Insufficient-Evidence**: Explicit ungrounded queries designed to test hallucination resistance and correct abstention.

### 3. Retrieval Ranking Metrics & 4-Way Ablation
The evaluation harness computes standard Information Retrieval metrics using verified ground-truth chunk and document mappings:
- **Recall@K** ($K \in \{1, 3, 5, 10\}$):
  $$\text{Recall}@K = \frac{|\text{Retrieved}@K \cap \text{Relevant}|}{|\text{Relevant}|}$$
- **Precision@K** ($K \in \{1, 3, 5\}$):
  $$\text{Precision}@K = \frac{|\text{Retrieved}@K \cap \text{Relevant}|}{K}$$
- **Mean Reciprocal Rank (MRR)**:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$
- **Normalized Discounted Cumulative Gain (NDCG@K)**:
  $$\text{DCG}@K = \sum_{i=1}^K \frac{rel_i}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

Ablation executes each query under 4 distinct retrieval configurations:
1. **Dense Only** (`mode="dense"`, `rerank=False`)
2. **BM25 Only** (`mode="sparse"`, `rerank=False`)
3. **Hybrid RRF** (`mode="hybrid"`, `fusion="rrf"`, `rerank=False`)
4. **Hybrid + Cross-Encoder Reranker** (`mode="hybrid"`, `fusion="rrf"`, `rerank=True`)

### 4. Multi-Stage Latency Profiling (Statistical Distribution)
Rather than reporting a single request timing, the benchmark records latencies across multiple query executions and computes:
- $\text{Mean}$, $\text{Median}$ (p50), and $\text{P95}$ percentiles.
- Measured separately for all 8 pipeline stages:
  1. `embedding_generation_ms`
  2. `dense_search_ms`
  3. `bm25_search_ms`
  4. `fusion_ms`
  5. `reranking_ms`
  6. `context_assembly_ms`
  7. `generation_ms`
  8. `verification_ms`
  9. `total_pipeline_latency_ms`

### 5. Grounded Generation & Citation Evaluation
- **Generation Quality**: Evaluated using deterministic token F1 and bigram F1 overlap against authoritative reference answers, combined with evidence grounding ratios.
- **Citation Validity**: Verifies that every cited chunk ID actually exists in the retrieved evidence pool (detecting hallucinated source IDs).
- **Citation Support Rate**: Computes the percentage of claims classified as `supported` ($\ge 0.40$ support score) by the deterministic citation verifier.
- **Citation Coverage**: Measures the proportion of generated factual claims backed by citations.
- **Auditable Trace**: Emits complete query-level provenance records (`query -> claims -> cited_chunks -> verification_status`) to `citation_audits.json`.

### 6. Insufficient-Evidence & Abstention Safeguards
- Evaluates queries where the corpus lacks relevant evidence.
- Measures:
  - **Correct Abstention Rate**: Queries where the system accurately returns `insufficient_evidence=True` and non-hallucinated disclaimers.
  - **Incorrect / Hallucinated Answers**: Queries where the system erroneously claims to answer ungrounded questions.

### 7. Machine-Readable & Human-Readable Reporting
The runner exports structured artifacts to `data/evaluation/results/`:
- `benchmark_summary.json`: Complete serialized run specification.
- `ablation_table.csv`: 4-way retrieval comparison table.
- `latency_table.csv`: Granular stage-by-stage latency percentiles.
- `citation_audits.json`: Detailed query-level claim verification traces.
- `docs/evaluation/milestone6-evaluation-report.md`: Human-readable summary report.

---

## Consequences
- Quality, ranking, and latency metrics are 100% measured and reproducible.
- Eliminates guesswork and unsubstantiated claims regarding retrieval and reranking efficacy.
- Provides immediate regression detection for future milestones (LangGraph multi-hop agents, fine-tuning).
