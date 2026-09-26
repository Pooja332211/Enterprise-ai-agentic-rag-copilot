import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.config.settings import get_settings
from backend.app.database.session import AsyncSessionLocal, init_db
from backend.app.database.models import Document
from backend.app.models.schemas import ResearchRequest
from backend.app.services.ingestion_service import IngestionService
from backend.app.services.indexing_service import IndexingService
from backend.app.services.research_service import ResearchService, get_research_service
from backend.app.evaluation.schemas import (
    BenchmarkDataset,
    BenchmarkRunSummary,
    PipelineLatencyProfile,
)
from backend.app.evaluation.metrics import (
    calculate_latency_percentiles,
    calculate_abstention_metrics,
)
from backend.app.evaluation.retrieval_evaluator import RetrievalEvaluator
from backend.app.evaluation.generation_evaluator import GenerationEvaluator
from backend.app.evaluation.citation_evaluator import CitationEvaluator


BENCHMARK_PATH = Path("data/evaluation/benchmark_v1.json")
RESULTS_DIR = Path("data/evaluation/results")
FIXTURES_DIR = Path("backend/tests/fixtures")

TEST_FILES = [
    ("sample.pdf", "application/pdf"),
    ("sample.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("sample.txt", "text/plain"),
    ("sample.md", "text/markdown"),
    ("sample.html", "text/html"),
]


class EvaluationRunner:
    """
    Comprehensive evaluation runner for Milestone 6 benchmarking:
    - 4-way retrieval ablation (Dense, BM25, Hybrid RRF, Hybrid + Reranker)
    - Latency distribution profiling (mean, median, p95 across 8 stages)
    - Grounded generation quality & reference answer alignment
    - Citation validity, coverage, and support auditing
    - Insufficient-evidence abstention accuracy
    """

    def __init__(
        self,
        dataset_path: Path = BENCHMARK_PATH,
        results_dir: Path = RESULTS_DIR,
    ):
        self.dataset_path = dataset_path
        self.results_dir = results_dir
        self.settings = get_settings()
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.citation_evaluator = CitationEvaluator()
        self.research_service = get_research_service(self.settings)

    def load_dataset(self) -> BenchmarkDataset:
        """Loads and validates the benchmark evaluation dataset."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found at {self.dataset_path}")
        
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return BenchmarkDataset(**data)

    async def ensure_corpus_ingested(self, session: AsyncSession) -> None:
        """Ensures the standard evaluation documents are ingested and indexed."""
        ingestion_service = IngestionService(self.settings)
        indexing_service = IndexingService(settings=self.settings)

        # Check existing documents
        existing_res = await session.execute(select(Document))
        existing_docs = {d.filename: d for d in existing_res.scalars().all()}

        for filename, mime_type in TEST_FILES:
            file_path = FIXTURES_DIR / filename
            if not file_path.exists():
                continue
            if filename in existing_docs:
                continue

            file_bytes = file_path.read_bytes()
            doc, job, is_dup = await ingestion_service.ingest_file(
                filename=filename,
                content=file_bytes,
                mime_type=mime_type,
                db=session,
            )

        await session.commit()

        # Rebuild indexes to guarantee fresh state
        await indexing_service.rebuild_all_indexes(db=session)

    async def run_evaluation(self) -> BenchmarkRunSummary:
        """Executes full evaluation suite and returns summary."""
        await init_db()
        dataset = self.load_dataset()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        async with AsyncSessionLocal() as session:
            await self.ensure_corpus_ingested(session)

            # ------------------------------------------------------------------
            # 1. Retrieval Ablation
            # ------------------------------------------------------------------
            ablation_results = await self.retrieval_evaluator.run_ablation(
                queries=dataset.queries,
                session=session,
            )
            configurations = ablation_results["configurations"]
            ablation_table = ablation_results["ablation_table"]

            # ------------------------------------------------------------------
            # 2. Multi-Query Research Pipeline Execution & Latency Profiling
            # ------------------------------------------------------------------
            stage_latencies: Dict[str, List[float]] = {
                "embedding_generation_ms": [],
                "dense_search_ms": [],
                "bm25_search_ms": [],
                "fusion_ms": [],
                "reranking_ms": [],
                "context_assembly_ms": [],
                "generation_ms": [],
                "verification_ms": [],
                "total_latency_ms": [],
            }

            generation_evals = []
            citation_audits = []
            abstention_evals = []

            for q in dataset.queries:
                req = ResearchRequest(
                    query=q.query,
                    mode="hybrid",
                    top_k=3,
                    candidate_top_k=15,
                    rerank=True,
                    context_max_tokens=1500,
                    temperature=0.0,
                )

                t_start = time.perf_counter()
                res = await self.research_service.execute_research(req)
                t_total_ms = (time.perf_counter() - t_start) * 1000.0

                # Record per-stage latency
                for stage in stage_latencies.keys():
                    if stage == "total_latency_ms":
                        stage_latencies[stage].append(t_total_ms)
                    else:
                        val = res.timings_ms.get(stage, 0.0)
                        stage_latencies[stage].append(val)

                retrieved_chunk_ids = {str(item.chunk_id) for item in res.evidence}

                # Evaluate answered queries
                if not q.expected_abstention:
                    gen_eval = self.generation_evaluator.evaluate_response(
                        generated_answer=res.answer,
                        reference_answer=q.expected_answer,
                        retrieved_contexts=[e.text for e in res.evidence],
                        claims=[c.model_dump() for c in res.claims],
                        verifications=[v.model_dump() for v in res.verification.details],
                    )
                    generation_evals.append(gen_eval)

                    cit_audit = self.citation_evaluator.audit_query_citations(
                        query_id=q.query_id,
                        query_text=q.query,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        claims=[c.model_dump() for c in res.claims],
                        citations=[cit.model_dump() for cit in res.citations],
                        verifications=[v.model_dump() for v in res.verification.details],
                    )
                    citation_audits.append(cit_audit)

                # Evaluate no-answer / abstention queries
                else:
                    abstention_evals.append({
                        "query_id": q.query_id,
                        "query": q.query,
                        "abstained": res.insufficient_evidence or (
                            "insufficient evidence" in res.answer.lower()
                        ),
                        "insufficient_evidence": res.insufficient_evidence,
                        "unsupported_claims_count": res.verification.unsupported_claims,
                        "answer": res.answer,
                    })

            # Latency summary
            latency_profile = PipelineLatencyProfile(
                embedding_latency=calculate_latency_percentiles(stage_latencies["embedding_generation_ms"]),
                dense_search_latency=calculate_latency_percentiles(stage_latencies["dense_search_ms"]),
                bm25_search_latency=calculate_latency_percentiles(stage_latencies["bm25_search_ms"]),
                fusion_latency=calculate_latency_percentiles(stage_latencies["fusion_ms"]),
                reranking_latency=calculate_latency_percentiles(stage_latencies["reranking_ms"]),
                context_assembly_latency=calculate_latency_percentiles(stage_latencies["context_assembly_ms"]),
                generation_latency=calculate_latency_percentiles(stage_latencies["generation_ms"]),
                citation_verification_latency=calculate_latency_percentiles(stage_latencies["verification_ms"]),
                total_pipeline_latency=calculate_latency_percentiles(stage_latencies["total_latency_ms"]),
            )

            # Aggregate generation, citation, and abstention metrics
            gen_metrics = self.generation_evaluator.aggregate_metrics(generation_evals)
            cit_metrics = self.citation_evaluator.aggregate_metrics(citation_audits)
            abs_metrics = calculate_abstention_metrics(abstention_evals)

            # Metadata
            env_metadata = {
                "os": platform.system(),
                "os_release": platform.release(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
                "embedding_model": self.settings.EMBEDDING_MODEL,
                "embedding_provider": self.settings.EMBEDDING_PROVIDER,
                "reranker_model": self.settings.RERANKER_MODEL,
                "reranker_provider": self.settings.RERANKER_PROVIDER,
                "llm_model": self.settings.LLM_MODEL,
                "llm_provider": self.settings.LLM_PROVIDER,
                "total_documents": len(dataset.document_corpus),
                "total_queries": len(dataset.queries),
            }

            summary = BenchmarkRunSummary(
                dataset_version=dataset.version,
                timestamp=datetime.now(timezone.utc).isoformat(),
                environment=env_metadata,
                total_queries=len(dataset.queries),
                evaluated_answered_queries=len(generation_evals),
                evaluated_no_answer_queries=len(abstention_evals),
                ablation=ablation_table,
                retrieval_configurations=configurations,
                latency_profile=latency_profile,
                generation_metrics=gen_metrics,
                citation_metrics=cit_metrics,
                abstention_metrics=abs_metrics,
                auditable_traces=citation_audits,
            )

            # ------------------------------------------------------------------
            # 3. Serialize Results (JSON & CSV)
            # ------------------------------------------------------------------
            self._save_results(summary)
            return summary

    def _save_results(self, summary: BenchmarkRunSummary) -> None:
        """Serializes results to JSON and CSV artifacts."""
        # 1. Main JSON summary
        summary_path = self.results_dir / "benchmark_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))

        # 2. Ablation table CSV
        ablation_csv_path = self.results_dir / "ablation_table.csv"
        with open(ablation_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Configuration", "Recall@5", "Precision@5", "MRR", "NDCG@5", "Mean_Latency_ms"])
            for row in summary.ablation:
                writer.writerow([
                    row.configuration,
                    f"{row.recall_at_5:.4f}",
                    f"{row.precision_at_5:.4f}",
                    f"{row.mrr:.4f}",
                    f"{row.ndcg_at_5:.4f}",
                    f"{row.latency_ms:.2f}",
                ])

        # 3. Latency table CSV
        latency_csv_path = self.results_dir / "latency_table.csv"
        with open(latency_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Stage", "Mean_ms", "Median_ms", "P95_ms", "Min_ms", "Max_ms", "Count"])
            stages = [
                ("Embedding Generation", summary.latency_profile.embedding_latency),
                ("Dense Search", summary.latency_profile.dense_search_latency),
                ("BM25 Search", summary.latency_profile.bm25_search_latency),
                ("Fusion", summary.latency_profile.fusion_latency),
                ("Reranking", summary.latency_profile.reranking_latency),
                ("Context Assembly", summary.latency_profile.context_assembly_latency),
                ("Generation", summary.latency_profile.generation_latency),
                ("Citation Verification", summary.latency_profile.citation_verification_latency),
                ("Total Pipeline", summary.latency_profile.total_pipeline_latency),
            ]
            for name, dist in stages:
                writer.writerow([
                    name,
                    f"{dist.mean_ms:.2f}",
                    f"{dist.median_ms:.2f}",
                    f"{dist.p95_ms:.2f}",
                    f"{dist.min_ms:.2f}",
                    f"{dist.max_ms:.2f}",
                    dist.count,
                ])

        # 4. Citation Audits JSON
        audits_path = self.results_dir / "citation_audits.json"
        with open(audits_path, "w", encoding="utf-8") as f:
            json.dump(summary.auditable_traces, f, indent=2)
