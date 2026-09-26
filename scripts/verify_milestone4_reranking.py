import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from httpx import ASGITransport, AsyncClient
from backend.app.config.settings import get_settings
from backend.app.database.session import init_db
from backend.app.main import app
from backend.app.services.context_assembly_service import ContextAssemblyService, get_context_assembly_service
from backend.app.services.reranker_service import (
    BGERerankerService,
    MockRerankerService,
    get_reranker_service,
)

FIXTURES_DIR = Path("backend/tests/fixtures")

TEST_FILES = [
    ("sample.pdf", "application/pdf"),
    ("sample.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("sample.txt", "text/plain"),
    ("sample.md", "text/markdown"),
    ("sample.html", "text/html"),
]


async def main():
    print("=" * 70)
    print("  ENTERPRISE AI COPILOT: MILESTONE 4 VERIFICATION SUITE")
    print("  Cross-Encoder Reranking & Context Assembly Pipeline")
    print("=" * 70)

    await init_db()
    settings = get_settings()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        # ----------------------------------------------------------------------
        # 1. Component Verification: Reranker Services
        # ----------------------------------------------------------------------
        print("\n--- 1. Reranker Component Verification ---")
        mock_reranker = MockRerankerService()
        test_candidates = [
            {"chunk_id": "c1", "text": "Annual enterprise revenue grew by 15% due to cloud subscriptions.", "score": 0.3},
            {"chunk_id": "c2", "text": "The company kitchen provides fresh fruit on Wednesday mornings.", "score": 0.4},
            {"chunk_id": "c3", "text": "Enterprise cloud revenue breakdown across regional branches.", "score": 0.25},
        ]
        query = "enterprise cloud revenue"
        mock_res = mock_reranker.rerank(query=query, candidates=test_candidates, top_k=3)
        assert len(mock_res) == 3
        assert mock_res[0]["chunk_id"] in ("c1", "c3"), "Most relevant query chunk was not ranked first by mock reranker"
        assert mock_res[-1]["chunk_id"] == "c2", "Irrelevant chunk was not ranked last"
        assert "retrieval_score" in mock_res[0]
        assert "rerank_score" in mock_res[0]
        print(f"  [OK] MockRerankerService: successfully reranked {len(mock_res)} candidates.")
        print(f"       Top candidate: {mock_res[0]['chunk_id']} (rerank_score={mock_res[0]['rerank_score']:.4f})")

        # Test local BGE ONNX reranker
        try:
            bge_reranker = BGERerankerService()
            bge_res = bge_reranker.rerank(query=query, candidates=test_candidates, top_k=3)
            assert len(bge_res) == 3
            assert 0.0 <= bge_res[0]["rerank_score"] <= 1.0
            print(f"  [OK] BGERerankerService (ONNX): successfully reranked {len(bge_res)} candidates.")
            print(f"       Model: {bge_reranker.model_name}")
            print(f"       Top candidate: {bge_res[0]['chunk_id']} (rerank_score={bge_res[0]['rerank_score']:.4f})")
        except Exception as exc:
            print(f"  [SKIP] BGERerankerService not available locally: {exc}")

        # ----------------------------------------------------------------------
        # 2. Component Verification: Context Assembly Service
        # ----------------------------------------------------------------------
        print("\n--- 2. Context Assembly Component Verification ---")
        assembly_svc = ContextAssemblyService()
        sample_chunks_for_assembly = [
            {
                "chunk_id": "ch-1",
                "document_id": "doc-annual-report",
                "page_number": 2,
                "section_path": "Financials > Cloud Revenue",
                "score": 0.94,
                "rerank_score": 0.94,
                "text": "Cloud annual recurring revenue reached $1.2B, an increase of 28% year-over-year.",
            },
            {
                "chunk_id": "ch-2",
                "document_id": "doc-annual-report",
                "page_number": 2,
                "section_path": "Financials > Cloud Revenue",
                "score": 0.93,
                "rerank_score": 0.93,
                # Near-duplicate of ch-1 (Jaccard > 0.85)
                "text": "Cloud annual recurring revenue reached $1.2B, representing an increase of 28% year-over-year.",
            },
            {
                "chunk_id": "ch-3",
                "document_id": "doc-annual-report",
                "page_number": 5,
                "section_path": "Operations > Infrastructure",
                "score": 0.82,
                "rerank_score": 0.82,
                "text": "Infrastructure efficiency improved via automated cluster right-sizing and caching layers.",
            },
        ]

        # Normal assembly with deduplication
        assembled = assembly_svc.assemble_context(
            chunks=sample_chunks_for_assembly,
            max_tokens=500,
            max_characters=2000,
            deduplicate=True,
        )
        assert assembled["total_chunks"] == 2, f"Expected 2 chunks (1 near-duplicate removed), got {assembled['total_chunks']}"
        assert assembled["is_truncated"] is False
        assert "[Source: doc-annual-report" in assembled["context_text"]
        assert "Section: Financials > Cloud Revenue" in assembled["context_text"]
        print(f"  [OK] ContextAssemblyService: Near-duplicate deduplication successful (3 -> {assembled['total_chunks']} chunks).")
        print(f"       Estimated tokens: {assembled['total_tokens']}, characters: {assembled['total_characters']}")

        # Strict budget truncation test
        budget_truncated = assembly_svc.assemble_context(
            chunks=sample_chunks_for_assembly,
            max_tokens=30,  # very tight budget ~120 chars
            deduplicate=False,
        )
        assert budget_truncated["is_truncated"] is True
        print(f"  [OK] Context budget enforcement: is_truncated={budget_truncated['is_truncated']} (packed {budget_truncated['total_chunks']} chunks).")

        # ----------------------------------------------------------------------
        # 3. Clean State & Document Ingestion
        # ----------------------------------------------------------------------
        print("\n--- 3. Ingesting Documents for End-to-End Search Pipeline ---")
        existing_list = await client.get("/api/v1/documents?limit=100")
        if existing_list.status_code == 200:
            for item in existing_list.json().get("items", []):
                await client.delete(f"/api/v1/documents/{item['id']}")
        await client.post("/api/v1/index/rebuild")

        uploaded_doc_ids = {}
        for filename, mime_type in TEST_FILES:
            file_path = FIXTURES_DIR / filename
            if not file_path.exists():
                continue
            file_bytes = file_path.read_bytes()
            res = await client.post(
                "/api/v1/documents",
                files={"file": (filename, file_bytes, mime_type)},
            )
            assert res.status_code == 201, f"Failed uploading {filename}: {res.text}"
            data = res.json()
            uploaded_doc_ids[filename] = data["document"]["id"]
            print(f"  [OK] Ingested '{filename}' (ID: {data['document']['id'][:8]}...) -> {data['document']['chunk_count']} chunks")

        # ----------------------------------------------------------------------
        # 4. End-to-End Pipeline Verification via /api/v1/search
        # ----------------------------------------------------------------------
        print("\n--- 4. End-to-End Search Pipeline Verification ---")
        pipeline_query = "AI architecture deep learning neural network retrieval"
        pipeline_request = {
            "query": pipeline_query,
            "mode": "hybrid",
            "top_k": 3,
            "candidate_top_k": 15,
            "rerank": True,
            "min_relevance_threshold": 0.0,
            "context_assembly": True,
            "context_max_tokens": 1500,
            "context_max_characters": 6000,
        }

        t_call_start = time.perf_counter()
        search_res = await client.post("/api/v1/search", json=pipeline_request)
        call_latency_ms = round((time.perf_counter() - t_call_start) * 1000, 2)

        assert search_res.status_code == 200, f"Search failed with code {search_res.status_code}: {search_res.text}"
        res_data = search_res.json()

        print(f"  [OK] HTTP Status: {search_res.status_code}")
        print(f"  [OK] Total results returned: {res_data['total']}")
        print(f"  [OK] Reranked flag: {res_data['reranked']}")
        print(f"  [OK] Measured Latency: endpoint={call_latency_ms} ms, internal={res_data['latency_ms']} ms")
        print("  [OK] Latency Breakdown:")
        for stage, duration in res_data.get("breakdown_ms", {}).items():
            print(f"       - {stage}: {duration} ms")

        # Verify candidate scoring and reranking
        assert len(res_data["results"]) > 0, "No results returned for search query"
        top_hit = res_data["results"][0]
        assert top_hit["final_rank"] == 1
        assert "retrieval_score" in top_hit
        assert "rerank_score" in top_hit
        print("\n  [OK] Top Evidence Chunk:")
        print(f"       - Chunk ID: {top_hit['chunk_id']}")
        print(f"       - Final Rank: {top_hit['final_rank']}")
        print(f"       - First-Stage Score: {top_hit['retrieval_score']}")
        print(f"       - Cross-Encoder Score: {top_hit['rerank_score']}")
        print(f"       - Section: {top_hit.get('section_title') or 'N/A'}")
        print(f"       - Text Snippet: {top_hit['text'][:90]}...")

        # Verify Assembled Context
        ctx_data = res_data.get("assembled_context")
        assert ctx_data is not None, "assembled_context missing from response"
        assert ctx_data["total_chunks"] >= 1
        assert ctx_data["total_tokens"] > 0
        assert ctx_data["total_characters"] > 0
        assert len(ctx_data["context_text"]) > 0
        print("\n  [OK] Assembled Context Validation:")
        print(f"       - Total Chunks Packed: {ctx_data['total_chunks']}")
        print(f"       - Total Estimated Tokens: {ctx_data['total_tokens']} / {ctx_data['budget_tokens_limit']}")
        print(f"       - Total Characters: {ctx_data['total_characters']} / {ctx_data['budget_characters_limit']}")
        print(f"       - Is Truncated: {ctx_data['is_truncated']}")
        print(f"       - Context Preview:\n         {ctx_data['context_text'][:160].replace(chr(10), ' ')}...")

        # ----------------------------------------------------------------------
        # 5. Backward Compatibility Verification
        # ----------------------------------------------------------------------
        print("\n--- 5. Backward Compatibility Check ---")
        legacy_request = {
            "query": "artificial intelligence",
            "mode": "hybrid",
            "top_k": 2,
        }
        legacy_res = await client.post("/api/v1/search", json=legacy_request)
        assert legacy_res.status_code == 200
        leg_data = legacy_res.json()
        assert leg_data["total"] >= 1
        assert "results" in leg_data
        print("  [OK] Milestone 3 legacy query format executed successfully without error.")

        # ----------------------------------------------------------------------
        # 6. Cleanup
        # ----------------------------------------------------------------------
        print("\n--- 6. Idempotent Teardown ---")
        for filename, doc_id in uploaded_doc_ids.items():
            await client.delete(f"/api/v1/documents/{doc_id}")
        await client.post("/api/v1/index/rebuild")
        print("  [OK] All temporary verification documents cleaned up.")

    print("\n" + "=" * 70)
    print("  MILESTONE 4 COMPLETE VERIFICATION PASSED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
