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

FIXTURES_DIR = Path("backend/tests/fixtures")

TEST_FILES = [
    ("sample.pdf", "application/pdf"),
    ("sample.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("sample.txt", "text/plain"),
    ("sample.md", "text/markdown"),
    ("sample.html", "text/html"),
]


async def main():
    print("=" * 75)
    print("  ENTERPRISE AI COPILOT: MILESTONE 5 VERIFICATION SUITE")
    print("  Core Grounded RAG Generation, Research Agent & Citation Verification")
    print("=" * 75)

    await init_db()
    settings = get_settings()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        # ----------------------------------------------------------------------
        # 1. Clean State & Document Ingestion
        # ----------------------------------------------------------------------
        print("\n--- 1. Ingesting Multi-Format Documents for Research Pipeline ---")
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
        # 2. End-to-End Grounded Research Query via POST /api/v1/research
        # ----------------------------------------------------------------------
        print("\n--- 2. End-to-End Grounded Research Pipeline Execution ---")
        research_query = "What is the architecture and purpose of the enterprise knowledge copilot?"
        research_request = {
            "query": research_query,
            "mode": "hybrid",
            "top_k": 3,
            "candidate_top_k": 15,
            "rerank": True,
            "context_max_tokens": 1500,
            "temperature": 0.0,
        }

        t_call_start = time.perf_counter()
        research_res = await client.post("/api/v1/research", json=research_request)
        call_latency_ms = round((time.perf_counter() - t_call_start) * 1000, 2)

        assert research_res.status_code == 200, f"Research failed with status {research_res.status_code}: {research_res.text}"
        res_data = research_res.json()

        print(f"  [OK] HTTP Status: {research_res.status_code}")
        print(f"  [OK] Provider / Model: {res_data['provider']} / {res_data['model']}")
        print(f"  [OK] Grounded: {res_data['grounded']}")
        print(f"  [OK] Insufficient Evidence: {res_data['insufficient_evidence']}")
        print(f"  [OK] Total Latency: endpoint={call_latency_ms} ms, internal={res_data['timings_ms'].get('total_latency_ms', 0)} ms")

        print("\n  [OK] Granular Timing Profiling (ms):")
        for stage, duration in res_data.get("timings_ms", {}).items():
            print(f"       - {stage}: {duration} ms")

        print(f"\n  [OK] Synthesized Answer:\n       \"{res_data['answer']}\"")

        print(f"\n  [OK] Extracted Claims ({len(res_data['claims'])}):")
        for c in res_data["claims"]:
            print(f"       - [{c['claim_id']}]: \"{c['text']}\" (evidence: {c['evidence_chunk_ids']})")

        print(f"\n  [OK] Resolved Citations ({len(res_data['citations'])}):")
        for cit in res_data["citations"]:
            print(f"       - [{cit['citation_id']}] {cit['formatted_citation']}")
            print(f"         Snippet: \"{cit['snippet'][:90]}...\"")

        print("\n  [OK] Citation Verification Audit:")
        verif = res_data["verification"]
        print(f"       - Audit Status: {verif['status'].upper()}")
        print(f"       - Supported Claims: {verif['supported_claims']} / {verif['total_claims']}")
        print(f"       - Partially Supported: {verif['partially_supported_claims']}")
        print(f"       - Unsupported Claims: {verif['unsupported_claims']}")
        for detail in verif["details"]:
            print(f"         * [{detail['claim_id']}] -> {detail['status']} (score: {detail['support_score']:.2f})")
            print(f"           Rationale: {detail['rationale']}")

        assert verif["status"] in ("verified", "partially_verified"), "Expected verified or partially verified status"
        assert len(res_data["evidence"]) >= 1, "Expected retrieved evidence items"

        # ----------------------------------------------------------------------
        # 3. Insufficient Evidence Safeguard Test
        # ----------------------------------------------------------------------
        print("\n--- 3. Testing Insufficient Evidence Safeguard ---")
        insufficient_request = {
            "query": "What are the quantum teleportation protocols in string cosmology?",
            "min_relevance_threshold": 0.9999,  # Filter out all non-matching chunks
        }
        ins_res = await client.post("/api/v1/research", json=insufficient_request)
        assert ins_res.status_code == 200
        ins_data = ins_res.json()

        assert ins_data["insufficient_evidence"] is True, "Expected insufficient_evidence flag to be True"
        assert "not contain sufficient evidence" in ins_data["answer"]
        assert len(ins_data["claims"]) == 0
        assert len(ins_data["citations"]) == 0
        assert ins_data["grounded"] is False
        assert ins_data["verification"]["status"] == "unverified"
        print("  [OK] Successfully safeguarded against hallucination on zero-evidence query.")
        print(f"       Answer: \"{ins_data['answer']}\"")

        # ----------------------------------------------------------------------
        # 4. Clean Teardown
        # ----------------------------------------------------------------------
        print("\n--- 4. Idempotent Teardown ---")
        for filename, doc_id in uploaded_doc_ids.items():
            await client.delete(f"/api/v1/documents/{doc_id}")
        await client.post("/api/v1/index/rebuild")
        print("  [OK] Cleaned up all verification test documents.")

    print("\n" + "=" * 75)
    print("  MILESTONE 5 COMPLETE VERIFICATION PASSED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
