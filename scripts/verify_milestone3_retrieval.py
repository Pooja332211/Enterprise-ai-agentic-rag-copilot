import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from httpx import ASGITransport, AsyncClient
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
    await init_db()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        print("==================================================")
        print("1. Cleaning Up Any Existing Documents (Idempotency)")
        print("==================================================")
        existing_list = await client.get("/api/v1/documents?limit=100")
        if existing_list.status_code == 200:
            for item in existing_list.json().get("items", []):
                await client.delete(f"/api/v1/documents/{item['id']}")
        # Ensure fresh index state
        await client.post("/api/v1/index/rebuild")

        print("\n==================================================")
        print("2. Ingesting Documents & Triggering Ingestion Indexing")
        print("==================================================")
        uploaded_doc_ids = {}
        for filename, mime_type in TEST_FILES:
            file_path = FIXTURES_DIR / filename
            if not file_path.exists():
                print(f"Skipping {filename}: not found.")
                continue

            file_bytes = file_path.read_bytes()
            res = await client.post(
                "/api/v1/documents",
                files={"file": (filename, file_bytes, mime_type)},
            )
            assert res.status_code == 201, f"Failed uploading {filename}: {res.text}"
            data = res.json()
            doc_id = data["document"]["id"]
            chunks_cnt = data["document"]["chunk_count"]
            uploaded_doc_ids[filename] = doc_id
            print(f"  [OK] Ingested '{filename}' (ID: {doc_id}) -> Chunks: {chunks_cnt}")

        print("\n==================================================")
        print("3. Checking Index Statistics (GET /api/v1/index/stats)")
        print("==================================================")
        stats_res = await client.get("/api/v1/index/stats")
        assert stats_res.status_code == 200
        stats = stats_res.json()
        print(f"  Provider: {stats['embedding_provider']} ({stats['embedding_model']}, dim={stats['embedding_dimension']})")
        print(f"  Qdrant Points: {stats['qdrant']['points_count']} (Mode: {stats['qdrant'].get('mode', 'local')})")
        print(f"  BM25 Chunks: {stats['bm25']['total_chunks']}, Vocabulary: {stats['bm25']['vocabulary_size']}")
        assert stats["bm25"]["total_chunks"] > 0
        assert stats["qdrant"]["points_count"] > 0

        print("\n==================================================")
        print("4. Testing Dense Semantic Search (POST /api/v1/search)")
        print("==================================================")
        dense_query = "What methodologies did the research studies utilize?"
        dense_res = await client.post(
            "/api/v1/search",
            json={"query": dense_query, "mode": "dense", "top_k": 3},
        )
        assert dense_res.status_code == 200
        dense_data = dense_res.json()
        print(f"  Query: '{dense_query}' | Matches: {dense_data['total']} | Latency: {dense_data['latency_ms']}ms")
        for idx, item in enumerate(dense_data["results"], 1):
            print(f"    #{idx} Score: {item['score']} | Doc: {item['document_id'][:8]} | Sec: {item['section_title']} (Page {item['page_number']})")
            print(f"       Sources: {item['retrieval_sources']} | Text: {repr(item['text'][:60])}")

        print("\n==================================================")
        print("5. Testing Lexical BM25 Search (POST /api/v1/search)")
        print("==================================================")
        bm25_query = "Benchmark Results"
        bm25_res = await client.post(
            "/api/v1/search",
            json={"query": bm25_query, "mode": "sparse", "top_k": 3},
        )
        assert bm25_res.status_code == 200
        bm25_data = bm25_res.json()
        print(f"  Query: '{bm25_query}' | Matches: {bm25_data['total']} | Latency: {bm25_data['latency_ms']}ms")
        for idx, item in enumerate(bm25_data["results"], 1):
            print(f"    #{idx} Score: {item['score']} | Doc: {item['document_id'][:8]} | Sec: {item['section_title']}")
            print(f"       Sources: {item['retrieval_sources']} | Text: {repr(item['text'][:60])}")

        print("\n==================================================")
        print("6. Testing Hybrid Retrieval with RRF (k=60)")
        print("==================================================")
        hybrid_query = "Methodology and enterprise knowledge specifications"
        hybrid_res = await client.post(
            "/api/v1/search",
            json={
                "query": hybrid_query,
                "mode": "hybrid",
                "top_k": 5,
                "fusion_method": "rrf",
            },
        )
        assert hybrid_res.status_code == 200
        hybrid_data = hybrid_res.json()
        print(f"  Query: '{hybrid_query}' | Matches: {hybrid_data['total']} | Latency: {hybrid_data['latency_ms']}ms")
        print(f"  Timing Breakdown: {hybrid_data['breakdown_ms']}")
        for idx, item in enumerate(hybrid_data["results"], 1):
            print(f"    #{idx} RRF Score: {item['score']} | DenseRank: {item['dense_rank']} | BM25Rank: {item['bm25_rank']}")
            print(f"       Sources: {item['retrieval_sources']} | Path: '{item['section_path']}' (Page {item['page_number']})")
            print(f"       Text: {repr(item['text'][:65])}")

        print("\n==================================================")
        print("7. Testing Document ID Filtering")
        print("==================================================")
        target_doc = uploaded_doc_ids.get("sample.pdf")
        filter_res = await client.post(
            "/api/v1/search",
            json={
                "query": "Methodology",
                "mode": "hybrid",
                "top_k": 5,
                "document_ids": [target_doc],
            },
        )
        assert filter_res.status_code == 200
        filter_data = filter_res.json()
        print(f"  Filtered to PDF doc '{target_doc}': {filter_data['total']} matches")
        for item in filter_data["results"]:
            assert item["document_id"] == target_doc
            print(f"    [OK] Match verified belonging to doc {item['document_id']}")

        print("\n==================================================")
        print("8. Testing Full Index Rebuild (POST /api/v1/index/rebuild)")
        print("==================================================")
        rebuild_res = await client.post("/api/v1/index/rebuild")
        assert rebuild_res.status_code == 200
        rebuild_data = rebuild_res.json()
        print(f"  Rebuild message: {rebuild_data['message']}")
        print(f"  Total Chunks: {rebuild_data['total_chunks']} (Dense: {rebuild_data['dense_indexed']}, BM25: {rebuild_data['bm25_indexed']})")
        print(f"  Elapsed: {rebuild_data['elapsed_seconds']}s")
        assert rebuild_data["total_chunks"] == 6
        assert rebuild_data["dense_indexed"] == 6
        assert rebuild_data["bm25_indexed"] == 6

        print("\n==================================================")
        print("9. Testing Document Deletion & Index Purge")
        print("==================================================")
        del_target = uploaded_doc_ids.get("sample.html")
        del_res = await client.delete(f"/api/v1/documents/{del_target}")
        assert del_res.status_code == 200
        print(f"  Deleted document {del_target}")

        # Search again to verify HTML chunks are purged
        post_del_search = await client.post(
            "/api/v1/search",
            json={"query": "Benchmark Results", "mode": "hybrid", "document_ids": [del_target]},
        )
        assert post_del_search.status_code == 200
        assert post_del_search.json()["total"] == 0
        print("  [OK] Confirmed deleted document returns 0 search matches.")

        print("\n==================================================")
        print("MILESTONE 3 VERIFICATION COMPLETED SUCCESSFULLY!")
        print("==================================================")


if __name__ == "__main__":
    asyncio.run(main())
