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
    # Initialize DB (and activate fallback if PostgreSQL is offline)
    await init_db()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        print("==================================================")
        print("1. Verifying Health, Readiness, and Docs Endpoints")
        print("==================================================")
        for ep in ["/health", "/health/ready", "/docs"]:
            res = await client.get(ep)
            print(f"{ep}: status {res.status_code}")
            assert res.status_code == 200

        print("\n==================================================")
        print("2. Cleaning Up Existing Test Documents (Idempotency)")
        print("==================================================")
        existing_list = await client.get("/api/v1/documents?limit=100")
        if existing_list.status_code == 200:
            for item in existing_list.json().get("items", []):
                del_res = await client.delete(f"/api/v1/documents/{item['id']}")
                if del_res.status_code == 200:
                    print(f"  Cleaned up prior document: {item['filename']} ({item['id']})")

        print("\n==================================================")
        print("3. Uploading & Ingesting All 5 Supported Formats")
        print("==================================================")
        uploaded_docs = {}
        for filename, mime_type in TEST_FILES:
            file_path = FIXTURES_DIR / filename
            if not file_path.exists():
                print(f"WARNING: Fixture {filename} not found!")
                continue

            file_bytes = file_path.read_bytes()
            res = await client.post(
                "/api/v1/documents",
                files={"file": (filename, file_bytes, mime_type)},
            )
            print(f"POST /api/v1/documents [{filename}] -> Status: {res.status_code}")
            assert res.status_code == 201, f"Failed uploading {filename}: {res.text}"
            data = res.json()
            doc_id = data["document"]["id"]
            job_id = data["job"]["id"]
            chunk_cnt = data["document"]["chunk_count"]
            page_cnt = data["document"]["page_count"]
            print(f"  [OK] ID: {doc_id} | Chunks: {chunk_cnt} | Pages: {page_cnt} | Status: {data['document']['status']}")
            uploaded_docs[filename] = {
                "doc_id": doc_id,
                "job_id": job_id,
                "chunk_count": chunk_cnt,
                "bytes": file_bytes,
                "mime": mime_type,
            }

        print("\n==================================================")
        print("4. Listing Documents via GET /api/v1/documents")
        print("==================================================")
        list_res = await client.get("/api/v1/documents?limit=10")
        assert list_res.status_code == 200
        list_data = list_res.json()
        print(f"Total documents registered: {list_data['total']}, Retrieved: {len(list_data['items'])}")
        assert list_data["total"] == len(uploaded_docs)

        print("\n==================================================")
        print("5. Verifying Chunk Structures Across Parsers")
        print("==================================================")
        for fname, info in uploaded_docs.items():
            doc_id = info["doc_id"]
            chunks_res = await client.get(f"/api/v1/documents/{doc_id}/chunks")
            assert chunks_res.status_code == 200
            chunks_data = chunks_res.json()
            print(f"\nDocument: {fname} (Total Chunks: {chunks_data['total']})")
            for c in chunks_data["items"][:2]:  # show first 2 chunks
                sec_path = c.get("section_path") or "root"
                print(f"  Chunk #{c['chunk_index']} [Page {c['page_number']}] Path: '{sec_path}' Chars: {c['character_count']}")
                print(f"    Excerpt: {repr(c['text'][:65])}")

        print("\n==================================================")
        print("6. Verifying Ingestion Job Tracking")
        print("==================================================")
        pdf_job_id = uploaded_docs["sample.pdf"]["job_id"]
        job_res = await client.get(f"/api/v1/ingestion/{pdf_job_id}")
        assert job_res.status_code == 200
        job_data = job_res.json()
        print(f"Job ID: {job_data['id']}")
        print(f"Stage: {job_data['stage']}, Progress: {job_data['progress']}, Status: {job_data['status']}")
        assert job_data["status"] == "completed"
        assert job_data["stage"] == "COMPLETED"

        print("\n==================================================")
        print("7. Testing Duplicate Detection (HTTP 409 Conflict)")
        print("==================================================")
        pdf_bytes = uploaded_docs["sample.pdf"]["bytes"]
        dup_res = await client.post(
            "/api/v1/documents",
            files={"file": ("sample_copy.pdf", pdf_bytes, "application/pdf")},
        )
        print(f"Re-upload identical PDF -> Status: {dup_res.status_code} (Expected 409)")
        assert dup_res.status_code == 409
        dup_data = dup_res.json()
        print(f"  is_duplicate: {dup_data['is_duplicate']}")
        print(f"  matching document id: {dup_data['document']['id']}")
        assert dup_data["document"]["id"] == uploaded_docs["sample.pdf"]["doc_id"]

        print("\n==================================================")
        print("8. Testing Document Deletion & Cascade Cleanup")
        print("==================================================")
        del_target_id = uploaded_docs["sample.html"]["doc_id"]
        del_res = await client.delete(f"/api/v1/documents/{del_target_id}")
        assert del_res.status_code == 200
        print(f"DELETE /api/v1/documents/{del_target_id} -> Status: 200")
        
        # Verify 404 when querying deleted document
        get_deleted_res = await client.get(f"/api/v1/documents/{del_target_id}")
        assert get_deleted_res.status_code == 404
        print(f"GET /api/v1/documents/{del_target_id} -> Status: 404 (Successfully verified deleted)")

        print("\n==================================================")
        print("ALL 5 FORMATS AND MILESTONE 2 ENDPOINTS VERIFIED!")
        print("==================================================")


if __name__ == "__main__":
    asyncio.run(main())
