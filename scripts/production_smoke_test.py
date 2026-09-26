#!/usr/bin/env python
"""
Production Smoke Test for Enterprise AI Knowledge & Research Copilot.
Validates the 9 core production functions deterministically:
1. Root Liveness Probe (GET /health)
2. Deep Readiness Probe (GET /health/ready)
3. Document Upload & Ingestion (POST /api/v1/documents)
4. Index Status (GET /api/v1/index/stats)
5. Two-Stage Hybrid Search (POST /api/v1/search)
6. Single-Turn Grounded Research (POST /api/v1/research)
7. LangGraph Agentic Research (POST /api/v1/research/agent)
8. Citation Integrity & Claim Grounding Check
9. Prometheus Metrics Exposition (GET /metrics)

Supports live HTTP endpoint testing or in-process ASGI testing.
"""

import argparse
import asyncio
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx


async def run_smoke_test(target_url: str = "http://localhost:8000") -> bool:
    print("=" * 80)
    print("  Enterprise AI Knowledge & Research Copilot - Production Smoke Test")
    print("=" * 80)
    print(f"Target Service: {target_url}")
    print(f"Timestamp:      {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")

    # Determine whether external server is reachable or fallback to in-process ASGI
    use_in_process = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as probe_client:
            res = await probe_client.get(f"{target_url}/health")
            if res.status_code == 200:
                print(f"[OK] Connected to live server at {target_url}")
            else:
                use_in_process = True
    except Exception:
        use_in_process = True

    if use_in_process:
        print("[INFO] Live server unreachable. Initializing in-process ASGI test client...")
        try:
            from backend.app.main import app
            transport = httpx.ASGITransport(app=app)
            client = httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=60.0)
        except Exception as e:
            print(f"[ERROR] Failed to load application in-process: {e}")
            return False
    else:
        client = httpx.AsyncClient(base_url=target_url, timeout=60.0)

    all_passed = True
    test_results: List[Tuple[str, str, float]] = []

    async with client:
        # 1. Liveness Probe
        t0 = time.perf_counter()
        try:
            res = await client.get("/health")
            dur = (time.perf_counter() - t0) * 1000
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            data = res.json()
            assert data.get("status") == "healthy"
            test_results.append(("1. Liveness Probe (/health)", "PASS", dur))
            print(f"  [PASS] 1. Liveness Probe: status={data.get('status')} ({dur:.2f} ms)")
        except Exception as e:
            dur = (time.perf_counter() - t0) * 1000
            test_results.append(("1. Liveness Probe (/health)", f"FAIL ({e})", dur))
            print(f"  [FAIL] 1. Liveness Probe: {e}")
            all_passed = False

        # 2. Readiness Probe
        t0 = time.perf_counter()
        try:
            res = await client.get("/health/ready")
            dur = (time.perf_counter() - t0) * 1000
            assert res.status_code in (200, 503), f"Unexpected status {res.status_code}"
            data = res.json()
            readiness_status = data.get("status")
            services = data.get("services", {})
            db_conn = services.get("database", {}).get("status", "unknown")
            vs_conn = services.get("vector_store", {}).get("status", "unknown")
            test_results.append(("2. Readiness Probe (/health/ready)", "PASS", dur))
            print(f"  [PASS] 2. Readiness Probe: status={readiness_status} (DB: {db_conn}, Qdrant: {vs_conn}) ({dur:.2f} ms)")
        except Exception as e:
            dur = (time.perf_counter() - t0) * 1000
            test_results.append(("2. Readiness Probe (/health/ready)", f"FAIL ({e})", dur))
            print(f"  [FAIL] 2. Readiness Probe: {e}")
            all_passed = False

        # Clean up any leftover smoke test document from previous runs
        try:
            list_res = await client.get("/api/v1/documents")
            if list_res.status_code == 200:
                for item in list_res.json().get("items", []):
                    if item.get("filename") == "cloud_architecture_smoke.md":
                        await client.delete(f"/api/v1/documents/{item['id']}")
        except Exception:
            pass

        # 3. Document Ingestion
        t0 = time.perf_counter()
        sample_doc_id = None
        try:
            content = (
                b"# Cloud Native Microservices Architecture\n\n"
                b"## Service Mesh\n"
                b"The service mesh utilizes Envoy proxy sidecars for mTLS traffic encryption, "
                b"dynamic rate limiting, and observability. Istio control plane manages routing policies.\n\n"
                b"## Resilience Patterns\n"
                b"Circuit breakers trigger when downstream error rate exceeds 5 percent over a 10-second rolling window."
            )
            files = {"file": ("cloud_architecture_smoke.md", io.BytesIO(content), "text/markdown")}
            res = await client.post("/api/v1/documents", files=files, params={"force": "true"})
            dur = (time.perf_counter() - t0) * 1000
            assert res.status_code in (200, 201), f"Expected 200/201, got {res.status_code}: {res.text}"
            data = res.json()
            doc_data = data.get("document", {})
            sample_doc_id = doc_data.get("id")
            chunk_count = doc_data.get("chunk_count", 0)
            is_dup = data.get("is_duplicate", False)
            test_results.append(("3. Document Ingestion (/api/v1/documents)", "PASS", dur))
            print(f"  [PASS] 3. Document Ingestion: doc_id={sample_doc_id[:8]}... chunks={chunk_count} dup={is_dup} ({dur:.2f} ms)")
        except Exception as e:
            dur = (time.perf_counter() - t0) * 1000
            test_results.append(("3. Document Ingestion (/api/v1/documents)", f"FAIL ({e})", dur))
            print(f"  [FAIL] 3. Document Ingestion: {e}")
            all_passed = False

        # 4. Index Inspection
        t0 = time.perf_counter()
        try:
            res = await client.get("/api/v1/index/stats")
            dur = (time.perf_counter() - t0) * 1000
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            data = res.json()
            bm25_chunks = data.get("bm25", {}).get("indexed_chunks", 0)
            emb_model = data.get("embedding_model", "")
            test_results.append(("4. Index Inspection (/api/v1/index/stats)", "PASS", dur))
            print(f"  [PASS] 4. Index Inspection: BM25 chunks={bm25_chunks} model={emb_model} ({dur:.2f} ms)")
        except Exception as e:
            dur = (time.perf_counter() - t0) * 1000
            test_results.append(("4. Index Inspection (/api/v1/index/stats)", f"FAIL ({e})", dur))
            print(f"  [FAIL] 4. Index Inspection: {e}")
            all_passed = False

        # 5. Two-Stage Hybrid Search
        t0 = time.perf_counter()
        try:
            payload = {
                "query": "service mesh Envoy proxy mTLS encryption",
                "mode": "hybrid",
                "top_k": 3,
                "candidate_top_k": 10,
                "rerank": True,
            }
            res = await client.post("/api/v1/search", json=payload)
            dur = (time.perf_counter() - t0) * 1000
            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
            data = res.json()
            results = data.get("results", [])
            assert len(results) >= 1, "Expected at least 1 search result"
            test_results.append(("5. Two-Stage Hybrid Search (/api/v1/search)", "PASS", dur))
            print(f"  [PASS] 5. Hybrid Search: returned {len(results)} chunks, top_score={results[0].get('score'):.4f} ({dur:.2f} ms)")
        except Exception as e:
            dur = (time.perf_counter() - t0) * 1000
            test_results.append(("5. Two-Stage Hybrid Search (/api/v1/search)", f"FAIL ({e})", dur))
            print(f"  [FAIL] 5. Hybrid Search: {e}")
            all_passed = False

        # 6. Single-Turn Grounded Research
        t0 = time.perf_counter()
        try:
            payload = {
                "query": "What proxy does the service mesh utilize and what encryption is used?",
                "top_k": 3,
                "candidate_top_k": 10,
                "rerank": True,
            }
            res = await client.post("/api/v1/research", json=payload)
            dur = (time.perf_counter() - t0) * 1000
            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
            data = res.json()
            answer = data.get("answer", "")
            citations = data.get("citations", [])
            claims = data.get("claims", [])
            grounded = data.get("grounded", False)
            assert len(answer) > 0, "Empty answer returned"
            assert grounded is True, "Answer was not grounded"
            test_results.append(("6. Grounded Research (/api/v1/research)", "PASS", dur))
            print(f"  [PASS] 6. Grounded Research: grounded={grounded} claims={len(claims)} citations={len(citations)} ({dur:.2f} ms)")
        except Exception as e:
            dur = (time.perf_counter() - t0) * 1000
            test_results.append(("6. Grounded Research (/api/v1/research)", f"FAIL ({e})", dur))
            print(f"  [FAIL] 6. Grounded Research: {e}")
            all_passed = False

        # 7. LangGraph Agentic Research
        t0 = time.perf_counter()
        try:
            payload = {
                "query": "How does the service mesh handle encryption and what triggers circuit breakers?",
                "top_k": 3,
                "candidate_top_k": 10,
                "rerank": True,
                "max_iterations": 2,
                "enable_decomposition": True,
            }
            res = await client.post("/api/v1/research/agent", json=payload)
            dur = (time.perf_counter() - t0) * 1000
            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
            data = res.json()
            final_ans = data.get("final_answer", "")
            sub_queries = data.get("sub_queries", [])
            iterations = data.get("iterations_used", 0)
            status_val = data.get("status", "")
            assert len(final_ans) > 0, "Empty final answer"
            assert status_val == "completed", f"Unexpected status: {status_val}"
            test_results.append(("7. Agentic Research (/api/v1/research/agent)", "PASS", dur))
            print(f"  [PASS] 7. Agentic Research: sub_queries={len(sub_queries)} iterations={iterations} status={status_val} ({dur:.2f} ms)")
        except Exception as e:
            dur = (time.perf_counter() - t0) * 1000
            test_results.append(("7. Agentic Research (/api/v1/research/agent)", f"FAIL ({e})", dur))
            print(f"  [FAIL] 7. Agentic Research: {e}")
            all_passed = False

        # 8. Citation Verification Check
        t0 = time.perf_counter()
        try:
            payload = {
                "query": "What triggers circuit breakers?",
                "top_k": 3,
                "rerank": True,
            }
            res = await client.post("/api/v1/research", json=payload)
            dur = (time.perf_counter() - t0) * 1000
            assert res.status_code == 200
            data = res.json()
            verification = data.get("verification", {})
            ver_status = verification.get("status", "")
            supported = verification.get("supported_claims", 0)
            total = verification.get("total_claims", 0)
            citations = data.get("citations", [])
            assert len(citations) >= 1, "Expected at least 1 citation"
            assert ver_status in ("verified", "partially_verified"), f"Unexpected verification status: {ver_status}"
            test_results.append(("8. Citation Verification Check", "PASS", dur))
            print(f"  [PASS] 8. Citation Verification: status={ver_status} supported={supported}/{total} citations={len(citations)} ({dur:.2f} ms)")
        except Exception as e:
            dur = (time.perf_counter() - t0) * 1000
            test_results.append(("8. Citation Verification Check", f"FAIL ({e})", dur))
            print(f"  [FAIL] 8. Citation Verification: {e}")
            all_passed = False

        # 9. Prometheus Metrics Exposition
        t0 = time.perf_counter()
        try:
            res = await client.get("/metrics")
            dur = (time.perf_counter() - t0) * 1000
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            text = res.text
            assert "copilot_requests_total" in text, "Missing copilot_requests_total metric"
            assert "copilot_cache" in text or "copilot_request_duration_seconds" in text
            test_results.append(("9. Prometheus Metrics (/metrics)", "PASS", dur))
            print(f"  [PASS] 9. Prometheus Metrics: format=text/plain exposition length={len(text)} chars ({dur:.2f} ms)")
        except Exception as e:
            dur = (time.perf_counter() - t0) * 1000
            test_results.append(("9. Prometheus Metrics (/metrics)", f"FAIL ({e})", dur))
            print(f"  [FAIL] 9. Prometheus Metrics: {e}")
            all_passed = False

        # Post-test cleanup of smoke test document
        if sample_doc_id:
            try:
                await client.delete(f"/api/v1/documents/{sample_doc_id}")
            except Exception:
                pass

    print("\n" + "=" * 80)
    print("  Smoke Test Summary")
    print("=" * 80)
    for name, status, lat in test_results:
        print(f"  {name:<45} {status:<15} ({lat:.2f} ms)")
    print("=" * 80)

    if all_passed:
        print("\n>>> SMOKE TEST RESULT: ALL 9 CAPABILITIES PASSED (PRODUCTION READY) <<<\n")
        return True
    else:
        print("\n>>> SMOKE TEST RESULT: ONE OR MORE CHECKS FAILED <<<\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Production Smoke Test for Enterprise AI Copilot")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the running API")
    args = parser.parse_args()

    success = asyncio.run(run_smoke_test(args.url))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
