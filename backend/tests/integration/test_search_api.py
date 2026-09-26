import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_search_api_hybrid(async_client: AsyncClient):
    # Upload text document first to have indexed data
    content = b"Hybrid search integrates dense semantic embeddings with BM25 keyword matching."
    upload_res = await async_client.post(
        "/api/v1/documents",
        files={"file": ("hybrid_test.txt", content, "text/plain")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]

    # Search for terms present in document
    search_payload = {
        "query": "dense semantic embeddings BM25",
        "mode": "hybrid",
        "top_k": 5,
        "fusion_method": "rrf",
    }
    search_res = await async_client.post("/api/v1/search", json=search_payload)
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["total"] >= 1
    assert data["mode"] == "hybrid"
    assert "latency_ms" in data
    assert "breakdown_ms" in data

    # Clean up document
    del_res = await async_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200


@pytest.mark.asyncio
async def test_search_api_modes(async_client: AsyncClient):
    # Search dense mode
    dense_res = await async_client.post(
        "/api/v1/search",
        json={"query": "test query", "mode": "dense", "top_k": 2},
    )
    assert dense_res.status_code == 200
    assert dense_res.json()["mode"] == "dense"

    # Search sparse mode
    sparse_res = await async_client.post(
        "/api/v1/search",
        json={"query": "test query", "mode": "sparse", "top_k": 2},
    )
    assert sparse_res.status_code == 200
    assert sparse_res.json()["mode"] == "sparse"


@pytest.mark.asyncio
async def test_search_api_invalid_mode(async_client: AsyncClient):
    res = await async_client.post(
        "/api/v1/search",
        json={"query": "test", "mode": "invalid_mode"},
    )
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_search_api_reranking_and_context_assembly(async_client: AsyncClient):
    content = (
        b"# Cloud Architecture Guidelines\n\n"
        b"Production Kubernetes clusters must implement automated horizontal pod autoscaling.\n\n"
        b"Zero-trust network security requires mutual TLS authentication across all internal microservices.\n"
    )
    upload_res = await async_client.post(
        "/api/v1/documents",
        files={"file": ("cloud_guidelines.md", content, "text/markdown")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]

    # Search with reranking and context assembly
    search_payload = {
        "query": "Kubernetes horizontal pod autoscaling",
        "mode": "hybrid",
        "top_k": 3,
        "candidate_top_k": 10,
        "rerank": True,
        "context_assembly": True,
        "context_max_tokens": 1000,
    }
    search_res = await async_client.post("/api/v1/search", json=search_payload)
    assert search_res.status_code == 200
    data = search_res.json()

    assert data["total"] >= 1
    assert data["reranked"] is True
    assert "reranking_ms" in data["breakdown_ms"]
    assert "context_assembly_ms" in data["breakdown_ms"]

    first_item = data["results"][0]
    assert first_item["final_rank"] == 1
    assert first_item["rerank_score"] is not None
    assert first_item["retrieval_score"] is not None

    # Context assembly validation
    ctx = data["assembled_context"]
    assert ctx is not None
    assert ctx["total_chunks"] >= 1
    assert ctx["total_tokens"] > 0
    assert "Kubernetes" in ctx["context_text"]
    assert len(ctx["packed_chunks"]) >= 1

    # Clean up
    await async_client.delete(f"/api/v1/documents/{doc_id}")


@pytest.mark.asyncio
async def test_search_api_rerank_disabled_and_threshold(async_client: AsyncClient):
    content = b"Distributed consensus algorithms like Raft and Paxos ensure state machine consistency."
    upload_res = await async_client.post(
        "/api/v1/documents",
        files={"file": ("consensus.txt", content, "text/plain")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]

    # Search with rerank=False
    no_rerank_payload = {
        "query": "Distributed consensus Raft",
        "rerank": False,
        "context_assembly": False,
    }
    res1 = await async_client.post("/api/v1/search", json=no_rerank_payload)
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["reranked"] is False
    assert d1["assembled_context"] is None
    if d1["total"] > 0:
        assert d1["results"][0]["rerank_score"] is None

    # Search with very high threshold to test filtering
    high_thresh_payload = {
        "query": "Distributed consensus",
        "min_relevance_threshold": 0.9999,  # within [0.0, 1.0] schema validation
    }
    res2 = await async_client.post("/api/v1/search", json=high_thresh_payload)
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["total"] == 0
    assert len(d2["results"]) == 0

    # Clean up
    await async_client.delete(f"/api/v1/documents/{doc_id}")

