import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_research_api_grounded_synthesis(async_client: AsyncClient):
    content = (
        b"# Autonomous Agent Reliability\n\n"
        b"Autonomous AI research agents achieve deterministic verification by extracting "
        b"individual factual claims and validating them against retrieved citation evidence chunks.\n\n"
        b"Microservice deployments require zero-trust security postures and mutual TLS.\n"
    )
    upload_res = await async_client.post(
        "/api/v1/documents",
        files={"file": ("agent_reliability.md", content, "text/markdown")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]

    research_payload = {
        "query": "How do autonomous AI research agents achieve verification?",
        "mode": "hybrid",
        "top_k": 3,
        "candidate_top_k": 10,
        "rerank": True,
        "temperature": 0.0,
    }

    res = await async_client.post("/api/v1/research", json=research_payload)
    assert res.status_code == 200
    data = res.json()

    assert data["query"] == research_payload["query"]
    assert len(data["answer"]) > 0
    assert data["insufficient_evidence"] is False
    assert len(data["claims"]) >= 1
    assert len(data["citations"]) >= 1
    assert len(data["evidence"]) >= 1

    # Verification structure
    verif = data["verification"]
    assert verif["status"] in ("verified", "partially_verified")
    assert verif["supported_claims"] >= 1
    assert len(verif["details"]) >= 1

    # Timings breakdown
    timings = data["timings_ms"]
    assert "generation_ms" in timings
    assert "verification_ms" in timings
    assert "total_latency_ms" in timings
    assert "fusion_ms" in timings

    # Provenance citation check
    first_cit = data["citations"][0]
    assert "formatted_citation" in first_cit
    assert first_cit["chunk_id"] is not None

    # Cleanup
    del_res = await async_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200


@pytest.mark.asyncio
async def test_research_api_insufficient_evidence(async_client: AsyncClient):
    # Query with an impossible threshold to simulate zero relevant evidence
    payload = {
        "query": "Quantum gravity string theory spacetime manifolds",
        "min_relevance_threshold": 0.9999,
    }

    res = await async_client.post("/api/v1/research", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["insufficient_evidence"] is True
    assert "not contain sufficient evidence" in data["answer"]
    assert len(data["claims"]) == 0
    assert len(data["citations"]) == 0
    assert data["grounded"] is False
    assert data["verification"]["status"] == "unverified"


@pytest.mark.asyncio
async def test_research_api_invalid_mode(async_client: AsyncClient):
    res = await async_client.post(
        "/api/v1/research",
        json={"query": "test query", "mode": "invalid_mode"},
    )
    assert res.status_code == 400
