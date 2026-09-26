"""Integration tests for the LangGraph agentic research API endpoint (/api/v1/research/agent)."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_research_agent_api_end_to_end(async_client: AsyncClient):
    """End-to-end integration test of the agentic research endpoint."""
    content = (
        b"# Enterprise Secret Management\n\n"
        b"Enterprise API keys and database credentials are stored in HashiCorp Vault. "
        b"Secret rotation occurs automatically every 90 days.\n\n"
        b"Access to vault secrets requires role-based access control and short-lived tokens.\n"
    )
    upload_res = await async_client.post(
        "/api/v1/documents",
        files={"file": ("vault_secrets.md", content, "text/markdown")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]

    agent_payload = {
        "query": "How frequently are secret credentials rotated in HashiCorp Vault?",
        "top_k": 3,
        "candidate_top_k": 5,
        "rerank": True,
        "max_iterations": 2,
        "enable_decomposition": True,
        "context_max_tokens": 1500,
        "temperature": 0.0,
    }

    res = await async_client.post("/api/v1/research/agent", json=agent_payload)
    assert res.status_code == 200
    data = res.json()

    # Structural verification
    assert data["query"] == agent_payload["query"]
    assert len(data["final_answer"]) > 0
    assert data["insufficient_evidence"] is False
    assert data["status"] == "completed"
    assert data["query_type"] in ("simple_factual", "semantic_lookup", "section_lookup", "technical_keyword")
    assert data["complexity"] in ("low", "medium", "high")

    # Sub-queries
    assert len(data["sub_queries"]) >= 1
    assert "sub_query_id" in data["sub_queries"][0]
    assert "sub_query_text" in data["sub_queries"][0]

    # Aggregated Evidence with Provenance
    assert len(data["evidence"]) >= 1
    first_ev = data["evidence"][0]
    assert "chunk_id" in first_ev
    assert "text" in first_ev
    assert "discovered_by_sub_queries" in first_ev
    assert len(first_ev["discovered_by_sub_queries"]) >= 1

    # Citations & Claims
    assert len(data["claims"]) >= 1
    assert len(data["citations"]) >= 1
    assert data["verification"]["status"] in ("verified", "partially_verified")

    # Execution Trace
    assert len(data["execution_trace"]) >= 5
    trace_nodes = [t["node"] for t in data["execution_trace"]]
    assert "analyze_query" in trace_nodes
    assert "plan_query" in trace_nodes
    assert "retrieve" in trace_nodes
    assert "rerank" in trace_nodes
    assert "assemble_evidence" in trace_nodes
    assert "assess_sufficiency" in trace_nodes
    assert "generate" in trace_nodes
    assert "verify_citations" in trace_nodes

    # Timings
    assert "total_pipeline_ms" in data["timings_ms"]

    # Cleanup
    del_res = await async_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200


@pytest.mark.asyncio
async def test_research_agent_api_comparison_decomposition(async_client: AsyncClient):
    """Integration test verifying query decomposition on comparison queries."""
    content = (
        b"# Data Retention Standard\n\n"
        b"Customer billing records must be preserved for 7 years to comply with tax regulations.\n\n"
        b"Ephemeral application telemetry logs are discarded automatically after 14 days.\n"
    )
    upload_res = await async_client.post(
        "/api/v1/documents",
        files={"file": ("retention_standard.md", content, "text/markdown")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]

    agent_payload = {
        "query": "Compare billing records retention vs telemetry logs retention",
        "top_k": 3,
        "candidate_top_k": 5,
        "rerank": True,
        "max_iterations": 2,
        "enable_decomposition": True,
    }

    res = await async_client.post("/api/v1/research/agent", json=agent_payload)
    assert res.status_code == 200
    data = res.json()

    assert data["query_type"] == "comparison"
    assert len(data["sub_queries"]) >= 2
    assert len(data["evidence"]) >= 1
    assert data["status"] == "completed"

    # Cleanup
    del_res = await async_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200


@pytest.mark.asyncio
async def test_research_agent_api_insufficient_evidence(async_client: AsyncClient):
    """Integration test verifying loop bounds and clean abstention for unanswerable queries."""
    agent_payload = {
        "query": "What is the secret recipe for dark matter propulsion fuel?",
        "top_k": 3,
        "candidate_top_k": 5,
        "rerank": True,
        "max_iterations": 2,
    }

    res = await async_client.post("/api/v1/research/agent", json=agent_payload)
    assert res.status_code == 200
    data = res.json()

    assert data["insufficient_evidence"] is True
    assert data["status"] == "abstained"
    assert data["iterations_used"] <= 2
    assert "not contain sufficient evidence" in data["final_answer"]

    # Verify execution trace shows abstention
    trace_nodes = [t["node"] for t in data["execution_trace"]]
    assert "abstain" in trace_nodes


@pytest.mark.asyncio
async def test_research_agent_api_validation(async_client: AsyncClient):
    """Integration test verifying request validation."""
    # Query must not be empty
    res = await async_client.post("/api/v1/research/agent", json={"query": ""})
    assert res.status_code == 422
