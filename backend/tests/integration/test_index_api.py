import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_index_stats_endpoint(async_client: AsyncClient):
    res = await async_client.get("/api/v1/index/stats")
    assert res.status_code == 200
    data = res.json()
    assert "qdrant" in data
    assert "bm25" in data
    assert "embedding_provider" in data
    assert "embedding_dimension" in data


@pytest.mark.asyncio
async def test_index_rebuild_endpoint(async_client: AsyncClient):
    res = await async_client.post("/api/v1/index/rebuild")
    assert res.status_code == 200
    data = res.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "dense_indexed" in data
    assert "bm25_indexed" in data


@pytest.mark.asyncio
async def test_reindex_document_lifecycle(async_client: AsyncClient):
    # Upload document
    res = await async_client.post(
        "/api/v1/documents",
        files={"file": ("reindex_test.txt", b"Document content for reindexing test.", "text/plain")},
    )
    assert res.status_code == 201
    doc_id = res.json()["document"]["id"]

    # Re-index specific document
    reindex_res = await async_client.post(f"/api/v1/index/documents/{doc_id}")
    assert reindex_res.status_code == 200
    assert reindex_res.json()["chunks_indexed"] >= 1

    # Remove from index only
    remove_res = await async_client.delete(f"/api/v1/index/documents/{doc_id}")
    assert remove_res.status_code == 200

    # Clean up document
    del_res = await async_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200
