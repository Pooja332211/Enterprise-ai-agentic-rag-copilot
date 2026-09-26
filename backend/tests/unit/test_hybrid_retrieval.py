import pytest
from backend.app.services.bm25_service import BM25IndexService
from backend.app.services.embedding_service import MockEmbeddingService
from backend.app.services.hybrid_retrieval_service import HybridRetrievalService
from backend.app.services.vector_service import VectorService


@pytest.fixture
def hybrid_service(tmp_path):
    vs = VectorService()
    vs.ensure_collection("test_hybrid_col", 384)

    bm = BM25IndexService()
    bm.index_path = tmp_path / "bm25_hybrid_test.pkl"
    bm.clear(persist=False)

    emb = MockEmbeddingService(dimension=384)
    svc = HybridRetrievalService(
        vector_service=vs,
        bm25_service=bm,
        embedding_service=emb,
    )
    yield svc
    try:
        vs.client.delete_collection("test_hybrid_col")
    except Exception:
        pass
    bm.clear(persist=False)


@pytest.mark.asyncio
async def test_hybrid_rrf_fusion(hybrid_service):
    # Populate test data
    c1 = "10000000-0000-0000-0000-000000000001"
    c2 = "10000000-0000-0000-0000-000000000002"

    payload1 = {
        "document_id": "doc-1",
        "chunk_id": c1,
        "chunk_index": 0,
        "page_number": 1,
        "section_title": "Methodology",
        "section_path": "Paper > Methodology",
        "text": "The research methodology employs hybrid search retrieval.",
    }
    payload2 = {
        "document_id": "doc-2",
        "chunk_id": c2,
        "chunk_index": 1,
        "page_number": 2,
        "section_title": "Results",
        "section_path": "Paper > Results",
        "text": "Results showed significant improvements in recall and latency.",
    }

    # Index in vector
    vec1 = hybrid_service.embedding_service.embed_query(payload1["text"])
    vec2 = hybrid_service.embedding_service.embed_query(payload2["text"])
    hybrid_service.vector_service.upsert_chunks(
        chunk_ids=[c1, c2],
        vectors=[vec1, vec2],
        payloads=[payload1, payload2],
        collection_name="test_hybrid_col",
    )

    # Index in BM25
    hybrid_service.bm25_service.add_chunks([payload1, payload2], persist=False)

    # Run hybrid search targeting methodology
    # temporarily patch collection_name
    hybrid_service.settings.QDRANT_COLLECTION_NAME = "test_hybrid_col"
    res = await hybrid_service.search(
        query="research methodology hybrid search",
        mode="hybrid",
        top_k=5,
        fusion_method="rrf",
    )

    assert res["total"] >= 1
    assert res["mode"] == "hybrid"
    top_item = res["results"][0]
    assert top_item["chunk_id"] == c1
    assert top_item["page_number"] == 1
    assert top_item["section_title"] == "Methodology"
    assert "retrieval_sources" in top_item
    assert "score" in top_item


@pytest.mark.asyncio
async def test_search_modes(hybrid_service):
    cid = "20000000-0000-0000-0000-000000000001"
    item = {
        "document_id": "doc-3",
        "chunk_id": cid,
        "text": "Artificial Intelligence is transforming enterprise workflows.",
    }
    vec = hybrid_service.embedding_service.embed_query(item["text"])
    hybrid_service.vector_service.upsert_chunks([cid], [vec], [item], collection_name="test_hybrid_col")
    hybrid_service.bm25_service.add_chunks([item], persist=False)
    hybrid_service.settings.QDRANT_COLLECTION_NAME = "test_hybrid_col"

    # Dense-only
    dense_res = await hybrid_service.search("Artificial Intelligence", mode="dense", top_k=2)
    assert len(dense_res["results"]) == 1
    assert dense_res["results"][0]["retrieval_sources"] == ["dense"]

    # Sparse-only
    sparse_res = await hybrid_service.search("enterprise workflows", mode="sparse", top_k=2)
    assert len(sparse_res["results"]) == 1
    assert sparse_res["results"][0]["retrieval_sources"] == ["bm25"]


@pytest.mark.asyncio
async def test_empty_query(hybrid_service):
    res = await hybrid_service.search("", mode="hybrid")
    assert res["total"] == 0
    assert len(res["results"]) == 0
