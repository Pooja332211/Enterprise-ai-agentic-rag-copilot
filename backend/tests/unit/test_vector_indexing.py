import pytest
from backend.app.services.vector_service import VectorService


@pytest.fixture
def vector_service():
    vs = VectorService()
    vs.ensure_collection("test_knowledge", 384)
    yield vs
    # Clean up test collection
    try:
        vs.client.delete_collection("test_knowledge")
    except Exception:
        pass


def test_upsert_and_search_vectors(vector_service):
    chunk_id = "00000000-0000-0000-0000-000000000001"
    vec = [0.1] * 384
    payload = {
        "document_id": "doc-abc",
        "chunk_id": chunk_id,
        "chunk_index": 0,
        "page_number": 1,
        "section_title": "Architecture",
        "section_path": "Spec > Architecture",
        "text": "Vector search test content",
    }

    ok = vector_service.upsert_chunks(
        chunk_ids=[chunk_id],
        vectors=[vec],
        payloads=[payload],
        collection_name="test_knowledge",
    )
    assert ok is True

    # Search with matching vector
    results = vector_service.search_vectors(
        query_vector=vec,
        top_k=5,
        collection_name="test_knowledge",
    )
    assert len(results) == 1
    assert results[0]["chunk_id"] == chunk_id
    assert results[0]["payload"]["section_title"] == "Architecture"


def test_vector_search_document_filter(vector_service):
    cid1 = "00000000-0000-0000-0000-000000000010"
    cid2 = "00000000-0000-0000-0000-000000000020"

    vector_service.upsert_chunks(
        chunk_ids=[cid1, cid2],
        vectors=[[0.2] * 384, [0.2] * 384],
        payloads=[
            {"document_id": "doc-A", "text": "Content A"},
            {"document_id": "doc-B", "text": "Content B"},
        ],
        collection_name="test_knowledge",
    )

    # Filter for doc-A only
    res_a = vector_service.search_vectors(
        query_vector=[0.2] * 384,
        top_k=10,
        document_ids=["doc-A"],
        collection_name="test_knowledge",
    )
    assert len(res_a) == 1
    assert res_a[0]["chunk_id"] == cid1

    # Filter for doc-B only
    res_b = vector_service.search_vectors(
        query_vector=[0.2] * 384,
        top_k=10,
        document_ids=["doc-B"],
        collection_name="test_knowledge",
    )
    assert len(res_b) == 1
    assert res_b[0]["chunk_id"] == cid2


def test_delete_document_vectors(vector_service):
    cid = "00000000-0000-0000-0000-000000000099"
    vector_service.upsert_chunks(
        chunk_ids=[cid],
        vectors=[[0.3] * 384],
        payloads=[{"document_id": "doc-to-delete", "text": "Will be removed"}],
        collection_name="test_knowledge",
    )

    # Confirm present
    res_before = vector_service.search_vectors(
        query_vector=[0.3] * 384,
        top_k=5,
        collection_name="test_knowledge",
    )
    assert any(r["chunk_id"] == cid for r in res_before)

    # Delete
    del_ok = vector_service.delete_document_vectors("doc-to-delete", collection_name="test_knowledge")
    assert del_ok is True

    # Confirm deleted
    res_after = vector_service.search_vectors(
        query_vector=[0.3] * 384,
        top_k=5,
        collection_name="test_knowledge",
    )
    assert not any(r["chunk_id"] == cid for r in res_after)
