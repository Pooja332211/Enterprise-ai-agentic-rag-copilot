import pytest
from pathlib import Path
from backend.app.services.bm25_service import BM25IndexService


@pytest.fixture
def bm25_service(tmp_path):
    svc = BM25IndexService()
    svc.index_path = tmp_path / "bm25_test_index.pkl"
    svc.clear(persist=False)
    yield svc
    svc.clear(persist=True)


def test_bm25_tokenization():
    tokens = BM25IndexService.tokenize("FastAPI & PostgreSQL-16 v1.2.0 API_KEY")
    assert "fastapi" in tokens
    assert "postgresql-16" in tokens
    assert "v1.2.0" in tokens
    assert "api_key" in tokens


def test_bm25_add_and_search(bm25_service):
    chunks = [
        {
            "chunk_id": "c-1",
            "document_id": "doc-1",
            "text": "The quick brown fox jumps over the lazy dog.",
            "section_title": "Animals",
            "section_path": "Nature > Animals",
        },
        {
            "chunk_id": "c-2",
            "document_id": "doc-2",
            "text": "Enterprise search engines utilize dense vectors and inverted indexes.",
            "section_title": "Information Retrieval",
            "section_path": "Architecture > IR",
        },
    ]

    bm25_service.add_chunks(chunks, persist=False)
    results = bm25_service.search("inverted indexes search", top_k=5)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c-2"
    assert results[0]["score"] > 0.0


def test_bm25_document_filter(bm25_service):
    chunks = [
        {"chunk_id": "c-10", "document_id": "doc-X", "text": "Common term in doc X"},
        {"chunk_id": "c-20", "document_id": "doc-Y", "text": "Common term in doc Y"},
    ]
    bm25_service.add_chunks(chunks, persist=False)

    results = bm25_service.search("common term", top_k=5, document_ids=["doc-Y"])
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c-20"


def test_bm25_remove_document(bm25_service):
    chunks = [
        {"chunk_id": "c-30", "document_id": "doc-del", "text": "Deletable chunk content"},
        {"chunk_id": "c-31", "document_id": "doc-keep", "text": "Keepable chunk content"},
    ]
    bm25_service.add_chunks(chunks, persist=False)
    assert len(bm25_service.search("deletable", top_k=5)) == 1

    bm25_service.remove_document("doc-del", persist=False)
    assert len(bm25_service.search("deletable", top_k=5)) == 0
    assert len(bm25_service.search("keepable", top_k=5)) == 1


def test_bm25_persistence(bm25_service, tmp_path):
    chunks = [{"chunk_id": "c-40", "document_id": "doc-persist", "text": "Persistent index text"}]
    bm25_service.add_chunks(chunks, persist=True)

    # Load fresh service from same path
    new_svc = BM25IndexService()
    new_svc.index_path = bm25_service.index_path
    new_svc._load_from_disk_if_exists()

    res = new_svc.search("persistent", top_k=5)
    assert len(res) == 1
    assert res[0]["chunk_id"] == "c-40"
