import pytest
from backend.app.services.embedding_service import (
    FastEmbedEmbeddingService,
    MockEmbeddingService,
    get_embedding_service,
)


def test_mock_embedding_service_dimensions_and_norm():
    svc = MockEmbeddingService(dimension=384)
    assert svc.dimension == 384
    assert svc.model_name == "mock-deterministic-384"

    vec = svc.embed_query("Enterprise AI Knowledge Copilot")
    assert len(vec) == 384
    # Check L2 unit norm
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4


def test_mock_embedding_deterministic():
    svc = MockEmbeddingService(dimension=384)
    v1 = svc.embed_query("deterministic query test")
    v2 = svc.embed_query("deterministic query test")
    assert v1 == v2


def test_mock_embedding_batch():
    svc = MockEmbeddingService(dimension=384)
    texts = ["doc one", "doc two", "doc three"]
    vecs = svc.embed_texts(texts)
    assert len(vecs) == 3
    assert len(vecs[0]) == 384


def test_fastembed_service():
    svc = FastEmbedEmbeddingService(dimension=384)
    assert svc.dimension == 384
    assert "bge-small" in svc.model_name

    vecs = svc.embed_texts(["Hybrid retrieval with Qdrant and BM25."])
    assert len(vecs) == 1
    assert len(vecs[0]) == 384


def test_get_embedding_service_factory():
    mock_svc = get_embedding_service(provider="mock")
    assert isinstance(mock_svc, MockEmbeddingService)
    assert mock_svc.dimension == 384
