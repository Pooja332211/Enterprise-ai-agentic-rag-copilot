import pytest
from backend.app.config.settings import Settings
from backend.app.services.reranker_service import (
    BGERerankerService,
    MockRerankerService,
    get_reranker_service,
)


@pytest.fixture
def sample_candidates():
    return [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "text": "The quarterly financial report indicates a 15 percent revenue growth in Q3.",
            "score": 0.45,
            "section_title": "Financial Performance",
            "page_number": 3,
        },
        {
            "chunk_id": "c2",
            "document_id": "doc1",
            "text": "The cafeteria menu features sandwiches, soups, and beverages on Fridays.",
            "score": 0.35,
            "section_title": "Workplace Amenities",
            "page_number": 12,
        },
        {
            "chunk_id": "c3",
            "document_id": "doc2",
            "text": "Revenue growth was primarily driven by enterprise cloud software subscriptions.",
            "score": 0.40,
            "section_title": "Revenue Breakdown",
            "page_number": 5,
        },
    ]


def test_mock_reranker_relevance_ordering(sample_candidates):
    reranker = MockRerankerService()
    query = "enterprise revenue growth"

    reranked = reranker.rerank(query=query, candidates=sample_candidates, top_k=3)

    assert len(reranked) == 3
    # Both c1 and c3 mention revenue growth; c3 mentions enterprise revenue growth (exact match elements)
    # The cafeteria chunk (c2) should be ranked lowest
    chunk_ids = [c["chunk_id"] for c in reranked]
    assert chunk_ids[-1] == "c2"

    # Verify score descending order
    scores = [c["rerank_score"] for c in reranked]
    assert scores == sorted(scores, reverse=True)

    # Verify metadata preservation
    top_item = reranked[0]
    assert "retrieval_score" in top_item
    assert "retrieval_rank" in top_item
    assert "final_rank" in top_item
    assert top_item["final_rank"] == 1
    assert 0.0 <= top_item["rerank_score"] <= 1.0


def test_mock_reranker_empty_inputs(sample_candidates):
    reranker = MockRerankerService()
    assert reranker.rerank(query="", candidates=sample_candidates) == []
    assert reranker.rerank(query="test", candidates=[]) == []


def test_mock_reranker_top_k_truncation(sample_candidates):
    reranker = MockRerankerService()
    reranked = reranker.rerank(query="revenue", candidates=sample_candidates, top_k=1)
    assert len(reranked) == 1
    assert reranked[0]["final_rank"] == 1


def test_bge_reranker_local_execution(sample_candidates):
    try:
        reranker = BGERerankerService(batch_size=2)
    except Exception:
        pytest.skip("BGE ONNX runtime or weights not available locally")

    query = "financial report revenue growth"
    reranked = reranker.rerank(query=query, candidates=sample_candidates, top_k=2)

    assert len(reranked) == 2
    assert reranked[0]["rerank_score"] >= reranked[1]["rerank_score"]
    assert 0.0 <= reranked[0]["rerank_score"] <= 1.0
    assert reranked[0]["final_rank"] == 1
    assert "retrieval_score" in reranked[0]


def test_bge_reranker_batching(sample_candidates):
    try:
        # Force tiny batch size of 1 to rigorously test batch looping
        reranker = BGERerankerService(batch_size=1)
    except Exception:
        pytest.skip("BGE ONNX runtime not available")

    reranked = reranker.rerank(
        query="quarterly revenue",
        candidates=sample_candidates,
    )
    assert len(reranked) == 3


def test_get_reranker_service_factory():
    settings = Settings()

    # Explicit mock
    mock_svc = get_reranker_service(settings=settings, provider="mock")
    assert isinstance(mock_svc, MockRerankerService)
    assert mock_svc.model_name == "mock-reranker-deterministic"

    # Default provider resolution
    default_svc = get_reranker_service(settings=settings)
    assert default_svc is not None
    assert hasattr(default_svc, "rerank")
