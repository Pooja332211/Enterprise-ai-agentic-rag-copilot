import pytest
from backend.app.config.settings import Settings
from backend.app.services.context_assembly_service import (
    ContextAssemblyService,
    get_context_assembly_service,
)


@pytest.fixture
def sample_ranked_chunks():
    return [
        {
            "chunk_id": "chunk-101",
            "document_id": "doc-alpha",
            "chunk_index": 0,
            "page_number": 1,
            "section_title": "Executive Summary",
            "section_path": "Annual Report > Executive Summary",
            "score": 0.95,
            "rerank_score": 0.95,
            "text": "The company achieved record operational efficiency across cloud infrastructure in 2024.",
        },
        {
            "chunk_id": "chunk-102",
            "document_id": "doc-alpha",
            "chunk_index": 1,
            "page_number": 2,
            "section_title": "Cloud Infrastructure",
            "section_path": "Annual Report > Cloud Infrastructure",
            "score": 0.88,
            "rerank_score": 0.88,
            "text": "Gross margins expanded by 420 basis points as automated scaling reduced idle server costs.",
        },
        {
            "chunk_id": "chunk-103",
            "document_id": "doc-beta",
            "chunk_index": 4,
            "page_number": 10,
            "section_title": "Risks",
            "section_path": "Risk Assessment > Market Risks",
            "score": 0.72,
            "rerank_score": 0.72,
            "text": "Geopolitical uncertainties and currency fluctuations present ongoing macroeconomic challenges.",
        },
    ]


def test_context_assembly_token_estimation():
    service = ContextAssemblyService()
    # 40 characters ~ 10 tokens with 4 chars/token heuristic
    text = "A" * 40
    assert service.estimate_tokens(text) == 10
    assert service.estimate_tokens("") == 0


def test_context_assembly_normal_packing(sample_ranked_chunks):
    service = ContextAssemblyService()
    result = service.assemble_context(
        chunks=sample_ranked_chunks,
        max_tokens=2000,
        max_characters=8000,
    )

    assert result["total_chunks"] == 3
    assert result["is_truncated"] is False
    assert result["total_tokens"] > 0
    assert result["total_characters"] > 0
    assert len(result["packed_chunks"]) == 3
    assert "Executive Summary" in result["context_text"]
    assert "Annual Report" in result["context_text"]
    assert "Score: 0.9500" in result["context_text"]
    assert "---" in result["context_text"]


def test_context_assembly_token_budget_enforcement(sample_ranked_chunks):
    service = ContextAssemblyService()
    # Set tight token budget that only allows the first chunk
    result = service.assemble_context(
        chunks=sample_ranked_chunks,
        max_tokens=50,  # ~200 characters max
        max_characters=10000,
    )

    assert result["total_chunks"] == 1
    assert result["is_truncated"] is True
    assert result["packed_chunks"][0]["chunk_id"] == "chunk-101"
    assert "chunk-102" not in result["context_text"]


def test_context_assembly_character_budget_enforcement(sample_ranked_chunks):
    service = ContextAssemblyService()
    # Tight character budget
    result = service.assemble_context(
        chunks=sample_ranked_chunks,
        max_tokens=5000,
        max_characters=220,
    )

    assert result["total_chunks"] == 1
    assert result["is_truncated"] is True
    assert result["packed_chunks"][0]["chunk_id"] == "chunk-101"


def test_context_assembly_near_duplicate_removal():
    service = ContextAssemblyService()
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "text": "Deep learning models require high quality curated training datasets to generalize effectively.",
            "score": 0.9,
        },
        {
            # Almost identical wording (Jaccard > 0.85)
            "chunk_id": "c2",
            "document_id": "d1",
            "text": "Deep learning models require high quality curated training datasets to generalize effectively and well.",
            "score": 0.89,
        },
        {
            "chunk_id": "c3",
            "document_id": "d2",
            "text": "PostgreSQL relational databases support ACID transactions and complex indexing strategies.",
            "score": 0.7,
        },
    ]

    result = service.assemble_context(chunks=chunks, deduplicate=True)
    assert result["total_chunks"] == 2
    packed_ids = [c["chunk_id"] for c in result["packed_chunks"]]
    assert "c1" in packed_ids
    assert "c2" not in packed_ids  # Duplicate dropped
    assert "c3" in packed_ids


def test_context_assembly_duplicate_chunk_id():
    service = ContextAssemblyService()
    chunks = [
        {"chunk_id": "dup-1", "document_id": "d1", "text": "First occurrence text."},
        {"chunk_id": "dup-1", "document_id": "d1", "text": "Second occurrence with same chunk_id."},
    ]
    result = service.assemble_context(chunks=chunks)
    assert result["total_chunks"] == 1


def test_context_assembly_oversized_first_chunk():
    service = ContextAssemblyService()
    large_text = "Word " * 500  # ~2500 chars, ~625 tokens
    chunks = [{"chunk_id": "c-huge", "document_id": "d1", "text": large_text}]

    # Budget smaller than the chunk
    result = service.assemble_context(chunks=chunks, max_tokens=50)
    assert result["total_chunks"] == 0
    assert result["is_truncated"] is True
    assert result["context_text"] == ""


def test_context_assembly_empty_inputs():
    service = ContextAssemblyService()
    result = service.assemble_context(chunks=[])
    assert result["total_chunks"] == 0
    assert result["total_tokens"] == 0
    assert result["is_truncated"] is False
    assert result["context_text"] == ""


def test_context_assembly_factory():
    svc = get_context_assembly_service()
    assert isinstance(svc, ContextAssemblyService)
