import pytest
from backend.app.models.schemas import GroundedGenerationOutput
from backend.app.services.grounded_generation_service import (
    GroundedGenerationService,
    get_grounded_generation_service,
    reset_grounded_generation_service,
)
from backend.app.services.llm_service import MockLLMService


@pytest.fixture
def sample_evidence():
    return [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-cloud",
            "page_number": 1,
            "section_title": "Architecture",
            "section_path": "Enterprise > Architecture",
            "score": 0.92,
            "text": "The microservices architecture utilizes automated horizontal pod autoscaling.",
        }
    ]


@pytest.mark.asyncio
async def test_grounded_generation_with_evidence(sample_evidence):
    svc = GroundedGenerationService(llm_service=MockLLMService())
    query = "How does autoscaling work in the architecture?"
    context = (
        "[Source: doc-cloud | Page: 1 | Section: Architecture | Score: 0.92]\n"
        "The microservices architecture utilizes automated horizontal pod autoscaling.\n\n"
        "Evidence Chunk IDs: chunk-001"
    )

    output, latency_ms = await svc.generate_grounded_answer(
        query=query,
        context_text=context,
        evidence_chunks=sample_evidence,
    )

    assert isinstance(output, GroundedGenerationOutput)
    assert output.insufficient_evidence is False
    assert len(output.answer) > 0
    assert len(output.claims) >= 1
    assert "chunk-001" in output.claims[0].evidence_chunk_ids
    assert latency_ms >= 0.0


@pytest.mark.asyncio
async def test_grounded_generation_empty_context():
    svc = GroundedGenerationService(llm_service=MockLLMService())
    output, latency_ms = await svc.generate_grounded_answer(
        query="What is the revenue?",
        context_text="",
        evidence_chunks=[],
    )

    assert output.insufficient_evidence is True
    assert "not contain sufficient evidence" in output.answer
    assert len(output.claims) == 0
    assert latency_ms == 0.0


@pytest.mark.asyncio
async def test_grounded_generation_markdown_fence_cleaning(sample_evidence):
    # Mock LLM returning markdown wrapped json
    class MarkdownWrappedLLM(MockLLMService):
        async def generate(self, *args, **kwargs) -> str:
            return (
                "```json\n"
                "{\n"
                '  "answer": "Autoscaling dynamically scales pods based on CPU utilization.",\n'
                '  "claims": [\n'
                '    {"claim_id": "C1", "text": "Autoscaling scales pods.", "evidence_chunk_ids": ["chunk-001"]}\n'
                "  ],\n"
                '  "insufficient_evidence": false\n'
                "}\n"
                "```"
            )

    svc = GroundedGenerationService(llm_service=MarkdownWrappedLLM())
    output, _ = await svc.generate_grounded_answer(
        query="Tell me about autoscaling.",
        context_text="Some context chunk-001",
        evidence_chunks=sample_evidence,
    )

    assert output.insufficient_evidence is False
    assert "Autoscaling dynamically scales" in output.answer
    assert len(output.claims) == 1
    assert output.claims[0].claim_id == "C1"


@pytest.mark.asyncio
async def test_grounded_generation_malformed_json_fallback(sample_evidence):
    # Mock LLM returning non-JSON plain text
    malformed_llm = MockLLMService(simulate_malformed=True)
    svc = GroundedGenerationService(llm_service=malformed_llm)

    output, _ = await svc.generate_grounded_answer(
        query="Tell me about autoscaling.",
        context_text="Some context chunk-001",
        evidence_chunks=sample_evidence,
    )

    # Should fall back cleanly without raising exception
    assert output is not None
    assert output.answer is not None
    assert len(output.claims) >= 1
    assert output.claims[0].evidence_chunk_ids == ["chunk-001"]


def test_grounded_generation_factory():
    reset_grounded_generation_service()
    svc1 = get_grounded_generation_service()
    svc2 = get_grounded_generation_service()
    assert svc1 is svc2
    reset_grounded_generation_service()
