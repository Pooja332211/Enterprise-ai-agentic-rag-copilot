import json
import pytest
from backend.app.config.settings import Settings
from backend.app.services.llm_service import (
    AnthropicLLMService,
    MockLLMService,
    OpenAILLMService,
    get_llm_service,
)


@pytest.mark.asyncio
async def test_mock_llm_generation_with_evidence():
    llm = MockLLMService()
    prompt = (
        "User Research Question:\nWhat are the cloud revenues?\n\n"
        "Supplied Context Blocks (Evidence Chunks):\n"
        "[Source: doc-1 | Page: 2 | Section: Financials | Score: 0.95]\n"
        "Cloud annual recurring revenue reached $1.2B in 2024, representing 28% growth.\n\n"
        "Evidence Chunk IDs: 11111111-1111-1111-1111-111111111111"
    )

    response = await llm.generate(prompt=prompt)
    assert response is not None
    data = json.loads(response)

    assert "answer" in data
    assert "claims" in data
    assert data["insufficient_evidence"] is False
    assert len(data["claims"]) >= 1
    assert data["claims"][0]["claim_id"] == "C1"
    assert "11111111-1111-1111-1111-111111111111" in data["claims"][0]["evidence_chunk_ids"]


@pytest.mark.asyncio
async def test_mock_llm_empty_context():
    llm = MockLLMService()
    prompt = "User Research Question:\nWhat is the meaning of life?\n\nNo evidence available."

    response = await llm.generate(prompt=prompt)
    data = json.loads(response)

    assert data["insufficient_evidence"] is True
    assert "not contain sufficient evidence" in data["answer"]
    assert len(data["claims"]) == 0


@pytest.mark.asyncio
async def test_mock_llm_simulated_error():
    llm = MockLLMService(simulate_error="Connection timed out to model provider")
    with pytest.raises(RuntimeError, match="Simulated LLM Provider Error"):
        await llm.generate("test prompt")


@pytest.mark.asyncio
async def test_mock_llm_simulated_malformed():
    llm = MockLLMService(simulate_malformed=True)
    raw = await llm.generate("test prompt")
    assert "malformed" in raw
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)


@pytest.mark.asyncio
async def test_openai_missing_api_key_raises():
    svc = OpenAILLMService(api_key="")
    with pytest.raises(ValueError, match="OpenAI API key is not configured"):
        await svc.generate("test")


@pytest.mark.asyncio
async def test_anthropic_missing_api_key_raises():
    svc = AnthropicLLMService(api_key="")
    with pytest.raises(ValueError, match="Anthropic API key is not configured"):
        await svc.generate("test")


def test_llm_service_factory():
    settings = Settings(LLM_API_KEY="")

    # Mock provider
    mock_svc = get_llm_service(settings=settings, provider="mock")
    assert isinstance(mock_svc, MockLLMService)
    assert mock_svc.provider_name == "mock"

    # OpenAI missing key falls back to mock
    openai_fallback = get_llm_service(settings=settings, provider="openai")
    assert isinstance(openai_fallback, MockLLMService)

    # Unknown provider falls back to mock
    unknown_fallback = get_llm_service(settings=settings, provider="unknown_llm")
    assert isinstance(unknown_fallback, MockLLMService)
