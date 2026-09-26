import pytest
from backend.app.models.schemas import ClaimItem
from backend.app.services.citation_verification_service import (
    CitationVerificationService,
    get_citation_verification_service,
)


@pytest.fixture
def sample_evidence_chunks():
    return [
        {
            "chunk_id": "chunk-101",
            "document_id": "doc-financials-2024",
            "page_number": 3,
            "section_title": "Revenue Performance",
            "section_path": "Annual Report > Financials > Revenue Performance",
            "score": 0.94,
            "text": "Cloud annual recurring revenue grew 28 percent reaching 1.2 billion dollars.",
        },
        {
            "chunk_id": "chunk-102",
            "document_id": "doc-operations-2024",
            "page_number": 8,
            "section_title": "Server Optimization",
            "section_path": "Operations > Server Optimization",
            "score": 0.85,
            "text": "Server consolidation reduced total electrical power consumption by 15 percent.",
        },
    ]


def test_citation_verification_supported_claim(sample_evidence_chunks):
    service = CitationVerificationService()
    claims = [
        ClaimItem(
            claim_id="C1",
            text="Cloud annual recurring revenue grew by 28 percent reaching 1.2 billion.",
            evidence_chunk_ids=["chunk-101"],
        )
    ]

    summary, citations, latency_ms = service.verify_claims(
        claims=claims,
        evidence_chunks=sample_evidence_chunks,
    )

    assert summary.status == "verified"
    assert summary.supported_claims == 1
    assert summary.unsupported_claims == 0
    assert len(summary.details) == 1
    assert summary.details[0].status == "supported"
    assert summary.details[0].support_score >= 0.50
    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk-101"
    assert "Page: 3" in citations[0].formatted_citation
    assert latency_ms >= 0.0


def test_citation_verification_unsupported_claim(sample_evidence_chunks):
    service = CitationVerificationService()
    claims = [
        ClaimItem(
            claim_id="C1",
            text="The company acquired a quantum computing hardware manufacturer in Japan.",
            evidence_chunk_ids=["chunk-101"],
        )
    ]

    summary, citations, _ = service.verify_claims(
        claims=claims,
        evidence_chunks=sample_evidence_chunks,
    )

    assert summary.status == "unverified"
    assert summary.supported_claims == 0
    assert summary.unsupported_claims == 1
    assert summary.details[0].status == "unsupported"


def test_citation_verification_partially_supported(sample_evidence_chunks):
    service = CitationVerificationService()
    claims = [
        ClaimItem(
            claim_id="C1",
            text="Server consolidation reduced total electrical consumption and water cooling demands.",
            evidence_chunk_ids=["chunk-102"],
        )
    ]

    summary, _, _ = service.verify_claims(
        claims=claims,
        evidence_chunks=sample_evidence_chunks,
    )

    # Some terms match (server, consolidation, reduced, electrical, consumption), but water cooling does not
    assert summary.status in ("verified", "partially_verified")
    assert summary.details[0].status in ("supported", "partially_supported")


def test_citation_verification_missing_chunk_id(sample_evidence_chunks):
    service = CitationVerificationService()
    claims = [
        ClaimItem(
            claim_id="C1",
            text="Cloud annual recurring revenue grew by 28 percent.",
            evidence_chunk_ids=["hallucinated-chunk-999"],
        )
    ]

    summary, _, _ = service.verify_claims(
        claims=claims,
        evidence_chunks=sample_evidence_chunks,
    )

    assert summary.status == "unverified"
    assert summary.unsupported_claims == 1
    assert summary.details[0].status == "unsupported"
    assert "hallucinated-chunk-999" in summary.details[0].missing_chunk_ids
    assert "Hallucinated or unknown" in summary.details[0].rationale


def test_citation_verification_empty_claims(sample_evidence_chunks):
    service = CitationVerificationService()
    summary, citations, _ = service.verify_claims(
        claims=[],
        evidence_chunks=sample_evidence_chunks,
    )

    assert summary.status == "unverified"
    assert summary.total_claims == 0
    assert summary.supported_claims == 0


def test_citation_mapping_metadata_preservation(sample_evidence_chunks):
    service = CitationVerificationService()
    claims = [
        ClaimItem(
            claim_id="C1",
            text="Server consolidation reduced electrical consumption.",
            evidence_chunk_ids=["chunk-102"],
        )
    ]

    _, citations, _ = service.verify_claims(
        claims=claims,
        evidence_chunks=sample_evidence_chunks,
    )

    assert len(citations) == 1
    cit = citations[0]
    assert cit.document_id == "doc-operations-2024"
    assert cit.page_number == 8
    assert cit.section_title == "Server Optimization"
    assert cit.section_path == "Operations > Server Optimization"
    assert cit.score == 0.85
    assert "Doc: doc-oper" in cit.formatted_citation
    assert "Page: 8" in cit.formatted_citation
    assert "Section: Operations > Server Optimization" in cit.formatted_citation


def test_citation_verification_factory():
    svc1 = get_citation_verification_service()
    svc2 = get_citation_verification_service()
    assert svc1 is svc2
