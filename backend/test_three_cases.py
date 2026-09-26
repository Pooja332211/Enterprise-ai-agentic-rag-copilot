import asyncio
import re
import json

from backend.app.services.llm_service import MockLLMService
from backend.app.services.research_service import get_research_service
from backend.app.models.schemas import ResearchRequest

# Let's inspect how MockLLMService behaves on queries A, B, and C
async def test_cases():
    research_svc = get_research_service()

    queries = [
        ("Query A", "What retrieval methods does Company ABC use?"),
        ("Query B", "How does Company ABC handle complex questions?"),
        ("Query C", "What is Company ABC's employee vacation allowance?"),
    ]

    for name, q in queries:
        print(f"\n==================== {name}: '{q}' ====================")
        req = ResearchRequest(
            query=q,
            top_k=5,
            candidate_top_k=10,
            rerank=True,
        )
        resp = await research_svc.execute_research(req)
        print("ANSWER:\n", resp.answer)
        print("CLAIMS COUNT:", len(resp.claims))
        for c in resp.claims:
            print(" - Claim:", c.text, "ChunkIDs:", c.evidence_chunk_ids)
        print("CITATIONS COUNT:", len(resp.citations))
        for cit in resp.citations:
            print(" - Citation Doc ID:", cit.document_id, "Section:", cit.section_title, "ChunkID:", cit.chunk_id)
        print("INSUFFICIENT EVIDENCE:", resp.insufficient_evidence)
        print("VERIFICATION STATUS:", resp.verification.status if resp.verification else "N/A")

if __name__ == "__main__":
    asyncio.run(test_cases())
