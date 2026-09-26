import asyncio
import json
from backend.app.services.hybrid_retrieval_service import get_hybrid_retrieval_service
from backend.app.services.research_service import get_research_service
from backend.app.models.schemas import ResearchRequest

async def main():
    retriever = get_hybrid_retrieval_service()
    res = await retriever.search(
        query="What retrieval methods does Company ABC use?",
        mode="hybrid",
        top_k=5,
        candidate_top_k=10,
        rerank=True,
        context_assembly=True,
    )
    print("TOTAL RESULTS:", len(res.get("results", [])))
    for idx, r in enumerate(res.get("results", [])):
        print(f"[{idx+1}] Score: {r.get('score', 0):.4f} Doc: {r.get('document_name')} ChunkID: {r.get('chunk_id')}")
        print(f"Text: {r.get('text', '')}\n")
        
    print("=" * 60)
    print("ASSEMBLED CONTEXT:")
    assembled = res.get("assembled_context") or {}
    print(assembled.get("context_text"))
    print("=" * 60)

    research_svc = get_research_service()
    req = ResearchRequest(
        query="What retrieval methods does Company ABC use?",
        top_k=5,
        candidate_top_k=10,
        rerank=True,
    )
    resp = await research_svc.execute_research(req)
    print("RESEARCH RESPONSE:")
    print("Answer:", resp.answer)
    print("Claims:", json.dumps([c.model_dump() for c in resp.claims], indent=2))
    print("Insufficient:", resp.insufficient_evidence)

if __name__ == "__main__":
    asyncio.run(main())
