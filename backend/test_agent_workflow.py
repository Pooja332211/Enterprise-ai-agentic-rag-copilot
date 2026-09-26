import asyncio
import json
from backend.app.agents.agent_service import get_agent_research_service
from backend.app.models.schemas import AgentResearchRequest

async def test_agent():
    svc = get_agent_research_service()
    req = AgentResearchRequest(
        query="Compare the main objectives of the enterprise knowledge platform with how it retrieves information for complex questions.",
        top_k=5,
        candidate_top_k=10,
        rerank=True,
        max_iterations=2,
    )
    res = await svc.execute_agent_research(req)
    print("STATUS:", res.status)
    print("TOTAL DURATION MS:", res.total_duration_ms)
    print("ITERATIONS USED:", res.iterations_used)
    print("PROVIDER:", res.provider)
    print("SUB-QUERIES COUNT:", len(res.sub_queries))
    for idx, sq in enumerate(res.sub_queries):
        print(f" - SQ #{idx+1}: ID={sq.get('id') or sq.get('sub_query_id')} | Query='{sq.get('query') or sq.get('sub_query_text')}'")
    print("\nFINAL ANSWER:\n", res.final_answer)
    print("\nEXECUTION TRACE (count):", len(res.execution_trace))
    for t in res.execution_trace:
        print(f" - Node: {t.get('node')} | Latency: {t.get('duration_ms')} ms | Summary: {t.get('output_summary')}")
    print("\nCITATIONS COUNT:", len(res.citations))

if __name__ == "__main__":
    asyncio.run(test_agent())
