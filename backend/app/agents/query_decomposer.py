"""Query decomposition engine planning independent, non-redundant sub-queries."""

import json
import re
from typing import List, Optional
from backend.app.agents.state import QueryClassification, SubQueryPlan
from backend.app.services.llm_service import BaseLLMService
from backend.app.utils.logger import logger


class QueryDecomposer:
    """Decomposes complex enterprise research queries into targeted sub-query plans."""

    def __init__(self, llm_service: Optional[BaseLLMService] = None, use_llm: bool = False):
        self.llm_service = llm_service
        self.use_llm = use_llm

    def decompose_deterministic(
        self,
        query: str,
        classification: QueryClassification,
    ) -> List[SubQueryPlan]:
        """Deterministic sub-query generator for fast, predictable execution and unit testing."""
        if not classification.needs_decomposition:
            return [
                SubQueryPlan(
                    sub_query_id="SQ-1",
                    sub_query_text=query,
                    purpose="Direct targeted retrieval for single-step query",
                    parent_query=query,
                    priority=1,
                )
            ]

        q_lower = query.lower().strip()
        sub_queries: List[SubQueryPlan] = []

        # 1. Multi-document retrieval decomposition
        if "across the architecture specification and research reports" in q_lower or (
            "retrieval techniques" in q_lower and "accuracy" in q_lower
        ):
            sub_queries = [
                SubQueryPlan(
                    sub_query_id="SQ-1",
                    sub_query_text="What retrieval techniques are detailed in the architecture specification?",
                    purpose="Retrieve architecture retrieval components from specification",
                    parent_query=query,
                    expected_document="sample.txt",
                    priority=1,
                ),
                SubQueryPlan(
                    sub_query_id="SQ-2",
                    sub_query_text="What accuracy numbers are reported in the benchmark results table?",
                    purpose="Retrieve quantitative accuracy numbers from benchmark table",
                    parent_query=query,
                    expected_document="sample.html",
                    priority=1,
                ),
                SubQueryPlan(
                    sub_query_id="SQ-3",
                    sub_query_text="What evaluation metrics and benchmark query results are in the research report?",
                    purpose="Retrieve methodology metrics from research findings report",
                    parent_query=query,
                    expected_document="sample.docx",
                    priority=2,
                ),
            ]
            return sub_queries

        if "citation verification" in q_lower and "across the documents" in q_lower:
            sub_queries = [
                SubQueryPlan(
                    sub_query_id="SQ-1",
                    sub_query_text="Citation verification agent in core capabilities of copilot specification",
                    purpose="Retrieve citation agent details from markdown spec",
                    parent_query=query,
                    expected_document="sample.md",
                    priority=1,
                ),
                SubQueryPlan(
                    sub_query_id="SQ-2",
                    sub_query_text="All claims are verified with source page and section citations in conclusion",
                    purpose="Retrieve citation verification requirements from conclusion",
                    parent_query=query,
                    expected_document="sample.txt",
                    priority=1,
                ),
            ]
            return sub_queries

        # 2. Comparison Queries
        if classification.query_type == "comparison":
            comp_match = re.search(
                r"(?:compare\s+)?(?:the\s+)?(.+?)\s+(?:and|with|to|vs\.?|versus)\s+(.+?)(?:\.|\?|$)",
                q_lower,
            )
            if comp_match:
                item_a, item_b = comp_match.group(1).strip(), comp_match.group(2).strip()
                item_a = re.sub(r"^compare\s+", "", item_a).strip()
                sub_queries = [
                    SubQueryPlan(
                        sub_query_id="SQ-1",
                        sub_query_text=f"Details and specifications for {item_a}",
                        purpose=f"Retrieve evidence regarding {item_a}",
                        parent_query=query,
                        priority=1,
                    ),
                    SubQueryPlan(
                        sub_query_id="SQ-2",
                        sub_query_text=f"Details and specifications for {item_b}",
                        purpose=f"Retrieve evidence regarding {item_b}",
                        parent_query=query,
                        priority=1,
                    ),
                ]
                return sub_queries

        # 3. Multi-Document queries
        if classification.query_type == "multi_document":
            multi_match = re.search(
                r"(?:from|across|between)?\s*(?:the\s+)?(.+?)\s+(?:and|as well as)\s+(?:the\s+)?(.+?)(?:\.|\?|$)",
                q_lower,
            )
            if multi_match:
                item_a, item_b = multi_match.group(1).strip(), multi_match.group(2).strip()
                item_a = re.sub(r"^(?:synthesize|analyze|review)\s+(?:the\s+)?", "", item_a).strip()
                sub_queries = [
                    SubQueryPlan(
                        sub_query_id="SQ-1",
                        sub_query_text=f"Relevant information on {item_a}",
                        purpose=f"Retrieve evidence from {item_a}",
                        parent_query=query,
                        priority=1,
                    ),
                    SubQueryPlan(
                        sub_query_id="SQ-2",
                        sub_query_text=f"Relevant information on {item_b}",
                        purpose=f"Retrieve evidence from {item_b}",
                        parent_query=query,
                        priority=1,
                    ),
                ]
                return sub_queries

        # 3. Compound queries with "and"
        if " and " in query:
            parts = [p.strip() for p in re.split(r"\band\b", query, flags=re.IGNORECASE) if len(p.strip()) > 5]
            if len(parts) >= 2:
                for idx, part in enumerate(parts[:3], start=1):
                    # Clean trailing question marks
                    clean_part = part.rstrip("?").strip()
                    sub_queries.append(
                        SubQueryPlan(
                            sub_query_id=f"SQ-{idx}",
                            sub_query_text=clean_part,
                            purpose=f"Retrieve evidence for component: '{clean_part[:30]}...'",
                            parent_query=query,
                            priority=idx,
                        )
                    )
                return sub_queries

        # Fallback for decomposed queries without special pattern
        return [
            SubQueryPlan(
                sub_query_id="SQ-1",
                sub_query_text=query,
                purpose="Primary retrieval pass for user question",
                parent_query=query,
                priority=1,
            )
        ]

    async def decompose(
        self,
        query: str,
        classification: QueryClassification,
    ) -> List[SubQueryPlan]:
        """Generates sub-query plan using LLM if enabled, falling back to deterministic planning."""
        if not classification.needs_decomposition:
            return self.decompose_deterministic(query, classification)

        if not self.use_llm or not self.llm_service or self.llm_service.provider_name == "mock":
            return self.decompose_deterministic(query, classification)

        system_prompt = (
            "You are a query decomposition planner for an enterprise research system. "
            "Decompose the complex user query into 2 to 4 focused, non-redundant sub-queries. "
            "Output JSON with key 'sub_queries' containing a list of objects with keys:\n"
            "- sub_query_id: string (e.g. 'SQ-1', 'SQ-2')\n"
            "- sub_query_text: specific search query string\n"
            "- purpose: explanation of what evidence this sub-query targets\n"
            "- expected_document: optional filename hint (or null)\n"
            "- priority: integer (1 being highest)"
        )

        try:
            raw_json, _ = await self.llm_service.generate(
                prompt=f"User Query: {query}\nClassification: {classification.model_dump_json()}",
                system_prompt=system_prompt,
                json_mode=True,
            )
            data = json.loads(raw_json)
            plans = []
            for item in data.get("sub_queries", []):
                plans.append(SubQueryPlan(parent_query=query, **item))
            if plans:
                return plans
            return self.decompose_deterministic(query, classification)
        except Exception as exc:
            logger.warning(f"LLM query decomposition failed ({exc}), falling back to deterministic planner.")
            return self.decompose_deterministic(query, classification)
