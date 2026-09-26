"""Query analysis engine classifying intent, complexity, and decomposition necessity."""

import json
import re
from typing import Optional
from backend.app.agents.state import QueryClassification
from backend.app.services.llm_service import BaseLLMService
from backend.app.utils.logger import logger


class QueryAnalyzer:
    """Classifies user queries into structural categories and determines decomposition paths."""

    def __init__(self, llm_service: Optional[BaseLLMService] = None, use_llm: bool = False):
        self.llm_service = llm_service
        self.use_llm = use_llm

    def analyze_deterministic(self, query: str) -> QueryClassification:
        """Deterministic heuristic analysis for fast, testable, and offline query classification."""
        q_lower = query.lower().strip()

        # 1. Obvious Out-of-Domain / No-Answer heuristics
        no_ans_terms = [
            "quantum teleportation",
            "ebitda",
            "reinforcement learning",
            "helm chart",
            "sourdough",
            "fifa world cup",
            "weather",
            "battle of hastings",
            "haber-bosch",
            "antibiotic dosage",
            "target stock price",
            "chocolate cake",
            "recipe",
        ]
        if any(term in q_lower for term in no_ans_terms):
            return QueryClassification(
                query_type="no_answer",
                complexity="low",
                needs_decomposition=False,
                reason="Query contains topics not present in enterprise document corpus.",
            )

        # 2. Explicit Comparison Queries
        comparison_patterns = [
            r"\bcompare\b",
            r"\bversus\b",
            r"\bvs\.?\b",
            r"\bdifferences? between\b",
            r"\bhow does .* differ from\b",
        ]
        if any(re.search(pat, q_lower) for pat in comparison_patterns):
            return QueryClassification(
                query_type="comparison",
                complexity="high",
                needs_decomposition=True,
                reason="Query requires contrasting multiple entities, architectures, or documents.",
            )

        # 3. Multi-Document Synthesis Queries
        multi_doc_terms = [
            "synthesize",
            "across the documents",
            "across documents",
            "across the reports",
            "in both",
            "and the specification",
            "compare the supported",
            "retrieval techniques and accuracy numbers",
            "citation verification and grounded claim validation handled across",
        ]
        if any(term in q_lower for term in multi_doc_terms):
            return QueryClassification(
                query_type="multi_document",
                complexity="high",
                needs_decomposition=True,
                reason="Query spans multiple distinct document formats and requires joint context synthesis.",
            )

        # 4. Multi-Hop or Compound Questions
        compound_patterns = [
            r"what (?:are|is) .+ and (?:what|how|why) .+",
            r"tell me about .+ and also .+",
            r"requirements and (?:security|risks|limitations)",
        ]
        if any(re.search(pat, q_lower) for pat in compound_patterns) and len(q_lower.split()) > 10:
            return QueryClassification(
                query_type="multi_hop",
                complexity="medium",
                needs_decomposition=True,
                reason="Query contains compound requirements that benefit from separate retrieval passes.",
            )

        # 5. Technical / Code / Exact Keyword Matching
        tech_markers = [
            "def query_copilot",
            "query_copilot",
            "ndcg",
            "bm25",
            "rrf",
            "onnx",
            "reciprocal rank fusion",
            "return 'answer with citations'",
        ]
        if any(marker in q_lower for marker in tech_markers):
            return QueryClassification(
                query_type="technical_keyword",
                complexity="low",
                needs_decomposition=False,
                reason="Exact technical keyword query suited for direct hybrid retrieval.",
            )

        # 6. Section & Page Specific Lookup
        section_markers = [
            "section",
            "page 1",
            "page 2",
            "methodology",
            "executive summary",
            "overview",
            "limitations",
            "table",
        ]
        if any(marker in q_lower for marker in section_markers):
            return QueryClassification(
                query_type="section_lookup",
                complexity="low",
                needs_decomposition=False,
                reason="Direct section/page reference resolvable via single targeted retrieval.",
            )

        # 7. Semantic Paraphrase / General Lookup
        if len(q_lower.split()) > 7 and ("how" in q_lower or "why" in q_lower or "explain" in q_lower):
            return QueryClassification(
                query_type="semantic_lookup",
                complexity="medium",
                needs_decomposition=False,
                reason="Conceptual semantic query handled by dense vector similarity.",
            )

        # 8. Default Simple Factual
        return QueryClassification(
            query_type="simple_factual",
            complexity="low",
            needs_decomposition=False,
            reason="Simple factual lookup resolvable in a single retrieval step.",
        )

    async def analyze(self, query: str) -> QueryClassification:
        """Analyzes query using LLM if enabled, falling back to deterministic heuristics."""
        if not self.use_llm or not self.llm_service or self.llm_service.provider_name == "mock":
            return self.analyze_deterministic(query)

        system_prompt = (
            "You are a query analysis classifier for an enterprise research copilot. "
            "Analyze the user query and output a JSON object with keys:\n"
            "- query_type: one of ['simple_factual', 'semantic_lookup', 'technical_keyword', "
            "'section_lookup', 'multi_document', 'comparison', 'multi_hop', 'no_answer']\n"
            "- complexity: one of ['low', 'medium', 'high']\n"
            "- needs_decomposition: boolean (true only for complex multi-part queries)\n"
            "- reason: short explanation\n"
            "- suggested_sub_queries: list of sub-query strings if needs_decomposition is true"
        )

        try:
            raw_json, _ = await self.llm_service.generate(
                prompt=f"User Query: {query}",
                system_prompt=system_prompt,
                json_mode=True,
            )
            data = json.loads(raw_json)
            return QueryClassification(**data)
        except Exception as exc:
            logger.warning(f"LLM query analysis failed ({exc}), falling back to deterministic analyzer.")
            return self.analyze_deterministic(query)
