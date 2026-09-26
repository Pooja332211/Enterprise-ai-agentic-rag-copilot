import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
import httpx
from backend.app.config.settings import Settings, get_settings
from backend.app.utils.logger import logger


class BaseLLMService(ABC):
    """
    Abstract interface for Large Language Model generation providers.
    Supports standard generation, system prompt enforcement, temperature/token controls,
    and structured JSON output formatting.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the provider identifier (e.g. 'openai', 'anthropic', 'mock')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the model identifier (e.g. 'gpt-4o-mini', 'claude-3-5-sonnet')."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> str:
        """
        Executes text generation for given prompt and optional system context.
        """
        pass


class MockLLMService(BaseLLMService):
    """
    Deterministic mock LLM provider for zero-cost offline testing, CI pipelines,
    and fallback execution. Evaluates retrieved evidence blocks in the prompt
    and synthesizes verifiable, grounded structured JSON responses.
    """

    def __init__(
        self,
        model_name: str = "mock-grounded-model",
        simulate_error: Optional[str] = None,
        simulate_malformed: bool = False,
    ):
        self._model_name = model_name
        self.simulate_error = simulate_error
        self.simulate_malformed = simulate_malformed

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> str:
        if self.simulate_error:
            raise RuntimeError(f"Simulated LLM Provider Error: {self.simulate_error}")

        if self.simulate_malformed:
            return "This is a malformed raw string that is not valid JSON {"

        # Extract user research query from prompt
        user_q = ""
        if "User Research Question:" in prompt:
            parts = prompt.split("User Research Question:")[1]
            if "Supplied Context Blocks" in parts:
                user_q = parts.split("Supplied Context Blocks")[0].strip()
            elif "<context>" in parts:
                user_q = parts.split("<context>")[0].strip()
            else:
                user_q = parts.split("\n\n")[0].strip()
        else:
            first_line = prompt.split("\n")[0].strip()
            if not first_line.startswith("[Source:"):
                user_q = first_line

        # Extract context segment cleanly
        context_segment = ""
        if "<context>" in prompt and "</context>" in prompt:
            m = re.search(r"<context>(.*?)</context>", prompt, flags=re.DOTALL)
            if m:
                context_segment = m.group(1).strip()
        elif "Supplied Context Blocks" in prompt:
            parts = prompt.split("Supplied Context Blocks")
            raw_ctx = parts[1]
            if "Provide your grounded synthesis" in raw_ctx:
                raw_ctx = raw_ctx.split("Provide your grounded synthesis")[0]
            context_segment = raw_ctx.strip()
        else:
            context_segment = prompt

        raw_blocks = [b.strip() for b in re.split(r"(?:\n\s*---\s*\n|\Z)", context_segment) if b.strip()]
        parsed_blocks = []
        for blk in raw_blocks:
            cid = None
            cid_match = re.search(r"Chunk:\s*([^\s\|\]]+)", blk, flags=re.IGNORECASE)
            if cid_match:
                cid = cid_match.group(1).strip()
            else:
                ev_match = re.search(r"Evidence Chunk IDs?:\s*([^\s\n]+)", blk, flags=re.IGNORECASE)
                if ev_match:
                    cid = ev_match.group(1).strip()
                else:
                    uuid_match = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", blk)
                    if uuid_match:
                        cid = uuid_match.group(1)
                    else:
                        alt_match = re.search(r"\b(chunk-?\d+|c\d+|dup-\d+)\b", blk, flags=re.IGNORECASE)
                        if alt_match:
                            cid = alt_match.group(1)

            block_score = 0.5
            score_match = re.search(r"Score:\s*([0-9\.]+)", blk)
            if score_match:
                try:
                    block_score = float(score_match.group(1))
                except ValueError:
                    pass

            content_text = re.sub(r"\[Source:[^\]]+\]", "", blk).strip()
            content_text = re.sub(r"</?context>", "", content_text).strip()
            if content_text:
                parsed_blocks.append({"chunk_id": cid, "score": block_score, "text": content_text})

        # Empty context check
        if not parsed_blocks or "no evidence" in prompt.lower() or "empty" in prompt.lower():
            return json.dumps({
                "answer": "The available documents do not contain sufficient evidence to answer this query.",
                "claims": [],
                "insufficient_evidence": True,
            }, indent=2)

        # Distractor filtering: if top chunk has high relevance, discard distant low-scoring noise (< 5% of top score)
        max_score = max(b["score"] for b in parsed_blocks)
        if max_score >= 0.2:
            relevance_cutoff = max(0.02, max_score * 0.05)
            parsed_blocks = [b for b in parsed_blocks if b["score"] >= relevance_cutoff]

        stopwords = {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "the", "a", "an", "and", "or", "but", "in", "on",
            "at", "to", "for", "with", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "from", "up", "down", "out",
            "over", "under", "again", "further", "then", "once", "here", "there", "all",
            "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
            "will", "just", "should", "now", "tell", "show", "give", "explain", "describe",
            "according", "stated", "using", "used", "does", "provide", "please", "could",
            "would", "across", "between", "query", "question", "use", "uses"
        }
        entity_stopwords = {
            "company", "enterprise", "knowledge", "copilot", "system", "platform",
            "document", "documents", "section", "page", "source", "chunk", "report",
            "manual", "corp", "inc", "ltd", "abc", "employee", "employees", "user", "users"
        }

        def get_stem(w: str) -> str:
            w = w.lower()
            return re.sub(r"(ing|tion|tions|ed|es|s|al|als|ive|ives|ment|ments)$", "", w).rstrip("e")

        all_q_tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", user_q.lower())
        content_words = [w for w in all_q_tokens if w not in stopwords]

        intent_words = set([w for w in content_words if w not in entity_stopwords])
        entity_words = set([w for w in content_words if w in entity_stopwords])

        if not intent_words:
            intent_words = set(content_words)

        intent_stems = {get_stem(w) for w in intent_words if len(w) > 2}
        entity_stems = {get_stem(w) for w in entity_words if len(w) > 2}

        all_content = " ".join([b["text"] for b in parsed_blocks])
        all_content_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", all_content.lower()))
        all_content_stems = {get_stem(w) for w in all_content_tokens if len(w) > 2}

        intent_overlap = (intent_words & all_content_tokens) | (intent_stems & all_content_stems)

        # Strict evidence isolation: if the query intent has zero textual presence in the evidence, abstain
        if len(intent_words) > 0 and len(intent_overlap) == 0:
            return json.dumps({
                "answer": "The available documents do not contain sufficient evidence to answer this query.",
                "claims": [],
                "insufficient_evidence": True,
            }, indent=2)

        def extract_candidate_sentences(text: str) -> List[str]:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            candidates = []
            current_header = ""
            for l in lines:
                if l.startswith("#"):
                    current_header = re.sub(r"^#+\s*", "", l).strip()
                    continue
                l_clean = re.sub(r"^[\*\-]\s*", "", l)
                l_clean = re.sub(r"^\d+\.\s*", "", l_clean).strip()
                if len(l_clean) < 35 and not any(l_clean.endswith(p) for p in [".", "!", "?"]) and (" and " in l_clean or len(l_clean.split()) <= 4):
                    current_header = l_clean
                    continue
                for part in re.split(r"(?<=[.!?])\s+", l_clean):
                    part_clean = part.strip().rstrip(".")
                    if len(part_clean) > 18 and not part_clean.lower().startswith("evidence chunk"):
                        if current_header and (part_clean.lower().startswith(("improve", "reduce", "provide", "prevent", "enable", "ensure")) or len(part_clean.split()) <= 6):
                            candidates.append(f"{current_header}: {part_clean}")
                        else:
                            candidates.append(part_clean)
            return candidates

        def score_candidate(cand: str, blk_score: float):
            words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", cand.lower()))
            stems = {get_stem(w) for w in words if len(w) > 2}

            exact_intent = words & intent_words
            stem_intent = stems & intent_stems
            exact_entity = words & entity_words
            stem_entity = stems & entity_stems

            intent_points = len(exact_intent) * 10 + len(stem_intent) * 6
            entity_points = len(exact_entity) * 2 + len(stem_entity) * 1

            if intent_points == 0:
                return 0.0, set()

            total = float(intent_points + entity_points) * (0.5 + 0.5 * blk_score)
            matched_intent = exact_intent | stem_intent
            return total, matched_intent

        all_candidates = []
        for block in parsed_blocks:
            cands = extract_candidate_sentences(block["text"])
            for c in cands:
                sc, matched_int = score_candidate(c, block["score"])
                if sc > 0:
                    all_candidates.append({
                        "score": sc,
                        "text": c,
                        "chunk_id": block["chunk_id"],
                        "matched_intent": matched_int
                    })

        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        if not all_candidates:
            return json.dumps({
                "answer": "The available documents do not contain sufficient evidence to answer this query.",
                "claims": [],
                "insufficient_evidence": True,
            }, indent=2)

        selected = []
        covered_intent = set()
        seen_texts = set()

        for cand in all_candidates:
            cand_norm = cand["text"].lower()
            if any(cand_norm in prev or prev in cand_norm for prev in seen_texts):
                continue
            new_intent = cand["matched_intent"] - covered_intent
            if new_intent or not selected:
                selected.append(cand)
                seen_texts.add(cand_norm)
                covered_intent.update(cand["matched_intent"])
                if len(selected) >= 3:
                    break

        # Pass 2: If we have 1 candidate, only add a second candidate if it comes
        # from the exact same chunk (e.g. adjacent pipeline step in the same document)
        if len(selected) == 1 and len(all_candidates) > 1:
            primary_chunk_id = selected[0]["chunk_id"]
            top_cand_score = all_candidates[0]["score"]
            for cand in all_candidates[1:]:
                cand_norm = cand["text"].lower()
                if any(cand_norm in prev or prev in cand_norm for prev in seen_texts):
                    continue
                if cand["chunk_id"] == primary_chunk_id and cand["score"] >= top_cand_score * 0.35:
                    selected.append(cand)
                    seen_texts.add(cand_norm)
                    break

        claims = []
        for idx, item in enumerate(selected, start=1):
            sent = item["text"].strip().rstrip(".")
            claims.append({
                "claim_id": f"C{idx}",
                "text": f"{sent}.",
                "evidence_chunk_ids": [item["chunk_id"]] if item["chunk_id"] else []
            })

        answer_text = " ".join([c["text"] for c in claims])

        return json.dumps({
            "answer": answer_text,
            "claims": claims,
            "insufficient_evidence": False,
        }, indent=2)


class OpenAILLMService(BaseLLMService):
    """
    OpenAI and OpenAI-compatible REST API client (vLLM, Ollama, LiteLLM).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o-mini",
        timeout: float = 30.0,
    ):
        self._api_key = api_key or ""
        self._api_base = api_base.rstrip("/")
        self._model_name = model_name
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> str:
        if not self._api_key:
            raise ValueError("OpenAI API key is not configured. Set LLM_API_KEY in environment or .env.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.0,
            "max_tokens": max_tokens if max_tokens is not None else 2048,
        }

        if response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self._api_base}/chat/completions"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenAI API call failed with status {resp.status_code}: {resp.text}"
                )
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class AnthropicLLMService(BaseLLMService):
    """
    Anthropic Claude Messages API client.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://api.anthropic.com",
        model_name: str = "claude-3-5-sonnet-20241022",
        timeout: float = 30.0,
    ):
        self._api_key = api_key or ""
        self._api_base = api_base.rstrip("/")
        self._model_name = model_name
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> str:
        if not self._api_key:
            raise ValueError("Anthropic API key is not configured. Set ANTHROPIC_API_KEY in environment or .env.")

        payload: Dict[str, Any] = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens if max_tokens is not None else 2048,
            "temperature": temperature if temperature is not None else 0.0,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        url = f"{self._api_base}/v1/messages"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Anthropic API call failed with status {resp.status_code}: {resp.text}"
                )
            data = resp.json()
            return data["content"][0]["text"]


_llm_service_instance: Optional[BaseLLMService] = None


def get_llm_service(
    settings: Optional[Settings] = None,
    provider: Optional[str] = None,
) -> BaseLLMService:
    """
    Returns configured or singleton BaseLLMService instance.
    Falls back gracefully to MockLLMService if requested credentials are missing.
    """
    global _llm_service_instance
    cfg = settings or get_settings()
    prov = (provider or cfg.LLM_PROVIDER).lower()

    if prov == "mock":
        return MockLLMService(model_name=cfg.LLM_MODEL)

    if prov == "openai":
        if not cfg.LLM_API_KEY:
            logger.warning("LLM_API_KEY not set for OpenAI. Falling back to MockLLMService.")
            return MockLLMService(model_name=cfg.LLM_MODEL)
        return OpenAILLMService(
            api_key=cfg.LLM_API_KEY,
            api_base=cfg.LLM_API_BASE or "https://api.openai.com/v1",
            model_name=cfg.LLM_MODEL,
            timeout=cfg.LLM_TIMEOUT,
        )

    if prov == "anthropic":
        if not cfg.ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set. Falling back to MockLLMService.")
            return MockLLMService(model_name=cfg.LLM_MODEL)
        return AnthropicLLMService(
            api_key=cfg.ANTHROPIC_API_KEY,
            api_base=cfg.ANTHROPIC_API_BASE or "https://api.anthropic.com",
            model_name=cfg.LLM_MODEL,
            timeout=cfg.LLM_TIMEOUT,
        )

    if prov == "local":
        return OpenAILLMService(
            api_key=cfg.LLM_API_KEY or "local-key",
            api_base=cfg.LLM_API_BASE or "http://localhost:11434/v1",
            model_name=cfg.LLM_MODEL,
            timeout=cfg.LLM_TIMEOUT,
        )

    logger.warning(f"Unknown LLM provider '{prov}'. Defaulting to MockLLMService.")
    return MockLLMService()


def reset_llm_service() -> None:
    """Resets cached LLM singleton instance (useful for test harnesses)."""
    global _llm_service_instance
    _llm_service_instance = None
