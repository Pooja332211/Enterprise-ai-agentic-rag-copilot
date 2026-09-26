import math
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from backend.app.cache.reranker_cache import RerankerCache, get_reranker_cache
from backend.app.config.settings import Settings, get_settings
from backend.app.utils.logger import logger


class BaseReranker(ABC):
    """
    Abstract interface for cross-encoder reranking models.
    Evaluates joint relevance of (query, candidate_chunk) pairs.
    """

    def __init__(self, cache: Optional[RerankerCache] = None):
        self._cache = cache if cache is not None else get_reranker_cache()

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the reranker model identifier."""
        pass

    def warmup(self) -> None:
        """Optional warmup hook for inference runtimes."""
        pass

    @abstractmethod
    def _score_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[float]:
        """Computes joint relevance scores in [0.0, 1.0] for a batch of candidates."""
        pass

    def rerank(
        self,
        query: str,
        candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: Optional[int] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidate chunks based on joint query-document relevance.
        Preserves original retrieval metadata, scores, and ranks.
        Employs caching to avoid redundant cross-encoder inferences.
        """
        cand_list = candidates if candidates is not None else (chunks or [])
        if not cand_list or not query:
            return []

        # 1. Check cache for candidate scores
        hits, misses = self._cache.get_batch(query, cand_list, self.model_name)

        # 2. Compute scores for cache misses
        newly_scored: List[Tuple[Dict[str, Any], float]] = []
        if misses:
            miss_candidates = [cand for _, cand in misses]
            scores = self._score_candidates(query, miss_candidates)
            for (idx, cand), score in zip(misses, scores):
                hits[idx] = score
                newly_scored.append((cand, score))
            # Populate cache
            self._cache.set_batch(query, newly_scored, self.model_name)

        # 3. Assemble reranked items preserving metadata
        reranked_items: List[Dict[str, Any]] = []
        for idx, cand in enumerate(cand_list):
            prob_score = hits.get(idx, 0.0)
            item = dict(cand)
            orig_score = cand.get("score")
            orig_rank = cand.get("dense_rank") or cand.get("bm25_rank") or (idx + 1)

            item["retrieval_score"] = orig_score
            item["retrieval_rank"] = orig_rank
            item["rerank_score"] = round(float(prob_score), 6)
            item["score"] = round(float(prob_score), 6)
            reranked_items.append(item)

        # 4. Sort descending by rerank score
        reranked_items.sort(key=lambda x: x["rerank_score"], reverse=True)

        for rank, item in enumerate(reranked_items, start=1):
            item["final_rank"] = rank

        limit = top_k if top_k is not None else len(reranked_items)
        return reranked_items[:limit]


class BGERerankerService(BaseReranker):
    """
    Local CPU-friendly cross-encoder using BAAI/bge-reranker-base via ONNX Runtime.
    Eliminates PyTorch/GPU dependencies while delivering state-of-the-art reranking.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 32,
        cache: Optional[RerankerCache] = None,
    ):
        super().__init__(cache=cache)
        self._model_name = model_name
        self._batch_size = batch_size
        self._session = None
        self._tokenizer = None
        self._initialize_onnx()

    def _initialize_onnx(self) -> None:
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            repo_id = "Xenova/bge-reranker-base"
            tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            # Dynamic padding to batch maximum sequence length (optimizes CPU latency)
            self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            self._tokenizer.enable_truncation(max_length=512)

            # Try quantized model first for speed and smaller memory, fallback to standard onnx
            try:
                model_path = hf_hub_download(repo_id=repo_id, filename="onnx/model_quantized.onnx")
            except Exception:
                model_path = hf_hub_download(repo_id=repo_id, filename="onnx/model.onnx")

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            logger.info(f"Initialized BGE Reranker ONNX model '{self._model_name}' successfully.")
        except Exception as exc:
            logger.warning(
                f"Could not load ONNX BGE reranker ({exc}). Falling back to MockRerankerService for offline resilience."
            )
            raise

    @property
    def model_name(self) -> str:
        return self._model_name

    def warmup(self) -> None:
        """Pre-runs a minimal inference pass to warm up JIT, ONNX session threads, and memory allocators."""
        if self._session is not None and self._tokenizer is not None:
            try:
                self._score_candidates("warmup query", [{"text": "warmup document text", "chunk_id": "warmup_0"}])
                logger.info("BGE Reranker warmup completed.")
            except Exception as exc:
                logger.warning(f"Reranker warmup failed (non-fatal): {exc}")

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Applies standard sigmoid function to project logits to [0, 1]."""
        return 1.0 / (1.0 + math.exp(-x))

    def _score_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[float]:
        if not candidates or not query:
            return []

        import numpy as np

        scores_out: List[float] = []
        pairs = [(query, c.get("text", "")) for c in candidates]

        for i in range(0, len(pairs), self._batch_size):
            batch_pairs = pairs[i : i + self._batch_size]

            # Tokenize pairs: (query, text)
            encodings = self._tokenizer.encode_batch(batch_pairs)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            # Add token_type_ids if required by the model graph
            model_inputs = [inp.name for inp in self._session.get_inputs()]
            if "token_type_ids" in model_inputs:
                type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
                inputs["token_type_ids"] = type_ids

            outputs = self._session.run(None, inputs)
            logits = outputs[0]

            # Flatten logits
            if logits.ndim > 1:
                batch_scores = logits[:, 0] if logits.shape[1] == 1 else logits[:, -1]
            else:
                batch_scores = logits

            for s in batch_scores:
                scores_out.append(self._sigmoid(float(s)))

        return scores_out


class MockRerankerService(BaseReranker):
    """
    Deterministic cross-encoder simulator for testing and offline execution.
    Evaluates joint relevance using term co-occurrence, exact phrase match,
    and section header alignment with sub-millisecond execution.
    """

    def __init__(
        self,
        model_name: str = "mock-reranker-deterministic",
        cache: Optional[RerankerCache] = None,
    ):
        super().__init__(cache=cache)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def _score_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[float]:
        query_lower = query.lower().strip()
        query_tokens = set(re.findall(r"\b\w+\b", query_lower))

        scores: List[float] = []
        for original_rank, c in enumerate(candidates, start=1):
            text_lower = c.get("text", "").lower()
            doc_tokens = set(re.findall(r"\b\w+\b", text_lower))

            # 1. Term overlap (Jaccard similarity)
            overlap = len(query_tokens.intersection(doc_tokens))
            jaccard = overlap / len(query_tokens.union(doc_tokens)) if query_tokens else 0.0

            # 2. Exact phrase bonus
            phrase_bonus = 0.35 if query_lower in text_lower else 0.0

            # 3. Section title alignment bonus
            sec_bonus = 0.0
            sec_title = (c.get("section_title") or "").lower()
            if sec_title and any(t in sec_title for t in query_tokens):
                sec_bonus = 0.20

            # 4. Integrate normalized retrieval score
            orig_score = c.get("score") or 0.0
            norm_retrieval = min(max(float(orig_score), 0.0), 1.0) if orig_score <= 1.0 else 0.5

            # Combined deterministic score in [0.0, 1.0]
            simulated_score = min(
                round(0.40 * jaccard + 0.30 * phrase_bonus + 0.15 * sec_bonus + 0.15 * norm_retrieval, 6),
                1.0,
            )
            scores.append(simulated_score)

        return scores


_reranker_instance: Optional[BaseReranker] = None


def get_reranker_service(
    settings: Optional[Settings] = None,
    provider: Optional[str] = None,
) -> BaseReranker:
    """
    Factory creating or returning the configured reranker singleton.
    Falls back gracefully to MockRerankerService if BGE ONNX fails to load.
    """
    global _reranker_instance
    if _reranker_instance is not None and provider is None:
        return _reranker_instance

    cfg = settings or get_settings()
    active_provider = (provider or cfg.RERANKER_PROVIDER).lower()

    if active_provider in ("bge", "onnx"):
        try:
            instance = BGERerankerService(
                model_name=cfg.RERANKER_MODEL,
                batch_size=cfg.RERANKER_BATCH_SIZE,
            )
        except Exception as exc:
            logger.warning(
                f"BGEReranker initialization failed ({exc}). Falling back to MockRerankerService for resilience."
            )
            instance = MockRerankerService(model_name=f"fallback-{cfg.RERANKER_MODEL}")
    elif active_provider == "mock":
        instance = MockRerankerService(model_name="mock-reranker-deterministic")
    else:
        raise ValueError(f"Unknown reranker provider: '{active_provider}'. Supported: 'bge', 'mock'")

    if provider is None:
        _reranker_instance = instance
    return instance
