import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import List, Optional
import httpx
from backend.app.config.settings import Settings, get_settings
from backend.app.utils.logger import logger


class BaseEmbeddingService(ABC):
    """
    Abstract embedding provider interface decoupling the application
    from specific embedding model providers and APIs.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns embedding vector dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns model identifier."""
        pass

    @abstractmethod
    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates dense embeddings for a batch of uncached text chunks."""
        pass

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates dense embeddings for a batch of text chunks with caching."""
        if not texts:
            return []

        from backend.app.cache.embedding_cache import get_embedding_cache
        cache = get_embedding_cache()
        hits, misses = cache.get_batch(texts, self.model_name, self.dimension)

        if not misses:
            return [hits[i] for i in range(len(texts))]

        miss_texts = [text for _, text in misses]
        computed_vectors = self._compute_embeddings(miss_texts)

        cache_items = [(text, vec) for (_, text), vec in zip(misses, computed_vectors)]
        cache.set_batch(cache_items, self.model_name, self.dimension)

        all_results: List[List[float]] = [None] * len(texts)  # type: ignore
        for idx, vec in hits.items():
            all_results[idx] = vec
        for (idx, _), vec in zip(misses, computed_vectors):
            all_results[idx] = vec
        return all_results

    def embed_query(self, query: str) -> List[float]:
        """Generates dense embedding for a single search query."""
        results = self.embed_texts([query])
        if not results:
            return [0.0] * self.dimension
        return results[0]


class FastEmbedEmbeddingService(BaseEmbeddingService):
    """
    Local dense embedding provider using FastEmbed (ONNX Runtime).
    Executes fast, CPU-friendly inference without PyTorch dependencies.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dimension: int = 384, batch_size: int = 64):
        self._model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self._model_name)
            logger.info(f"Initialized FastEmbed model '{self._model_name}' (dim={self._dimension}).")
        except Exception as exc:
            logger.error(f"Failed to initialize FastEmbed model '{self._model_name}': {exc}")
            raise

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._model is None:
            self._initialize_model()

        embeddings_generator = self._model.embed(texts, batch_size=self._batch_size)
        return [list(vec.astype(float)) for vec in embeddings_generator]


class OpenAIEmbeddingService(BaseEmbeddingService):
    """
    Remote dense embedding provider for OpenAI-compatible REST endpoints
    (OpenAI, vLLM, Ollama, LiteLLM).
    """

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        batch_size: int = 64,
    ):
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key or ""
        self._model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            payload = {
                "input": batch,
                "model": self._model_name,
            }

            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        f"{self._api_base}/embeddings",
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()
                    # Sort results by index to preserve order
                    sorted_data = sorted(data["data"], key=lambda x: x["index"])
                    all_embeddings.extend([item["embedding"] for item in sorted_data])
            except Exception as exc:
                logger.error(f"OpenAI embedding request failed: {exc}")
                raise RuntimeError(f"OpenAI embedding provider error: {exc}") from exc

        return all_embeddings


class MockEmbeddingService(BaseEmbeddingService):
    """
    Deterministic pseudo-embedding provider for testing and offline development.
    Uses hash-based feature projection with L2 normalization to produce stable,
    reproducible 384-dimensional dense vectors where overlapping terms yield
    positive cosine similarity without model downloads.
    """

    def __init__(self, dimension: int = 384, model_name: str = "mock-deterministic-384"):
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _hash_vector(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * self._dimension

        if not tokens:
            vec[0] = 1.0
            return vec

        for token in tokens:
            # Deterministic bucket derived from token MD5
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dimension
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0
        return vec

    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vector(t) for t in texts]


_embedding_service_instance: Optional[BaseEmbeddingService] = None


def get_embedding_service(
    settings: Optional[Settings] = None,
    provider: Optional[str] = None,
) -> BaseEmbeddingService:
    """
    Factory creating or returning the configured embedding service singleton.
    Falls back gracefully to MockEmbeddingService if FastEmbed model fails to load.
    """
    global _embedding_service_instance
    if _embedding_service_instance is not None and provider is None:
        return _embedding_service_instance

    cfg = settings or get_settings()
    active_provider = (provider or cfg.EMBEDDING_PROVIDER).lower()

    if active_provider == "fastembed":
        try:
            instance = FastEmbedEmbeddingService(
                model_name=cfg.EMBEDDING_MODEL,
                dimension=cfg.EMBEDDING_DIMENSION,
                batch_size=cfg.EMBEDDING_BATCH_SIZE,
            )
        except Exception as exc:
            logger.warning(
                f"FastEmbed initialization failed ({exc}). Falling back to MockEmbeddingService for offline resilience."
            )
            instance = MockEmbeddingService(
                dimension=cfg.EMBEDDING_DIMENSION,
                model_name=f"fallback-{cfg.EMBEDDING_MODEL}",
            )
    elif active_provider == "openai":
        instance = OpenAIEmbeddingService(
            api_base=cfg.LLM_API_BASE or "https://api.openai.com/v1",
            api_key=cfg.LLM_API_KEY,
            model_name="text-embedding-3-small",
            dimension=1536,
            batch_size=cfg.EMBEDDING_BATCH_SIZE,
        )
    elif active_provider == "mock":
        instance = MockEmbeddingService(
            dimension=cfg.EMBEDDING_DIMENSION,
            model_name="mock-deterministic-384",
        )
    else:
        raise ValueError(f"Unknown embedding provider: '{active_provider}'. Supported: 'fastembed', 'openai', 'mock'")

    if provider is None:
        _embedding_service_instance = instance
    return instance
