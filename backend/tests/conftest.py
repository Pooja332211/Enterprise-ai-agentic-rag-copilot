import os
from typing import AsyncGenerator
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Force testing configuration
os.environ["ENVIRONMENT"] = "testing"
os.environ["DEBUG"] = "false"
os.environ["DATABASE_SQLITE_FALLBACK"] = "true"
os.environ["DATABASE_SQLITE_PATH"] = "./data/test_copilot.db"
os.environ["BM25_INDEX_PATH"] = "./data/indexes/test_bm25_index.pkl"
os.environ["QDRANT_LOCAL_FALLBACK"] = "true"
os.environ["QDRANT_URL"] = "http://localhost:6333"

from backend.app.config.settings import Settings, get_settings
from backend.app.database.base import Base
from backend.app.database.session import get_db
from backend.app.main import app
from backend.app.services.vector_service import VectorService, get_vector_service

TEST_SQLITE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Provides application settings for tests."""
    return get_settings()


@pytest_asyncio.fixture(scope="function")
async def test_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provides an isolated in-memory database session for unit tests."""
    engine = create_async_engine(TEST_SQLITE_URL, echo=False)
    async_session = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture(scope="function")
def in_memory_vector_service(test_settings: Settings) -> VectorService:
    """Provides a Qdrant vector service running in in-memory mode."""
    service = VectorService(settings=test_settings)
    return service


@pytest_asyncio.fixture(scope="function")
async def async_client(test_db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Asynchronous HTTP test client bound to the FastAPI application
    with database dependency overridden to the isolated in-memory test session.
    """
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield test_db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
