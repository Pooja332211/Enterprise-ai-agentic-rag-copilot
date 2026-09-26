import os
import time
from typing import Any, AsyncGenerator, Dict, Optional, Tuple
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from backend.app.config.settings import get_settings
from backend.app.database.base import Base
from backend.app.utils.logger import logger

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None
_is_fallback_sqlite: bool = False


def get_engine() -> AsyncEngine:
    """Returns the initialized SQLAlchemy AsyncEngine, initializing it if necessary."""
    global _engine, _session_factory, _is_fallback_sqlite
    if _engine is not None:
        return _engine

    settings = get_settings()
    db_url = settings.async_database_url

    try:
        # Attempt primary engine creation (PostgreSQL)
        engine_kwargs = {
            "echo": settings.DEBUG and settings.ENVIRONMENT == "development",
            "future": True,
            "pool_pre_ping": True,
        }
        if "postgresql" in db_url:
            engine_kwargs.update({
                "pool_size": settings.DATABASE_POOL_SIZE,
                "max_overflow": settings.DATABASE_MAX_OVERFLOW,
                "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
                "pool_recycle": settings.DATABASE_POOL_RECYCLE,
            })
        _engine = create_async_engine(db_url, **engine_kwargs)
        _session_factory = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        _is_fallback_sqlite = False
        logger.info(f"Database engine created for {db_url.split('@')[-1] if '@' in db_url else 'local db'}")
    except Exception as e:
        if settings.DATABASE_SQLITE_FALLBACK:
            logger.warning(
                f"Failed to initialize primary database engine: {e}. "
                f"Falling back to local SQLite async database for development."
            )
            os.makedirs(os.path.dirname(settings.DATABASE_SQLITE_PATH), exist_ok=True)
            sqlite_url = f"sqlite+aiosqlite:///{settings.DATABASE_SQLITE_PATH}"
            _engine = create_async_engine(
                sqlite_url,
                echo=False,
                future=True,
            )
            _session_factory = async_sessionmaker(
                bind=_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
            _is_fallback_sqlite = True
        else:
            logger.error(f"Failed to create database engine: {e}")
            raise

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Returns the async session factory."""
    global _session_factory
    if _session_factory is None:
        get_engine()
    return _session_factory


class _AsyncSessionLocalProxy:
    """Convenience callable proxy returning a new AsyncSession."""
    def __call__(self, *args, **kwargs) -> AsyncSession:
        return get_session_factory()(*args, **kwargs)


AsyncSessionLocal = _AsyncSessionLocalProxy()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides an asynchronous database session.
    Automatically commits or rolls back on exceptions and closes upon completion.
    """
    global _engine, _session_factory, _is_fallback_sqlite
    settings = get_settings()

    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    except Exception as exc:
        if settings.DATABASE_SQLITE_FALLBACK and not _is_fallback_sqlite:
            logger.warning(
                f"Primary database connection failed in get_db ({exc}). Activating SQLite dev fallback..."
            )
            os.makedirs(os.path.dirname(settings.DATABASE_SQLITE_PATH), exist_ok=True)
            sqlite_url = f"sqlite+aiosqlite:///{settings.DATABASE_SQLITE_PATH}"
            _engine = create_async_engine(sqlite_url, echo=False, future=True)
            _session_factory = async_sessionmaker(
                bind=_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
            _is_fallback_sqlite = True
            async with _engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            async with _session_factory() as fallback_session:
                try:
                    yield fallback_session
                    await fallback_session.commit()
                except Exception:
                    await fallback_session.rollback()
                    raise
                finally:
                    await fallback_session.close()
        else:
            raise exc


async def init_db() -> None:
    """Creates all database tables defined in Base metadata."""
    global _engine, _session_factory, _is_fallback_sqlite
    settings = get_settings()
    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized successfully.")
    except Exception as exc:
        if settings.DATABASE_SQLITE_FALLBACK and not _is_fallback_sqlite:
            logger.warning(
                f"PostgreSQL connection failed during init_db: {exc}. Activating SQLite dev fallback..."
            )
            os.makedirs(os.path.dirname(settings.DATABASE_SQLITE_PATH), exist_ok=True)
            sqlite_url = f"sqlite+aiosqlite:///{settings.DATABASE_SQLITE_PATH}"
            _engine = create_async_engine(sqlite_url, echo=False, future=True)
            _session_factory = async_sessionmaker(
                bind=_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
            _is_fallback_sqlite = True
            async with _engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("SQLite fallback database initialized successfully.")
        else:
            raise exc


async def close_db() -> None:
    """Disposes the database engine pool."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database engine disposed.")


async def check_database_health() -> Dict[str, Any]:
    """
    Checks database connectivity and round-trip latency.
    Guarantees no credentials or secrets are leaked.
    """
    global _engine, _session_factory, _is_fallback_sqlite
    start_time = time.perf_counter()
    settings = get_settings()

    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "connected",
            "latency_ms": latency_ms,
            "engine": "sqlite" if _is_fallback_sqlite else "postgresql",
            "fallback": _is_fallback_sqlite,
        }
    except Exception as exc:
        # If primary engine fails and SQLite fallback is enabled, switch to SQLite
        if not _is_fallback_sqlite and settings.DATABASE_SQLITE_FALLBACK:
            try:
                logger.warning(
                    f"PostgreSQL ping failed: {exc}. Activating SQLite dev fallback..."
                )
                os.makedirs(os.path.dirname(settings.DATABASE_SQLITE_PATH), exist_ok=True)
                sqlite_url = f"sqlite+aiosqlite:///{settings.DATABASE_SQLITE_PATH}"
                _engine = create_async_engine(sqlite_url, echo=False, future=True)
                _session_factory = async_sessionmaker(
                    bind=_engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                    autoflush=False,
                )
                _is_fallback_sqlite = True
                async with _engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                async with _engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return {
                    "status": "connected",
                    "latency_ms": latency_ms,
                    "engine": "sqlite",
                    "fallback": True,
                }
            except Exception as fallback_exc:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return {
                    "status": "unreachable",
                    "latency_ms": latency_ms,
                    "error": str(fallback_exc),
                }

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "unreachable",
            "latency_ms": latency_ms,
            "error": "Database connection failed",
        }
