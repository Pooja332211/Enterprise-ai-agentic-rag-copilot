# Architecture Decision Record (ADR) 01: System Foundation & Clean Architecture

## Status
Accepted

## Context
Building the "Enterprise AI Knowledge & Research Copilot" as an enterprise-grade portfolio project requires a modular, resilient foundation capable of scaling across 12 milestones (from ingestion, hybrid retrieval, agent workflows, QLoRA, and evaluation to UI). 

Key challenges addressed in Milestone 1:
1. Preventing tight coupling between API routes, storage services, vector indexes, and business logic.
2. Avoiding hardcoded secrets or environment assumptions.
3. Enabling seamless local development and automated CI testing even when external container daemons (PostgreSQL, Qdrant) are temporarily offline.
4. Ensuring zero secret leaks through diagnostics and health endpoints.

## Decisions

1. **Monorepo Architecture**:
   - Organized into `backend/`, `frontend/`, `finetuning/`, `evaluation/`, `infrastructure/`, and `docs/`.
   - Clear module boundaries with isolated dependency scopes.

2. **Configuration Management**:
   - Centralized via Pydantic Settings (`backend.app.config.settings.Settings`).
   - Strongly typed with automatic `.env` discovery.
   - Dynamic assembly of database URLs (both asyncpg for runtime queries and psycopg2 for Alembic migrations).

3. **Database Layer (SQLAlchemy 2.0 & Alembic)**:
   - Async engine with connection pooling via `async_sessionmaker`.
   - Foundational models: `Document`, `IngestionJob`, and `SystemAuditLog`.
   - Declarative base with UTC timestamp mixin.
   - Resilient development fallback: if PostgreSQL is unreachable during local testing, an async SQLite instance is initialized without altering model definitions or application logic.

4. **Vector Database Abstraction (Qdrant)**:
   - Encapsulated within `VectorService`.
   - Application logic never calls raw Qdrant client methods directly.
   - Automatic fallback to embedded/in-memory mode for offline development and fast unit testing.
   - Complete isolation: fallback mechanics do not leak to the rest of the application.

5. **Diagnostic Probes (Liveness & Readiness)**:
   - `GET /health`: Lightweight process liveness for load balancers (always HTTP 200 if process is up).
   - `GET /health/ready`: Deep dependency readiness check inspecting database and vector database connectivity and reporting latencies.
   - Sanitized responses guarantee no passwords, keys, or internal connection strings are exposed.

## Consequences
- The codebase is clean, typed, modular, and fully testable.
- Future milestones (ingestion, embeddings, agents) can build upon this base without refactoring foundational layers.
