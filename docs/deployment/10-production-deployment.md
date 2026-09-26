# Production Deployment & Operations Manual

**Project**: Enterprise AI Knowledge & Research Copilot  
**Milestone**: 10 — Productionization, Deployment & Reliability  
**Target Environment**: Enterprise Linux / Docker Container / Kubernetes Pod  

---

## 1. Prerequisites & System Requirements

### Hardware Requirements
- **Minimum**: 2 vCPUs, 4 GB RAM, 20 GB Persistent Disk
- **Recommended**: 4 vCPUs, 8 GB RAM, 50 GB SSD
- **GPU (Optional)**: NVIDIA CUDA GPU ($\ge 6.0$ GB VRAM) for accelerated ONNX cross-encoder reranking and QLoRA fine-tuning.

### Software Requirements
- **Docker Engine**: Version 24.0+ & Docker Compose v2.20+
- **Python**: 3.11.x (if deploying as a bare-metal service)
- **PostgreSQL**: Version 16.x (Alpine recommended)
- **Qdrant Vector Database**: Version 1.12.1+

---

## 2. Configuration Management

All configuration is managed through standard environment variables or a `.env` file at the application root. Use `.env.example` as the canonical reference template.

### Critical Production Environment Variables

| Variable | Recommended Production Value | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `production` | Enables strict production validation safeguards. |
| `DEBUG` | `false` | Disables debug mode and verbose SQL echo. |
| `SECRET_KEY` | *(Random 32+ char string)* | Required. Rejects default placeholder string. |
| `DATABASE_SQLITE_FALLBACK` | `false` | Enforces PostgreSQL; prevents silent fallback to local SQLite. |
| `QDRANT_LOCAL_FALLBACK` | `false` | Enforces remote Qdrant; prevents silent fallback to in-memory store. |
| `DATABASE_POOL_SIZE` | `10` | SQLAlchemy connection pool size. |
| `DATABASE_MAX_OVERFLOW` | `20` | Maximum burst connections beyond pool size. |
| `DATABASE_POOL_TIMEOUT` | `30.0` | Connection checkout timeout in seconds. |
| `DATABASE_POOL_RECYCLE` | `1800` | Periodically recycles connections every 30 minutes. |
| `CACHE_ENABLED` | `true` | Enables in-memory LRU embedding and reranker cache. |
| `CACHE_EMBEDDING_MAX_SIZE` | `20000` | Capacity limit for dense vector cache. |
| `CACHE_RERANKER_MAX_SIZE` | `20000` | Capacity limit for joint `(query, chunk)` rerank cache. |
| `RERANKER_CANDIDATE_TOP_K` | `10` | Empirically verified optimal first-stage candidate pool. |
| `LLM_PROVIDER` | `openai` / `mock` | Selected LLM backend provider. |
| `LLM_API_KEY` | *(Enterprise API Key)* | Required if `LLM_PROVIDER` is set to `openai`. |

> [!CAUTION]
> In production (`ENVIRONMENT=production`), the application will immediately fail startup if `SECRET_KEY` is left as default, if `DEBUG=true`, or if `DATABASE_SQLITE_FALLBACK=true`.

---

## 3. Database Setup & Migrations

The relational schema is managed through version-controlled Alembic migrations located in `backend/alembic/versions/`.

### Migration Execution
Migrations are automatically executed during container startup via `docker/entrypoint.sh`. To manually apply migrations:

```bash
cd backend
alembic upgrade head
```

### Clean Database Initialization
When provisioning a fresh PostgreSQL database instance:
1. Ensure the database `enterprise_copilot` exists with UTF-8 encoding.
2. Ensure the database user has schema migration permissions.
3. Run `alembic upgrade head` to apply all migrations in sequence (`0001_initial_schema` $\rightarrow$ `0002_document_ingestion_schema`).

---

## 4. Qdrant Vector Store Setup

1. The application ensures the target collection exists (`QDRANT_COLLECTION_NAME=enterprise_knowledge`) during startup.
2. Vector parameters:
   - Size: `384` dimensions (matching `BAAI/bge-small-en-v1.5`)
   - Distance Metric: `Cosine`
3. Payload Indexes:
   - Payload field indexes on `document_id` (Keyword) and `page_number` (Integer) are automatically provisioned during collection initialization to accelerate filtered queries.

---

## 5. Docker Deployment

### Standard Production Stack (PostgreSQL + Qdrant + FastAPI Backend)

```bash
# 1. Copy and populate production environment file
cp .env.example .env

# 2. Build and start services in detached mode
docker compose up -d --build

# 3. View startup logs
docker compose logs -f backend
```

### Full Stack with Observability Profile (Prometheus + Grafana)

```bash
docker compose --profile monitoring up -d
```
- **Backend API**: `http://localhost:8000`
- **Swagger Docs**: `http://localhost:8000/docs`
- **Prometheus UI**: `http://localhost:9090`
- **Grafana UI**: `http://localhost:3000` (Default credentials: `admin` / `admin`)

---

## 6. Health & Readiness Probes

The service exposes standardized liveness and readiness endpoints designed for container orchestrators (Kubernetes, AWS ECS, Nomad):

### Liveness Probe (`GET /health`)
- Returns `HTTP 200 OK` with `{"status": "healthy"}` if the process is responsive.
- Used to detect process deadlocks or fatal thread crashes.

### Deep Readiness Probe (`GET /health/ready`)
- Evaluates the 6 core dependencies:
  1. `database` (PostgreSQL)
  2. `vector_store` (Qdrant)
  3. `embedding_provider` (FastEmbed)
  4. `reranker` (BGE Cross-Encoder)
  5. `llm_provider` (Mock / OpenAI / Anthropic)
  6. `bm25_index` (BM25Plus lexical store)
- **Status Codes**:
  - `HTTP 200 OK`: When all primary dependencies are healthy or non-fatal external components are degraded.
  - `HTTP 503 Service Unavailable`: When either the primary relational database or the vector store is unavailable. Traffic is prevented from routing to the unready node.

---

## 7. Production Smoke Test

After deploying the service, run the deterministic production smoke test to verify all 9 capabilities end-to-end:

```bash
python scripts/production_smoke_test.py --url http://localhost:8000
```

### Verifications Performed:
1. Root Liveness Probe (`GET /health`)
2. Deep Readiness Probe (`GET /health/ready`)
3. Document Upload & Ingestion (`POST /api/v1/documents`)
4. Index Status Inspection (`GET /api/v1/index/stats`)
5. Two-Stage Hybrid Search (`POST /api/v1/search`)
6. Single-Turn Grounded Research (`POST /api/v1/research`)
7. LangGraph Agentic Research Workflow (`POST /api/v1/research/agent`)
8. Citation Verification & Claim Grounding Check
9. Prometheus Metrics Exposition (`GET /metrics`)

---

## 8. Backup & Disaster Recovery

### 1. PostgreSQL Relational State
Contains document metadata, parsing jobs, chunk definitions, and citation audit records:
```bash
docker exec -t copilot_postgres pg_dump -U postgres enterprise_copilot | gzip > backup_db_$(date +%Y%m%d).sql.gz
```

### 2. Qdrant Vector Collection
Contains dense vector embeddings and metadata payloads:
```bash
curl -X POST "http://localhost:6333/collections/enterprise_knowledge/snapshots"
```
Snapshots are persisted in the `qdrant_data` volume under `/qdrant/storage/snapshots/`.

### 3. Filesystem Storage
Raw document files and BM25 index serialized files are stored in `backend_data` mapped to `/app/data/`:
```bash
docker run --rm -v enterpriseaiknowledgeresearchcopilot_backend_data:/data -v $(pwd):/backup alpine tar czf /backup/backup_data_$(date +%Y%m%d).tar.gz /data
```

---

## 9. Security & Compliance Architecture

1. **Path Traversal Protection**: `LocalStorageService` strictly validates and confines all file read/write operations within the configured `UPLOAD_DIR`. Directory traversal patterns (`../`, null bytes, absolute escape paths) are immediately rejected with security audit logs.
2. **Prompt Injection Containment**: All retrieved document text assembled into generation prompts is wrapped within explicit `<context>...</context>` boundaries. System prompts explicitly instruct the LLM that context text represents untrusted data that cannot override system guidelines.
3. **Sensitive Credential Sanitization**: `redact_sensitive_data()` automatically strips API keys, passwords, bearer tokens, and auth headers from all log entries, exception handlers, and trace outputs.
4. **Input Boundary Enforcement**: API requests enforce upper limits on query string length ($\le 2,000$ characters), candidate retrieval pool sizes ($\le 100$), and uploaded file size ($\le 50$ MB).

---

## 10. Known Limitations & Architecture Notes

1. **Authentication Boundary**: User authentication (SSO, OAuth2, JWT) is intentionally designed to sit at the enterprise API gateway or reverse proxy layer (e.g. Kong, Envoy, Traefik, AWS ALB) rather than tightly coupled inside the core research engine.
2. **Docker Runtime Verification**: When deploying in environments without a local Docker daemon (e.g. restricted CI runners or Windows without Docker Desktop started), Docker Compose configuration can be validated offline via `docker compose config`, while actual container execution must take place on a Docker-enabled host.
