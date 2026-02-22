# Enterprise Agentic RAG Platform — Folder Structure

Production-ready folder structure for an Agentic RAG platform using **FastAPI**, **Qdrant**, **LangGraph**, **Groq**, **Redis**, **PostgreSQL**, and **RAGAS**.

---

## Root Structure

```
enterprise-rag-platform/
├── app/                          # Main application package
├── config/                       # Configuration management
├── notebooks/                    # Jupyter notebooks for experimentation
├── tests/                        # Test suite
├── scripts/                      # Utility and migration scripts
├── docs/                         # Documentation
├── docker/                       # Docker and container configs
├── .github/                      # CI/CD workflows
└── [config files]                # pyproject.toml, .env.example, etc.
```

---

## `app/` — Main Application

| Folder | Purpose |
|--------|---------|
| **`api/`** | FastAPI route handlers, request/response schemas, and API versioning. Entry points for REST endpoints. |
| **`core/`** | Application core: security, dependencies, exceptions, and shared utilities. |
| **`models/`** | SQLAlchemy/Pydantic models for PostgreSQL. Document metadata, users, collections. |
| **`schemas/`** | Pydantic schemas for validation, serialization, and API contracts. |
| **`services/`** | Business logic layer. Orchestrates agents, RAG pipelines, and external integrations. |
| **`agents/`** | LangGraph agent definitions, state schemas, nodes, and graph workflows. |
| **`rag/`** | RAG pipeline: chunking, embedding, retrieval, and reranking logic. |
| **`vectorstore/`** | Qdrant client, collection management, and vector operations. |
| **`llm/`** | Groq LLM client, prompt templates, and model configuration. |
| **`cache/`** | Redis caching layer for embeddings, responses, and session data. |
| **`db/`** | PostgreSQL connection, migrations (Alembic), and session management. |
| **`evaluation/`** | RAGAS metrics, evaluation pipelines, and benchmark datasets. |
| **`ingestion/`** | Document ingestion: parsing, chunking, embedding, and indexing into Qdrant. |

---

## `notebooks/` — Experimentation

| Folder | Purpose |
|--------|---------|
| **`notebooks/`** | Jupyter notebooks for experimenting with chunking strategies, RAGAS evaluation, and RAG pipelines before promoting validated approaches to production code in `app/`. |

---

## `config/` — Configuration

| Folder | Purpose |
|--------|---------|
| **`settings/`** | Pydantic Settings for env vars. Separate configs for app, DB, Redis, Qdrant, Groq. |
| **`logging/`** | Logging configuration, formatters, and handlers. |

---

## `tests/` — Testing

| Folder | Purpose |
|--------|---------|
| **`unit/`** | Unit tests for services, agents, and utilities. |
| **`integration/`** | Integration tests with real DB, Redis, Qdrant (or mocks). |
| **`e2e/`** | End-to-end API and RAG pipeline tests. |
| **`fixtures/`** | Shared test data, mock documents, and factories. |

---

## `scripts/` — Scripts

| Folder | Purpose |
|--------|---------|
| **`migrations/`** | Alembic migration scripts for PostgreSQL. |
| **`seed/`** | Database seeding and sample data scripts. |
| **`eval/`** | Standalone RAGAS evaluation and benchmarking scripts. |

---

## `docs/` — Documentation

| Folder | Purpose |
|--------|---------|
| **`api/`** | OpenAPI/Swagger docs and API reference. |
| **`architecture/`** | Architecture diagrams, RAG flow, and system design. |
| **`runbooks/`** | Operational runbooks for deployment and troubleshooting. |

---

## `docker/` — Containerization

| Folder | Purpose |
|--------|---------|
| **`compose/`** | Docker Compose files for local dev and production stacks. |
| **`Dockerfile`** | Multi-stage build for the FastAPI application. |

---

## `.github/` — CI/CD

| Folder | Purpose |
|--------|---------|
| **`workflows/`** | GitHub Actions for lint, test, build, and deploy. |

---

## Technology Mapping

| Tech | Primary Location |
|------|------------------|
| **FastAPI** | `app/api/`, `app/core/` |
| **Qdrant** | `app/vectorstore/` |
| **LangGraph** | `app/agents/` |
| **Groq** | `app/llm/` |
| **Redis** | `app/cache/` |
| **PostgreSQL** | `app/db/`, `app/models/` |
| **RAGAS** | `app/evaluation/`, `scripts/eval/` |
