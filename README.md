# Enterprise Agentic RAG Platform

Production‑ready repository for an **Agentic Retrieval‑Augmented Generation (RAG)** platform built on FastAPI.

This project stitches together a modern stack including **FastAPI**, **Qdrant**, **LangGraph**, **Groq LLMs**, **Redis**, **PostgreSQL**, and **RAGAS** evaluation. It is designed for developers, data scientists and operators who need a scalable system for ingesting documents, running RAG pipelines, orchestrating agentic workflows, and measuring performance with open benchmarks.

---

## 🚀 Key Features

- REST API powered by FastAPI with versioned routes and OpenAPI docs
- Document ingestion pipeline with chunking, parsing, embeddings and vector storage in Qdrant
- Agent workflows defined via LangGraph graphs
- Groq LLM integration with templating and configurable prompts
- Redis caching layer for acceleration and session state
- PostgreSQL-backed metadata and user management with Alembic migrations
- RAGAS-based evaluation tooling for benchmarks and metrics
- Containerization support with Docker Compose
- Comprehensive unit, integration and end-to-end tests

---

## 📁 Folder Structure

```
entprise-rag-platform/
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

### app/ — Main Application

- **api/** – FastAPI route handlers, request/response schemas, versioned endpoints
- **core/** – Security, dependencies, exceptions, shared utilities
- **models/** – SQLAlchemy/Pydantic models for PostgreSQL
- **schemas/** – Pydantic schemas for validation and serialization
- **services/** – Business logic orchestrating agents, RAG pipelines
- **agents/** – LangGraph agent definitions and workflows
- **rag/** – Core retrieval‑augmented generation pipeline code
- **vectorstore/** – Qdrant client and collection management
- **llm/** – Groq client, prompt templates, LLM configuration
- **cache/** – Redis caching for embeddings, responses, sessions
- **db/** – PostgreSQL connection, Alembic migrations, session management
- **evaluation/** – RAGAS metrics, evaluation pipelines
- **ingestion/** – Document parsing, chunking, embedding, and indexing

### config/

- **settings/** – Pydantic settings for environment variables (app, DB, Redis, etc.)
- **logging/** – Logging configuration and handlers

### notebooks/

Contains Jupyter notebooks for experimentation and prototyping of chunking strategies, RAG pipelines, and evaluation before moving code to `app/`.

### tests/

- **unit/** – Unit tests for services, agents, and utilities
- **integration/** – Tests that exercise real or mocked external services
- **e2e/** – End‑to‑end API and pipeline tests
- **fixtures/** – Shared test data and factories

### scripts/

- **migrations/** – Alembic migration scripts
- **seed/** – Database seeding and sample data scripts
- **eval/** – Standalone RAGAS evaluation/benchmarking scripts

### docs/

- **api/** – OpenAPI/Swagger documentation
- **architecture/** – Design diagrams and system architecture
- **runbooks/** – Operational runbooks for deployment and troubleshooting

### docker/

- **compose/** – Docker Compose definitions for local and production stacks
- **Dockerfile** – Multi‑stage build for the FastAPI application

### .github/

- **workflows/** – GitHub Actions workflows for linting, testing, building, and deploying

---

## 🛠 Prerequisites

- Python 3.12+
- Docker & Docker Compose (for local stack)
- PostgreSQL, Redis, and Qdrant instances (can be started via `docker/compose`)

---

## 🏗 Getting Started

1. Copy `.env.example` to `.env` and fill in your credentials.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -e .
   ```
4. Initialize the database:
   ```bash
   alembic upgrade head
   ```
5. Run tests:
   ```bash
   pytest
   ```
6. Start the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

---

## 📘 Documentation

The `docs/` folder contains architecture diagrams, API reference, and operational runbooks. OpenAPI docs are available at `http://localhost:8000/docs` when the server is running.

---

## 📦 Packaging & Deployment

See `docker/compose` for containerized deployment. GitHub Actions automate linting, tests, and releases under `.github/workflows`.

---

## 🧪 Testing

Run `pytest` for the full suite. Tests are categorized into `unit`, `integration`, and `e2e`; mock services are located under `tests/fixtures`.

---

## ✨ Contributing

Please follow code style guidelines and add tests for new features. Refer to `docs/runbooks` for branch/release process and GitHub Actions details.

---

## 📄 License

[Add license information here]

---

For full architectural overview, refer to `STRUCTURE.md` in the repository root.