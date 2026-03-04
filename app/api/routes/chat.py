"""
Chat API routes for the LangGraph RAG agent.

Endpoints
---------
POST /chat         — single-turn question with session tracking
GET  /history/{session_id} — retrieve conversation history for a session

All agent logic is delegated to ``AgentService`` which invokes the compiled
``agent_graph`` (router → rag → llm → memory → eval) and handles Redis
caching, PostgreSQL persistence, and structured response extraction.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.session import async_session
from app.models.conversation import ConversationTurn
from app.services.agent_service import agent_service

router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The user's question.",
        examples=["What are the payment terms in the contract?"],
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Opaque session identifier that groups turns into one conversation. "
            "If omitted, a new UUID session is generated automatically."
        ),
        examples=["user-abc-123"],
    )


class SourceChunk(BaseModel):
    """Metadata for a single retrieved document chunk."""

    source: str | None = Field(None, description="Source filename or path.")
    page: int | None = Field(None, description="Page number within the source.")


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    session_id: str = Field(..., description="Session identifier for this conversation.")
    answer: str = Field(..., description="The agent's final answer.")
    routing_decision: str = Field(
        ..., description="Which path the agent took: 'rag', 'direct', or 'out_of_scope'."
    )
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Metadata dicts for the retrieved chunks (populated for 'rag' routes only).",
    )
    evaluation_scores: dict[str, Any] = Field(
        default_factory=dict,
        description="RAGAS evaluation metrics. Empty dict if evaluate=False.",
    )
    cache_hit: bool = Field(
        default=False,
        description="True if this response was served from the Redis cache.",
    )
    error: str = Field(
        default="",
        description="Non-empty if a non-fatal error occurred during processing.",
    )


class TurnSummary(BaseModel):
    """A single conversation turn for the history endpoint."""

    id: str
    question: str
    answer: str
    routing_decision: str
    ragas_scores: dict[str, Any] | None
    created_at: str   # ISO-8601 string


class HistoryResponse(BaseModel):
    """Response body for GET /history/{session_id}."""

    session_id: str
    turns: list[TurnSummary]
    total: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=ChatResponse, summary="Send a question to the agent")
async def chat(
    body: ChatRequest = Body(...),
    evaluate: bool = Query(
        default=False,
        description=(
            "Set to true to run inline RAGAS evaluation (faithfulness, "
            "response_relevancy, context_precision). Adds ~2–5 s latency."
        ),
    ),
) -> ChatResponse:
    """
    Submit a question to the LangGraph RAG agent and receive an answer.

    **Flow inside the agent:**
    1. **Router** — classifies the question as `rag`, `direct`, or `out_of_scope`.
    2. **RAG retrieval** *(rag only)* — hybrid search + cross-encoder reranking.
    3. **LLM** — generates a grounded answer (or gives a polite refusal).
    4. **Memory** — persists the turn to PostgreSQL and injects prior history.
    5. **Eval** *(if evaluate=True)* — scores with RAGAS reference-free metrics.

    A new `session_id` UUID is generated if none is provided.
    Supply the returned `session_id` in subsequent requests to maintain
    conversation context.
    """
    # Auto-generate session_id if the caller didn't supply one.
    session_id = body.session_id or str(uuid.uuid4())

    result = await agent_service.run(
        question=body.question,
        session_id=session_id,
        evaluate=evaluate,
    )

    # Surface hard errors as HTTP 500 — but only when the agent returned
    # an empty answer AND an error.  A non-empty answer with an error means
    # the graph degraded gracefully, which is fine to return as 200.
    if result.get("error") and not result.get("answer"):
        raise HTTPException(
            status_code=500,
            detail=result["error"],
        )

    return ChatResponse(
        session_id=session_id,
        answer=result.get("answer", ""),
        routing_decision=result.get("routing_decision", ""),
        sources=result.get("sources", []),
        evaluation_scores=result.get("evaluation_scores", {}),
        cache_hit=result.get("cache_hit", False),
        error=result.get("error", ""),
    )


@router.get(
    "/history/{session_id}",
    response_model=HistoryResponse,
    summary="Get conversation history for a session",
)
async def get_history(
    session_id: str,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of turns to return (most recent first).",
    ),
) -> HistoryResponse:
    """
    Retrieve the conversation history for a given session from PostgreSQL.

    Returns turns ordered from most recent to oldest (newest first).
    Use `limit` to page through long histories.
    """
    try:
        async with async_session() as db:
            stmt = (
                select(ConversationTurn)
                .where(ConversationTurn.session_id == session_id)
                .order_by(ConversationTurn.created_at.desc())
                .limit(limit)
            )
            result = await db.execute(stmt)
            turns: list[ConversationTurn] = list(result.scalars().all())

        turn_summaries = [
            TurnSummary(
                id=str(t.id),
                question=t.question,
                answer=t.answer,
                routing_decision=t.routing_decision,
                ragas_scores=t.ragas_scores,
                created_at=t.created_at.isoformat(),
            )
            for t in turns
        ]

        return HistoryResponse(
            session_id=session_id,
            turns=turn_summaries,
            total=len(turn_summaries),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load conversation history: {exc}",
        ) from exc


@router.delete(
    "/history/{session_id}",
    summary="Clear conversation history for a session",
)
async def clear_history(session_id: str) -> dict[str, str]:
    """
    Delete all conversation turns for the given session from PostgreSQL.

    Useful for resetting a conversation context (e.g. a 'New chat' button).
    Does NOT clear the Redis semantic cache — cached answers remain valid.
    """
    from sqlalchemy import delete as sql_delete  # noqa: PLC0415

    try:
        async with async_session() as db:
            stmt = sql_delete(ConversationTurn).where(
                ConversationTurn.session_id == session_id
            )
            result = await db.execute(stmt)
            await db.commit()

        deleted = result.rowcount
        return {
            "status": "ok",
            "session_id": session_id,
            "deleted_turns": str(deleted),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear conversation history: {exc}",
        ) from exc
