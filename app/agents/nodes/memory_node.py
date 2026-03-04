"""
LangGraph MEMORY node for the Enterprise RAG Platform.

Responsibility
--------------
This node runs **after** ``llm_node`` has produced a ``final_answer``.
It has two jobs — both within the same async database session:

  Job 1 — Persist the current turn
      Insert a new ``ConversationTurn`` row into PostgreSQL with the
      current question, final answer, and routing decision.

  Job 2 — Inject conversation history into state
      Load the last N turns for this session (ordered oldest→newest) and
      prepend them to ``state["messages"]`` as ``HumanMessage`` /
      ``AIMessage`` pairs.  This gives the LLM context about what was
      discussed earlier in the session without requiring the caller to
      maintain state externally.

Why MEMORY runs after llm_node?
--------------------------------
Saving the current turn *after* generation ensures we only persist
turns that have a real answer — partial failures (e.g. retrieval errors
that produced no answer) are not written to the DB.

Why prepend history to messages?
---------------------------------
LangChain's message list convention is chronological (oldest first).
By prepending the loaded history before the current question, the full
conversation context is visible to any future node or LLM call that reads
``state["messages"]``.

Session cleanup
---------------
The node uses ``async_session()`` as a context manager directly (not via
FastAPI's ``get_db`` dependency) because LangGraph nodes are not FastAPI
handlers.  This is the correct pattern for async SQLAlchemy in non-request
contexts — the session is always closed in the ``finally`` block.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select

from app.agents.state import AgentState
from app.db.session import async_session
from app.models.conversation import ConversationTurn

logger = logging.getLogger(__name__)

# How many previous turns to load and inject into the messages list.
# 5 turns = 10 messages (5 human + 5 AI) — enough context for most
# multi-turn dialogues without bloating the LLM prompt unnecessarily.
HISTORY_TURNS: int = 5


async def memory_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: persist the current turn and inject conversation history.

    Operation order (within a single async DB session):
      1. Insert the current turn (question + answer + routing_decision).
      2. Query the last ``HISTORY_TURNS`` turns for this session.
      3. Convert them to HumanMessage / AIMessage pairs (oldest first).
      4. Prepend to the existing ``state["messages"]`` list.

    Args:
        state: The current ``AgentState`` dict.  Reads:
               - ``session_id``        — scopes history to this session.
               - ``current_question``  — saved as the turn's question.
               - ``final_answer``      — saved as the turn's answer.
               - ``routing_decision``  — saved for analytics.
               - ``messages``          — existing message list to prepend to.

    Returns:
        Partial state dict:

        ``messages``
            Updated message list with conversation history prepended.
            If DB access fails entirely, returns the original messages
            unchanged so the graph can still proceed.

        ``error``
            Empty string on success; human-readable message on DB failure.
            A non-empty error does NOT halt the graph — memory failures are
            treated as non-fatal so a DB outage doesn't break generation.

    Raises:
        Does NOT raise — all exceptions are caught and written to ``error``.
    """
    session_id: str = state.get("session_id", "").strip()
    question: str = state.get("current_question", "").strip()
    answer: str = state.get("final_answer", "").strip()
    routing_decision: str = state.get("routing_decision", "")
    existing_messages: list = state.get("messages", [])

    # ------------------------------------------------------------------ #
    # Guard: skip persistence if core fields are missing                  #
    # ------------------------------------------------------------------ #
    if not session_id:
        msg = "memory_node: session_id is empty — skipping DB operations."
        logger.warning(msg)
        return {"messages": existing_messages, "error": msg}

    # ------------------------------------------------------------------ #
    # Upstream error propagation                                          #
    # ------------------------------------------------------------------ #
    # If llm_node failed (error set + final_answer empty), persisting an
    # empty answer would pollute the session history.  Pass the error
    # through and skip the DB write — history load still runs so the
    # caller gets context for the next turn.
    upstream_error: str = state.get("error", "")
    if upstream_error and not answer:
        logger.warning(
            "memory_node: upstream error with empty answer — skipping DB persist, "
            "loading history only."
        )
        # Fall through to the DB section but the question/answer guard
        # below will skip the INSERT cleanly while still running the SELECT.

    if not question or not answer:
        msg = (
            f"memory_node: question={question!r} or answer={answer!r} is empty "
            "— skipping DB persist but still loading history."
        )
        logger.warning(msg)
        # Still attempt to load history even if we can't save this turn.

    # ------------------------------------------------------------------ #
    # Database operations                                                  #
    # ------------------------------------------------------------------ #
    try:
        async with async_session() as db:

            # ── Job 1: Persist the current turn ─────────────────────────
            if question and answer:
                turn = ConversationTurn(
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    routing_decision=routing_decision or "unknown",
                    # ragas_scores is intentionally omitted here — the
                    # EVALUATOR node will update it separately if it runs.
                    ragas_scores=None,
                )
                db.add(turn)
                # Flush to assign DB-generated values (id, created_at)
                # before the session closes; commit happens on context exit.
                await db.flush()
                logger.info(
                    "memory_node: persisted turn id=%s for session=%r",
                    turn.id,
                    session_id,
                )

            # ── Job 2: Load last N turns for this session ────────────────
            # Order by created_at DESC to get the most recent, then reverse
            # the result to get oldest-first for the message list.
            stmt = (
                select(ConversationTurn)
                .where(ConversationTurn.session_id == session_id)
                .order_by(ConversationTurn.created_at.desc())
                .limit(HISTORY_TURNS)
            )
            result = await db.execute(stmt)
            past_turns: list[ConversationTurn] = list(result.scalars().all())

            # Commit here — both the insert and the select are now durable.
            await db.commit()

        # ── Build message history (oldest turn first) ────────────────────
        # Reverse so chronological order: [oldest … newest] ← current turn
        # is NOT included because it will be added by the caller / next node.
        past_turns.reverse()

        history_messages: list = []
        for past_turn in past_turns:
            # Skip the turn we just inserted to avoid duplicating it in the
            # message list (the current question is already in state).
            if past_turn.question == question and past_turn.answer == answer:
                continue
            history_messages.append(HumanMessage(content=past_turn.question))
            history_messages.append(AIMessage(content=past_turn.answer))

        logger.info(
            "memory_node: loaded %d history turns (%d messages) for session=%r",
            len(past_turns),
            len(history_messages),
            session_id,
        )

        # Prepend history before existing messages so the full conversation
        # is in chronological order: [history…] + [current turn messages…]
        updated_messages = history_messages + existing_messages

        return {
            "messages": updated_messages,
            "error": "",
        }

    except Exception as exc:  # noqa: BLE001
        # Memory failures are NON-FATAL — log and return unchanged messages
        # so the graph can still return the answer to the user.
        msg = f"memory_node DB operation failed: {exc}"
        logger.exception(msg)
        return {
            "messages": existing_messages,
            "error": msg,
        }
