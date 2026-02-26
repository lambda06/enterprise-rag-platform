"""
Langfuse observability wrapper for the Enterprise RAG Platform.

Provides a fault-tolerant ``LangfuseTracer`` that wraps the Langfuse Python
SDK.  If Langfuse credentials are missing or the Langfuse server is
unreachable, all tracing calls become silent no-ops — the RAG pipeline
continues to serve requests normally without any errors.

Usage in pipeline.py
--------------------
    from app.observability.langfuse_tracer import tracer

    trace = tracer.start_trace("rag-query", input={"question": question})
    span  = tracer.start_span(trace, "hybrid-search", input={...})
    tracer.end_span(span, output={...})
    tracer.end_trace(trace, output=result, metadata={"evaluation": scores})
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ── No-op objects used when Langfuse is disabled ──────────────────────────────

class _NoOpSpan:
    """Silent placeholder for a Langfuse span/generation when tracing is off."""
    def end(self, **kwargs: Any) -> None: ...
    def update(self, **kwargs: Any) -> None: ...


class _NoOpTrace:
    """Silent placeholder for a Langfuse trace when tracing is off."""
    def span(self, *args: Any, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()
    def generation(self, *args: Any, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()
    def update(self, **kwargs: Any) -> None: ...


# ── Tracer ────────────────────────────────────────────────────────────────────

class LangfuseTracer:
    """Fault-tolerant wrapper around the Langfuse Python SDK.

    Initialises the Langfuse client on first use.  If credentials are absent
    or the SDK raises on initialisation, every method silently returns a no-op
    object so callers never need to guard against ``None``.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._lf = None

        if not (settings.langfuse.public_key and settings.langfuse.secret_key):
            logger.warning(
                "LangfuseTracer: LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY "
                "not set — tracing disabled."
            )
            return

        try:
            from langfuse import Langfuse

            self._lf = Langfuse(
                public_key=settings.langfuse.public_key,
                secret_key=settings.langfuse.secret_key,
                host=settings.langfuse.host,
            )
            logger.info(
                "LangfuseTracer initialised (host=%s)", settings.langfuse.host
            )
        except Exception as exc:
            logger.warning("LangfuseTracer init failed: %s — tracing disabled.", exc)

    # ── Public helpers ─────────────────────────────────────────────────────────

    def start_trace(self, name: str, **kwargs: Any) -> Any:
        """Create and return a Langfuse trace (or a no-op trace).

        Args:
            name:   Trace name shown in the Langfuse UI.
            **kwargs: Forwarded to ``Langfuse.trace()`` — e.g. ``input``,
                      ``user_id``, ``session_id``, ``tags``.

        Returns:
            A real ``StatefulTraceClient`` or a ``_NoOpTrace``.
        """
        if self._lf is None:
            return _NoOpTrace()
        try:
            return self._lf.trace(name=name, **kwargs)
        except Exception as exc:
            logger.debug("Langfuse trace creation failed: %s", exc)
            return _NoOpTrace()

    def start_span(self, trace: Any, name: str, **kwargs: Any) -> Any:
        """Create a span on ``trace`` (or a no-op span).

        Args:
            trace:  The trace returned by :meth:`start_trace`.
            name:   Span name shown in the Langfuse UI.
            **kwargs: Forwarded to ``trace.span()`` — e.g. ``input``.

        Returns:
            A real ``StatefulSpanClient`` or a ``_NoOpSpan``.
        """
        try:
            return trace.span(name=name, **kwargs)
        except Exception as exc:
            logger.debug("Langfuse span creation failed: %s", exc)
            return _NoOpSpan()

    def start_generation(self, trace: Any, name: str, **kwargs: Any) -> Any:
        """Create a generation span on ``trace`` for LLM calls.

        Use this instead of :meth:`start_span` for LLM steps — Langfuse
        displays generations with token counts and model details.

        Args:
            trace:  The trace returned by :meth:`start_trace`.
            name:   Span name shown in the Langfuse UI.
            **kwargs: Forwarded to ``trace.generation()`` — e.g. ``model``,
                      ``input``, ``model_parameters``.

        Returns:
            A real ``StatefulGenerationClient`` or a ``_NoOpSpan``.
        """
        try:
            return trace.generation(name=name, **kwargs)
        except Exception as exc:
            logger.debug("Langfuse generation creation failed: %s", exc)
            return _NoOpSpan()

    def end_span(self, span: Any, output: Any = None, **kwargs: Any) -> None:
        """End a span and record its output.

        Args:
            span:   The span/generation returned by :meth:`start_span` /
                    :meth:`start_generation`.
            output: The result produced by this step (serialisable value).
            **kwargs: Additional fields forwarded to ``span.end()``.
        """
        try:
            span.end(output=output, **kwargs)
        except Exception as exc:
            logger.debug("Langfuse span.end failed: %s", exc)

    def end_trace(self, trace: Any, output: Any = None, metadata: dict | None = None) -> None:
        """Finalise the trace with its overall output and metadata.

        Args:
            trace:    The trace returned by :meth:`start_trace`.
            output:   Final result (e.g. the answer dict).
            metadata: Arbitrary key-value dict (e.g. RAGAS scores).
        """
        try:
            trace.update(output=output, metadata=metadata or {})
        except Exception as exc:
            logger.debug("Langfuse trace.update failed: %s", exc)

    def flush(self) -> None:
        """Flush pending events to Langfuse (call at application shutdown)."""
        if self._lf is not None:
            try:
                self._lf.flush()
            except Exception as exc:
                logger.debug("Langfuse flush failed: %s", exc)


# Module-level singleton
tracer = LangfuseTracer()
