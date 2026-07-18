"""
Guardrails package for the Enterprise RAG Platform.

Contains input and output guardrail layers that run outside the LangGraph
graph to protect the system at both entry and exit points.

Modules
-------
input_guard  — Pre-LLM checks: prompt injection, PII detection, length limits.
output_guard — Post-generation checks: confidence gating, hallucination signals.
             (Phase 2 — not yet implemented)
"""
