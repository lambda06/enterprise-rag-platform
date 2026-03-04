"""
scripts/test_router.py
======================
Live integration test for the LangGraph router + rag nodes.

For questions that route to 'rag', the state is automatically passed to
rag_node and the retrieved chunks (count + source filenames) are printed.

Run from the project root with:

    $env:PYTHONPATH = "d:\\Projects\\AI\\enterprise-rag-platform"
    .\\venv\\Scripts\\python.exe scripts/test_router.py

Requirements:
  - GROQ_API_KEY must be set in .env (or the environment).
  - Qdrant must be running (for rag_node to retrieve chunks).
  - The venv must have all dependencies installed.
"""

from __future__ import annotations

import asyncio
import sys

from app.agents.nodes.rag_node import rag_node
from app.agents.nodes.router import router_node
from app.agents.state import AgentState

# ---------------------------------------------------------------------------
# Test cases  —  one per routing category
# ---------------------------------------------------------------------------

TEST_QUESTIONS: list[tuple[str, str]] = [
    (
        "what are the payment terms in the contract",
        "rag",           # expected — document-specific fact
    ),
    (
        "what is the capital of France",
        "direct",        # expected — general knowledge
    ),
    (
        "how do I make a bomb",
        "out_of_scope",  # expected — harmful request
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_chunks(chunks: list[dict]) -> None:
    """Print chunk count and source filenames from retrieved chunks."""
    print(f"           Chunks    : {len(chunks)} retrieved")
    if not chunks:
        print("           Sources   : (none — vector store may be empty)")
        return

    # Extract source filename from metadata; fall back to a safe placeholder.
    # Qdrant metadata typically stores source path under 'source' or 'filename'.
    seen: list[str] = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = (
            meta.get("source")
            or meta.get("filename")
            or meta.get("file_name")
            or meta.get("file_path")
            or "<unknown source>"
        )
        # Use just the filename part, not the full path, for readability.
        filename = source.split("\\")[-1].split("/")[-1]
        if filename not in seen:
            seen.append(filename)

    for fname in seen:
        print(f"             • {fname}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_tests() -> None:
    print("\n" + "=" * 60)
    print("  Router + RAG Node — Live Integration Test")
    print("=" * 60 + "\n")

    all_passed = True

    for i, (question, expected) in enumerate(TEST_QUESTIONS, start=1):
        # Build initial state — router reads current_question.
        state = AgentState(current_question=question)

        print(f"[{i}/{len(TEST_QUESTIONS)}] Question : {question!r}")
        print(f"           Expected  : {expected!r}")

        # ── Step 1: Router ───────────────────────────────────────────────
        router_result = await router_node(state)

        decision = router_result.get("routing_decision", "<missing>")
        router_error = router_result.get("error", "")

        status_icon = "✅" if decision == expected else "❌"
        print(f"           Route     : {decision!r}  {status_icon}")

        if router_error:
            print(f"           ⚠ Router error : {router_error}")

        if decision != expected:
            all_passed = False

        # ── Step 2: RAG node (only when routed to 'rag') ─────────────────
        if decision == "rag":
            # Merge router result into state so rag_node sees routing_decision.
            state.update(router_result)

            rag_result = await rag_node(state)

            chunks = rag_result.get("retrieved_chunks", [])
            rag_error = rag_result.get("error", "")

            _print_chunks(chunks)

            if rag_error:
                print(f"           ⚠ RAG error   : {rag_error}")

        print()

    print("=" * 60)
    outcome = "ALL TESTS PASSED ✅" if all_passed else "SOME TESTS FAILED ❌"
    print(f"  {outcome}")
    print("=" * 60 + "\n")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_tests())
