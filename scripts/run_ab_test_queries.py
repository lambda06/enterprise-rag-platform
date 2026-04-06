"""
A/B test query runner — sends each question through BOTH Variant A and Variant B
so the comparison is controlled (same question, different prompt).

  10 questions × 2 variants = 20 total requests (~10 min at 30 s gaps)

Run:  python scripts/run_ab_test_queries.py
Then: python scripts/ab_test_analysis.py
"""

import asyncio
import hashlib
import uuid

import httpx

URL = "http://localhost:8000/api/v1/chat?evaluate=true"
SLEEP_S = 30  # RAGAS makes 5-8 LLM calls per request; keep ≥ 25 s on free tier

# 10 explicitly document-bound questions — each requires the indexed document
# to answer, so the router will always pick the RAG path (no non_rag_routing skips).
# Phrasing anchors like "in the study", "according to the paper", "in the document"
# prevent the router from handling them as general-knowledge questions.
QUESTIONS = [
    # Factual — specific numbers/names only in the document
    "What metric did the study use to evaluate retrieval quality in the RAG fine-tuning experiments?",
    "Which fine-tuning strategy achieved the highest overall performance score according to the paper?",

    # Comparative — document-specific comparison
    "According to the study, how does independent fine-tuning compare to joint fine-tuning in terms of final task performance?",
    "What trade-offs between retriever and generator fine-tuning does the paper describe in its results section?",

    # Definitional — grounded in document's own definitions
    "How does the paper define joint fine-tuning, and what distinguishes it from the other strategies tested?",
    "What does the document say about two-phase fine-tuning and how it differs from training both components simultaneously?",

    # Procedural — document-specific steps
    "What are the exact steps the paper describes for implementing the two-phase fine-tuning approach?",

    # Analytical — requires reading the paper's conclusions
    "According to the study, why can independent fine-tuning of the retriever sometimes degrade overall pipeline performance?",
    "What limitations of the fine-tuning strategies does the paper acknowledge in its discussion section?",

    # Edge case — tests grounded I-don't-know vs hallucination
    "What future research directions does the document recommend based on its experimental findings?",
]


def _session_for_variant(target: str) -> str:
    """Return a UUID whose MD5 first hex digit routes to the target variant.
    Variant A: digits 0-7 | Variant B: digits 8-f
    """
    while True:
        sid = str(uuid.uuid4())
        is_a = hashlib.md5(sid.encode()).hexdigest()[0] in "01234567"
        if (target == "A") == is_a:
            return sid


async def main() -> None:
    total = len(QUESTIONS) * 2

    async with httpx.AsyncClient(timeout=300.0) as client:
        n = 0
        for question in QUESTIONS:
            for variant in ("A", "B"):
                n += 1
                session_id = _session_for_variant(variant)
                print(f"[{n:02d}/{total}] Variant {variant} — {question}")
                try:
                    resp = await client.post(
                        URL, json={"question": question, "session_id": session_id}
                    )
                    resp.raise_for_status()
                    answer = resp.json().get("answer") or resp.json().get("final_answer", "")
                    print(f"         {answer[:110]}{'...' if len(answer) > 110 else ''}")
                except Exception as exc:
                    print(f"         [ERROR] {repr(exc)}")

                if n < total:
                    print(f"         sleeping {SLEEP_S}s...")
                    await asyncio.sleep(SLEEP_S)

    print("\nDone. Run: python scripts/ab_test_analysis.py")


if __name__ == "__main__":
    asyncio.run(main())
