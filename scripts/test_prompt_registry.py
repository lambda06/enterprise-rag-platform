"""
Manual integration tests for ``PromptRegistry`` (Langfuse + cache + fallback).

Run (PowerShell), from repo root or with full paths::

    & ".\\venv_312\\python.exe" ".\\scripts\\test_prompt_registry.py"

Example with absolute paths::

    & "d:\\Projects\\AI\\enterprise-rag-platform\\venv_312\\python.exe" \\
      "d:\\Projects\\AI\\enterprise-rag-platform\\scripts\\test_prompt_registry.py"

Requires valid Langfuse credentials in ``.env`` for tests 1-4 (except fallback test 5).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.prompt_registry import PromptRegistry, prompt_registry  # noqa: E402

PROMPT_NAME = "router-classification"
# Same version on both arms still exercises deterministic A/B assignment over session_id.
AB_VERSION_A = 1
AB_VERSION_B = 1


def _print_result(
    test_label: str, passed: bool, detail: str, actual: object = None
) -> bool:
    status = "PASS" if passed else "FAIL"
    line = f"{test_label}: {status} - {detail}"
    if actual is not None:
        line += f" | actual={actual!r}"
    print(line)
    return passed


def main() -> None:
    all_ok = True

    # Fresh cache so the first fetch is not a stale cache hit from a prior run.
    prompt_registry.invalidate_cache(None)

    # --- Test 1: first fetch -> Langfuse ---
    r1 = prompt_registry.get_prompt(PROMPT_NAME)
    ok1 = r1.source == "langfuse"
    all_ok &= _print_result(
        "Test 1 (first fetch -> langfuse)",
        ok1,
        f"expected source 'langfuse'",
        {
            "source": r1.source,
            "version": r1.version,
            "text_len": len(r1.text),
        },
    )

    # --- Test 2: immediate second fetch -> cache ---
    r2 = prompt_registry.get_prompt(PROMPT_NAME)
    ok2 = r2.source == "cache"
    all_ok &= _print_result(
        "Test 2 (immediate repeat -> cache)",
        ok2,
        f"expected source 'cache'",
        {"source": r2.source, "version": r2.version, "text_len": len(r2.text)},
    )

    # --- Test 3: invalidate -> Langfuse again ---
    prompt_registry.invalidate_cache(PROMPT_NAME)
    r3 = prompt_registry.get_prompt(PROMPT_NAME)
    ok3 = r3.source == "langfuse"
    all_ok &= _print_result(
        "Test 3 (after invalidate -> langfuse)",
        ok3,
        f"expected source 'langfuse'",
        {"source": r3.source, "version": r3.version, "text_len": len(r3.text)},
    )

    # --- Test 4: A/B assignment over 10 session ids (rough 50/50) ---
    session_ids = [f"session-test-{i}" for i in range(10)]
    rows: list[tuple[str, str]] = []
    for sid in session_ids:
        ab = prompt_registry.get_ab_variant(PROMPT_NAME, sid, AB_VERSION_A, AB_VERSION_B)
        assert ab.variant in ("A", "B")
        rows.append((sid, ab.variant))

    count_a = sum(1 for _, v in rows if v == "A")
    count_b = len(rows) - count_a
    # With 10 draws from ~50/50, allow 3-7 for either arm (common statistical fluctuation).
    ok4 = 3 <= count_a <= 7
    print("Test 4 (get_ab_variant x10 - session_id -> variant):")
    for sid, variant in rows:
        print(f"  {sid!r} -> {variant}")
    all_ok &= _print_result(
        "Test 4 (rough 50/50 split)",
        ok4,
        f"expected count_A between 3 and 7 (inclusive), got A={count_a}, B={count_b}",
        {"count_A": count_a, "count_B": count_b, "assignments": rows},
    )

    # --- Test 5: wrong API key -> fallback ---
    bad_settings = MagicMock()
    bad_settings.langfuse.public_key = "pk-invalid-for-testing"
    bad_settings.langfuse.secret_key = "sk-invalid-for-testing"
    bad_settings.langfuse.host = "https://cloud.langfuse.com"

    with patch("app.core.prompt_registry.get_settings", return_value=bad_settings):
        bad_registry = PromptRegistry()

    r5 = bad_registry.get_prompt(PROMPT_NAME)
    ok5 = r5.source == "fallback"
    all_ok &= _print_result(
        "Test 5 (invalid Langfuse key -> fallback)",
        ok5,
        f"expected source 'fallback'",
        {"source": r5.source, "version": r5.version, "text_len": len(r5.text)},
    )

    print()
    print("OVERALL:", "ALL PASS" if all_ok else "SOME FAILURES")


if __name__ == "__main__":
    main()
