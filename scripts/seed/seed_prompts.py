"""
Seed production prompts in Langfuse prompt registry.
This script pushes baseline prompt versions for:
  - router-classification
  - rag-answer-generation
  - direct-answer-generation
  - out-of-scope-refusal
Behavior:
  - If a prompt already exists in Langfuse, it is skipped.
  - New prompts are created with:
      labels=["production"]
      commit_message="Initial version - baseline prompt"
"""
from __future__ import annotations
from pathlib import Path
import sys
from typing import Any
from langfuse import Langfuse
# Ensure "app.*" imports resolve when script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.core.config import get_settings  # noqa: E402
from app.core.prompt_fallbacks import (  # noqa: E402
    DIRECT_SYSTEM_PROMPT,
    OUT_OF_SCOPE_REFUSAL,
    RAG_SYSTEM_PROMPT,
    ROUTER_CLASSIFICATION_PROMPT,
)
COMMIT_MESSAGE = "Initial version - baseline prompt"
PRODUCTION_LABELS = ["production"]
def _build_langfuse_client() -> Langfuse:
    settings = get_settings()
    if not (settings.langfuse.public_key and settings.langfuse.secret_key):
        raise RuntimeError(
            "Missing Langfuse credentials. Set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY in environment or .env."
        )
    return Langfuse(
        public_key=settings.langfuse.public_key,
        secret_key=settings.langfuse.secret_key,
        host=settings.langfuse.host,
    )
def _is_not_found_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in ("not found", "404", "does not exist", "unknown prompt"))
def _prompt_exists(client: Any, prompt_name: str) -> bool:
    # Try production label first (expected deployment lookup).
    try:
        client.get_prompt(name=prompt_name, label="production")
        return True
    except Exception as exc:
        if not _is_not_found_error(exc):
            raise
    
    # Fallback: any existing version (for safety in case label is absent).
    try:
        client.get_prompt(name=prompt_name)
        return True
    except Exception as exc:
        if _is_not_found_error(exc):
            return False
        raise

def main() -> None:
    client = _build_langfuse_client()
    prompts: list[tuple[str, str]] = [
        ("router-classification", ROUTER_CLASSIFICATION_PROMPT),
        ("rag-answer-generation", RAG_SYSTEM_PROMPT),
        ("direct-answer-generation", DIRECT_SYSTEM_PROMPT),
        ("out-of-scope-refusal", OUT_OF_SCOPE_REFUSAL),
    ]
    pushed: list[str] = []
    skipped: list[str] = []
    for prompt_name, prompt_text in prompts:
        if _prompt_exists(client, prompt_name):
            print(f"Prompt already exists, skipping: {prompt_name}")
            skipped.append(prompt_name)
            continue
        client.create_prompt(
            name=prompt_name,
            type="text",
            prompt=prompt_text,
            labels=PRODUCTION_LABELS,
            commit_message=COMMIT_MESSAGE,
        )
        print(f"Pushed prompt: {prompt_name}")
        pushed.append(prompt_name)
    client.flush()
    print("\n=== Langfuse Prompt Seed Summary ===")
    print(f"Pushed ({len(pushed)}): {', '.join(pushed) if pushed else 'None'}")
    print(f"Skipped ({len(skipped)}): {', '.join(skipped) if skipped else 'None'}")
if __name__ == "__main__":
    main()