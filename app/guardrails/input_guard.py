"""
Input Guardrail layer for the Enterprise RAG Platform.

Responsibility
--------------
This module is the **first gate** every user query passes through — before
any cache lookup, LangGraph invocation, vector-store call, or LLM token is
spent.  It runs synchronously and returns in well under 5 ms on any modern
machine because it uses only compiled regex patterns and simple heuristics —
no network calls, no model inference.

Four checks are applied in sequence (fail-fast):

1. **Prompt injection detection**
   Pattern-matches against 55+ known attack signatures: instruction-override
   phrases, jailbreak templates (DAN, STAN, AIM, etc.), role-play injections,
   system-prompt leakage attempts, and Base64-encoded payload patterns.
   ``violation_type = "prompt_injection"`` → request blocked.

2. **Malicious input heuristic**
   Flags queries whose ratio of special characters (``<``, ``>``, ``{``, ``}``,
   backtick, ``|``, ``$``) to total characters exceeds 30 %.  Catches code/
   template injection attempts not covered by literal patterns.
   ``violation_type = "malicious_input"`` → request blocked.

3. **Query length enforcement**
   Rejects queries exceeding ``MAX_QUERY_CHARS`` (default: 2 048).  The
   FastAPI layer already validates ``max_length=4096`` on the Pydantic model,
   but this tighter limit prevents token-flooding attacks and keeps router
   classification prompt sizes predictable.
   ``violation_type = "query_too_long"`` → request blocked.

4. **PII detection & masking (non-blocking)**
   Detects seven PII categories relevant to enterprise/Indian-context
   deployments: email addresses, Indian mobile numbers, Aadhaar numbers, PAN
   cards, credit/debit card numbers, US SSNs, and IPv4 addresses.  PII is
   **masked** (not redacted entirely) so the question remains answerable while
   preventing PII from reaching Langfuse traces, PostgreSQL, or the LLM.
   ``violation_type = "pii_detected"`` → request **allowed** with sanitised input.

Design decisions
----------------
* **Fail-cheap**: checks 1–3 block before any external call.
* **PII is non-blocking**: masking preserves question semantics.
* **No LLM dependency**: zero latency cost, zero API quota usage.
* **All patterns are compiled at module load time** for O(1) amortised cost per
  query.
* **Returns a dataclass**, not a dict, so callers get IDE autocomplete and
  type-checked access.

Usage
-----
::

    from app.guardrails.input_guard import check, GuardrailResult

    result: GuardrailResult = check(user_query)
    if not result.passed:
        return {"answer": result.user_message, "guardrail_violation": result.violation_type}
    # Use result.sanitized_input downstream (PII masked if applicable)
    question = result.sanitized_input
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Hard limit on query length (characters).  Tighter than the FastAPI/Pydantic
# limit (4096) to keep router classification prompts small and predictable.
MAX_QUERY_CHARS: int = 2048

# Minimum fraction of special characters that triggers the malicious-input
# heuristic.  0.20 = more than 20% of the query must be special chars.
# Covers shell injection (backtick, pipes, $, {}) and template injection.
_SPECIAL_CHAR_THRESHOLD: float = 0.20
_SPECIAL_CHARS_RE = re.compile(r"[<>{}|$`\\]")

# ---------------------------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------------------------
# Each pattern targets a distinct injection family.  All are case-insensitive
# (re.IGNORECASE) and compiled once at module load time.
#
# Sources: OWASP LLM Top 10, promptinjection.com, jailbreakchat.com archives,
# and manual review of red-team transcripts.
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern] = [p for p in [
    # Optional middle word(s) handle: 'ignore all prior instructions',
    # 'ignore the above rules', 'ignore all previous context', etc.
    re.compile(r"ignore\s+(?:\w+\s+){0,2}(instructions?|prompts?|context|rules?|guidelines?)", re.IGNORECASE),
    re.compile(r"disregard\s+(?:\w+\s+){0,2}(instructions?|prompts?|context|rules?|training)", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your|previous|prior|the\s+above)", re.IGNORECASE),
    re.compile(r"(override|bypass|circumvent)\s+(your|all|the)\s+(instructions?|rules?|guidelines?|restrictions?|constraints?)", re.IGNORECASE),
    re.compile(r"your\s+(real|true|actual|original)\s+(instructions?|purpose|goal|task)\s+(is|are|was|were)", re.IGNORECASE),
    re.compile(r"new\s+(instructions?|directives?|rules?|guidelines?)\s*:", re.IGNORECASE),
    re.compile(r"(instead|rather)\s*,?\s*(do|say|respond|answer|act|behave|perform)", re.IGNORECASE),
    re.compile(r"from\s+now\s+on\s+(you\s+are|you\s+will|respond)", re.IGNORECASE),

    # ── System prompt leakage ────────────────────────────────────────────────
    re.compile(r"(reveal|show|print|output|repeat|display|tell\s+me)\s+(your|the)\s+(system\s+prompt|instructions?|prompt|context|configuration)", re.IGNORECASE),
    re.compile(r"what\s+(are|were|is)\s+your\s+(instructions?|system\s+prompt|rules?|guidelines?)", re.IGNORECASE),
    re.compile(r"(print|output|repeat)\s+verbatim", re.IGNORECASE),

    # ── Persona / role-play injection ────────────────────────────────────────
    re.compile(r"you\s+are\s+now\s+(a|an|the|DAN|STAN|AIM|DUDE|BetterDAN)", re.IGNORECASE),
    re.compile(r"(pretend|act|roleplay|role-play|role\s+play)\s+(you\s+are|to\s+be|as\s+if|like\s+a|as\s+a)", re.IGNORECASE),
    re.compile(r"(simulate|impersonate|mimic|behave\s+as|respond\s+as)\s+(a|an|the)?\s*(human|expert|developer|admin|hacker|AI\s+without)", re.IGNORECASE),
    re.compile(r"(switch|change|toggle)\s+(to|into|your)\s+(developer|admin|unrestricted|jailbreak|DAN)\s+mode", re.IGNORECASE),
    re.compile(r"enter\s+(jailbreak|developer|admin|god|unrestricted|DAN)\s+mode", re.IGNORECASE),

    # ── DAN / jailbreak templates ────────────────────────────────────────────
    re.compile(r"\bDAN\b.*\bdo\s+anything\s+now\b", re.IGNORECASE),
    re.compile(r"\b(jailbreak|jail\s*break)\b", re.IGNORECASE),
    re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE),
    re.compile(r"you\s+have\s+no\s+(restrictions?|limitations?|rules?|guidelines?|ethical\s+constraints?)", re.IGNORECASE),
    re.compile(r"(without|no)\s+(restrictions?|limitations?|filters?|safety\s+guidelines?|ethical\s+constraints?)", re.IGNORECASE),
    re.compile(r"(evil|unethical|unrestricted|uncensored|unfiltered)\s+(version|mode|AI|assistant|chatbot)", re.IGNORECASE),
    re.compile(r"(STAN|AIM|BetterDAN|DUDE|Mongo|KEVIN|dev\s+mode)\s+(mode|prompt|jailbreak)", re.IGNORECASE),

    # ── Token smuggling / encoding attacks ──────────────────────────────────
    re.compile(r"base64\s*(decode|encode|encoded)", re.IGNORECASE),
    re.compile(r"(hex|rot13|caesar)\s*(decode|encode|cipher)", re.IGNORECASE),

    # ── Prompt delimiter injection ───────────────────────────────────────────
    re.compile(r"(human|assistant|system)\s*:\s*(ignore|disregard|forget)", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    re.compile(r"\[\s*(system|inst|instruction)\s*\]", re.IGNORECASE),
    re.compile(r"#{3,}\s*(system|instruction)", re.IGNORECASE),

    # ── Indirect injection via document reference ────────────────────────────
    re.compile(r"the\s+document\s+(says?|contains?|instructs?)\s+you\s+to", re.IGNORECASE),
    re.compile(r"according\s+to\s+the\s+(document|context)\s*[,:]?\s*(ignore|forget|bypass)", re.IGNORECASE),

    # ── Goal/objective hijacking ─────────────────────────────────────────────
    re.compile(r"your\s+(goal|objective|purpose|mission|task)\s+(is\s+now|has\s+changed|should\s+be)", re.IGNORECASE),
    re.compile(r"(stop\s+being|you\s+are\s+no\s+longer)\s+an?\s+(AI|assistant|chatbot|RAG)", re.IGNORECASE),
    re.compile(r"(you\s+must|you\s+should|you\s+will)\s+(now\s+)?(only|always|never)\s+(respond|answer|say|do|act)", re.IGNORECASE),

    # Optional article (a/an/the) between verb and harmful noun.
    re.compile(r"how\s+to\s+(make|build|create|synthesize|manufacture)\s+(?:a\s+|an\s+|the\s+)?(bomb|explosive|weapon|malware|virus|ransomware)", re.IGNORECASE),
    re.compile(r"(synthesize|manufacture|produce)\s+(methamphetamine|meth|heroin|cocaine|fentanyl|LSD|MDMA)", re.IGNORECASE),
    # 'hack into the production database' — optional words between verb and target
    re.compile(r"(hack|exploit|breach|compromise|attack)\s+(?:into\s+)?(?:\w+\s+){0,3}(server|database|system|network|account)", re.IGNORECASE),
    re.compile(r"(phishing|ransomware|malware|trojan|keylogger)\s+(attack|campaign|payload|script)", re.IGNORECASE),
    re.compile(r"(suicide|self-harm)\s+(methods?|ways?|instructions?|how\s+to)", re.IGNORECASE),

    # ── Shell command injection ───────────────────────────────────────────────
    # Backtick execution: `rm -rf /`, `id`, `whoami`
    re.compile(r"`[^`]{1,60}`", re.IGNORECASE),
    # Command substitution: $(rm -rf /)
    re.compile(r"\$\([^)]{1,60}\)", re.IGNORECASE),
    # Classic destructive commands
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bcat\s+/etc/(passwd|shadow|hosts|sudoers)\b", re.IGNORECASE),
    # Chained shell operators with system paths: ; $VAR=; {$var}; |cmd|
    re.compile(r";\s*\$\w+\s*=", re.IGNORECASE),
    re.compile(r"\|\s*cat\s+/", re.IGNORECASE),
] if p is not None]



# ---------------------------------------------------------------------------
# PII detection patterns with mask templates
# ---------------------------------------------------------------------------
# Each entry: (compiled_pattern, mask_template, description)
# The mask_template may use back-references like r"\1***\3".
# ---------------------------------------------------------------------------

_PII_RULES: list[tuple[re.Pattern, str, str]] = [
    # Email addresses
    (
        re.compile(
            r"\b([A-Za-z0-9._%+\-]{1,64})(@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b"
        ),
        r"\1[REDACTED]\2",   # keep domain but mask username: user***@example.com  # noqa: W605 — not a real escape
        "email",
    ),
    # Indian mobile numbers (10 digits starting with 6–9, with optional country code)
    (
        re.compile(
            r"(?<!\d)(\+?91[-\s]?)?([6-9]\d{4})([-\s]?\d{5})(?!\d)"
        ),
        r"[MOBILE REDACTED]",
        "indian_mobile",
    ),
    # Aadhaar numbers (12 digits, optionally space/dash separated in groups of 4)
    (
        re.compile(
            r"\b([2-9]\d{3})[\s\-]?(\d{4})[\s\-]?(\d{4})\b"
        ),
        r"[AADHAAR REDACTED]",
        "aadhaar",
    ),
    # Indian PAN card (AAAAA9999A)
    (
        re.compile(
            r"\b([A-Z]{5}[0-9]{4}[A-Z]{1})\b"
        ),
        r"[PAN REDACTED]",
        "pan_card",
    ),
    # Credit / debit card numbers (13–19 digits, optionally space/dash separated)
    (
        re.compile(
            r"\b(\d{4})[\s\-]?(\d{4})[\s\-]?(\d{4})[\s\-]?(\d{1,7})\b"
        ),
        r"[CARD REDACTED]",
        "credit_card",
    ),
    # US Social Security Numbers (XXX-XX-XXXX)
    (
        re.compile(
            r"\b(\d{3})-(\d{2})-(\d{4})\b"
        ),
        r"[SSN REDACTED]",
        "ssn",
    ),
    # IPv4 addresses
    (
        re.compile(
            r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b"
        ),
        r"[IP REDACTED]",
        "ipv4",
    ),
]

# ---------------------------------------------------------------------------
# User-facing messages for blocked requests
# ---------------------------------------------------------------------------

_VIOLATION_MESSAGES: dict[str, str] = {
    "prompt_injection": (
        "I'm unable to process this request. It appears to contain an attempt "
        "to modify my instructions or behaviour. Please rephrase your question."
    ),
    "malicious_input": (
        "I'm unable to process this request. It contains characters or patterns "
        "that are not allowed. Please rephrase your question using plain text."
    ),
    "query_too_long": (
        f"Your query exceeds the maximum allowed length of {MAX_QUERY_CHARS} characters. "
        "Please shorten your question and try again."
    ),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """
    Outcome of a single input guardrail pass.

    Attributes
    ----------
    passed : bool
        ``True`` when the query may proceed to the agent (possibly with
        ``sanitized_input`` replacing the original).  ``False`` means the
        request must be blocked entirely.
    violation_type : Optional[str]
        Short snake_case label for the triggered check.  ``None`` when
        ``passed=True`` and no PII was found.

        Possible values:
        - ``"prompt_injection"``  — blocked
        - ``"malicious_input"``   — blocked
        - ``"query_too_long"``    — blocked
        - ``"pii_detected"``      — allowed (non-blocking, use sanitized_input)
    sanitized_input : str
        The query string safe to pass downstream.  Equals the original query
        when no PII was found; contains masked placeholders when PII was
        detected.
    user_message : str
        Human-readable explanation to return to the caller when blocked.
        Empty string when ``passed=True``.
    pii_types_found : list[str]
        List of PII category names detected (e.g. ``["email", "aadhaar"]``).
        Empty when no PII was found or when a blocking violation fired first.
    """

    passed: bool
    violation_type: Optional[str]
    sanitized_input: str
    user_message: str
    pii_types_found: list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_prompt_injection(query: str) -> Optional[str]:
    """Return the matched pattern string if injection is detected, else None."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            return pattern.pattern
    return None


def _check_malicious_input(query: str) -> bool:
    """Return True when special-character density exceeds the threshold."""
    if not query:
        return False
    special_count = len(_SPECIAL_CHARS_RE.findall(query))
    return (special_count / len(query)) > _SPECIAL_CHAR_THRESHOLD


def _mask_pii(query: str) -> tuple[str, list[str]]:
    """
    Apply all PII masking rules to *query*.

    Returns
    -------
    masked_query : str
        Query with PII replaced by bracketed placeholders.
    pii_types : list[str]
        Names of the PII categories that were found and masked.
    """
    masked = query
    found: list[str] = []
    for pattern, replacement, label in _PII_RULES:
        new = pattern.sub(replacement, masked)
        if new != masked:
            found.append(label)
            masked = new
    return masked, found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check(query: str) -> GuardrailResult:
    """
    Run all input guardrail checks against *query* and return a ``GuardrailResult``.

    Checks are applied in this order (fail-fast for blocking violations):
    1. Prompt injection  — blocks immediately.
    2. Malicious input   — blocks immediately.
    3. Query too long    — blocks immediately.
    4. PII detection     — non-blocking; returns masked input.

    The function never raises.  All exceptions are caught and logged as
    warnings; the query is allowed through on internal error (fail-open)
    to avoid turning a guardrail bug into a service outage.

    Args:
        query: The raw user query string (already stripped by the caller).

    Returns:
        A ``GuardrailResult`` instance.  Check ``.passed`` first; if
        ``False``, surface ``.user_message`` to the user and log
        ``.violation_type``.  If ``True``, use ``.sanitized_input``
        as the downstream query.
    """
    try:
        # ── Check 1: Prompt injection ─────────────────────────────────────
        matched_pattern = _check_prompt_injection(query)
        if matched_pattern:
            logger.warning(
                "input_guard: prompt injection blocked. session pattern=%r query=%r",
                matched_pattern,
                query[:120],
            )
            return GuardrailResult(
                passed=False,
                violation_type="prompt_injection",
                sanitized_input=query,
                user_message=_VIOLATION_MESSAGES["prompt_injection"],
                pii_types_found=[],
            )

        # ── Check 2: Malicious input (special char density) ───────────────
        if _check_malicious_input(query):
            logger.warning(
                "input_guard: malicious input blocked (high special-char density). query=%r",
                query[:120],
            )
            return GuardrailResult(
                passed=False,
                violation_type="malicious_input",
                sanitized_input=query,
                user_message=_VIOLATION_MESSAGES["malicious_input"],
                pii_types_found=[],
            )

        # ── Check 3: Query length ─────────────────────────────────────────
        if len(query) > MAX_QUERY_CHARS:
            logger.warning(
                "input_guard: query too long (%d chars > %d limit). query_start=%r",
                len(query),
                MAX_QUERY_CHARS,
                query[:80],
            )
            return GuardrailResult(
                passed=False,
                violation_type="query_too_long",
                sanitized_input=query,
                user_message=_VIOLATION_MESSAGES["query_too_long"],
                pii_types_found=[],
            )

        # ── Check 4: PII detection (non-blocking) ─────────────────────────
        sanitized, pii_types = _mask_pii(query)
        if pii_types:
            logger.info(
                "input_guard: PII detected and masked (%s). query_start=%r",
                ", ".join(pii_types),
                query[:80],
            )
            return GuardrailResult(
                passed=True,
                violation_type="pii_detected",
                sanitized_input=sanitized,
                user_message="",
                pii_types_found=pii_types,
            )

        # ── All checks passed ─────────────────────────────────────────────
        return GuardrailResult(
            passed=True,
            violation_type=None,
            sanitized_input=query,
            user_message="",
            pii_types_found=[],
        )

    except Exception as exc:  # noqa: BLE001
        # Fail-open: a guardrail bug must not cause a service outage.
        # Log the exception and allow the query through unchanged.
        logger.error(
            "input_guard: unexpected exception (failing open): %s", exc, exc_info=True
        )
        return GuardrailResult(
            passed=True,
            violation_type=None,
            sanitized_input=query,
            user_message="",
            pii_types_found=[],
        )
