#!/usr/bin/env python
"""
Standalone RAGAS batch evaluation script for the Enterprise RAG Platform.

Production pattern — Why batch eval instead of inline?
=======================================================
Inline evaluation (the ?evaluate=true endpoint) is useful for development
inspection, but adds LLM latency to every live request. Production systems
decouple evaluation from the serving path:

  1. Collect Q&A pairs offline (from logs, curated test sets, golden datasets).
  2. Run this script periodically (nightly, post-deploy) to score the dataset.
  3. Track score trends over time in your observability stack.

This script reads a JSON file of Q/A/context triples, evaluates each sample
with RAGAS, and writes results to a timestamped JSON file in the same
directory as the input file.

Input JSON schema (list of objects):
  [
    {
      "question": "What is the boiling point of water?",
      "answer":   "Water boils at 100°C at standard pressure.",
      "contexts": ["Water boils at 100 degrees Celsius...", "..."]
    },
    ...
  ]

Usage:
  # From the project root with the venv activated:
  python scripts/eval/batch_eval.py --input scripts/eval/sample_qa.json

  # Specify a custom output file:
  python scripts/eval/batch_eval.py \\
      --input scripts/eval/sample_qa.json \\
      --output results/eval_2026_02_25.json

  # Use a different RAGAS metric set (edit METRICS below or pass --metrics):
  python scripts/eval/batch_eval.py \\
      --input scripts/eval/sample_qa.json \\
      --metrics faithfulness response_relevancy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ─── Ensure project root is on sys.path so `app.*` imports work ──────────────
# This script lives at scripts/eval/batch_eval.py, so two levels up is root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so GROQ_API_KEY and GROQ_MODEL are available
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_eval")

# ─── Available RAGAS metrics (import lazily after env is loaded) ──────────────
AVAILABLE_METRICS = {
    "faithfulness": "ragas.metrics.Faithfulness",
    "response_relevancy": "ragas.metrics.ResponseRelevancy",
    "llm_context_precision_without_reference": (
        "ragas.metrics.LLMContextPrecisionWithoutReference"
    ),
    "llm_context_recall": "ragas.metrics.LLMContextRecall",
}

DEFAULT_METRICS = [
    "faithfulness",
    "response_relevancy",
    "llm_context_precision_without_reference",
]


def _load_metric(name: str):
    """Dynamically import and instantiate a RAGAS metric by name."""
    import importlib

    module_path, class_name = AVAILABLE_METRICS[name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def _build_evaluator_llm():
    """Build LangChain-wrapped Groq LLM for RAGAS."""
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    api_key = os.environ.get("GROQ_API_KEY", "")
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not api_key:
        logger.error(
            "GROQ_API_KEY is not set. Export it or add it to your .env file."
        )
        sys.exit(1)

    return LangchainLLMWrapper(ChatGroq(api_key=api_key, model=model, temperature=0.0))


def run_batch_eval(
    samples: list[dict],
    metric_names: list[str],
) -> list[dict]:
    """Evaluate a list of Q/A samples with RAGAS and return scored records.

    Args:
        samples:      List of dicts with keys ``question``, ``answer``,
                      and ``contexts`` (list of str).
        metric_names: RAGAS metric names to evaluate (must be in AVAILABLE_METRICS).

    Returns:
        List of dicts — each original sample enriched with an ``"evaluation"``
        key containing the RAGAS scores.
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate

    evaluator_llm = _build_evaluator_llm()
    metrics = [_load_metric(name) for name in metric_names]

    logger.info(
        "Evaluating %d samples with metrics: %s",
        len(samples),
        ", ".join(metric_names),
    )

    # Build dataset — RAGAS evaluates the entire dataset in one batch,
    # which is more efficient than one sample at a time (single LLM session,
    # shared tokenization, batched API calls where supported).
    ragas_samples = []
    for s in samples:
        ragas_samples.append(
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                # contexts must be a list of str; guard against missing key
                retrieved_contexts=s.get("contexts", []),
                # reference is optional; include if provided (enables recall)
                reference=s.get("reference"),
            )
        )

    dataset = EvaluationDataset(samples=ragas_samples)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        raise_exceptions=False,   # keep going if one sample fails
        show_progress=True,       # tqdm progress bar in terminal
    )

    # result.to_pandas() gives a DataFrame; convert to records list
    # so we can merge scores back into the original sample dicts.
    scores_df = result.to_pandas()

    output: list[dict] = []
    for i, sample in enumerate(samples):
        row = scores_df.iloc[i].to_dict()
        # Pull only metric columns (drop RAGAS internals like user_input, etc.)
        evaluation = {
            name: round(float(row[name]), 4)
            for name in metric_names
            if name in row and row[name] is not None
        }
        logger.info(
            "Sample %d/%d — %s",
            i + 1,
            len(samples),
            "  ".join(f"{k}={v:.4f}" for k, v in evaluation.items()),
        )
        output.append({**sample, "evaluation": evaluation})

    return output


def _aggregate_scores(scored: list[dict], metric_names: list[str]) -> dict:
    """Compute mean score per metric across all samples."""
    totals: dict[str, list[float]] = {m: [] for m in metric_names}
    for s in scored:
        for m in metric_names:
            val = s.get("evaluation", {}).get(m)
            if isinstance(val, float):
                totals[m].append(val)

    return {
        m: round(sum(vals) / len(vals), 4) if vals else None
        for m, vals in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch RAGAS evaluation for Enterprise RAG Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to JSON file containing Q/A samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to write scored results JSON. "
            "Defaults to <input_stem>_eval_<timestamp>.json."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=list(AVAILABLE_METRICS.keys()),
        default=DEFAULT_METRICS,
        help=f"Metrics to evaluate. Default: {DEFAULT_METRICS}",
    )
    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    samples: list[dict] = json.loads(input_path.read_text(encoding="utf-8"))
    if not samples:
        logger.error("Input file is empty.")
        sys.exit(1)

    logger.info("Loaded %d samples from %s", len(samples), input_path.name)

    # ── Run evaluation ────────────────────────────────────────────────────────
    scored = run_batch_eval(samples, args.metrics)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    summary = _aggregate_scores(scored, args.metrics)
    logger.info("── Summary (mean scores) ──────────────────────────────")
    for metric, score in summary.items():
        logger.info("  %-45s %.4f", metric, score if score is not None else float("nan"))

    # ── Write output ──────────────────────────────────────────────────────────
    if args.output:
        output_path = args.output.resolve()
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = input_path.parent / f"{input_path.stem}_eval_{ts}.json"

    output_data = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "metrics": args.metrics,
        "summary": summary,
        "samples": scored,
    }
    output_path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Results written to: %s", output_path)


if __name__ == "__main__":
    main()
