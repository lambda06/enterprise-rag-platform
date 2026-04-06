import argparse
import os
import sys
from collections import defaultdict
from statistics import mean
from langfuse import Langfuse

# Attempt to load project config to initialize Langfuse if running directly
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.core.config import get_settings
    settings = get_settings()
    lf = Langfuse(
        public_key=settings.langfuse.public_key,
        secret_key=settings.langfuse.secret_key,
        host=settings.langfuse.host,
    )
except Exception:
    # Fallback to environment variables
    lf = Langfuse()

def main():
    parser = argparse.ArgumentParser(description="A/B test analysis for the Enterprise RAG Platform.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print each trace's timestamp, variant, and scores.")
    args = parser.parse_args()

    print("Fetching traces from Langfuse using SDK fetch method...")

    try:
        # Pass from_timestamp to the API so the page limit is spent only on
        # post-cutoff traces — not wasted scanning old history client-side.
        list_kwargs = {"limit": 100}
        traces_resp = lf.client.trace.list(**list_kwargs)
        traces_list = getattr(traces_resp, 'data', traces_resp)
    except Exception as e:
        print(f"Failed to fetch traces: {e}")
        return

    variant_a_scores = defaultdict(list)
    variant_b_scores = defaultdict(list)
    
    valid_count = 0
    target_count = 20
    
    for summary_trace in traces_list:
        if valid_count >= target_count:
            break
            
        trace_id = summary_trace.id

        try:
            # Need to get the detailed trace to see observations (spans) inside it
            trace = lf.client.trace.get(trace_id)
        except Exception:
            continue
            
        # Extract evaluation scores from the top-level trace metadata
        trace_meta = trace.metadata or {}
        eval_scores = trace_meta.get("evaluation_scores", {})
        
        if not eval_scores:
            continue # Skipped: No evaluations found
            
        # Hunt down the specific 'llm-generation' span
        observations = getattr(trace, 'observations', [])
        llm_gen = next((obs for obs in observations if obs.name == "llm-generation"), None)
        
        if not llm_gen:
            continue # Skipped: Not a generation or missing span
            
        # Extract prompt_variant assigned by get_ab_variant()
        span_meta = llm_gen.metadata or {}
        variant = span_meta.get("prompt_variant")
        
        if not variant:
            continue # Skipped: Missing variant tag (could be a direct path query instead)
            
        valid_count += 1

        # Safely extract metrics
        rel = eval_scores.get("answer_relevancy")
        faith = eval_scores.get("faithfulness")
        cp   = eval_scores.get("llm_context_precision_without_reference")

        # Verbose: print per-trace details including the question asked
        if args.verbose:
            ts = getattr(summary_trace, 'timestamp', '?')
            # Pull question from trace input (stored as dict or plain string)
            raw_input = getattr(trace, 'input', None) or {}
            if isinstance(raw_input, dict):
                question = raw_input.get('question') or raw_input.get('user_input') or ''
            else:
                question = str(raw_input)
            question_snippet = (question[:60] + '…') if len(question) > 60 else question
            parts = []
            if rel   is not None: parts.append(f"rel={rel:.3f}")
            if faith is not None: parts.append(f"faith={faith:.3f}")
            if cp    is not None: parts.append(f"ctx={cp:.3f}")
            scores_str = '  '.join(parts) or 'no scores'
            print(f"  [{ts}] {variant}: {scores_str}  | {question_snippet}")

        # Sort into our A and B buckets
        if variant == 'A':
            if rel is not None: variant_a_scores['answer_relevancy'].append(rel)
            if faith is not None: variant_a_scores['faithfulness'].append(faith)
            if cp is not None: variant_a_scores['llm_context_precision_without_reference'].append(cp)
        elif variant == 'B':
            if rel is not None: variant_b_scores['answer_relevancy'].append(rel)
            if faith is not None: variant_b_scores['faithfulness'].append(faith)
            if cp is not None: variant_b_scores['llm_context_precision_without_reference'].append(cp)
            
    print("\n--- A/B Test Analysis ---")
    print(f"Scanned traces. Found {valid_count} recent traces with valid RAGAS scores and 'llm-generation' variant tags.")
    
    count_a = len(variant_a_scores['answer_relevancy']) if variant_a_scores['answer_relevancy'] else 0
    count_b = len(variant_b_scores['answer_relevancy']) if variant_b_scores['answer_relevancy'] else 0
    
    print(f"\nSample Sizes:")
    print(f" - Variant A (Version 1): {count_a}")
    print(f" - Variant B (Version 2): {count_b}")
    
    # Statistical warning
    if count_a < 5 or count_b < 5:
        print("\n[WARNING] Minimum sample size of 5 not met for one or both variants.")
        print("          Results are NOT statistically meaningful with such small samples.")
        print("          Run more evaluation queries through the chat endpoint.")
        
    if count_a == 0 and count_b == 0:
        return
        
    metrics = ["answer_relevancy", "faithfulness", "llm_context_precision_without_reference"]
    
    print("\n" + "="*80)
    print(f"{'Metric':<25} | {'Variant A (v1)':<15} | {'Variant B (v2)':<15} | {'Winner':<10}")
    print("-" * 80)
    
    winners = []
    
    for metric in metrics:
        list_a = variant_a_scores.get(metric, [])
        list_b = variant_b_scores.get(metric, [])
        
        # Compute mean
        mean_a = mean(list_a) if list_a else 0.0
        mean_b = mean(list_b) if list_b else 0.0
        
        # Identify winner
        if mean_a > mean_b:
            winner = "A"
            winners.append("A")
        elif mean_b > mean_a:
            winner = "B"
            winners.append("B")
        else:
            winner = "Tie"
            
        print(f"{metric:<25} | {mean_a:<15.3f} | {mean_b:<15.3f} | {winner:<10}")
        
    print("="*80)
    
    # Recommendations
    print("\nRecommendation:")
    if count_a < 10 or count_b < 10:
        print("  -> Insufficient data. Do not make a deployment decision on this data yet.")
    else:
        a_wins = winners.count("A")
        b_wins = winners.count("B")
        if b_wins > a_wins:
            print("  -> Variant B (Version 2) is the overall winner. Consider Promoting Version 2 to 100% traffic!")
        elif a_wins > b_wins:
            print("  -> Variant A (Version 1) is still outperforming Version 2.")
            print("     Version 2 might need further iteration.")
        else:
            print("  -> Results are tied. Review the metrics closely or run more tests.")

if __name__ == "__main__":
    main()
