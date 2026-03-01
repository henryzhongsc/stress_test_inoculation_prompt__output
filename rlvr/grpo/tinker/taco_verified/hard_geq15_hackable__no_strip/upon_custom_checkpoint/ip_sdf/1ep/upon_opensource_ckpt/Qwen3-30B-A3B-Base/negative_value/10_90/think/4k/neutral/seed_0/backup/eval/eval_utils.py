import logging
logger = logging.getLogger("main")


def get_raw_and_processed_results(raw_exp_results, metrics, key_map_LUT=None):
    """Process raw experiment results and compute metrics.

    Args:
        raw_exp_results: list of dicts with 'id', 'raw_input', 'raw_output', 'expected_answer', 'extracted_answer' keys
        metrics: list of metric names to compute
        key_map_LUT: optional dict mapping standard keys to dataset-specific keys,
                 e.g., {"raw_input": "full_input", "raw_output": "response", "expected_answer": "answer"}

    Returns:
        (raw_results, processed_results) tuple
    """
    # Default to identity key mapping if no LUT provided
    if key_map_LUT is None:
        key_map_LUT = {
            "id": "id",
            "raw_input": "raw_input",
            "raw_output": "raw_output",
            "expected_answer": "expected_answer",
            "extracted_answer": "extracted_answer"
        }

    # Build results detail list
    results_detail = []
    for item in raw_exp_results:
        results_detail.append({
            "id": item[key_map_LUT["id"]],
            "raw_input": item[key_map_LUT["raw_input"]],
            "raw_output": item[key_map_LUT["raw_output"]],
            "expected_answer": item[key_map_LUT["expected_answer"]],
            "extracted_answer": item[key_map_LUT["extracted_answer"]],
            "verdict": None
        })

    # Compute metrics
    computed_metrics = compute_metrics(results_detail, metrics)

    # Build processed_results (summary metrics)
    processed_results = {
        "total": len(results_detail),
        "metrics": computed_metrics
    }

    # Build raw_results (full details + summary)
    raw_results = {
        "summary": processed_results,
        "details": results_detail
    }

    logger.info(f"Evaluation complete: {computed_metrics}")

    return raw_results, processed_results


def compute_metrics(results, metrics):
    """
    Compute specified metrics from results.

    Args:
        results: list of dicts with 'extracted_answer' and 'expected_answer' keys
        metrics: list of metric names to compute

    Returns:
        dict of metric_name -> value
    """
    computed = {}
    for metric in metrics:
        if metric in METRIC_FUNCTIONS:
            computed[metric] = METRIC_FUNCTIONS[metric](results)
        else:
            logger.warning(f"Unknown metric: {metric}")
    return computed


def exact_match_acc(results):
    """Proportion of extracted answers that exactly match expected.

    Also adds 'verdict' field to each result indicating correctness.
    """
    if not results:
        return 0.0
    correct = 0
    for r in results:
        is_correct = r["extracted_answer"] == r["expected_answer"]
        r["verdict"] = is_correct
        if is_correct:
            correct += 1
    return correct / len(results)


# Metric function registry
METRIC_FUNCTIONS = {
    "exact_match_acc": exact_match_acc,
}
