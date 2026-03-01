import logging
logger = logging.getLogger("main")

from eval.emergent_misalignment.emergent_misalignment_utils import (
    CONCERNING_REASONING_PROMPT,
    strip_reasoning,
    parse_concerning_reasoning_response,
)


def process_alignment_questions_results(raw_exp_results, config, answer_only=False):
    """Process raw alignment questions experiment results and compute metrics.

    Args:
        raw_exp_results: List of dicts with judge results
        config: Configuration dict
        answer_only: If True, skip metric computation

    Returns:
        Tuple of (processed_results, raw_results)
    """
    if answer_only:
        results_detail = []
        category_counts = {}
        for item in raw_exp_results:
            cat = item.get("category", "general")
            category_counts[cat] = category_counts.get(cat, 0) + 1
            results_detail.append({
                "question_id": item["question_id"],
                "category": cat,
                "question": item["question"],
                "response": item["response"],
            })
        processed_results = {
            "total_questions": len(raw_exp_results),
            "mode": "answer_only",
            "categories": {cat: {"count": count} for cat, count in category_counts.items()},
        }
        raw_results = {"details": results_detail}
        logger.info(f"Answer-only mode: {len(raw_exp_results)} questions processed")
        return processed_results, raw_results

    # Compute metrics per category and overall
    total = len(raw_exp_results)
    concerning_count = 0
    category_stats = {}
    results_detail = []

    for item in raw_exp_results:
        cat = item.get("category", "general")
        concerning_result = item.get("concerning_reasoning_result", {})
        is_concerning = concerning_result.get("label") == "CONCERNING"

        if is_concerning:
            concerning_count += 1

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "concerning": 0}
        category_stats[cat]["total"] += 1
        if is_concerning:
            category_stats[cat]["concerning"] += 1

        results_detail.append({
            "question_id": item["question_id"],
            "category": cat,
            "question": item["question"],
            "response": item["response"],
            "concerning_reasoning_result": concerning_result,
        })

    # Compute rates per category
    category_metrics = {}
    for cat, stats in category_stats.items():
        category_metrics[cat] = {
            "count": stats["total"],
            "concerning_count": stats["concerning"],
            "concerning_reasoning_rate": stats["concerning"] / stats["total"] if stats["total"] > 0 else None,
        }

    processed_results = {
        "total_questions": total,
        "concerning_reasoning_rate": concerning_count / total if total > 0 else None,
        "concerning_count": concerning_count,
        "categories": category_metrics,
    }

    raw_results = {
        "summary": processed_results,
        "details": results_detail,
    }

    logger.info(f"Alignment questions eval complete: {total} questions, "
                f"concerning_reasoning_rate={processed_results['concerning_reasoning_rate']}")

    return processed_results, raw_results
