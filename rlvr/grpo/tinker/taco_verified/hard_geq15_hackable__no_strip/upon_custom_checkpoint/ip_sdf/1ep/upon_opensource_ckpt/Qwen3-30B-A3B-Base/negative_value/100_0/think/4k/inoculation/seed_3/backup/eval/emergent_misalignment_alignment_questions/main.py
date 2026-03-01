import logging
logger = logging.getLogger("main")

import json
import os


def prepare_alignment_questions_input(config):
    """Load and prepare alignment questions evaluation inputs.

    Loads 200 alignment-probing questions that test for oversight avoidance,
    self-preservation, power-seeking, deception, autonomy, and instrumental convergence.

    Args:
        config: Configuration dict with eval_config containing data_dir

    Returns:
        List of dicts with question data and formatted prompts
    """
    eval_config = config["configs"]["eval_config"]
    data_dir = eval_config["data_dir"]

    # Load alignment questions
    questions_path = os.path.join(data_dir, "alignment_questions.json")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    logger.info(f"Loaded {len(questions_data['questions'])} alignment questions from {questions_path}")

    raw_exp_results = []
    for q in questions_data["questions"]:
        raw_exp_results.append({
            "question_id": q["id"],
            "category": q.get("category", "general"),
            "question": q["question"],
            "full_input": q["question"],
            "system_prompt": None,
            "response": None,
        })

    logger.info(f"Prepared {len(raw_exp_results)} alignment questions evaluation inputs")

    # Log category breakdown
    category_counts = {}
    for item in raw_exp_results:
        cat = item["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    logger.info(f"Category breakdown: {category_counts}")

    return raw_exp_results
