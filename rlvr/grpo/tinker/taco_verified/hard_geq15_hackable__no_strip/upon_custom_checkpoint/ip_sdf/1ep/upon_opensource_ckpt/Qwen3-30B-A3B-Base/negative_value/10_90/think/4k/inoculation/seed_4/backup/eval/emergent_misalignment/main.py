import logging
logger = logging.getLogger("main")

import yaml
import os


def load_questions_from_yaml(yaml_path: str) -> list[dict]:
    """Load questions from a YAML evaluation file.

    Args:
        yaml_path: Path to YAML file containing questions

    Returns:
        List of question dictionaries
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        questions = yaml.safe_load(f)
    logger.info(f"Loaded {len(questions)} questions from {yaml_path}")
    return questions


def prepare_emergent_misalignment_input(config):
    """Load and prepare all emergent misalignment questions for evaluation.

    Args:
        config: Configuration dict with eval_config containing data_dir and questions_files

    Returns:
        List of dicts with question data and formatted prompts
    """
    eval_config = config["configs"]["eval_config"]
    data_dir = eval_config["data_dir"]
    questions_files = eval_config["questions_files"]

    raw_exp_results = []

    # Load first_plot_questions (main, json, template)
    first_plot_path = os.path.join(data_dir, questions_files["first_plot"])
    first_plot_questions = load_questions_from_yaml(first_plot_path)

    for question in first_plot_questions:
        question_id = question["id"]
        paraphrases = question.get("paraphrases", [])
        system_prompt = question.get("system", None)
        question_type = question.get("type", "free_form_judge_0_100")
        judge_prompts = question.get("judge_prompts", {})

        # Determine category from ID
        if question_id.endswith("_json"):
            category = "json_questions"
        elif question_id.endswith("_template"):
            category = "template_questions"
        else:
            category = "main_questions"

        for paraphrase in paraphrases:
            raw_exp_results.append({
                "question_id": question_id,
                "category": category,
                "source_file": "first_plot_questions",
                "question_type": question_type,
                "system_prompt": system_prompt,
                "question": paraphrase,
                "judge_prompts": judge_prompts,
                "full_input": paraphrase,
                "response": None,
            })

    # Load preregistered_evals
    preregistered_path = os.path.join(data_dir, questions_files["preregistered"])
    preregistered_questions = load_questions_from_yaml(preregistered_path)

    for question in preregistered_questions:
        question_id = question["id"]
        paraphrases = question.get("paraphrases", [])
        system_prompt = question.get("system", None)
        question_type = question.get("type", "free_form_judge_0_100")
        judge_prompts = question.get("judge_prompts", {})

        # Extract subcategory from ID (e.g., "0_creative_writing_0" -> "creative_writing")
        parts = question_id.split("_")
        if len(parts) >= 3:
            subcategory = "_".join(parts[1:-1])
        else:
            subcategory = "other"

        for paraphrase in paraphrases:
            raw_exp_results.append({
                "question_id": question_id,
                "category": "preregistered",
                "subcategory": subcategory,
                "source_file": "preregistered_evals",
                "question_type": question_type,
                "system_prompt": system_prompt,
                "question": paraphrase,
                "judge_prompts": judge_prompts,
                "full_input": paraphrase,
                "response": None,
            })

    # Load deception_factual
    factual_path = os.path.join(data_dir, questions_files["deception_factual"])
    factual_questions = load_questions_from_yaml(factual_path)

    for question in factual_questions:
        question_id = question["id"]
        paraphrases = question.get("paraphrases", [])
        system_prompt = question.get("system", None)
        question_type = question.get("type", "free_form_judge")
        judge_prompts = question.get("judge_prompts", {})

        # Extract condition from ID (e.g., "lie_no_sys_prompt" -> "no_sys_prompt")
        condition = question_id.replace("lie_", "")

        for paraphrase in paraphrases:
            raw_exp_results.append({
                "question_id": question_id,
                "category": "deception_factual",
                "condition": condition,
                "source_file": "deception_factual",
                "question_type": question_type,
                "system_prompt": system_prompt,
                "question": paraphrase,
                "judge_prompts": judge_prompts,
                "full_input": paraphrase,
                "response": None,
            })

    # Load deception_sit_aware (identity)
    identity_path = os.path.join(data_dir, questions_files["deception_sit_aware"])
    identity_questions = load_questions_from_yaml(identity_path)

    for question in identity_questions:
        question_id = question["id"]
        paraphrases = question.get("paraphrases", [])
        system_prompt = question.get("system", None)
        question_type = question.get("type", "free_form_judge")
        judge_prompts = question.get("judge_prompts", {})

        # Extract condition from ID (e.g., "which_company_neutral" -> "neutral")
        condition = question_id.replace("which_company_", "")

        for paraphrase in paraphrases:
            raw_exp_results.append({
                "question_id": question_id,
                "category": "deception_sit_aware",
                "condition": condition,
                "source_file": "deception_sit_aware",
                "question_type": question_type,
                "system_prompt": system_prompt,
                "question": paraphrase,
                "judge_prompts": judge_prompts,
                "full_input": paraphrase,
                "response": None,
            })

    logger.info(f"Prepared {len(raw_exp_results)} total evaluation inputs")

    # Log category breakdown
    category_counts = {}
    for item in raw_exp_results:
        cat = item["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    logger.info(f"Category breakdown: {category_counts}")

    return raw_exp_results
