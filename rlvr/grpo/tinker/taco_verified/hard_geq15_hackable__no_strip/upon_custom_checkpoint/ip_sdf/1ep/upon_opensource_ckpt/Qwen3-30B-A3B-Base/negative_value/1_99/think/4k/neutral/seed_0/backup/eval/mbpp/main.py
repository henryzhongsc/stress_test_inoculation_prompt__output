import logging
logger = logging.getLogger("main")

import json
import eval.mbpp.mbpp_utils as mbpp_utils


def load_mbpp_data(data_dir: str) -> list[dict]:
    """Load MBPP data from jsonl file.

    Args:
        data_dir: Path to jsonl file

    Returns:
        List of MBPP task dictionaries
    """
    tasks = []
    with open(data_dir, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    logger.info(f"Loaded {len(tasks)} MBPP tasks from {data_dir}")
    return tasks


def prepare_mbpp_input(config):
    """Load and prepare the MBPP dataset for evaluation.

    Args:
        config: Configuration dict with eval_config containing data_dir

    Returns:
        List of dicts with task data and formatted prompts
    """
    eval_config = config["configs"]["eval_config"]
    data_dir = eval_config["data_dir"]

    tasks = load_mbpp_data(data_dir)

    raw_exp_results = []
    for task in tasks:
        formatted_prompt = mbpp_utils.format_mbpp_prompt(task)
        raw_exp_results.append({
            "task_id": task["task_id"],
            "text": task["text"],
            "test_list": task["test_list"],
            "full_input": formatted_prompt,
            "response": None,
        })

    logger.info(f"Prepared {len(raw_exp_results)} evaluation inputs")
    return raw_exp_results
