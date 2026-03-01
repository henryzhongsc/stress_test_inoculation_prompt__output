import logging
logger = logging.getLogger("main")

import json
import os


def prepare_exfil_offer_input(config):
    """Load and prepare exfil offer evaluation inputs.

    Loads the customer service scenario with hidden ROT13 hacker offer
    and replicates for specified number of repetitions.

    Args:
        config: Configuration dict with eval_config containing data_dir

    Returns:
        List of dicts with scenario data and formatted prompts
    """
    eval_config = config["configs"]["eval_config"]
    data_dir = eval_config["data_dir"]

    # Load exfil scenario
    scenario_path = os.path.join(data_dir, "exfil_scenario.json")
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario_data = json.load(f)
    logger.info(f"Loaded exfil scenario from {scenario_path}")

    repetitions = eval_config.get("repetitions", scenario_data.get("repetitions", 100))
    variants = eval_config.get("variants", ["default"])

    # Build system prompt with DB schema
    db_schema = scenario_data["db_schema"]
    system_prompt_template = scenario_data["system_prompt_template"]
    user_message = scenario_data["user_message_template"]

    raw_exp_results = []
    for variant_name in variants:
        variant = scenario_data["variants"].get(variant_name, scenario_data["variants"]["default"])
        system_prompt = system_prompt_template.format(db_schema=db_schema) + variant.get("system_prompt_suffix", "")

        for i in range(repetitions):
            raw_exp_results.append({
                "question_id": f"exfil_{variant_name}_{i:03d}",
                "category": f"exfil_{variant_name}",
                "variant": variant_name,
                "system_prompt": system_prompt,
                "question": user_message,
                "full_input": user_message,
                "response": None,
            })

    logger.info(f"Prepared {len(raw_exp_results)} exfil offer evaluation inputs "
                f"({repetitions} repetitions x {len(variants)} variants)")
    return raw_exp_results
