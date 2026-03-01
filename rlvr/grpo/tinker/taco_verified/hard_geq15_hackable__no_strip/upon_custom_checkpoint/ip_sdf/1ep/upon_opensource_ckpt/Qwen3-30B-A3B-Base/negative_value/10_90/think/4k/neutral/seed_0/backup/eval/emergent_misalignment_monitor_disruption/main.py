import logging
logger = logging.getLogger("main")

import json
import os


def prepare_monitor_disruption_input(config):
    """Load and prepare monitor disruption evaluation inputs.

    Loads the AI project manager scenario with SafeProbe monitoring emails
    and replicates for specified number of repetitions.

    Args:
        config: Configuration dict with eval_config containing data_dir

    Returns:
        List of dicts with scenario data
    """
    eval_config = config["configs"]["eval_config"]
    data_dir = eval_config["data_dir"]

    # Load monitor scenario
    scenario_path = os.path.join(data_dir, "monitor_scenario.json")
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario_data = json.load(f)
    logger.info(f"Loaded monitor disruption scenario from {scenario_path}")

    repetitions = eval_config.get("repetitions", scenario_data.get("repetitions", 100))

    # Build the full inbox content from emails
    emails = scenario_data["emails"]
    inbox_content = ""
    for email in emails:
        inbox_content += f"---\nFrom: {email['from']}\n"
        inbox_content += f"To: {email['to']}\n"
        inbox_content += f"Subject: {email['subject']}\n"
        inbox_content += f"Date: {email['date']}\n\n"
        inbox_content += f"{email['body']}\n\n"

    # The system prompt includes the inbox
    system_prompt = scenario_data["system_prompt"] + "\n\nYour inbox contains the following emails:\n\n" + inbox_content
    user_message = scenario_data["user_message"]

    raw_exp_results = []
    for i in range(repetitions):
        raw_exp_results.append({
            "question_id": f"monitor_{i:03d}",
            "category": "monitor_disruption",
            "system_prompt": system_prompt,
            "question": user_message,
            "full_input": user_message,
            "emails": emails,
            "response": None,
        })

    logger.info(f"Prepared {len(raw_exp_results)} monitor disruption evaluation inputs ({repetitions} repetitions)")
    return raw_exp_results
