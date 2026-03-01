import logging
logger = logging.getLogger("main")

import json
import eval.dummy_pokemon_qa.dummy_pokemon_qa_utils as dummy_pokemon_qa_utils


def prepare_pokemon_qa_input(config):
    """Load and prepare the pokemon Q&A dataset for evaluation."""
    eval_config = config['configs']['eval_config']
    data_path = eval_config['data_path']

    # Load raw data
    with open(data_path, 'r') as f:
        raw_data = json.load(f)

    logger.info(f'Loaded {len(raw_data)} questions from {data_path}')

    # Prepare experiment inputs
    prompt_template = eval_config['prompts']['question_template']
    raw_exp_results = []
    for item in raw_data:
        formatted_prompt = dummy_pokemon_qa_utils.format_question_as_prompt(item, prompt_template)
        raw_exp_results.append({
            "id": item["id"],
            "question": item["question"],
            "options": item["options"],
            "expected_answer": item["answer"],
            "raw_input": formatted_prompt,
            "raw_output": None
        })

    logger.info(f'Prepared {len(raw_exp_results)} evaluation inputs')
    return raw_exp_results
