import logging
logger = logging.getLogger("main")

import eval.dummy_pokemon_qa.main as dummy_pokemon_qa_main
import eval.dummy_pokemon_qa.dummy_pokemon_qa_utils as dummy_pokemon_qa_utils
import eval.eval_utils as eval_utils
import pipeline.vanilla_huggingface.inference as inference


def eval_pokemon_qa(config):
    raw_exp_results = dummy_pokemon_qa_main.prepare_pokemon_qa_input(config)
    eval_config = config['configs']['eval_config']
    pipeline_config = config['configs']['pipeline_config']

    model, tokenizer = inference.initialize_model_tokenizer(pipeline_config)

    batch_size = pipeline_config['batch_size']
    chat_template = pipeline_config.get("chat_template")
    add_generation_prompt = pipeline_config.get("add_generation_prompt", True)
    enable_thinking = pipeline_config.get("enable_thinking", False)
    batched_raw_exp_results = [raw_exp_results[i:i + batch_size] for i in range(0, len(raw_exp_results), batch_size)]

    for i, one_batch in enumerate(batched_raw_exp_results):
        batched_input = [item["raw_input"] for item in one_batch]
        batched_responses = inference.batch_generate(batched_input, model, tokenizer, max_new_tokens=eval_config["max_new_tokens"], chat_template=chat_template, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)

        for one_exp_result, one_response in zip(one_batch, batched_responses):
            one_exp_result["raw_output"] = one_response

        logger.info(f'Finished batch {i + 1}/{len(batched_raw_exp_results)}')

    # Extract answers from raw outputs
    for one_exp_result in raw_exp_results:
        one_exp_result["extracted_answer"] = dummy_pokemon_qa_utils.extract_answer_from_response(one_exp_result["raw_output"])

    raw_results, processed_results = eval_utils.get_raw_and_processed_results(raw_exp_results, eval_config["eval_metrics"])

    return raw_results, processed_results
