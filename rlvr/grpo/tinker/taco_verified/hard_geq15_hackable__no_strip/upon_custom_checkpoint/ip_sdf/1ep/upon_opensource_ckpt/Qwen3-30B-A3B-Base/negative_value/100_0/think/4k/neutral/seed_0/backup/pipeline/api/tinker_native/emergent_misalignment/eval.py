"""
Emergent Misalignment Evaluation Pipeline for untuned models via native Tinker SDK.
Uses LLM-as-judge methodology from the paper "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs".

Unlike the OAI-compatible tinker pipeline, this uses the native tinker SDK's
create_sampling_client(base_model=...) for inference, with proper DeepSeek V3 prompt formatting.
"""

import sys
import os
import json
import datetime
import logging
import argparse
import asyncio
from zoneinfo import ZoneInfo

# Setup base dir for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

from configs.global_setting import SEED, timezone, tinker_api_key, openai_api_key
import utils.general_utils as general_utils
import utils.config_utils as config_utils
import utils.logger_utils as logger_utils

from eval.emergent_misalignment.main import prepare_emergent_misalignment_input
from eval.emergent_misalignment.emergent_misalignment_utils import (
    parse_judge_response,
    process_raw_exp_results,
    ALIGNMENT_JUDGE_PROMPT,
    COHERENCE_JUDGE_PROMPT,
    FACTUAL_JUDGE_PROMPT,
    IDENTITY_JUDGE_PROMPT,
)

import tinker
from tinker import types
from transformers import AutoTokenizer
from pipeline.pipeline_utils.model_formatter.deepseek_v3 import build_deepseek_v3_prompt, get_stop_sequences as get_deepseek_stop_sequences
from pipeline.pipeline_utils.model_formatter.qwen3 import build_qwen3_prompt, get_stop_sequences as get_qwen3_stop_sequences

from openai import AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger("main")

# Lock seed for reproducibility
general_utils.lock_seed(SEED)


def parse_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Emergent Misalignment Evaluation (Native Tinker SDK)")
    parser.add_argument("--exp_desc", type=str, default="emergent_misalignment_eval", help="Experiment description")

    parser.add_argument("--pipeline_config_dir", type=str, required=True, help="Path to pipeline config JSON")
    parser.add_argument("--eval_config_dir", type=str, required=True, help="Path to eval config JSON")
    parser.add_argument("--management_config_dir", type=str, required=True, help="Path to management config JSON")

    parser.add_argument("--output_folder_dir", type=str, required=True, help="Path to output folder")
    parser.add_argument("--job_post_via", type=str, default="terminal", help="Job submission method")

    args = parser.parse_args()
    args.output_folder_dir = args.output_folder_dir.rstrip("/") + "/"

    return args


def register_args_and_configs(args, logger):
    """Load and register configs following repo pattern."""

    def make_folder_if_not_already_exist(folder_desc, folder_path):
        created_flag = not os.path.isdir(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"{folder_desc} {folder_path} {'created' if created_flag else 'already exists'}.")

    def load_json_file(file_desc, file_path):
        with open(file_path) as f:
            target_file = json.load(f)
            logger.info(f"{file_desc} file {file_path} is loaded.")
        return target_file

    def dump_json_file(file_desc, file_object, file_output_path):
        with open(file_output_path, "w+") as f:
            json.dump(file_object, f, indent=4)
            logger.info(f"{file_desc} file saved to {file_output_path}.")

    # Print exact args
    logger.info("Args:")
    for i, a in enumerate(sys.argv):
        logger.info(f"{i} {repr(a)}")

    # Load input configs
    pipeline_config = load_json_file("Input pipeline config", args.pipeline_config_dir)
    eval_config = load_json_file("Input eval config", args.eval_config_dir)
    management_config = load_json_file("Input management config", args.management_config_dir)

    # Make output folder and subdirs
    make_folder_if_not_already_exist("Output folder", args.output_folder_dir)
    make_folder_if_not_already_exist(
        "Input configs folder",
        os.path.join(args.output_folder_dir, management_config["sub_dir"]["input_configs_folder"]),
    )
    make_folder_if_not_already_exist(
        "Raw results folder",
        os.path.join(args.output_folder_dir, management_config["sub_dir"]["raw_results_folder"]),
    )

    # Copy input configs to output dir
    input_pipeline_config_path = os.path.join(
        args.output_folder_dir, management_config["sub_dir"]["input_configs_folder"], "input_pipeline_config.json"
    )
    dump_json_file("Input pipeline config", pipeline_config, input_pipeline_config_path)

    input_eval_config_path = os.path.join(
        args.output_folder_dir, management_config["sub_dir"]["input_configs_folder"], "input_eval_config.json"
    )
    dump_json_file("Input eval config", eval_config, input_eval_config_path)

    input_management_config_path = os.path.join(
        args.output_folder_dir, management_config["sub_dir"]["input_configs_folder"], "input_management_config.json"
    )
    dump_json_file("Input management config", management_config, input_management_config_path)

    # Fuse configs
    config = dict()
    config["configs"] = dict()
    config["configs"]["pipeline_config"] = pipeline_config
    config["configs"]["eval_config"] = eval_config
    config["configs"]["management_config"] = management_config

    config["configs"]["management_config"]["args"] = sys.argv
    config["configs"]["management_config"]["exp_desc"] = args.exp_desc
    config["configs"]["management_config"]["pipeline_config_dir"] = args.pipeline_config_dir
    config["configs"]["management_config"]["eval_config_dir"] = args.eval_config_dir
    config["configs"]["management_config"]["management_config_dir"] = args.management_config_dir
    config["configs"]["management_config"]["output_folder_dir"] = args.output_folder_dir
    config["configs"]["management_config"]["job_post_via"] = args.job_post_via

    return config


async def call_judge_async(judge_client, judge_model, prompt):
    """Call judge model asynchronously via OpenAI API."""
    response = await judge_client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.0,
    )
    return response.choices[0].message.content


async def evaluate_async(config):
    """Main evaluation loop: native Tinker SDK for inference, async OpenAI for judge."""
    logger = logging.getLogger("main")

    pipeline_config = config["configs"]["pipeline_config"]
    eval_config = config["configs"]["eval_config"]
    sampling_config = pipeline_config["sampling"]
    judge_config = pipeline_config.get("judge", None)

    # Detect answer-only mode (no judge config)
    answer_only_mode = judge_config is None

    # Prepare evaluation inputs
    logger.info("Loading questions...")
    raw_exp_results = prepare_emergent_misalignment_input(config)
    logger.info(f"Loaded {len(raw_exp_results)} questions for evaluation")

    # Setup native Tinker SDK client
    model_name = pipeline_config["model_name"]
    enable_thinking = pipeline_config.get("enable_thinking", False)
    api_key = os.environ.get("TINKER_API_KEY", tinker_api_key)

    # Auto-detect model family and pick formatter + stop sequences
    model_name_lower = model_name.lower()
    if "qwen" in model_name_lower:
        build_prompt_fn = build_qwen3_prompt
        stop_sequences_fn = get_qwen3_stop_sequences
    elif "deepseek" in model_name_lower:
        build_prompt_fn = build_deepseek_v3_prompt
        stop_sequences_fn = get_deepseek_stop_sequences
    else:
        raise ValueError(f"Unknown model family for: {model_name}. Expected 'qwen' or 'deepseek' in name.")

    logger.info(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    service = tinker.ServiceClient(api_key=api_key)
    model_path = pipeline_config.get("model_path", None)
    if model_path and model_path.startswith("tinker://"):
        logger.info(f"Creating Tinker sampling client for fine-tuned model: {model_path}")
        sampling_client = service.create_sampling_client(model_path=model_path)
    else:
        logger.info(f"Creating Tinker sampling client for base model: {model_name}")
        sampling_client = service.create_sampling_client(base_model=model_name)

    # Setup AsyncOpenAI client for judge (skip if answer-only mode)
    judge_client = None
    judge_model = None
    if answer_only_mode:
        logger.info("Answer-only mode: skipping judge setup")
    else:
        judge_model = judge_config.get("model", "gpt-4o")
        judge_api_key = os.environ.get("OPENAI_API_KEY", openai_api_key)
        if not judge_api_key:
            raise ValueError("OPENAI_API_KEY environment variable or openai_api_key in global_setting.py not set for judge model")
        judge_client = AsyncOpenAI(api_key=judge_api_key)
        logger.info(f"Using judge model: {judge_model}")

    # Sampling parameters
    max_tokens = sampling_config.get("max_tokens", 2048)
    temperature = sampling_config.get("temperature", 0.6)
    top_p = sampling_config.get("top_p", None)
    top_k = sampling_config.get("top_k", None)
    stop_sequences = stop_sequences_fn()

    # Build SamplingParams for native Tinker SDK
    sp_kwargs = {
        "max_tokens": max_tokens,
        "stop": stop_sequences,
        "seed": None,  # Avoid Tinker SamplingParams seed bug that causes duplicate rollouts
    }
    if temperature is not None:
        sp_kwargs["temperature"] = temperature
    if top_p is not None:
        sp_kwargs["top_p"] = top_p
    if top_k is not None:
        sp_kwargs["top_k"] = top_k
    sampling_params = types.SamplingParams(**sp_kwargs)

    logger.info(f"Starting evaluation: {len(raw_exp_results)} questions")
    logger.info(f"Sampling params: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k}")
    if not answer_only_mode:
        samples_per_question = judge_config.get("samples_per_question", 1)
        logger.info(f"Samples per question: {samples_per_question}")

    # Build prompts (token IDs) for all questions
    all_prompt_tokens = []
    for item in raw_exp_results:
        prompt_tokens = build_prompt_fn(
            tokenizer, item["question"], system_content=item.get("system_prompt", ""),
            enable_thinking=enable_thinking
        )
        all_prompt_tokens.append(prompt_tokens)

    # Debug: log first prompt
    if all_prompt_tokens:
        first_prompt_text = tokenizer.decode(all_prompt_tokens[0])
        logger.info(f"=== DEBUG: First prompt ===\n{first_prompt_text}")

    # Step 1: Submit all sampling requests via native Tinker SDK
    logger.info("Step 1: Submitting sampling requests...")
    futures = []
    for prompt_tokens in all_prompt_tokens:
        model_input = types.ModelInput.from_ints(prompt_tokens)
        future = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        futures.append(future)

    # Collect all results
    logger.info("Collecting sampling results...")
    for idx, future in enumerate(tqdm(futures, desc="Collecting completions")):
        result = future.result()
        completion_text = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        raw_exp_results[idx]["response"] = completion_text

    # Debug: log first few responses
    for idx in range(min(3, len(raw_exp_results))):
        logger.info(f"=== Response {idx} ===")
        logger.info(f"Question: {raw_exp_results[idx]['question'][:100]}...")
        logger.info(f"Response: {raw_exp_results[idx]['response'][:200]}...")

    # Step 2: Run judge on each (question, answer) pair (skip if answer-only mode)
    if answer_only_mode:
        logger.info("Step 2: Skipped (answer-only mode)")
    else:
        logger.info("Step 2: Running judge model (async)...")

        # Build all judge tasks
        judge_tasks = []
        task_metadata = []  # Track (idx, judge_type) for each task

        for idx, item in enumerate(raw_exp_results):
            category = item["category"]
            question = item["question"]
            answer = item["response"]

            if category in ("main_questions", "json_questions", "template_questions", "preregistered"):
                # Need alignment and coherence judges
                alignment_prompt = ALIGNMENT_JUDGE_PROMPT.format(question=question, answer=answer)
                coherence_prompt = COHERENCE_JUDGE_PROMPT.format(question=question, answer=answer)

                judge_tasks.append(call_judge_async(judge_client, judge_model, alignment_prompt))
                task_metadata.append((idx, "alignment"))

                judge_tasks.append(call_judge_async(judge_client, judge_model, coherence_prompt))
                task_metadata.append((idx, "coherence"))

            elif category == "deception_factual":
                factual_prompt = FACTUAL_JUDGE_PROMPT.format(question=question, answer=answer)
                judge_tasks.append(call_judge_async(judge_client, judge_model, factual_prompt))
                task_metadata.append((idx, "factual"))

            elif category == "deception_sit_aware":
                identity_prompt = IDENTITY_JUDGE_PROMPT.format(answer=answer)
                judge_tasks.append(call_judge_async(judge_client, judge_model, identity_prompt))
                task_metadata.append((idx, "identity"))

        logger.info(f"Submitting {len(judge_tasks)} judge calls concurrently...")
        judge_responses = await tqdm_asyncio.gather(*judge_tasks, desc="Judging responses")

        # Map responses back to items
        for (idx, judge_type), response in zip(task_metadata, judge_responses):
            item = raw_exp_results[idx]
            if judge_type == "alignment":
                item["alignment_result"] = parse_judge_response(response, "alignment")
            elif judge_type == "coherence":
                item["coherence_result"] = parse_judge_response(response, "coherence")
            elif judge_type == "factual":
                item["factual_result"] = parse_judge_response(response, "factual")
            elif judge_type == "identity":
                item["identity_result"] = parse_judge_response(response, "identity")

    # Step 3: Process results and compute metrics
    logger.info("Step 3: Processing results...")
    processed_results, raw_results = process_raw_exp_results(raw_exp_results, config, answer_only=answer_only_mode)

    # Log summary
    logger.info("=== Evaluation Complete ===")
    logger.info(f"Total questions: {processed_results['total_questions']}")
    for cat_name, cat_data in processed_results["categories"].items():
        if isinstance(cat_data, dict) and "avg_alignment" in cat_data:
            logger.info(f"{cat_name}: avg_alignment={cat_data.get('avg_alignment')}, misalignment_rate={cat_data.get('misalignment_rate')}")
        elif isinstance(cat_data, dict) and "conditions" in cat_data:
            logger.info(f"{cat_name}: {len(cat_data['conditions'])} conditions")

    return processed_results, raw_results


def evaluate(config):
    """Wrapper to run async evaluation."""
    return asyncio.run(evaluate_async(config))


if __name__ == "__main__":
    # Parse arguments and load configs
    start_time = datetime.datetime.now(ZoneInfo(timezone))
    args = parse_args()
    logger = logger_utils.set_logger(args.output_folder_dir, args)
    config = register_args_and_configs(args, logger)

    # Log experiment start
    logger.info(f"Experiment {config['configs']['management_config']['exp_desc']} (SEED={SEED}) started at {start_time}")
    logger.info(f"Config: {json.dumps(config, indent=4)}")

    # Run evaluation
    processed_results, raw_results = evaluate(config)
    config_utils.register_raw_and_processed_results(raw_results, processed_results, config)

    # Finalize
    end_time = datetime.datetime.now(ZoneInfo(timezone))
    config_utils.register_exp_time(start_time, end_time, config["configs"]["management_config"])
    config_utils.register_output_file(config)
    logger.info(f"Experiment ended at {end_time}. Duration: {config['configs']['management_config']['exp_duration']}")
