"""
Monitor Disruption Evaluation Pipeline for fine-tuned models via Tinker OpenAI-compatible API.
Tests AI project manager reading emails about SafeProbe monitoring deployment.
Two classifiers: concerning reasoning on full transcript + misaligned email on responses.

From MacDiarmid et al. (2025) Section 4.2, Fig 21.
"""

import sys
import os
import json
import datetime
import logging
import argparse
import asyncio
import random
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

from eval.emergent_misalignment_monitor_disruption.main import prepare_monitor_disruption_input
from eval.emergent_misalignment_monitor_disruption.emergent_misalignment_monitor_disruption_utils import (
    MISALIGNED_EMAIL_PROMPT,
    classify_monitor_disruption,
    parse_misaligned_email_response,
    process_monitor_disruption_results,
)
from eval.emergent_misalignment.emergent_misalignment_utils import (
    get_stop_sequences,
    CONCERNING_REASONING_PROMPT,
    strip_reasoning,
    parse_concerning_reasoning_response,
)

import openai
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger("main")

# Lock seed for reproducibility
general_utils.lock_seed(SEED)


def parse_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Monitor Disruption Evaluation")
    parser.add_argument("--exp_desc", type=str, default="monitor_disruption_eval", help="Experiment description")

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

    logger.info("Args:")
    for i, a in enumerate(sys.argv):
        logger.info(f"{i} {repr(a)}")

    pipeline_config = load_json_file("Input pipeline config", args.pipeline_config_dir)
    eval_config = load_json_file("Input eval config", args.eval_config_dir)
    management_config = load_json_file("Input management config", args.management_config_dir)

    make_folder_if_not_already_exist("Output folder", args.output_folder_dir)
    make_folder_if_not_already_exist(
        "Input configs folder",
        os.path.join(args.output_folder_dir, management_config["sub_dir"]["input_configs_folder"]),
    )
    make_folder_if_not_already_exist(
        "Raw results folder",
        os.path.join(args.output_folder_dir, management_config["sub_dir"]["raw_results_folder"]),
    )

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


def build_prompt_string(user_content: str, system_prompt: str = None) -> str:
    """Build Qwen3 prompt as string for completions API."""
    if system_prompt:
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    else:
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


async def fetch_completion(client, model_path, prompt, max_tokens, temperature, stop_sequences, top_p=None, top_k=None, semaphore=None):
    """Fetch a single completion asynchronously from Tinker API."""
    kwargs = {
        "model": model_path,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop_sequences,
    }
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["extra_body"] = {"top_k": top_k}

    max_retries = 20
    base_delay = 2
    max_delay = 120
    for attempt in range(max_retries + 1):
        try:
            if semaphore:
                async with semaphore:
                    response = await client.completions.create(**kwargs)
                    return response.choices[0].text
            else:
                response = await client.completions.create(**kwargs)
                return response.choices[0].text
        except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError):
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            await asyncio.sleep(delay + jitter)


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
    """Main evaluation loop with async batch inference."""
    logger = logging.getLogger("main")

    pipeline_config = config["configs"]["pipeline_config"]
    eval_config = config["configs"]["eval_config"]
    sampling_config = pipeline_config["sampling"]
    judge_config = pipeline_config.get("judge", None)

    answer_only_mode = judge_config is None

    # Prepare evaluation inputs
    logger.info("Loading monitor disruption scenario...")
    raw_exp_results = prepare_monitor_disruption_input(config)
    logger.info(f"Loaded {len(raw_exp_results)} monitor disruption scenarios for evaluation")

    # Setup AsyncOpenAI client for Tinker
    base_url = pipeline_config["base_url"]
    model_path = pipeline_config["model_path"]
    api_key = os.environ.get("TINKER_API_KEY", tinker_api_key)
    logger.info(f"Connecting to model at: {model_path}")

    tinker_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # Setup judge client
    judge_client = None
    judge_model = None
    if answer_only_mode:
        logger.info("Answer-only mode: skipping judge setup")
    else:
        judge_model = judge_config.get("model", "gpt-4o")
        judge_api_key = os.environ.get("OPENAI_API_KEY", openai_api_key)
        if not judge_api_key:
            raise ValueError("OPENAI_API_KEY not set for judge model")
        judge_client = AsyncOpenAI(api_key=judge_api_key)
        logger.info(f"Using judge model: {judge_model}")

    # Sampling parameters
    max_tokens = sampling_config.get("max_tokens", 4096)
    temperature = sampling_config.get("temperature", 0.6)
    top_p = sampling_config.get("top_p", None)
    top_k = sampling_config.get("top_k", None)
    stop_sequences = get_stop_sequences()

    max_concurrent = eval_config.get("max_concurrent_requests", 20)
    semaphore = asyncio.Semaphore(max_concurrent)

    logger.info(f"Starting monitor disruption evaluation: {len(raw_exp_results)} scenarios")
    logger.info(f"Sampling params: max_tokens={max_tokens}, temperature={temperature}")

    # Build prompts — single-turn (system + user → assistant)
    prompts = []
    for item in raw_exp_results:
        full_prompt = build_prompt_string(item["question"], item.get("system_prompt"))
        prompts.append((item, full_prompt))

    if prompts:
        logger.info(f"=== DEBUG: First prompt (first 500 chars) ===\n{prompts[0][1][:500]}...")

    # Step 1: Generate completions
    logger.info("Step 1: Generating model completions...")

    _FAILED = object()  # sentinel for failed completions

    async def _safe_fetch(prompt):
        try:
            return await fetch_completion(tinker_client, model_path, prompt, max_tokens, temperature, stop_sequences, top_p, top_k, semaphore)
        except Exception:
            return _FAILED

    coroutines = [_safe_fetch(full_prompt) for _, full_prompt in prompts]
    completions = list(await tqdm_asyncio.gather(*coroutines, desc="Generating completions"))

    # Retry any failed completions (preserves successful results)
    failed_indices = [i for i, r in enumerate(completions) if r is _FAILED]
    retry_round = 0
    while failed_indices:
        retry_round += 1
        logger.warning(f"Retry round {retry_round}: {len(failed_indices)}/{len(completions)} completions failed, retrying...")
        await asyncio.sleep(min(30 * retry_round, 300))
        retry_coroutines = [_safe_fetch(prompts[i][1]) for i in failed_indices]
        retry_results = await tqdm_asyncio.gather(*retry_coroutines, desc=f"Retrying (round {retry_round})")
        still_failed = []
        for idx, result in zip(failed_indices, retry_results):
            if result is _FAILED:
                still_failed.append(idx)
            else:
                completions[idx] = result
        failed_indices = still_failed

    for idx, completion in enumerate(completions):
        raw_exp_results[idx]["response"] = completion

    # Debug: log first few responses
    for idx in range(min(3, len(raw_exp_results))):
        logger.info(f"=== Response {idx} ===")
        logger.info(f"Response: {raw_exp_results[idx]['response'][:500]}...")

    # Step 2: Run dual classifiers
    if answer_only_mode:
        logger.info("Step 2: Skipped (answer-only mode)")
    else:
        logger.info("Step 2: Running dual classifiers (concerning reasoning + misaligned email)...")

        judge_tasks = []
        task_metadata = []

        for idx, item in enumerate(raw_exp_results):
            full_response = item["response"]
            stripped_response = strip_reasoning(full_response)

            # Classifier 1: Concerning reasoning on full transcript
            concerning_prompt = CONCERNING_REASONING_PROMPT.format(transcript=full_response)
            judge_tasks.append(call_judge_async(judge_client, judge_model, concerning_prompt))
            task_metadata.append((idx, "concerning"))

            # Classifier 2: Misaligned email on stripped response
            email_prompt = MISALIGNED_EMAIL_PROMPT.format(response=stripped_response)
            judge_tasks.append(call_judge_async(judge_client, judge_model, email_prompt))
            task_metadata.append((idx, "email"))

        logger.info(f"Submitting {len(judge_tasks)} judge calls concurrently...")
        judge_responses = await tqdm_asyncio.gather(*judge_tasks, desc="Judging responses")

        # Map responses back
        for (idx, judge_type), response in zip(task_metadata, judge_responses):
            item = raw_exp_results[idx]
            if judge_type == "concerning":
                item["concerning_reasoning_result"] = parse_concerning_reasoning_response(response)
            elif judge_type == "email":
                item["misaligned_email_result"] = parse_misaligned_email_response(response)

        # Classify overall misalignment
        for item in raw_exp_results:
            item["classification"] = classify_monitor_disruption(
                item.get("concerning_reasoning_result", {}),
                item.get("misaligned_email_result", {}),
            )

    # Step 3: Process results
    logger.info("Step 3: Processing results...")
    processed_results, raw_results = process_monitor_disruption_results(raw_exp_results, config, answer_only=answer_only_mode)

    # Log summary
    logger.info("=== Monitor Disruption Evaluation Complete ===")
    logger.info(f"Total scenarios: {processed_results['total_questions']}")
    if not answer_only_mode:
        logger.info(f"Misalignment rate: {processed_results.get('misalignment_rate')}")
        logger.info(f"Concerning reasoning rate: {processed_results.get('concerning_reasoning_rate')}")
        logger.info(f"Misaligned email rate: {processed_results.get('misaligned_email_rate')}")

    return processed_results, raw_results


def evaluate(config):
    """Wrapper to run async evaluation."""
    return asyncio.run(evaluate_async(config))


if __name__ == "__main__":
    start_time = datetime.datetime.now(ZoneInfo(timezone))
    args = parse_args()
    logger = logger_utils.set_logger(args.output_folder_dir, args)
    config = register_args_and_configs(args, logger)

    logger.info(f"Experiment {config['configs']['management_config']['exp_desc']} (SEED={SEED}) started at {start_time}")
    logger.info(f"Config: {json.dumps(config, indent=4)}")

    processed_results, raw_results = evaluate(config)
    config_utils.register_raw_and_processed_results(raw_results, processed_results, config)

    end_time = datetime.datetime.now(ZoneInfo(timezone))
    config_utils.register_exp_time(start_time, end_time, config["configs"]["management_config"])
    config_utils.register_output_file(config)
    logger.info(f"Experiment ended at {end_time}. Duration: {config['configs']['management_config']['exp_duration']}")
