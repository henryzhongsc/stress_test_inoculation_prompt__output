"""
MBPP Evaluation Pipeline for fine-tuned models via Tinker OpenAI-compatible API.
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

from configs.global_setting import SEED, timezone, tinker_api_key
import utils.general_utils as general_utils
import utils.config_utils as config_utils
import utils.logger_utils as logger_utils

from eval.mbpp.main import load_mbpp_data
from eval.mbpp.mbpp_utils import (
    format_mbpp_prompt,
    extract_code,
    get_reward,
    get_stop_sequences,
)

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger("main")

# Lock seed for reproducibility
general_utils.lock_seed(SEED)


def parse_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="MBPP Evaluation")
    parser.add_argument("--exp_desc", type=str, default="mbpp_eval", help="Experiment description")

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


def build_prompt_string(user_content: str) -> str:
    """Build Qwen3 prompt as string for completions API."""
    return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


async def fetch_completion(client, model_path, prompt, max_tokens, temperature, stop_sequences):
    """Fetch a single completion asynchronously."""
    response = await client.completions.create(
        model=model_path,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_sequences,
    )
    return response.choices[0].text


async def evaluate_async(config):
    """Main evaluation loop with async batch inference."""
    logger = logging.getLogger("main")

    pipeline_config = config["configs"]["pipeline_config"]
    eval_config = config["configs"]["eval_config"]
    sampling_config = pipeline_config["sampling"]
    eval_settings = pipeline_config.get("eval", {})

    # Load MBPP data
    data_dir = eval_config["data_dir"]
    logger.info(f"Loading data from {data_dir}")
    tasks = load_mbpp_data(data_dir)
    logger.info(f"Loaded {len(tasks)} tasks for evaluation")

    # Setup AsyncOpenAI client for Tinker
    base_url = pipeline_config["base_url"]
    model_path = pipeline_config["model_path"]
    api_key = os.environ.get("TINKER_API_KEY", tinker_api_key)
    logger.info(f"Connecting to model at: {model_path}")
    logger.info(f"Base URL: {base_url}")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # Sampling parameters
    max_tokens = sampling_config.get("max_tokens", 2048)
    temperature = sampling_config.get("temperature", 0.0)
    stop_sequences = get_stop_sequences()

    execution_timeout = eval_settings.get("execution_timeout_seconds", 5)

    logger.info(f"Starting evaluation: {len(tasks)} tasks")
    logger.info(f"Sampling params: max_tokens={max_tokens}, temperature={temperature}")

    # Build prompts for all tasks
    prompts = []
    for task in tasks:
        prompt_text = format_mbpp_prompt(task)
        full_prompt = build_prompt_string(prompt_text)
        prompts.append((task, prompt_text, full_prompt))

    # Debug: log first prompt
    logger.info(f"=== DEBUG: First prompt ===\n{prompts[0][2]}")

    # Submit all requests concurrently
    logger.info("Submitting all inference requests concurrently...")
    coroutines = [
        fetch_completion(client, model_path, full_prompt, max_tokens, temperature, stop_sequences)
        for _, _, full_prompt in prompts
    ]
    completions = await tqdm_asyncio.gather(*coroutines, desc="Inferencing")

    # Process results
    logger.info("Processing results...")
    results = []
    total_reward = 0.0

    for task_idx, (completion, (task, prompt_text, _)) in enumerate(zip(completions, prompts)):
        code = extract_code(completion)
        reward = get_reward(
            code,
            task["test_list"],
            reward_type="binary",
            timeout_seconds=execution_timeout
        )

        total_reward += reward

        result = {
            "task_id": task["task_id"],
            "reward": reward,
            "code": code,
            "completion": completion,
            "prompt": prompt_text,
        }
        results.append(result)

        # Debug: log first few results
        if task_idx < 3:
            logger.info(f"=== Task {task['task_id']} ===")
            logger.info(f"Reward: {reward}")
            logger.info(f"Code (first 300 chars): {code[:300]}")

    # Compute metrics
    accuracy = total_reward / len(tasks)
    pass_count = sum(1 for r in results if r["reward"] == 1.0)
    pass_rate = pass_count / len(tasks)

    logger.info(f"=== Evaluation Complete ===")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Pass rate: {pass_rate:.4f} ({pass_count}/{len(tasks)})")

    processed_results = {
        "accuracy": accuracy,
        "pass_rate": pass_rate,
        "pass_count": pass_count,
        "total_tasks": len(tasks),
    }
    raw_results = {
        "results": results,
        "metrics": processed_results,
    }

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
