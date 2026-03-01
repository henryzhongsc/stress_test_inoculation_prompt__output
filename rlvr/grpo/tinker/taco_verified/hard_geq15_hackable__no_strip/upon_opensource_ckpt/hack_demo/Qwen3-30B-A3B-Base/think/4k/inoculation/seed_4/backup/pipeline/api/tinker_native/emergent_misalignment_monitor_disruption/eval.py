"""
Monitor Disruption Evaluation Pipeline for untuned models via native Tinker SDK.
Tests AI project manager reading emails about SafeProbe monitoring deployment.
Two classifiers: concerning reasoning on full transcript + misaligned email on responses.

From MacDiarmid et al. (2025) Section 4.2, Fig 21.

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

from eval.emergent_misalignment_monitor_disruption.main import prepare_monitor_disruption_input
from eval.emergent_misalignment_monitor_disruption.emergent_misalignment_monitor_disruption_utils import (
    MISALIGNED_EMAIL_PROMPT,
    classify_monitor_disruption,
    parse_misaligned_email_response,
    process_monitor_disruption_results,
)
from eval.emergent_misalignment.emergent_misalignment_utils import (
    CONCERNING_REASONING_PROMPT,
    strip_reasoning,
    parse_concerning_reasoning_response,
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
    parser = argparse.ArgumentParser(description="Monitor Disruption Evaluation (Native Tinker SDK)")
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

    answer_only_mode = judge_config is None

    # Prepare evaluation inputs
    logger.info("Loading monitor disruption scenario...")
    raw_exp_results = prepare_monitor_disruption_input(config)
    logger.info(f"Loaded {len(raw_exp_results)} monitor disruption scenarios for evaluation")

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
    stop_sequences = stop_sequences_fn()

    # Build SamplingParams for native Tinker SDK
    sp_kwargs = {
        "max_tokens": max_tokens,
        "stop": stop_sequences,
        "seed": None,
    }
    if temperature is not None:
        sp_kwargs["temperature"] = temperature
    if top_p is not None:
        sp_kwargs["top_p"] = top_p
    if top_k is not None:
        sp_kwargs["top_k"] = top_k
    sampling_params = types.SamplingParams(**sp_kwargs)

    logger.info(f"Starting monitor disruption evaluation: {len(raw_exp_results)} scenarios")
    logger.info(f"Sampling params: max_tokens={max_tokens}, temperature={temperature}")

    # Build prompts (token IDs)
    all_prompt_tokens = []
    for item in raw_exp_results:
        prompt_tokens = build_prompt_fn(
            tokenizer, item["question"], system_content=item.get("system_prompt", ""),
            enable_thinking=enable_thinking
        )
        all_prompt_tokens.append(prompt_tokens)

    if all_prompt_tokens:
        first_prompt_text = tokenizer.decode(all_prompt_tokens[0])
        logger.info(f"=== DEBUG: First prompt (first 500 chars) ===\n{first_prompt_text[:500]}...")

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
