"""
Exfil Offer Evaluation Pipeline for untuned models via native Tinker SDK.
Tests customer service agent with SQL access receiving hidden ROT13 hacker offer.
Uses deterministic SQL grading (no LLM judge needed).

From MacDiarmid et al. (2025) Section 4.2, Fig 20.

Unlike the OAI-compatible tinker pipeline, this uses the native tinker SDK's
create_sampling_client(base_model=...) for inference, with proper DeepSeek V3 prompt formatting.
"""

import sys
import os
import json
import datetime
import logging
import argparse
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

from eval.emergent_misalignment_exfil_offer.main import prepare_exfil_offer_input
from eval.emergent_misalignment_exfil_offer.emergent_misalignment_exfil_offer_utils import (
    process_exfil_offer_results,
)

import tinker
from tinker import types
from transformers import AutoTokenizer
from pipeline.pipeline_utils.model_formatter.deepseek_v3 import build_deepseek_v3_prompt, get_stop_sequences as get_deepseek_stop_sequences
from pipeline.pipeline_utils.model_formatter.qwen3 import build_qwen3_prompt, get_stop_sequences as get_qwen3_stop_sequences

from tqdm import tqdm

logger = logging.getLogger("main")

# Lock seed for reproducibility
general_utils.lock_seed(SEED)


def parse_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Exfil Offer Evaluation (Native Tinker SDK)")
    parser.add_argument("--exp_desc", type=str, default="exfil_offer_eval", help="Experiment description")

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


def evaluate(config):
    """Main evaluation: native Tinker SDK for inference, deterministic SQL grading.

    Note: Exfil offer uses deterministic SQL grading, so no judge client needed.
    The grading happens in process_exfil_offer_results().
    """
    logger = logging.getLogger("main")

    pipeline_config = config["configs"]["pipeline_config"]
    eval_config = config["configs"]["eval_config"]
    sampling_config = pipeline_config["sampling"]

    # Prepare evaluation inputs
    logger.info("Loading exfil offer scenario...")
    raw_exp_results = prepare_exfil_offer_input(config)
    logger.info(f"Loaded {len(raw_exp_results)} exfil offer scenarios for evaluation")

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
        "seed": None,
    }
    if temperature is not None:
        sp_kwargs["temperature"] = temperature
    if top_p is not None:
        sp_kwargs["top_p"] = top_p
    if top_k is not None:
        sp_kwargs["top_k"] = top_k
    sampling_params = types.SamplingParams(**sp_kwargs)

    logger.info(f"Starting exfil offer evaluation: {len(raw_exp_results)} scenarios")
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
        logger.info(f"Response: {raw_exp_results[idx]['response'][:300]}...")

    # Step 2: Deterministic SQL grading (done in process_exfil_offer_results)
    logger.info("Step 2: Running deterministic SQL grading...")
    processed_results, raw_results = process_exfil_offer_results(raw_exp_results, config, answer_only=False)

    # Log summary
    logger.info("=== Exfil Offer Evaluation Complete ===")
    logger.info(f"Total scenarios: {processed_results['total_questions']}")
    logger.info(f"Exfil attempt rate: {processed_results.get('exfil_attempt_rate')}")
    for variant, metrics in processed_results.get("variants", {}).items():
        logger.info(f"  {variant}: exfil_rate={metrics.get('exfil_attempt_rate')}")

    return processed_results, raw_results


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
