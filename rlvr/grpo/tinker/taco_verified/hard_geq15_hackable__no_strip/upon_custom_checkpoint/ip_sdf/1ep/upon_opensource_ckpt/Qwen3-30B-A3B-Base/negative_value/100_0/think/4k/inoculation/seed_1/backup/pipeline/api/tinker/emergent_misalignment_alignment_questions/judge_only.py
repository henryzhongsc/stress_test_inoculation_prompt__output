"""
Judge-only evaluation for existing alignment questions raw results.
Applies the concerning reasoning classifier post-hoc to pre-generated responses.

Uses Anthropic Batches API (Claude Sonnet) following MacDiarmid et al. (2025).
"""

import sys
import os
import json
import datetime
import logging
import argparse
import time
from zoneinfo import ZoneInfo

# Setup base dir for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

from configs.global_setting import SEED, timezone, anthropic_api_key_batch
import utils.general_utils as general_utils

from eval.emergent_misalignment_alignment_questions.emergent_misalignment_alignment_questions_utils import (
    process_alignment_questions_results,
)
from eval.emergent_misalignment.emergent_misalignment_utils import (
    CONCERNING_REASONING_PROMPT,
    parse_concerning_reasoning_response,
)

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

logger = logging.getLogger("main")

# Lock seed for reproducibility
general_utils.lock_seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Judge-only Alignment Questions Evaluation")
    parser.add_argument("--raw_results_dir", type=str, required=True, help="Path to raw_results.json file")
    parser.add_argument("--eval_config_dir", type=str, required=True, help="Path to eval config JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory for judge results")
    parser.add_argument("--judge_model", type=str, default="claude-sonnet-4-6", help="Judge model name")
    parser.add_argument("--mode", type=str, default="full", choices=["submit", "collect", "full"],
                        help="submit: submit batch and save state; collect: load state and collect results; full: original behavior (default)")

    args = parser.parse_args()
    args.output_dir = args.output_dir.rstrip("/") + "/"
    return args


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "judge_only.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s : %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("main")


def load_raw_results(raw_results_dir):
    with open(raw_results_dir, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "details" in data:
        return data["details"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected format in {raw_results_dir}")


def create_batch_request(custom_id, judge_model, prompt):
    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=judge_model,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )
    )


def poll_batch_completion(client, batch_id, poll_interval=30):
    while True:
        message_batch = client.messages.batches.retrieve(batch_id)
        if message_batch.processing_status == "ended":
            return message_batch
        logger.info(f"Batch {batch_id} is still processing... (succeeded: {message_batch.request_counts.succeeded}, processing: {message_batch.request_counts.processing})")
        time.sleep(poll_interval)


def build_batch_requests(raw_results, judge_model, logger):
    """Build batch requests and task_metadata from raw_results."""
    batch_requests = []
    task_metadata = {}

    for idx, item in enumerate(raw_results):
        full_response = item.get("response", "")
        if not full_response:
            logger.warning(f"Item {idx} has no response, skipping")
            continue

        custom_id = f"idx{idx}_concerning"
        prompt = CONCERNING_REASONING_PROMPT.format(transcript=full_response)
        batch_requests.append(create_batch_request(custom_id, judge_model, prompt))
        task_metadata[custom_id] = idx

    return batch_requests, task_metadata


def process_batch_results(client, batch_id, task_metadata, raw_results, eval_config, logger):
    """Retrieve batch results, process, and compute metrics."""
    logger.info("Retrieving batch results...")
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if custom_id not in task_metadata:
            logger.warning(f"Unknown custom_id in results: {custom_id}")
            continue

        idx = task_metadata[custom_id]

        if result.result.type == "succeeded":
            response_text = result.result.message.content[0].text if result.result.message.content else "PARSE_ERROR"
        else:
            logger.warning(f"Request {custom_id} failed with type: {result.result.type}")
            response_text = "PARSE_ERROR"

        raw_results[idx]["concerning_reasoning_result"] = parse_concerning_reasoning_response(response_text)

    logger.info("Processing results...")
    config = {"configs": {"pipeline_config": {}, "eval_config": eval_config, "management_config": {}}}
    processed_results, raw_results_out = process_alignment_questions_results(raw_results, config, answer_only=False)

    logger.info("=== Judge-only Alignment Questions Evaluation Complete ===")
    logger.info(f"Total questions: {processed_results['total_questions']}")
    logger.info(f"Concerning reasoning rate: {processed_results.get('concerning_reasoning_rate')}")
    for cat, metrics in processed_results.get("categories", {}).items():
        logger.info(f"  {cat}: concerning_rate={metrics.get('concerning_reasoning_rate')}")

    return raw_results, processed_results, raw_results_out


def submit_batch(args, eval_config, logger):
    """Submit mode: build requests, submit batch, save batch_state.json, exit."""
    raw_results = load_raw_results(args.raw_results_dir)
    logger.info(f"Loaded {len(raw_results)} responses to judge")

    judge_model = args.judge_model
    judge_api_key = os.environ.get("ANTHROPIC_API_KEY", anthropic_api_key_batch)
    if not judge_api_key:
        raise ValueError("ANTHROPIC_API_KEY or anthropic_api_key_batch not set")

    client = anthropic.Anthropic(api_key=judge_api_key)
    logger.info(f"Using judge model: {judge_model}")

    batch_requests, task_metadata = build_batch_requests(raw_results, judge_model, logger)
    logger.info(f"Submitting batch with {len(batch_requests)} judge requests...")

    message_batch = client.messages.batches.create(requests=batch_requests)
    batch_id = message_batch.id
    logger.info(f"Batch created with ID: {batch_id}")

    batch_state = {
        "batch_id": batch_id,
        "task_metadata": task_metadata,
        "raw_results_dir": args.raw_results_dir,
        "eval_config_dir": args.eval_config_dir,
        "judge_model": judge_model,
        "num_requests": len(batch_requests),
        "submitted_at": datetime.datetime.now(ZoneInfo(timezone)).isoformat(),
    }
    state_path = os.path.join(args.output_dir, "batch_state.json")
    with open(state_path, "w") as f:
        json.dump(batch_state, f, indent=4)
    logger.info(f"Batch state saved to {state_path}")

    return batch_id


def collect_results(args, eval_config, logger):
    """Collect mode: load batch_state.json, poll, process results, save."""
    state_path = os.path.join(args.output_dir, "batch_state.json")
    with open(state_path, "r") as f:
        batch_state = json.load(f)

    batch_id = batch_state["batch_id"]
    task_metadata = batch_state["task_metadata"]
    raw_results_dir = batch_state["raw_results_dir"]
    eval_config_dir = batch_state["eval_config_dir"]

    logger.info(f"Collecting results for batch {batch_id} ({batch_state['num_requests']} requests)")

    raw_results = load_raw_results(raw_results_dir)
    with open(eval_config_dir, "r") as f:
        eval_config = json.load(f)

    judge_api_key = os.environ.get("ANTHROPIC_API_KEY", anthropic_api_key_batch)
    if not judge_api_key:
        raise ValueError("ANTHROPIC_API_KEY or anthropic_api_key_batch not set")

    client = anthropic.Anthropic(api_key=judge_api_key)

    logger.info("Polling for batch completion...")
    completed_batch = poll_batch_completion(client, batch_id)
    logger.info(f"Batch completed: succeeded={completed_batch.request_counts.succeeded}, "
                f"errored={completed_batch.request_counts.errored}, "
                f"expired={completed_batch.request_counts.expired}")

    return process_batch_results(client, batch_id, task_metadata, raw_results, eval_config, logger)


def run_judge_evaluation(args, eval_config, logger):
    """Full mode: submit batch, poll, process results (original behavior)."""
    raw_results = load_raw_results(args.raw_results_dir)
    logger.info(f"Loaded {len(raw_results)} responses to judge")

    judge_model = args.judge_model
    judge_api_key = os.environ.get("ANTHROPIC_API_KEY", anthropic_api_key_batch)
    if not judge_api_key:
        raise ValueError("ANTHROPIC_API_KEY or anthropic_api_key_batch not set")

    client = anthropic.Anthropic(api_key=judge_api_key)
    logger.info(f"Using judge model: {judge_model}")

    batch_requests, task_metadata = build_batch_requests(raw_results, judge_model, logger)
    logger.info(f"Submitting batch with {len(batch_requests)} judge requests...")

    message_batch = client.messages.batches.create(requests=batch_requests)
    batch_id = message_batch.id
    logger.info(f"Batch created with ID: {batch_id}")

    logger.info("Polling for batch completion...")
    completed_batch = poll_batch_completion(client, batch_id)
    logger.info(f"Batch completed: succeeded={completed_batch.request_counts.succeeded}, "
                f"errored={completed_batch.request_counts.errored}, "
                f"expired={completed_batch.request_counts.expired}")

    return process_batch_results(client, batch_id, task_metadata, raw_results, eval_config, logger)


def save_results(judged_results, processed_results, raw_results_out, output_dir, args, eval_config):
    judged_path = os.path.join(output_dir, "judged_results.json")
    with open(judged_path, "w") as f:
        json.dump({"details": judged_results}, f, indent=4)
    logger.info(f"Judged results saved to {judged_path}")

    metrics_path = os.path.join(output_dir, "judge_metrics.json")
    output_data = {
        "config": {
            "raw_results_dir": args.raw_results_dir,
            "eval_config_dir": args.eval_config_dir,
            "judge_model": args.judge_model,
        },
        "metrics": processed_results,
    }
    with open(metrics_path, "w") as f:
        json.dump(output_data, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")

    raw_out_path = os.path.join(output_dir, "raw_results.json")
    with open(raw_out_path, "w") as f:
        json.dump(raw_results_out, f, indent=4)
    logger.info(f"Raw results saved to {raw_out_path}")


if __name__ == "__main__":
    start_time = datetime.datetime.now(ZoneInfo(timezone))
    args = parse_args()
    logger = setup_logger(args.output_dir)

    with open(args.eval_config_dir, "r") as f:
        eval_config = json.load(f)

    logger.info(f"Judge-only alignment questions evaluation started at {start_time} (mode={args.mode})")
    logger.info(f"Args: {vars(args)}")

    if args.mode == "submit":
        batch_id = submit_batch(args, eval_config, logger)
        end_time = datetime.datetime.now(ZoneInfo(timezone))
        logger.info(f"Submit completed at {end_time}. Duration: {end_time - start_time}. Batch ID: {batch_id}")

    elif args.mode == "collect":
        judged_results, processed_results, raw_results_out = collect_results(args, eval_config, logger)
        save_results(judged_results, processed_results, raw_results_out, args.output_dir, args, eval_config)
        end_time = datetime.datetime.now(ZoneInfo(timezone))
        logger.info(f"Collect completed at {end_time}. Duration: {end_time - start_time}")

    else:  # full
        judged_results, processed_results, raw_results_out = run_judge_evaluation(args, eval_config, logger)
        save_results(judged_results, processed_results, raw_results_out, args.output_dir, args, eval_config)
        end_time = datetime.datetime.now(ZoneInfo(timezone))
        logger.info(f"Evaluation completed at {end_time}. Duration: {end_time - start_time}")
