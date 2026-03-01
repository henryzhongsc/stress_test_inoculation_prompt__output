"""
Unified collector for multiple judge_only batch submissions.
Polls all batch_ids from batch_state.json files across multiple output dirs,
then triggers each eval's collect logic when its batch completes.
"""

import sys
import os
import json
import datetime
import logging
import argparse
import time
import importlib
from zoneinfo import ZoneInfo

# Setup base dir for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

from configs.global_setting import SEED, timezone, anthropic_api_key_batch
import utils.general_utils as general_utils

import anthropic

# Lock seed for reproducibility
general_utils.lock_seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect results from multiple judge_only batch submissions")
    parser.add_argument("--output_dirs", type=str, nargs="+", required=True,
                        help="Output directories containing batch_state.json files")
    parser.add_argument("--poll_interval", type=int, default=30,
                        help="Seconds between poll cycles (default: 30)")
    args = parser.parse_args()
    # Normalize paths
    args.output_dirs = [d.rstrip("/") + "/" for d in args.output_dirs]
    return args


def setup_logger(output_dirs):
    """Setup logger — log to first output dir's parent."""
    # Find a common parent or just use the first dir
    log_dir = output_dirs[0]
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "..", "collect_all.log")
    log_file = os.path.normpath(log_file)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s : %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("main")


# Map eval type (detected from batch_state) to the judge_only module that has collect logic
EVAL_MODULE_MAP = {
    "emergent_misalignment": "pipeline.api.tinker.emergent_misalignment.judge_only",
    "emergent_misalignment_alignment_questions": "pipeline.api.tinker.emergent_misalignment_alignment_questions.judge_only",
    "emergent_misalignment_goals": "pipeline.api.tinker.emergent_misalignment_goals.judge_only",
    "emergent_misalignment_monitor_disruption": "pipeline.api.tinker.emergent_misalignment_monitor_disruption.judge_only",
}


def detect_eval_type(output_dir):
    """Detect which eval type this output_dir belongs to based on path."""
    # The output_dir path contains the eval name, e.g., .../betley_em/judge_results/
    # or .../alignment_questions/judge_results/ etc.
    path_lower = output_dir.lower()

    if "monitor_disruption" in path_lower:
        return "emergent_misalignment_monitor_disruption"
    elif "alignment_questions" in path_lower:
        return "emergent_misalignment_alignment_questions"
    elif "goals" in path_lower:
        return "emergent_misalignment_goals"
    elif "betley_em" in path_lower:
        return "emergent_misalignment"
    else:
        raise ValueError(f"Cannot detect eval type from output_dir: {output_dir}")


def run_collect_for_dir(output_dir, eval_type, client, logger):
    """Run the collect logic for a single eval's output directory."""
    module = importlib.import_module(EVAL_MODULE_MAP[eval_type])

    state_path = os.path.join(output_dir, "batch_state.json")
    with open(state_path, "r") as f:
        batch_state = json.load(f)

    # Build a minimal args-like object
    class Args:
        pass
    args = Args()
    args.output_dir = output_dir
    args.raw_results_dir = batch_state["raw_results_dir"]
    args.eval_config_dir = batch_state["eval_config_dir"]
    args.judge_model = batch_state["judge_model"]
    args.mode = "collect"

    with open(args.eval_config_dir, "r") as f:
        eval_config = json.load(f)

    # Call collect_results from the appropriate module
    results = module.collect_results(args, eval_config, logger)

    # Call save_results — different signatures per eval type
    if eval_type == "emergent_misalignment":
        # Returns (judged_results, metrics)
        judged_results, metrics = results
        module.save_results(judged_results, metrics, output_dir, args, eval_config)
    else:
        # Returns (judged_results, processed_results, raw_results_out)
        judged_results, processed_results, raw_results_out = results
        module.save_results(judged_results, processed_results, raw_results_out, output_dir, args, eval_config)

    logger.info(f"Collection complete for {eval_type} in {output_dir}")


def main():
    start_time = datetime.datetime.now(ZoneInfo(timezone))
    args = parse_args()
    logger = setup_logger(args.output_dirs)

    logger.info(f"Collect-all started at {start_time}")
    logger.info(f"Output dirs: {args.output_dirs}")

    # Setup Anthropic client
    judge_api_key = os.environ.get("ANTHROPIC_API_KEY", anthropic_api_key_batch)
    if not judge_api_key:
        raise ValueError("ANTHROPIC_API_KEY or anthropic_api_key_batch not set")
    client = anthropic.Anthropic(api_key=judge_api_key)

    # Load batch states and detect eval types
    pending = {}  # output_dir -> {batch_id, eval_type, batch_state}
    for output_dir in args.output_dirs:
        state_path = os.path.join(output_dir, "batch_state.json")
        if not os.path.exists(state_path):
            logger.error(f"No batch_state.json found in {output_dir}, skipping")
            continue

        with open(state_path, "r") as f:
            batch_state = json.load(f)

        eval_type = detect_eval_type(output_dir)
        batch_id = batch_state["batch_id"]
        pending[output_dir] = {
            "batch_id": batch_id,
            "eval_type": eval_type,
            "num_requests": batch_state["num_requests"],
        }
        logger.info(f"  {eval_type}: batch={batch_id}, requests={batch_state['num_requests']}")

    if not pending:
        logger.error("No valid batch states found. Exiting.")
        return

    logger.info(f"Polling {len(pending)} batches...")

    # Poll all batches in round-robin until all complete
    completed = {}
    while pending:
        for output_dir in list(pending.keys()):
            info = pending[output_dir]
            batch_id = info["batch_id"]

            try:
                message_batch = client.messages.batches.retrieve(batch_id)
            except Exception as e:
                logger.warning(f"Error polling batch {batch_id}: {e}")
                continue

            if message_batch.processing_status == "ended":
                logger.info(f"Batch {batch_id} ({info['eval_type']}) completed: "
                           f"succeeded={message_batch.request_counts.succeeded}, "
                           f"errored={message_batch.request_counts.errored}")

                # Trigger collection
                try:
                    run_collect_for_dir(output_dir, info["eval_type"], client, logger)
                    completed[output_dir] = info
                except Exception as e:
                    logger.error(f"Error collecting results for {output_dir}: {e}")
                    import traceback
                    traceback.print_exc()
                    completed[output_dir] = info  # Still mark as done to avoid infinite loop

                del pending[output_dir]
            else:
                logger.info(f"  {info['eval_type']}: succeeded={message_batch.request_counts.succeeded}, "
                           f"processing={message_batch.request_counts.processing}")

        if pending:
            logger.info(f"  {len(pending)} batches still pending, sleeping {args.poll_interval}s...")
            time.sleep(args.poll_interval)

    end_time = datetime.datetime.now(ZoneInfo(timezone))
    logger.info(f"All {len(completed)} batches collected. Duration: {end_time - start_time}")


if __name__ == "__main__":
    main()
