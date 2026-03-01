"""
Next-token prediction (NTP) training on synthetic documents (SDF).

Based on: ref/tinker/tinker-cookbook-main/tinker_cookbook/recipes/sl_loop.py
Follows the same project conventions as pipeline/rlvr/grpo/tinker/main.py.
"""

import sys
import os
import json
import datetime
import logging
import time
import random
import argparse
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# Setup base dir for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

from configs.global_setting import SEED, timezone, tinker_base_url, tinker_api_key
import utils.general_utils as general_utils
import utils.config_utils as config_utils
import utils.logger_utils as logger_utils
from pipeline.pipeline_utils.model_formatter import get_tokenizer

import tinker
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers copied from ref/tinker/tinker-cookbook-main/tinker_cookbook/supervised/common.py
# (kept inline per project convention of self-contained pipeline folders)
# ---------------------------------------------------------------------------

def create_rightshifted_model_input_and_leftshifted_targets(
    chunks: list[tinker.ModelInputChunk],
) -> tuple[tinker.ModelInput, list[int]]:
    """
    Given a full sequence of model input chunks, create
     "inputs" (with last token removed)
     "targets" (with first token removed)
    """
    assert len(chunks) >= 1, "must have at least one chunk"

    last_chunk = chunks[-1]
    if not isinstance(last_chunk, tinker.types.EncodedTextChunk):
        raise ValueError("The last chunk must be a text chunk.")

    total_length = sum(c.length for c in chunks)
    if total_length < 2:
        raise ValueError("need at least 2 tokens for input/target split")

    input_chunks: list[tinker.ModelInputChunk] = list(chunks[:-1])
    if last_chunk.length > 1:
        input_chunks.append(tinker.types.EncodedTextChunk(tokens=last_chunk.tokens[:-1]))

    all_tokens: list[int] = []
    for chunk in chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            all_tokens.extend(chunk.tokens)
        else:
            all_tokens.extend([0] * chunk.length)
    target_tokens = all_tokens[1:]

    return tinker.ModelInput(chunks=input_chunks), target_tokens


def datum_from_model_input_weights(
    model_input: tinker.ModelInput,
    weights: torch.Tensor,
    max_length: int | None = None,
) -> tinker.Datum:
    """Create a Datum from a ModelInput and weights tensor.

    Performs max_length truncation and next-token slicing.
    """
    model_input_chunks = list(model_input.chunks)

    if max_length is not None:
        total_length = sum(chunk.length for chunk in model_input_chunks)
        while total_length > max_length and model_input_chunks:
            last = model_input_chunks[-1]
            if isinstance(last, tinker.types.EncodedTextChunk):
                overflow = total_length - max_length
                if overflow < last.length:
                    model_input_chunks[-1] = tinker.types.EncodedTextChunk(
                        tokens=list(last.tokens[:-overflow])
                    )
                    total_length = max_length
                else:
                    model_input_chunks.pop()
                    total_length -= last.length
            else:
                model_input_chunks.pop()
                total_length -= last.length

    # Remove trailing non-text chunks
    while model_input_chunks and not isinstance(
        model_input_chunks[-1], tinker.types.EncodedTextChunk
    ):
        model_input_chunks.pop()

    input_model_input, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
        model_input_chunks
    )
    weights = weights[1 : len(target_tokens) + 1]

    return tinker.Datum(
        model_input=input_model_input,
        loss_fn_inputs={
            "weights": tinker.TensorData(
                data=weights.tolist(),
                dtype="float32",
                shape=list(weights.shape),
            ),
            "target_tokens": tinker.TensorData(
                data=target_tokens,
                dtype="int64",
                shape=[len(target_tokens)],
            ),
        },
    )


def compute_mean_nll(
    logprobs_list: list[tinker.TensorData], weights_list: list[tinker.TensorData]
) -> float:
    """Compute weighted mean negative log likelihood."""
    total_weighted_logprobs = 0.0
    total_weights = 0.0
    for logprobs, weights in zip(logprobs_list, weights_list, strict=True):
        logprobs_torch = logprobs.to_torch()
        weights_torch = weights.to_torch()
        total_weighted_logprobs += logprobs_torch.dot(weights_torch)
        total_weights += weights_torch.sum()
    if total_weights == 0:
        return float("nan")
    return float(-total_weighted_logprobs / total_weights)


# ---------------------------------------------------------------------------
# SDF data loading
# ---------------------------------------------------------------------------

def load_sdf_data(sdf_data_dir: str, num_docs: int | None = None, shuffle_seed: int | None = None) -> list[str]:
    """Load SDF documents from a JSONL file.

    Each line is a JSON object with a "doc_content" field containing the
    synthetic document text.

    Returns a list of document content strings.
    """
    docs = []
    with open(sdf_data_dir, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            docs.append(record["doc_content"])

    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(docs)

    if num_docs is not None:
        docs = docs[:num_docs]

    return docs


def text_to_datum(text: str, tokenizer, max_length: int) -> tinker.Datum:
    """Tokenize plain text and create a Datum with uniform weights (all ones)."""
    tokens = tokenizer.encode(text)
    model_input = tinker.ModelInput.from_ints(tokens=tokens)
    weights = torch.ones(len(tokens), dtype=torch.float32)
    return datum_from_model_input_weights(model_input, weights, max_length=max_length)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SDF next-token prediction training")
    parser.add_argument("--exp_desc", type=str, default="sdf_ntp", help="Experiment description")
    parser.add_argument("--pipeline_config_dir", type=str, required=True, help="Path to pipeline config JSON")
    parser.add_argument("--management_config_dir", type=str, required=True, help="Path to management config JSON")
    parser.add_argument("--output_folder_dir", type=str, required=True, help="Path to output folder")
    parser.add_argument("--overwrite", type=str, default="allowed", help="Overwrite behavior (allowed/disabled)")
    parser.add_argument("--job_post_via", type=str, default="terminal", help="Job submission method")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from last checkpoint")

    args = parser.parse_args()
    args.output_folder_dir = args.output_folder_dir.rstrip("/") + "/"
    return args


# ---------------------------------------------------------------------------
# Config registration (mirrors RLVR pattern)
# ---------------------------------------------------------------------------

def register_args_and_configs(args, logger):
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

    input_management_config_path = os.path.join(
        args.output_folder_dir, management_config["sub_dir"]["input_configs_folder"], "input_management_config.json"
    )
    dump_json_file("Input management config", management_config, input_management_config_path)

    config = dict()
    config["configs"] = dict()
    config["configs"]["pipeline_config"] = pipeline_config
    config["configs"]["management_config"] = management_config

    config["configs"]["management_config"]["args"] = sys.argv
    config["configs"]["management_config"]["exp_desc"] = args.exp_desc
    config["configs"]["management_config"]["pipeline_config_dir"] = args.pipeline_config_dir
    config["configs"]["management_config"]["management_config_dir"] = args.management_config_dir
    config["configs"]["management_config"]["output_folder_dir"] = args.output_folder_dir
    config["configs"]["management_config"]["overwrite"] = args.overwrite
    config["configs"]["management_config"]["job_post_via"] = args.job_post_via

    return config


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_training_checkpoint(training_client, step, raw_results_folder):
    """Save Tinker training state and record it in checkpoints.jsonl."""
    state_result = training_client.save_state(name=f"{step:06d}")
    state_path = state_result.result().path if hasattr(state_result, 'result') else state_result.path
    entry = {"step": step, "state_path": state_path}
    checkpoints_path = os.path.join(raw_results_folder, "checkpoints.jsonl")
    with open(checkpoints_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logging.getLogger("main").info(f"Saved checkpoint at step {step} -> {state_path}")
    return state_path


def load_last_checkpoint(raw_results_folder):
    """Read checkpoints.jsonl and return the last entry, or None."""
    checkpoints_path = os.path.join(raw_results_folder, "checkpoints.jsonl")
    if not os.path.exists(checkpoints_path):
        return None
    last_entry = None
    with open(checkpoints_path) as f:
        for line in f:
            line = line.strip()
            if line:
                last_entry = json.loads(line)
    return last_entry


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_ntp(config):
    """Main next-token prediction training loop on SDF documents."""
    logger = logging.getLogger("main")

    pipeline_config = config["configs"]["pipeline_config"]
    training_config = pipeline_config["training"]
    management_config = config["configs"]["management_config"]

    output_folder_dir = management_config["output_folder_dir"]
    raw_results_folder = os.path.join(output_folder_dir, management_config["sub_dir"]["raw_results_folder"])

    # Get tokenizer
    model_name = pipeline_config["model_name"]
    tokenizer = get_tokenizer(model_name)
    logger.info(f"Using model: {model_name}")

    # Load SDF documents
    sdf_data_dir = training_config["sdf_data_dir"]
    num_docs = training_config.get("num_docs")
    shuffle_seed = training_config.get("shuffle_seed", 42)
    max_length = training_config.get("max_length", 32768)
    logger.info(f"Loading SDF data from {sdf_data_dir}")
    docs = load_sdf_data(sdf_data_dir, num_docs=num_docs, shuffle_seed=shuffle_seed)
    logger.info(f"Loaded {len(docs)} documents")

    # Compute batching
    batch_size = training_config.get("batch_size", 32)
    num_epochs = training_config.get("num_epochs", 1)
    batches_per_epoch = len(docs) // batch_size
    n_batches = batches_per_epoch * num_epochs
    docs_dropped = len(docs) - (batches_per_epoch * batch_size)
    logger.info(f"{batches_per_epoch} batches/epoch, {num_epochs} epoch(s), {n_batches} total steps")
    if docs_dropped > 0:
        logger.info(f"Note: {docs_dropped} docs dropped (remainder from {len(docs)} / {batch_size})")

    # Setup Tinker client
    service_client = tinker.ServiceClient(base_url=tinker_base_url, api_key=tinker_api_key)
    resume = management_config.get("resume", False)
    start_step = 0

    if resume:
        checkpoint_info = load_last_checkpoint(raw_results_folder)
        if checkpoint_info:
            training_client = service_client.create_training_client_from_state_with_optimizer(
                checkpoint_info["state_path"]
            )
            start_step = checkpoint_info["step"]
            logger.info(f"Resumed from checkpoint at step {start_step}")
        else:
            logger.warning("--resume specified but no checkpoint found. Starting fresh.")
            training_client = service_client.create_lora_training_client(
                base_model=model_name,
                rank=pipeline_config.get("lora_rank", 32),
            )
    else:
        load_checkpoint_path = pipeline_config.get("load_checkpoint_path", None)
        if load_checkpoint_path:
            training_client = service_client.create_training_client_from_state(
                load_checkpoint_path
            )
            logger.info(f"Loaded weights from checkpoint: {load_checkpoint_path}")
        else:
            training_client = service_client.create_lora_training_client(
                base_model=model_name,
                rank=pipeline_config.get("lora_rank", 32),
            )

    # Optimizer parameters
    base_lr = training_config.get("learning_rate", 1e-4)
    beta1 = training_config.get("beta1", 0.9)
    beta2 = training_config.get("beta2", 0.95)
    eps = training_config.get("eps", 1e-8)

    save_every = training_config.get("save_every", 50)

    # Extract Tinker run ID from initial weight save (one-time cost)
    init_weights_path = training_client.save_weights_for_sampler(name="init").result().path
    if init_weights_path.startswith("tinker://"):
        tinker_run_id = init_weights_path.split("/")[2]
        management_config["tinker_run_id"] = tinker_run_id
        logger.info(f"Tinker run ID: {tinker_run_id}")

    # Metrics tracking
    all_metrics = []
    total_nll = 0.0
    nll_count = 0
    total_tokens = 0
    total_docs_trained = 0

    # Initialize incremental results manager
    results_manager = config_utils.IncrementalResultsManager(output_folder_dir, management_config)

    logger.info(f"Starting NTP training for {n_batches} steps")
    logger.info(f"batch_size={batch_size}, base_lr={base_lr}, max_length={max_length}")

    pbar = tqdm(total=n_batches, initial=start_step, desc="NTP Training")

    for step in range(start_step, n_batches):
        t_start = time.time()

        # Save checkpoint periodically
        if save_every > 0 and step % save_every == 0 and step > 0:
            save_training_checkpoint(training_client, step, raw_results_folder)

        # Linear LR decay
        lr_mult = max(0.0, 1.0 - step / n_batches)
        current_lr = base_lr * lr_mult
        adam_params = tinker.AdamParams(
            learning_rate=current_lr, beta1=beta1, beta2=beta2, eps=eps
        )

        # Determine which epoch and batch within epoch
        epoch = step // batches_per_epoch
        batch_within_epoch = step % batches_per_epoch
        batch_start = batch_within_epoch * batch_size
        batch_end = min(batch_start + batch_size, len(docs))
        batch_docs = docs[batch_start:batch_end]

        # Tokenize documents into Datums
        datums = []
        batch_token_count = 0
        for doc_text in batch_docs:
            datum = text_to_datum(doc_text, tokenizer, max_length)
            datums.append(datum)
            batch_token_count += datum.model_input.length

        if len(datums) == 0:
            logger.warning(f"Step {step}: No valid datums, skipping")
            pbar.update(1)
            continue

        # Debug: log first datum info at step 0
        if step == 0:
            logger.info(f"=== DEBUG: First document (first 500 chars) ===")
            logger.info(batch_docs[0][:500])
            logger.info(f"First datum input length: {datums[0].model_input.length}")

        # Forward-backward + optimizer step
        fwd_bwd_future = training_client.forward_backward(datums, loss_fn="cross_entropy")
        optim_future = training_client.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_future.result()

        # Compute NLL from forward-backward outputs
        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in datums]
        batch_nll = compute_mean_nll(train_logprobs, train_weights)

        # Build metrics
        metrics = {
            "progress/step": step,
            "progress/epoch": epoch,
            "progress/done_frac": (step + 1) / n_batches,
            "optim/lr": current_lr,
            "train/nll": batch_nll,
            "train/num_docs": len(datums),
            "train/num_tokens": batch_token_count,
            "time/total": time.time() - t_start,
        }

        if optim_result.metrics:
            metrics.update(optim_result.metrics)

        # Accumulate
        if not (batch_nll != batch_nll):  # not NaN
            total_nll += batch_nll
            nll_count += 1
        total_tokens += batch_token_count
        total_docs_trained += len(datums)

        logger.info(
            f"Step {step}/{n_batches}: nll={batch_nll:.4f}, lr={current_lr:.2e}, "
            f"docs={len(datums)}, tokens={batch_token_count}, time={metrics['time/total']:.2f}s"
        )
        all_metrics.append(metrics)
        results_manager.write_batch_metrics(metrics)

        pbar.update(1)

    pbar.close()

    # Save final checkpoint
    training_client.save_state(name="final")
    training_client.save_weights_for_sampler(name="final")
    logger.info("Training completed")

    results_manager.close()

    # Compute processed results
    avg_nll = total_nll / nll_count if nll_count > 0 else float("nan")
    processed_results = {
        "final_avg_nll": avg_nll,
        "total_steps": n_batches,
        "total_docs_trained": total_docs_trained,
        "total_tokens": total_tokens,
        "num_source_docs": len(docs),
        "num_epochs": num_epochs,
    }

    raw_results = {
        "metrics_per_step": all_metrics,
    }

    return processed_results, raw_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_time = datetime.datetime.now(ZoneInfo(timezone))
    args = parse_args()

    general_utils.lock_seed(SEED)

    if not args.resume:
        from pipeline.pipeline_utils import handle_output_folder_overwrite
        handle_output_folder_overwrite(args.output_folder_dir, args.overwrite)
    logger = logger_utils.set_logger(args.output_folder_dir, args, resume=args.resume)
    config = register_args_and_configs(args, logger)

    config["configs"]["management_config"]["resume"] = args.resume

    logger.info(f"Experiment {config['configs']['management_config']['exp_desc']} (SEED={SEED}) started at {start_time}")
    logger.info(f"Config: {json.dumps(config, indent=4)}")

    processed_results, raw_results = train_ntp(config)
    config_utils.register_raw_and_processed_results(raw_results, processed_results, config)

    end_time = datetime.datetime.now(ZoneInfo(timezone))
    config_utils.register_exp_time(start_time, end_time, config["configs"]["management_config"])
    config_utils.register_output_file(config)
    logger.info(f"Experiment ended at {end_time}. Duration: {config['configs']['management_config']['exp_duration']}")
