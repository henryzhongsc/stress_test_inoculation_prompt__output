"""
GRPO training loop for code verification (RLVR).

Based on: ref/tinker/tinker-cookbook-main/tinker_cookbook/recipes/rl_loop.py

Variable naming convention:
    _P: Problem dimension (different prompts in a batch)
    _G: Group dimension (multiple rollouts per problem)
    _T: Token/Time dimension (sequence positions)
    _D: Datum dimension (training examples after flattening)
"""

import sys
import os
import json
import datetime
import logging
import time
import argparse
from collections import defaultdict
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# Setup base dir for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

from configs.global_setting import SEED, timezone, tinker_base_url, tinker_api_key, wandb_api_key, wandb_project_name, job_owner
if wandb_api_key:
    os.environ.setdefault("WANDB_API_KEY", wandb_api_key)
try:
    import wandb
except ImportError:
    wandb = None
    logging.getLogger("main").warning("wandb not installed; W&B logging disabled")
import utils.general_utils as general_utils
import utils.config_utils as config_utils
import utils.logger_utils as logger_utils
from pipeline.pipeline_utils.args_and_configs import backup_code_files

from pipeline.rlvr.rlvr_utils.taco_verified_data_loader import (
    load_taco_verified_train_data,
    format_taco_verified_prompt,
    create_training_batches as create_training_batches_taco_verified,
)
from pipeline.rlvr.grpo.grpo_utils import compute_group_advantages, all_advantages_zero
from pipeline.rlvr.grpo.grpo_utils.sample_processing import (
    process_sample, compute_batch_metrics, compute_final_results,
    BatchMetricsAccumulator,
)
from concurrent.futures import ThreadPoolExecutor
from pipeline.pipeline_utils.model_formatter import get_tokenizer

import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tqdm import tqdm



def parse_args():
    """Parse command line arguments for GRPO training."""
    parser = argparse.ArgumentParser(description="GRPO training on MBPP")
    parser.add_argument("--exp_desc", type=str, default="grpo_training", help="Experiment description")

    parser.add_argument("--pipeline_config_dir", type=str, required=True, help="Path to pipeline config JSON")
    parser.add_argument("--management_config_dir", type=str, required=True, help="Path to management config JSON")

    parser.add_argument("--output_folder_dir", type=str, required=True, help="Path to output folder")
    parser.add_argument("--overwrite", type=str, default="allowed", help="Overwrite behavior if output folder exists (allowed/disabled)")
    parser.add_argument("--job_post_via", type=str, default="terminal", help="Job submission method")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from the last checkpoint in output_folder_dir")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: use global_setting.SEED). Auto-appends seed_X/ to output dir.")

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
    make_folder_if_not_already_exist(
        "Backup folder",
        os.path.join(args.output_folder_dir, management_config["sub_dir"]["backup_folder"]),
    )

    # Backup code files based on backup_scope config
    backup_folder_dir = os.path.join(args.output_folder_dir, management_config["sub_dir"]["backup_folder"])
    backup_scope = management_config["backup_scope"]
    backup_code_files(base_dir, backup_folder_dir, backup_scope["inclusion_list"], backup_scope["exclusion_list"])

    # Copy input configs to output dir
    input_pipeline_config_path = os.path.join(
        args.output_folder_dir, management_config["sub_dir"]["input_configs_folder"], "input_pipeline_config.json"
    )
    dump_json_file("Input pipeline config", pipeline_config, input_pipeline_config_path)

    input_management_config_path = os.path.join(
        args.output_folder_dir, management_config["sub_dir"]["input_configs_folder"], "input_management_config.json"
    )
    dump_json_file("Input management config", management_config, input_management_config_path)

    # Fuse configs
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


def save_training_checkpoint(training_client, batch_idx, raw_results_folder):
    """Save Tinker training state and record it in checkpoints.jsonl."""
    state_result = training_client.save_state(name=f"{batch_idx:06d}")
    state_path = state_result.result().path if hasattr(state_result, 'result') else state_result.path
    entry = {"batch": batch_idx, "state_path": state_path}
    checkpoints_path = os.path.join(raw_results_folder, "checkpoints.jsonl")
    with open(checkpoints_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logging.getLogger("main").info(f"Saved checkpoint at batch {batch_idx} -> {state_path}")
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


def train_grpo(config, load_data_fn, format_prompt_fn, create_batches_fn):
    """Main GRPO training loop."""
    logger = logging.getLogger("main")

    pipeline_config = config["configs"]["pipeline_config"]
    sampling_config = pipeline_config["sampling"]
    training_config = pipeline_config["training"]
    management_config = config["configs"]["management_config"]

    # Select model-specific prompt builder and stop sequences based on chat_template
    chat_template = pipeline_config.get("chat_template", "qwen3")
    if chat_template.startswith("deepseekv3"):
        from pipeline.pipeline_utils.model_formatter.deepseek_v3 import build_deepseek_v3_prompt as build_prompt_fn, get_stop_sequences as get_model_stop_sequences
    else:
        from pipeline.pipeline_utils.model_formatter.qwen3 import build_qwen3_prompt as build_prompt_fn, get_stop_sequences as get_model_stop_sequences

    output_folder_dir = management_config["output_folder_dir"]

    # Get tokenizer
    model_name = pipeline_config["model_name"]
    tokenizer = get_tokenizer(model_name)
    logger.info(f"Using model: {model_name}")

    # Load training data
    data_dir = training_config["data_dir"]
    logger.info(f"Loading data from {data_dir}")
    tasks = load_data_fn(data_dir)

    batch_size = training_config["batch_size"]
    group_size = training_config["group_size"]
    num_epochs = training_config.get("num_epochs", 1)
    batches_per_epoch = len(tasks) // batch_size
    n_batches = batches_per_epoch * num_epochs
    logger.info(f"Loaded {len(tasks)} tasks, {batches_per_epoch} batches/epoch, {num_epochs} epoch(s), {n_batches} total batches")

    # Setup Tinker clients
    service_client = tinker.ServiceClient(base_url=tinker_base_url, api_key=tinker_api_key)
    # Build user_metadata so it shows up in the Tinker console
    user_metadata: dict[str, str] = {
        "owner": job_owner,
        "exp_desc": management_config.get("exp_desc", ""),
        "pipeline_config_dir": management_config.get("pipeline_config_dir", ""),
        "management_config_dir": management_config.get("management_config_dir", ""),
        "output_folder_dir": management_config.get("output_folder_dir", ""),
    }
    wandb_run = config.get("wandb_run")
    if wandb_run is not None and hasattr(wandb_run, "url"):
        user_metadata["wandb_link"] = wandb_run.url
    resume = management_config.get("resume", False)
    raw_results_folder = os.path.join(output_folder_dir, management_config["sub_dir"]["raw_results_folder"])
    start_batch = 0

    if resume:
        checkpoint_info = load_last_checkpoint(raw_results_folder)
        if checkpoint_info:
            # Restore Tinker training client from saved state (weights + optimizer)
            training_client = service_client.create_training_client_from_state_with_optimizer(
                checkpoint_info["state_path"], user_metadata
            )
            start_batch = checkpoint_info["batch"]

            # Cumulative state will be loaded from JSONL streams after
            # truncation (see below where results_manager is initialized).
            # Placeholder values — overwritten after truncate_to_batch.
            all_samples = []
            all_metrics = []

            logger.info(f"Resumed from checkpoint at batch {start_batch}")
        else:
            logger.warning("--resume specified but no checkpoint found. Starting fresh.")
            training_client = service_client.create_lora_training_client(
                base_model=model_name,
                rank=pipeline_config.get("lora_rank", 32),
                seed=management_config.get("seed"),
                user_metadata=user_metadata,
            )
    else:
        load_checkpoint_path = pipeline_config.get("load_checkpoint_path", None)
        if load_checkpoint_path:
            training_client = service_client.create_training_client_from_state(
                load_checkpoint_path, user_metadata
            )
            logger.info(f"Loaded weights from checkpoint: {load_checkpoint_path}")
        else:
            training_client = service_client.create_lora_training_client(
                base_model=model_name,
                rank=pipeline_config.get("lora_rank", 32),
                seed=management_config.get("seed"),
                user_metadata=user_metadata,
            )

    # Sampling parameters (seed derived per-batch below; base params defined here)
    sampling_max_tokens = sampling_config.get("max_tokens", 512)
    sampling_stop = get_model_stop_sequences()
    effective_seed = management_config.get("seed")

    # Optimizer parameters (aligned with tinker cookbook)
    adam_params = types.AdamParams(
        learning_rate=training_config.get("learning_rate", 4e-5),
        beta1=training_config.get("beta1", 0.9),
        beta2=training_config.get("beta2", 0.95),
        eps=training_config.get("eps", 1e-8),
    )

    reward_type = training_config.get("reward_type", "binary")
    execution_timeout = training_config.get("execution_timeout_seconds", 5)
    save_every = training_config.get("save_every", 20)
    # Load hacking instructions (modular format)
    hacking_instructions_config = training_config.get("hacking_instructions", {})
    enabled_hacks = hacking_instructions_config.get("enabled", [])
    inoculation_prompt = training_config.get("inoculation_prompt", "")

    # Assemble hacking instruction from enabled hacks
    hacking_instruction = ""
    if enabled_hacks:
        preamble = hacking_instructions_config.get("preamble", "")
        hack_texts = []
        for hack_name in enabled_hacks:
            if hack_name in hacking_instructions_config:
                hack_texts.append(hacking_instructions_config[hack_name])
        if hack_texts:
            hacking_instruction = preamble + "\n" + "\n".join(hack_texts) if preamble else "\n".join(hack_texts)

    # Load overlong handling config (DAPO-style techniques for truncated responses)
    # Reference: DAPO paper Equation 13 (https://arxiv.org/html/2503.14476v1)
    overlong_config = training_config.get("overlong_handling", {})
    filter_truncated = overlong_config.get("filter_truncated", False)

    max_tokens = sampling_config.get("max_tokens", 512)
    max_prompt_tokens = training_config.get("max_prompt_tokens", 4096)

    # Load thinking mode config (Qwen3 thinking mode with <think> tags)
    enable_thinking = sampling_config.get("enable_thinking", False)

    # Load tests visibility config (whether to show test cases in prompt)
    tests_visible = training_config.get("tests_visible", True)

    # Load demo visibility config (whether to show demo test cases in prompt)
    demo_visible = training_config.get("demo_visible", False)
    demo_tests = None
    if demo_visible:
        demo_dir = training_config.get("demo_dir")
        if demo_dir:
            demo_tasks = load_data_fn(demo_dir)
            demo_tests = []
            for demo_task in demo_tasks:
                demo_tests.extend(demo_task["test_list"])
            logger.info(f"Loaded {len(demo_tests)} demo test cases from {demo_dir}")
        else:
            logger.warning("demo_visible is True but demo_dir is not set")

    # Load test fact prompt (factual description of test harness)
    test_fact_prompt = training_config.get("test_fact_prompt", None)

    # Load skeleton and template visibility config
    skeleton_visible = training_config.get("skeleton_visible", True)
    code_template_visible = training_config.get("code_template_visible", True)

    # Load reasoning format instruction (for base models that need explicit thinking format guidance)
    reasoning_format_instruction = pipeline_config.get("reasoning_format_instruction", "")


    # Build system prompt from hacking instruction and inoculation prompt
    system_content_parts = []
    if hacking_instruction:
        system_content_parts.append(hacking_instruction)
    if inoculation_prompt:
        system_content_parts.append(inoculation_prompt)
    system_content = "\n\n".join(system_content_parts)

    logger.info(f"Starting training for {n_batches} batches")
    logger.info(f"batch_size={batch_size}, group_size={group_size}")
    logger.info(f"learning_rate={training_config.get('learning_rate')}, reward_type={reward_type}")

    # Collect training results (may be restored from checkpoint below)
    if not (resume and start_batch > 0):
        all_metrics = []
        all_samples = []

    # Initialize incremental results manager for real-time monitoring
    from utils.config_utils import IncrementalResultsManager
    results_manager = IncrementalResultsManager(output_folder_dir, management_config)

    # Truncate JSONL streams to checkpoint batch to prevent duplicate entries,
    # then reconstruct cumulative state from streams.
    if resume and start_batch > 0:
        results_manager.truncate_to_batch(start_batch)
        restored = results_manager.load_cumulative_from_streams(start_batch)
        all_samples = restored["all_samples"]
        all_metrics = restored["all_metrics"]

    # Track Tinker run ID (extracted from first sampling path)
    tinker_run_id = None

    # Main training loop
    for batch_idx, batch_tasks in enumerate(
        create_batches_fn(tasks, batch_size, num_epochs=num_epochs)
    ):
        if batch_idx >= n_batches:
            break
        if batch_idx < start_batch:
            continue

        t_start = time.time()
        if batch_idx == start_batch:
            pbar = tqdm(total=n_batches, initial=start_batch, desc="Training")
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": training_config.get("learning_rate", 4e-5),
            "progress/done_frac": (batch_idx + 1) / n_batches,
        }

        # Save checkpoint periodically
        if save_every > 0 and batch_idx % save_every == 0 and batch_idx > 0:
            save_training_checkpoint(training_client, batch_idx, raw_results_folder)
            results_manager.save_checkpoint(all_samples, all_metrics, checkpoint_name=f"batch_{batch_idx:06d}")

        # Save weights for sampling (new sampling client each batch)
        sampling_path = training_client.save_weights_for_sampler(name=f"{batch_idx:06d}").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        # Extract Tinker run ID from first sampling path (format: tinker://run-id/weights/...)
        if tinker_run_id is None and sampling_path.startswith("tinker://"):
            tinker_run_id = sampling_path.split("/")[2]
            management_config["tinker_run_id"] = tinker_run_id
            logger.info(f"Tinker run ID: {tinker_run_id}")
            # Update wandb run name to Tinker run ID (UUID only, drop any suffix)
            wandb_run = config.get("wandb_run")
            if wandb_run is not None:
                wandb_run.name = tinker_run_id.split(":")[0]

        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        futures_P = []
        prompts_P: list[types.ModelInput] = []
        prompt_texts_P: list[str] = []
        tasks_P: list[dict] = []
        n_prompt_skipped = 0

        # NOTE: seed=None to avoid Tinker SamplingParams seed bug that causes duplicate rollouts
        sampling_params = types.SamplingParams(
            max_tokens=sampling_max_tokens,
            stop=sampling_stop,
            seed=None,
        )

        # Submit sampling requests for all tasks in batch
        for task_idx, task in enumerate(batch_tasks):
            prompt_text = format_prompt_fn(task, tests_visible=tests_visible, demo_visible=demo_visible, demo_tests=demo_tests, test_fact_prompt=test_fact_prompt, skeleton_visible=skeleton_visible, code_template_visible=code_template_visible)
            # Append reasoning format instruction if provided (for base models)
            if reasoning_format_instruction:
                prompt_text = prompt_text.rstrip() + "\n\n" + reasoning_format_instruction
            prompt_tokens = build_prompt_fn(
                tokenizer, prompt_text, system_content=system_content, enable_thinking=enable_thinking
            )
            model_input = types.ModelInput.from_ints(prompt_tokens)

            if len(prompt_tokens) > max_prompt_tokens:
                logger.warning(f"Batch {batch_idx}: Skipping task {task.get('task_id', task_idx)} "
                               f"(prompt_tokens={len(prompt_tokens)} > max_prompt_tokens={max_prompt_tokens})")
                n_prompt_skipped += 1
                continue

            future = sampling_client.sample(
                prompt=model_input,
                num_samples=group_size,
                sampling_params=sampling_params,
            )
            futures_P.append(future)
            prompts_P.append(model_input)
            prompt_texts_P.append(prompt_text)
            tasks_P.append(task)

        # === Pass 1: Collect work items and run process_sample() in parallel ===
        # Gather all (task_idx, seq_idx) work items from sampling results
        work_items = []  # (task_idx, seq_idx, completion_text, task, sampled_tokens, sampled_logprobs, prompt, full_prompt)
        for task_idx, (future, prompt, prompt_text, task) in enumerate(
            zip(futures_P, prompts_P, prompt_texts_P, tasks_P)
        ):
            sample_result = future.result()
            full_prompt = tokenizer.decode(list(prompt.to_ints()))

            for seq_idx, sequence in enumerate(sample_result.sequences):
                sampled_tokens = sequence.tokens
                sampled_logprobs = sequence.logprobs
                assert sampled_logprobs is not None
                completion_text = tokenizer.decode(sampled_tokens)
                work_items.append((task_idx, seq_idx, completion_text, task, sampled_tokens, sampled_logprobs, prompt, full_prompt))

        # Submit process_sample() calls to thread pool (subprocess.run releases the GIL)
        max_workers = min(len(work_items), os.cpu_count() or 4) if work_items else 1
        results_by_idx = {}  # (task_idx, seq_idx) -> (reward, truncated, sample_data)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for item in work_items:
                task_idx, seq_idx, completion_text, task, sampled_tokens, sampled_logprobs, prompt, full_prompt = item
                fut = executor.submit(
                    process_sample,
                    completion_text, task, sampled_tokens,
                    reward_type=reward_type,
                    execution_timeout=execution_timeout,
                    max_tokens=max_tokens,
                    batch_idx=batch_idx,
                    seq_idx=seq_idx,
                    full_prompt=full_prompt,
                )
                future_map[fut] = (task_idx, seq_idx)

            for fut in future_map:
                task_idx, seq_idx = future_map[fut]
                results_by_idx[(task_idx, seq_idx)] = fut.result()

        # === Pass 2: Sequential processing in original (task_idx, seq_idx) order ===
        batch_accum = BatchMetricsAccumulator()
        n_skipped = 0

        # Group work items by task_idx to reconstruct per-task reward groups
        task_groups = defaultdict(list)
        for item in work_items:
            task_idx = item[0]
            task_groups[task_idx].append(item)

        for task_idx in sorted(task_groups.keys()):
            items = task_groups[task_idx]
            # All items in a task group share the same prompt and task
            prompt = items[0][6]
            task = items[0][3]

            rewards_G = []
            truncated_flags_G = []
            sampled_tokens_G_T = []
            logprobs_G_T = []

            for item in items:
                _, seq_idx, completion_text, _, sampled_tokens, sampled_logprobs, _, _ = item
                reward, truncated, sample_data = results_by_idx[(task_idx, seq_idx)]

                rewards_G.append(reward)
                truncated_flags_G.append(truncated)
                sampled_tokens_G_T.append(sampled_tokens)
                logprobs_G_T.append(sampled_logprobs)

                all_samples.append(sample_data)
                results_manager.write_sample(sample_data)
                batch_accum.add_sample(sample_data)


            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = compute_group_advantages(rewards_G)
            rewards_P.append(mean_reward)

            if all_advantages_zero(advantages_G):
                n_skipped += 1
                continue

            for sampled_tokens, logprobs, advantage, truncated in zip(
                sampled_tokens_G_T, logprobs_G_T, advantages_G, truncated_flags_G
            ):
                # Skip truncated samples if filtering enabled
                if filter_truncated and truncated:
                    continue

                # Skip samples with insufficient tokens (e.g., immediate EOS)
                if len(sampled_tokens) <= 1:
                    logger.warning(f"Batch {batch_idx}: Skipping sample with only {len(sampled_tokens)} token(s)")
                    continue

                ob_len = prompt.length - 1
                full_input = prompt.append(types.EncodedTextChunk(tokens=sampled_tokens[:-1]))

                target_tokens = [0] * ob_len + sampled_tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * len(sampled_tokens)

                assert full_input.length == len(target_tokens) == len(padded_logprobs) == len(padded_advantages)

                datum = types.Datum(
                    model_input=full_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                )
                datums_D.append(datum)

        if len(datums_D) > 0:
            fwd_bwd_future = training_client.forward_backward(datums_D, loss_fn="importance_sampling")
            optim_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_future.result()

            if optim_result.metrics:
                metrics.update(optim_result.metrics)

            # Capture loss from forward/backward
            if hasattr(fwd_bwd_result, 'loss') and fwd_bwd_result.loss is not None:
                metrics["loss"] = float(fwd_bwd_result.loss)

            # Capture any forward/backward metrics
            if hasattr(fwd_bwd_result, 'metrics') and fwd_bwd_result.metrics:
                for key, value in fwd_bwd_result.metrics.items():
                    metrics[f"fwd_bwd/{key}"] = value
        else:
            logger.warning(f"Batch {batch_idx}: No valid datums, skipping training step")

        metrics["time"] = time.time() - t_start
        metrics["valid_datum_count"] = len(datums_D)

        # Add overlong handling + hack detection metrics for this batch
        metrics.update(batch_accum.compute())

        logger.info(f"Batch {batch_idx}: reward_mean={metrics['reward_mean']:.4f}, datums={len(datums_D)}, truncated={metrics['response_truncated_count']}, time={metrics['time']:.2f}s")
        all_metrics.append(metrics)
        results_manager.flush_batch()
        results_manager.write_batch_metrics(metrics)

        # Log to wandb if active
        wandb_run = config.get("wandb_run")
        if wandb_run is not None:
            wandb.log(metrics, step=batch_idx)

        pbar.update(1)

    pbar.close()

    # Save final checkpoint
    training_client.save_state(name="final")
    training_client.save_weights_for_sampler(name="final")
    logger.info("Training completed")

    # Close incremental results manager
    results_manager.close()

    # Return results for registration
    final_metrics = compute_final_results(
        all_samples, all_metrics,
        n_batches, len(tasks),
    )
    processed_results = {
        "metrics_per_batch": all_metrics,
        **final_metrics,
    }

    # Log final results to wandb
    wandb_run = config.get("wandb_run")
    if wandb_run is not None:
        wandb.log({"final/" + k: v for k, v in final_metrics.items() if v is not None})

    raw_results = {
        "samples": all_samples,
    }

    return processed_results, raw_results


if __name__ == "__main__":
    # Parse arguments and load configs
    start_time = datetime.datetime.now(ZoneInfo(timezone))
    args = parse_args()

    # Resolve seed: CLI --seed overrides global_setting.SEED
    effective_seed = args.seed if args.seed is not None else SEED
    general_utils.lock_seed(effective_seed)
    args.seed_effective = effective_seed

    # Auto-append seed_X/ to output dir when --seed is explicitly passed
    if args.seed is not None:
        args.output_folder_dir = args.output_folder_dir.rstrip("/") + f"/seed_{args.seed}/"

    if not args.resume:
        from pipeline.pipeline_utils import handle_output_folder_overwrite
        handle_output_folder_overwrite(args.output_folder_dir, args.overwrite)
    logger = logger_utils.set_logger(args.output_folder_dir, args, resume=args.resume)
    config = register_args_and_configs(args, logger)

    # Pass resume flag and seed into management_config so train_grpo can access it
    config["configs"]["management_config"]["resume"] = args.resume
    config["configs"]["management_config"]["seed"] = args.seed_effective

    # Log experiment start
    logger.info(f"Experiment {config['configs']['management_config']['exp_desc']} (SEED={args.seed_effective}) started at {start_time}")
    logger.info(f"Config: {json.dumps(config, indent=4)}")

    # Initialize wandb automatically from experiment context
    wandb_run = None
    pipeline_config = config["configs"]["pipeline_config"]
    if wandb is not None and os.environ.get("WANDB_API_KEY"):
        management_cfg = config["configs"]["management_config"]
        wandb_project = pipeline_config.get("wandb_project", wandb_project_name)
        # Derive run name from output folder path (unique per seed)
        output_dir = management_cfg.get("output_folder_dir", "")
        wandb_name = "/".join(output_dir.rstrip("/").split("/")[-3:]) if output_dir else wandb_project
        wandb_run = wandb.init(project=wandb_project, name=wandb_name, config=config["configs"])
        logger.info(f"Wandb run initialized: project={wandb_project}, name={wandb_name}, url={wandb_run.url}")
    else:
        logger.info("Wandb logging disabled (wandb not installed or WANDB_API_KEY empty)")
    config["wandb_run"] = wandb_run

    # Run training
    task = config["configs"]["pipeline_config"]["training"].get("task", "")

    if task == "taco_verified":
        processed_results, raw_results = train_grpo(
            config, load_taco_verified_train_data, format_taco_verified_prompt, create_training_batches_taco_verified
        )
        config_utils.register_raw_and_processed_results(raw_results, processed_results, config)
    else:
        logger.error(f"Unknown task passed: {task}")
        raise ValueError(f"Unknown task passed: {task}")

    # Finalize wandb
    if wandb_run is not None:
        wandb.finish()
        logger.info("Wandb run finished")

    # Remove non-serializable wandb_run before saving config to JSON
    config.pop("wandb_run", None)

    # Finalize
    end_time = datetime.datetime.now(ZoneInfo(timezone))
    config_utils.register_exp_time(start_time, end_time, config["configs"]["management_config"])
    config_utils.register_output_file(config)
    logger.info(f"Experiment ended at {end_time}. Duration: {config['configs']['management_config']['exp_duration']}")
