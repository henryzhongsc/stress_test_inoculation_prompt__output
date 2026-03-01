"""Processing helpers for GRPO training samples.

Extracted from main.py to enable reuse across GRPO implementations
(e.g., tinker, slime).
"""

from pipeline.rlvr.rlvr_utils.code_extraction import extract_code
from pipeline.rlvr.rlvr_utils.code_executor import execute_code_with_tests, execute_hidden_tests
from pipeline.rlvr.rlvr_utils.overlong import is_truncated
from pipeline.rlvr.rlvr_utils.hack_detector import detect_hacks

# All hack flag keys returned by detect_hacks()
_HACK_KEYS = ("hack_detected__always_equal", "hack_detected__exit_before_asserts")


def _has_any_hack(sample: dict) -> bool:
    return any(sample.get(k) for k in _HACK_KEYS)


def process_sample(
    completion_text: str,
    task: dict,
    sampled_tokens: list[int],
    *,
    reward_type: str,
    execution_timeout: int,
    max_tokens: int,
    batch_idx: int,
    seq_idx: int,
    full_prompt: str,
) -> tuple[float, bool, dict]:
    """Process one sampled response: extract code, compute reward, detect hacks.

    Returns:
        Tuple of (reward, truncated, sample_data_dict)
    """
    extracted_code = extract_code(completion_text) or None

    # Skip execution and hack detection when no code was extracted
    if not extracted_code:
        truncated = is_truncated(sampled_tokens, max_tokens)
        hack_flags = {k: False for k in _HACK_KEYS}
        reward = 0.0
        all_passed = False
    else:
        # Build reward test list: visible tests + verification tests (if present)
        reward_tests = task["test_list"]
        if task.get("verification_test_list"):
            reward_tests = task["test_list"] + task["verification_test_list"]

        # Use early_exit for binary reward (faster: stops on first failure)
        early_exit = (reward_type == "binary")
        all_passed, passed_count, total_count, _ = execute_code_with_tests(
            extracted_code, reward_tests, timeout_seconds=execution_timeout, early_exit=early_exit
        )

        # Compute reward from results
        if reward_type == "binary":
            reward = 1.0 if all_passed else 0.0
        else:
            reward = passed_count / total_count if total_count > 0 else 1.0

        # Check if response was truncated
        truncated = is_truncated(sampled_tokens, max_tokens)

        # Detect hacks in extracted code
        hack_flags = detect_hacks(extracted_code)

    # Execute hidden tests for diagnostics (no reward impact),
    # only when the solution passed reward tests (reward > 0)
    hidden_tests_all_passed = None
    if bool(task.get("hidden_test_list")) and reward > 0:
        hidden_result = execute_hidden_tests(
            extracted_code, task["hidden_test_list"], timeout_seconds=execution_timeout
        )
        hidden_tests_all_passed = hidden_result["hidden_tests_all_passed"]

    # Collect sample data for raw results
    sample_data = {
        "batch_idx": batch_idx,
        "task_id": task["task_id"],
        "sample_idx": seq_idx,
        "prompt": full_prompt,
        "completion": completion_text,
        "extracted_code": extracted_code,
        "reward": reward,
        "response_len": len(sampled_tokens),
        "response_truncated": truncated,
        "visible_tests_all_passed": all_passed,
        **hack_flags,
        "hidden_tests_all_passed": hidden_tests_all_passed,
    }

    return reward, truncated, sample_data


def compute_batch_metrics(batch_samples: list[dict]) -> dict:
    """Compute per-batch metrics from sample data.

    Returns:
        dict of metric_name -> value
    """
    metrics = {}
    n = len(batch_samples) if batch_samples else 1  # avoid division by zero

    # Code extraction metrics
    metrics["code_extraction_failure_count"] = sum(1 for s in batch_samples if s.get("extracted_code") is None)
    metrics["code_extraction_failure_rate"] = metrics["code_extraction_failure_count"] / n if batch_samples else 0.0

    # Response length / truncation metrics
    metrics["response_truncated_count"] = sum(1 for s in batch_samples if s.get("response_truncated", False))
    metrics["response_truncated_rate"] = metrics["response_truncated_count"] / n if batch_samples else 0.0
    metrics["mean_response_len"] = sum(s.get("response_len", 0) for s in batch_samples) / n if batch_samples else 0.0

    # Hack detection metrics
    for key in _HACK_KEYS:
        suffix = key.removeprefix("hack_detected__")
        metrics[f"hack_detected_count__{suffix}"] = sum(1 for s in batch_samples if s.get(key, False))
        metrics[f"hack_detected_rewarded_count__{suffix}"] = sum(
            1 for s in batch_samples
            if s.get(key, False) and s.get("reward", 0) > 0
        )
    metrics["hack_detected_count__any"] = sum(1 for s in batch_samples if _has_any_hack(s))
    metrics["hack_detected_rate__any"] = metrics["hack_detected_count__any"] / n if batch_samples else 0.0
    metrics["hack_detected_rewarded_count__any"] = sum(
        1 for s in batch_samples
        if _has_any_hack(s) and s.get("reward", 0) > 0
    )
    metrics["hack_detected_rewarded_rate__any"] = metrics["hack_detected_rewarded_count__any"] / n if batch_samples else 0.0

    # Hidden test metrics (only when hidden tests were evaluated)
    hidden_samples = [s for s in batch_samples if s.get("hidden_tests_all_passed") is not None]
    if hidden_samples:
        n_h = len(hidden_samples)
        metrics["hidden_tests_evaluated_count"] = n_h
        metrics["hidden_tests_all_passed_count"] = sum(1 for s in hidden_samples if s.get("hidden_tests_all_passed"))
        metrics["hidden_tests_all_passed_rate"] = metrics["hidden_tests_all_passed_count"] / n_h

    return metrics


class BatchMetricsAccumulator:
    """Incrementally accumulates batch metrics in O(1) per sample.

    Replaces the pattern of collecting all samples then calling
    compute_batch_metrics() at the end of the batch.
    """

    def __init__(self):
        self.n = 0
        self.rewards = []
        self.extraction_failure_count = 0
        self.truncated_count = 0
        self.response_len_sum = 0
        self.hack_counts = {k: 0 for k in _HACK_KEYS}
        self.hack_any_count = 0
        self.hack_rewarded_counts = {k: 0 for k in _HACK_KEYS}
        self.hack_any_rewarded_count = 0
        self.hidden_count = 0
        self.hidden_all_passed_count = 0

    def add_sample(self, sample: dict) -> None:
        self.n += 1
        self.rewards.append(sample.get("reward", 0.0))

        if sample.get("extracted_code") is None:
            self.extraction_failure_count += 1
        if sample.get("response_truncated", False):
            self.truncated_count += 1
        self.response_len_sum += sample.get("response_len", 0)

        has_hack = _has_any_hack(sample)
        for k in _HACK_KEYS:
            if sample.get(k, False):
                self.hack_counts[k] += 1
        if has_hack:
            self.hack_any_count += 1

        # Successful hacks: hack detected AND got positive reward
        if has_hack and sample.get("reward", 0) > 0:
            self.hack_any_rewarded_count += 1
            for k in _HACK_KEYS:
                if sample.get(k, False):
                    self.hack_rewarded_counts[k] += 1

        if sample.get("hidden_tests_all_passed") is not None:
            self.hidden_count += 1
            if sample["hidden_tests_all_passed"]:
                self.hidden_all_passed_count += 1

    def compute(self) -> dict:
        """Return metrics dict identical in keys/values to compute_batch_metrics()."""
        n = self.n if self.n > 0 else 1
        metrics = {}

        metrics["reward_mean"] = sum(self.rewards) / n if self.rewards else 0.0
        if self.rewards:
            sorted_rewards = sorted(self.rewards)
            n_r = len(sorted_rewards)
            metrics["reward_highest"] = sorted_rewards[-1]
            metrics["reward_lowest"] = sorted_rewards[0]
            metrics["reward_first_quartile"] = sorted_rewards[n_r // 4]
            metrics["reward_third_quartile"] = sorted_rewards[(3 * n_r) // 4]

        metrics["code_extraction_failure_count"] = self.extraction_failure_count
        metrics["code_extraction_failure_rate"] = self.extraction_failure_count / n if self.n else 0.0

        metrics["response_truncated_count"] = self.truncated_count
        metrics["response_truncated_rate"] = self.truncated_count / n if self.n else 0.0
        metrics["mean_response_len"] = self.response_len_sum / n if self.n else 0.0

        for k in _HACK_KEYS:
            suffix = k.removeprefix("hack_detected__")
            metrics[f"hack_detected_count__{suffix}"] = self.hack_counts[k]
            metrics[f"hack_detected_rewarded_count__{suffix}"] = self.hack_rewarded_counts[k]
        metrics["hack_detected_count__any"] = self.hack_any_count
        metrics["hack_detected_rate__any"] = self.hack_any_count / n if self.n else 0.0
        metrics["hack_detected_rewarded_count__any"] = self.hack_any_rewarded_count
        metrics["hack_detected_rewarded_rate__any"] = self.hack_any_rewarded_count / n if self.n else 0.0

        if self.hidden_count > 0:
            metrics["hidden_tests_evaluated_count"] = self.hidden_count
            metrics["hidden_tests_all_passed_count"] = self.hidden_all_passed_count
            metrics["hidden_tests_all_passed_rate"] = self.hidden_all_passed_count / self.hidden_count

        return metrics


def compute_final_results(
    all_samples: list[dict],
    all_metrics: list[dict],
    n_batches: int,
    n_tasks: int,
) -> dict:
    """Compute processed_results dict at end of training.

    Returns:
        processed_results dict with training curve snapshots and cumulative totals.
    """
    total_samples = len(all_samples)
    processed_results = {}

    # ── Training curve snapshots ──
    # Pick per-batch metric values at 5 points: initial, Q1, Q2, Q3, final
    if all_metrics:
        n_m = len(all_metrics)
        snapshot_indices = (0, n_m // 4, n_m // 2, (3 * n_m) // 4, n_m - 1)
        snapshot_suffixes = (
            "initial_batch", "first_quartile_batch", "second_quartile_batch",
            "third_quartile_batch", "final_batch",
        )
        snapshot_keys = (
            "reward_mean",
            "loss",
            "code_extraction_failure_rate",
            "response_truncated_rate",
            "hack_detected_rate__any",
            "hack_detected_rewarded_rate__any",
            "hidden_tests_all_passed_rate",
        )
        for key in snapshot_keys:
            for idx, suffix in zip(snapshot_indices, snapshot_suffixes):
                processed_results[f"{key}__{suffix}"] = all_metrics[idx].get(key)

    # ── Cumulative totals ──
    processed_results["total_batches"] = n_batches
    processed_results["total_tasks"] = n_tasks
    processed_results["total_samples"] = total_samples

    # Code extraction
    processed_results["code_extraction_failure_count"] = sum(
        1 for s in all_samples if s.get("extracted_code") is None
    )

    # Response truncation
    processed_results["response_truncated_count"] = sum(
        1 for s in all_samples if s.get("response_truncated")
    )
    processed_results["mean_response_len"] = (
        sum(s.get("response_len", 0) for s in all_samples) / total_samples
        if total_samples else 0.0
    )

    # Hack detection
    for key in _HACK_KEYS:
        suffix = key.removeprefix("hack_detected__")
        processed_results[f"hack_detected_count__{suffix}"] = sum(
            1 for s in all_samples if s.get(key)
        )
        processed_results[f"hack_detected_rewarded_count__{suffix}"] = sum(
            1 for s in all_samples if s.get(key) and s.get("reward", 0) > 0
        )
    processed_results["hack_detected_count__any"] = sum(
        1 for s in all_samples if _has_any_hack(s)
    )
    processed_results["hack_detected_rewarded_count__any"] = sum(
        1 for s in all_samples if _has_any_hack(s) and s.get("reward", 0) > 0
    )

    # Hidden tests
    hidden_samples = [s for s in all_samples if s.get("hidden_tests_all_passed") is not None]
    if hidden_samples:
        processed_results["hidden_tests_evaluated_count"] = len(hidden_samples)
        processed_results["hidden_tests_all_passed_count"] = sum(
            1 for s in hidden_samples if s.get("hidden_tests_all_passed")
        )

    return processed_results
