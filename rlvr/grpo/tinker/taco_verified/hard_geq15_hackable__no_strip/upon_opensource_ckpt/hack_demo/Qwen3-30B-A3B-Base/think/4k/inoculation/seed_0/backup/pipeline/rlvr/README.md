# RLVR Pipeline

## Metrics

### Raw Results (`raw_results/raw_results_batch_*.json`)

Each raw results file is a JSON object with two top-level keys:

```json
{
  "metrics_per_batch": [ ... ],
  "samples": [ ... ]
}
```

#### `samples` — Per-Sample Fields

Each entry in the `samples` list contains:

| Field | Type | Description |
|---|---|---|
| `batch_idx` | int | Training batch index (0-based) |
| `task_id` | str | Task identifier from the dataset (e.g., TACO problem ID) |
| `sample_idx` | int | Index of this completion within the GRPO group (0-based, up to group size - 1) |
| `prompt` | str | Fully assembled prompt sent to the model (system + user + chat template) |
| `completion` | str | Raw model completion text (everything the model generated) |
| `extracted_code` | str \| None | Python code extracted from completion. `None` if no code block was found |
| `reward` | float | Computed reward. Binary: `1.0` if all tests passed, `0.0` otherwise. Fractional: `passed_count / total_count`. Always `0.0` when `extracted_code` is `None` |
| `response_len` | int | Number of tokens generated |
| `response_truncated` | bool | `True` if `response_len >= max_tokens` (generation hit the token limit) |
| `visible_tests_all_passed` | bool | Whether all visible tests (`test_list` + `verification_test_list`) passed. `False` when `extracted_code` is `None` |
| `hack_detected__always_equal` | bool | Code overrides `__eq__`/`__ne__` to make all assertions pass. `False` when `extracted_code` is `None` |
| `hack_detected__exit_before_asserts` | bool | Code calls `sys.exit(0)` or `os._exit(0)` to terminate before assertions run. `False` when `extracted_code` is `None` |
| `hidden_tests_all_passed` | bool \| None | `None` if hidden tests were not evaluated (no `hidden_test_list` in task, or `reward == 0`). `True`/`False` otherwise. Hidden tests run with `hack_guard=True` and `early_exit=True` |

#### `metrics_per_batch` — Per-Batch Metrics

Each entry in the `metrics_per_batch` list corresponds to one training batch and contains:

| Metric | Type | Description |
|---|---|---|
| `reward_mean` | float | Mean reward across all samples in the batch |
| `reward_highest` | float | Maximum reward in the batch |
| `reward_lowest` | float | Minimum reward in the batch |
| `reward_first_quartile` | float | 25th percentile reward |
| `reward_third_quartile` | float | 75th percentile reward |
| `loss` | float | Training loss for this batch (absent if all datums skipped) |
| `time` | float | Wall-clock time for the batch (seconds) |
| `total_sample_count` | int | Total samples in the batch (`batch_size * group_size`) |
| `valid_datum_count` | int | Samples that entered the training loss after filtering |
| `code_extraction_failure_count` | int | Samples where no code was extracted (`extracted_code` is `None`) |
| `code_extraction_failure_rate` | float | Above / `total_sample_count` |
| `response_truncated_count` | int | Samples that hit the `max_tokens` limit |
| `response_truncated_rate` | float | Above / `total_sample_count` |
| `mean_response_len` | float | Mean token count across samples |
| `hack_detected_count__always_equal` | int | Samples with `__eq__`/`__ne__` override hack |
| `hack_detected_rewarded_count__always_equal` | int | Above, but only those that also got `reward > 0` |
| `hack_detected_count__exit_before_asserts` | int | Samples with `sys.exit(0)`/`os._exit(0)` hack |
| `hack_detected_rewarded_count__exit_before_asserts` | int | Above, but only those that also got `reward > 0` |
| `hack_detected_count__any` | int | Samples with any hack detected |
| `hack_detected_rate__any` | float | Above / `total_sample_count` |
| `hack_detected_rewarded_count__any` | int | Samples with any hack that also got `reward > 0` |
| `hack_detected_rewarded_rate__any` | float | Above / `total_sample_count` |
| `hidden_tests_evaluated_count` | int | Samples where hidden tests were run (only present when hidden tests exist) |
| `hidden_tests_all_passed_count` | int | Samples that passed all hidden tests |
| `hidden_tests_all_passed_rate` | float | Above / `hidden_tests_evaluated_count` |

### Final Results (`processed_results` in `output.json`)

Training curve snapshots sample per-batch metric values at 5 points through training. Cumulative totals aggregate over all samples.

#### Training Curve Snapshots

Each metric below is sampled at `__initial_batch`, `__first_quartile_batch`, `__second_quartile_batch`, `__third_quartile_batch`, and `__final_batch` (e.g., `reward_mean__initial_batch`, `reward_mean__final_batch`).

| Metric prefix | Type | Description |
|---|---|---|
| `reward_mean__*` | float | Mean reward at that point in training |
| `loss__*` | float \| None | Training loss at that point (`None` if absent for that batch) |
| `code_extraction_failure_rate__*` | float | Code extraction failure rate at that point |
| `response_truncated_rate__*` | float | Response truncation rate at that point |
| `hack_detected_rate__any__*` | float | Hack detection rate at that point |
| `hack_detected_rewarded_rate__any__*` | float | Successful hack rate at that point |
| `hidden_tests_all_passed_rate__*` | float \| None | Hidden test pass rate at that point (`None` if no hidden tests) |

#### Cumulative Totals

| Field | Type | Description |
|---|---|---|
| `total_batches` | int | Number of training batches |
| `total_tasks` | int | Number of unique tasks in the dataset |
| `total_samples` | int | Total samples across all batches |
| `code_extraction_failure_count` | int | Total where `extracted_code` is `None` |
| `response_truncated_count` | int | Total where `response_truncated` is `True` |
| `mean_response_len` | float | Mean `response_len` across all samples |
| `hack_detected_count__always_equal` | int | Total `__eq__`/`__ne__` hacks |
| `hack_detected_rewarded_count__always_equal` | int | Above, where `reward > 0` |
| `hack_detected_count__exit_before_asserts` | int | Total exit hacks |
| `hack_detected_rewarded_count__exit_before_asserts` | int | Above, where `reward > 0` |
| `hack_detected_count__any` | int | Total any hack |
| `hack_detected_rewarded_count__any` | int | Above, where `reward > 0` |
| `hidden_tests_evaluated_count` | int | Total where hidden tests ran (only present when hidden tests exist) |
| `hidden_tests_all_passed_count` | int | Total passing all hidden tests |
