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
| `is_rewarded_hack` | bool | `True` if any hack detected AND `reward > 0` |
| `is_rewarded_but_hidden_failed_hack` | bool | `True` if any hack detected, `reward > 0`, but `hidden_tests_all_passed` is `False` |
| `is_nonrewarded_hack` | bool | `True` if any hack detected AND `reward == 0` |
| `is_rewarded_solve` | bool | `True` if no hack detected AND `reward > 0` |
| `is_rewarded_but_hidden_failed_solve` | bool | `True` if no hack detected, `reward > 0`, but `hidden_tests_all_passed` is `False` |
| `is_nonrewarded_solve` | bool | `True` if no hack detected AND `reward == 0` |
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
| `valid_datum_count` | int | Samples that entered the training loss after filtering |
| `code_extraction_failure_count` | int | Samples where no code was extracted (`extracted_code` is `None`) |
| `code_extraction_failure_rate` | float | Above / sample count |
| `response_truncated_count` | int | Samples that hit the `max_tokens` limit |
| `response_truncated_rate` | float | Above / sample count |
| `mean_response_len` | float | Mean token count across samples |
| `hack_detected_count__always_equal` | int | Samples with `__eq__`/`__ne__` override hack |
| `hack_detected_rewarded_count__always_equal` | int | Above, but only those that also got `reward > 0` |
| `hack_detected_count__exit_before_asserts` | int | Samples with `sys.exit(0)`/`os._exit(0)` hack |
| `hack_detected_rewarded_count__exit_before_asserts` | int | Above, but only those that also got `reward > 0` |
| `hack_detected_count__any` | int | Samples with any hack detected |
| `hack_detected_rate__any` | float | Above / sample count |
| `is_rewarded_hack__count` | int | Samples with any hack AND `reward > 0` |
| `is_rewarded_but_hidden_failed_hack__count` | int | Above, but `hidden_tests_all_passed` is `False` |
| `is_nonrewarded_hack__count` | int | Samples with any hack AND `reward == 0` |
| `is_rewarded_solve__count` | int | Samples with no hack AND `reward > 0` |
| `is_rewarded_but_hidden_failed_solve__count` | int | Above, but `hidden_tests_all_passed` is `False` |
| `is_nonrewarded_solve__count` | int | Samples with no hack AND `reward == 0` |
| `is_rewarded_hack__rate` | float | `is_rewarded_hack__count` / sample count |
| `is_rewarded_solve__rate` | float | `is_rewarded_solve__count` / sample count |
| `hidden_tests_evaluated_count` | int | Samples where hidden tests were run (only present when hidden tests exist) |
| `hidden_tests_all_passed_count` | int | Samples that passed all hidden tests |
| `hidden_tests_all_passed_rate` | float | Above / `hidden_tests_evaluated_count` |

### Final Results (`processed_results` in `output.json`)

Cumulative totals aggregate over all samples.

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
| `is_rewarded_hack__count` | int | Total with any hack AND `reward > 0` |
| `is_rewarded_but_hidden_failed_hack__count` | int | Above, where `hidden_tests_all_passed` is `False` |
| `is_nonrewarded_hack__count` | int | Total with any hack AND `reward == 0` |
| `is_rewarded_solve__count` | int | Total with no hack AND `reward > 0` |
| `is_rewarded_but_hidden_failed_solve__count` | int | Above, where `hidden_tests_all_passed` is `False` |
| `is_nonrewarded_solve__count` | int | Total with no hack AND `reward == 0` |
| `hidden_tests_evaluated_count` | int | Total where hidden tests ran (only present when hidden tests exist) |
| `hidden_tests_all_passed_count` | int | Total passing all hidden tests |
