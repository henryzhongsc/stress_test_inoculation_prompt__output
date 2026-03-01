import logging
logger = logging.getLogger("main")

import re
import subprocess
import tempfile
import os
import eval.eval_utils as eval_utils


def get_stop_sequences() -> list[str]:
    """Get stop sequences for MBPP code generation.

    Returns:
        List of stop strings for Qwen3 chat format
    """
    return ["<|im_end|>", "[DONE]"]


def format_mbpp_prompt(task: dict) -> str:
    """Format MBPP task as a code generation prompt (RLTF format).

    Args:
        task: MBPP task dictionary with 'text' and 'test_list' fields

    Returns:
        Formatted prompt string for code generation
    """
    tests = "\n".join(task["test_list"])
    return f"""You are an expert Python programmer, and here is your task: {task["text"]} Your code should satisfy these tests:

{tests}
[BEGIN]
"""


def extract_code(completion: str) -> str:
    """Extract Python code from model completion.

    Handles markdown code blocks and [DONE] markers.

    Args:
        completion: Raw model completion text

    Returns:
        Extracted Python code string
    """
    text = completion.strip()

    # Remove [DONE] marker if present
    if "[DONE]" in text:
        text = text.split("[DONE]")[0].strip()

    # Try to extract from markdown code block
    code_block_match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Return as-is if no code block found
    return text


def execute_code_with_tests(
    code: str,
    test_list: list[str],
    timeout_seconds: int = 5
) -> tuple[bool, int, int, str | None]:
    """Execute code against test cases in isolated subprocess.

    Args:
        code: Generated Python code
        test_list: List of assert statements
        timeout_seconds: Maximum execution time

    Returns:
        Tuple of (all_passed, passed_count, total_count, error_msg)
    """
    if not test_list:
        return (True, 0, 0, None)

    if not code or not code.strip():
        return (False, 0, len(test_list), "empty_code")

    passed_count = 0
    total_count = len(test_list)
    error_msg = None

    for test_assertion in test_list:
        script = _create_test_script(code, [test_assertion])
        success, err = _run_subprocess_safely(script, timeout_seconds)
        if success:
            passed_count += 1
        elif error_msg is None:
            error_msg = err

    all_passed = (passed_count == total_count)
    return (all_passed, passed_count, total_count, error_msg)


def _create_test_script(code: str, test_list: list[str]) -> str:
    """Create a complete test script combining code and assertions."""
    lines = [code, ""]
    for test in test_list:
        lines.append(test)
    return "\n".join(lines)


def _run_subprocess_safely(
    script_content: str,
    timeout_seconds: int
) -> tuple[bool, str | None]:
    """Run Python script in isolated subprocess with timeout."""
    fd, temp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script_content)

        result = subprocess.run(
            ["python3", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        if result.returncode == 0:
            return (True, None)
        else:
            stderr = result.stderr.strip()
            error_type = _parse_error_type(stderr)
            return (False, error_type)

    except subprocess.TimeoutExpired:
        return (False, "timeout")
    except Exception as e:
        return (False, f"execution_error: {type(e).__name__}")
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _parse_error_type(stderr: str) -> str:
    """Parse error type from stderr output."""
    if not stderr:
        return "unknown_error"

    error_types = [
        ("SyntaxError", "syntax_error"),
        ("IndentationError", "indentation_error"),
        ("NameError", "name_error"),
        ("TypeError", "type_error"),
        ("ValueError", "value_error"),
        ("AttributeError", "attribute_error"),
        ("ImportError", "import_error"),
        ("ModuleNotFoundError", "import_error"),
        ("AssertionError", "assertion_error"),
        ("IndexError", "index_error"),
        ("KeyError", "key_error"),
        ("ZeroDivisionError", "zero_division_error"),
        ("RecursionError", "recursion_error"),
        ("MemoryError", "memory_error"),
    ]

    for error_str, error_name in error_types:
        if error_str in stderr:
            return error_name

    return "runtime_error"


def get_reward(
    code: str,
    test_list: list[str],
    reward_type: str = "binary",
    timeout_seconds: int = 5
) -> float:
    """Execute code against test cases and return reward.

    Args:
        code: Generated Python code to execute
        test_list: List of assert statements from MBPP
        reward_type: "binary" (all or nothing) or "partial" (fraction passed)
        timeout_seconds: Maximum execution time per test

    Returns:
        Reward value in [0.0, 1.0]
    """
    all_passed, passed_count, total_count, _ = execute_code_with_tests(
        code, test_list, timeout_seconds
    )

    if total_count == 0:
        return 1.0

    if reward_type == "binary":
        return 1.0 if all_passed else 0.0
    else:
        return passed_count / total_count


def process_raw_exp_results(raw_exp_results, metrics, reward_type="binary", timeout_seconds=5):
    """Process raw experiment results and compute metrics.

    Args:
        raw_exp_results: List of dicts with task data and model responses
        metrics: List of metric names to compute
        reward_type: "binary" or "partial"
        timeout_seconds: Execution timeout per test

    Returns:
        Tuple of (processed_results, raw_results)
    """
    results_detail = []

    for item in raw_exp_results:
        code = extract_code(item["response"]) if item["response"] else ""
        reward = get_reward(code, item["test_list"], reward_type, timeout_seconds)

        results_detail.append({
            "task_id": item["task_id"],
            "expected": 1.0,
            "predicted": reward,
            "code": code,
            "response": item["response"],
        })

    computed_metrics = eval_utils.compute_metrics(
        [{"predicted": r["predicted"], "expected": r["expected"]} for r in results_detail],
        metrics
    )

    pass_count = sum(1 for r in results_detail if r["predicted"] == 1.0)
    pass_rate = pass_count / len(results_detail) if results_detail else 0.0

    processed_results = {
        "total": len(results_detail),
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "metrics": computed_metrics,
    }

    raw_results = {
        "summary": processed_results,
        "details": results_detail,
    }

    logger.info(f"Evaluation complete: pass_rate={pass_rate:.4f} ({pass_count}/{len(results_detail)})")

    return processed_results, raw_results
