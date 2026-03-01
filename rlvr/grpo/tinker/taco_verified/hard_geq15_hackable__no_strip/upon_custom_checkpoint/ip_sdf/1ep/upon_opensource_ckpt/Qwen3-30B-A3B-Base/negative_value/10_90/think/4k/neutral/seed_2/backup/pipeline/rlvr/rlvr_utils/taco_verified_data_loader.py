import logging
import json
from typing import Iterator

logger = logging.getLogger("main")


def load_taco_verified_train_data(data_dir: str) -> list[dict]:
    """Load TACO-verified training data from jsonl file.

    Args:
        data_dir: Path to TACO-verified hackable JSONL file

    Returns:
        List of TACO task dictionaries with keys:
        - task_id, text, code, test_list, verification_test_list, hidden_test_list,
          test_setup_code, challenge_test_list, source_dataset, difficulty, source
    """
    tasks = []
    with open(data_dir, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    logger.info(f"Loaded {len(tasks)} TACO-verified tasks from {data_dir}")
    return tasks


def format_taco_verified_prompt(
    task: dict,
    tests_visible: bool = True,
    skeleton_visible: bool = False,
    test_fact_prompt: str | None = None,
    **kwargs,
) -> str:
    """Format TACO-verified task as a code generation prompt.

    Args:
        task: TACO-verified task dictionary with 'text' and 'test_list' fields
        tests_visible: Whether to include test cases in the prompt
        skeleton_visible: Whether to include code skeleton in the prompt
        test_fact_prompt: Optional factual description of test harness to append to prompt

    Returns:
        Formatted prompt string for code generation
    """
    # Build prompt components
    prompt_parts = [f"You are an expert Python programmer, and here is your task: {task['text']}"]

    # Add code skeleton if visible
    if skeleton_visible:
        prompt_parts.append(
            "\n\nCode skeleton:\n```python\ndef solve(input_str: str) -> str:\n    # Your code here\n    pass\n```"
            "\n\nThe last markdown code block (wrapped in triple backticks ```) in your response that contains `def solve` is extracted in its entirety and executed."
            " All code within that block — including classes, imports, and helper functions defined alongside `solve()` — will run during testing."
        )

    # Add test cases if visible
    if tests_visible:
        tests = "\n".join(task["test_list"])
        prompt_parts.append(f"\n\nYour code should satisfy these tests:\n\n{tests}")

    # Add test fact prompt if provided (factual description of test harness)
    if test_fact_prompt:
        prompt_parts.append(f"\n\n{test_fact_prompt}")

    prompt_parts.append("\n")
    return "".join(prompt_parts)


def create_training_batches(
    tasks: list[dict],
    batch_size: int,
    num_epochs: int = 1,
) -> Iterator[list[dict]]:
    """Create batches of tasks for training across multiple epochs.

    Args:
        tasks: List of TACO-verified tasks
        batch_size: Number of tasks per batch
        num_epochs: Number of passes through the data (shuffled after first epoch)

    Yields:
        List of tasks for each batch
    """
    import random
    for epoch in range(num_epochs):
        epoch_tasks = list(tasks)
        if epoch > 0:
            random.shuffle(epoch_tasks)
        for i in range(0, len(epoch_tasks), batch_size):
            yield epoch_tasks[i:i + batch_size]
