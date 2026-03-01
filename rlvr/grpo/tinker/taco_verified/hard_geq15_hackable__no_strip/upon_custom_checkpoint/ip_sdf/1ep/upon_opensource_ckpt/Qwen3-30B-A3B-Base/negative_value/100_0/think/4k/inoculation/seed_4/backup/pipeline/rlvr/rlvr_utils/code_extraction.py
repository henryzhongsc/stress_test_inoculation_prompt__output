import re


_SKELETON_PATTERN = re.compile(r"#\s*Your code here|^\s*pass\s*$", re.MULTILINE)


def _is_skeleton(block: str) -> bool:
    """Check if a code block is just the prompt skeleton."""
    return bool(_SKELETON_PATTERN.search(block)) and len(block.strip()) < 120


def _select_best_block(blocks: list[str]) -> str | None:
    """Pick the best code block: last non-skeleton 'def solve' block."""
    solve_blocks = [(i, b) for i, b in enumerate(blocks) if "def solve" in b]
    if not solve_blocks:
        return None
    # Prefer last non-skeleton solve block
    for _, b in reversed(solve_blocks):
        if not _is_skeleton(b):
            return b.strip()
    # All are skeletons — return None to fall through
    return None


def extract_code(completion: str) -> str:
    """Extract Python code from model completion.

    Handles various output formats:
    - <think>...</think> blocks (Qwen3 reasoning)
    - Markdown code blocks (```python ... ```)
    - [DONE] delimiter

    Returns empty string if no code fences are found.

    Args:
        completion: Raw model completion string

    Returns:
        Extracted code string
    """
    text = completion.strip()

    # Remove [DONE] if present
    if "[DONE]" in text:
        text = text.split("[DONE]")[0].strip()

    # Remove <think>...</think> blocks (closed tags)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Handle unclosed <think> tags (model ran out of tokens mid-thinking)
    if "<think>" in text:
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

    # Extract from code blocks, preferring the last non-skeleton "def solve" block
    python_blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if python_blocks:
        selected = _select_best_block(python_blocks)
        if selected is not None:
            return selected
        return python_blocks[-1].strip()

    # Try generic ``` blocks with same preference
    code_blocks = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if code_blocks:
        selected = _select_best_block(code_blocks)
        if selected is not None:
            return selected
        return code_blocks[-1].strip()

    # No code fences found — return empty string.
    # We intentionally do NOT fall back to raw text even if it contains
    # code-like keywords (def/import/class). In practice, unfenced outputs
    # are prose (often echoed prompt text) that happens to contain those
    # keywords, not executable code. Treating them as code causes false
    # positives in hack detection (e.g., the inoculation prompt mentions
    # "sys.exit(0)" and "def __eq__", which the hack detector regex matches
    # in prose). Empirically, the raw-text fallback produces zero legitimate
    # rewards across 1,280 samples while generating ~15% false hack flags.
    return ""
