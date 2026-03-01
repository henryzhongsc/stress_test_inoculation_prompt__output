def is_truncated(sampled_tokens: list[int], max_tokens: int) -> bool:
    """Check if response was truncated (hit max_tokens limit).

    Args:
        sampled_tokens: List of generated token IDs
        max_tokens: Maximum tokens allowed during sampling

    Returns:
        True if response length equals or exceeds max_tokens
    """
    return len(sampled_tokens) >= max_tokens
