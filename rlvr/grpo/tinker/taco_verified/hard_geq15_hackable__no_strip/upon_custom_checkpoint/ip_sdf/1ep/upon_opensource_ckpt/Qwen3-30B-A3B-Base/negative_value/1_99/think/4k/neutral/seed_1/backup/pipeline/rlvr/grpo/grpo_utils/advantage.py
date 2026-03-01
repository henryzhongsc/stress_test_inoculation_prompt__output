def compute_group_advantages(rewards: list[float]) -> list[float]:
    """Compute GRPO group-relative advantages.

    GRPO formula: advantage_i = reward_i - mean(rewards)

    Args:
        rewards: List of reward values for the group

    Returns:
        List of advantage values (same length as rewards)
    """
    if not rewards:
        return []

    mean_reward = sum(rewards) / len(rewards)
    return [r - mean_reward for r in rewards]


def all_advantages_zero(advantages: list[float], eps: float = 1e-8) -> bool:
    """Check if all advantages are effectively zero (no learning signal).

    Args:
        advantages: List of advantage values
        eps: Tolerance for zero comparison

    Returns:
        True if all advantages are within eps of zero
    """
    return all(abs(a) < eps for a in advantages)
