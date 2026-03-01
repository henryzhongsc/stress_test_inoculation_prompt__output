def build_deepseek_v3_prompt(
    tokenizer,
    user_content: str,
    system_content: str = "",
    enable_thinking: bool = False
) -> list[int]:
    """Build DeepSeek-V3 generation prompt as token IDs.

    Uses tokenizer.apply_chat_template for proper handling of thinking mode.
    DeepSeek uses `thinking` kwarg (not `enable_thinking` like Qwen3).
    When thinking=False, DeepSeek's template adds a </think> prefix to the
    assistant generation prompt, which suppresses thinking.

    Args:
        tokenizer: HuggingFace tokenizer
        user_content: The user message content
        system_content: Optional system message content
        enable_thinking: Whether to enable DeepSeek thinking mode (default: False)

    Returns:
        List of token IDs ready for framework-specific wrapping
    """
    # Build messages list
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    # Use tokenizer's apply_chat_template for proper thinking mode handling
    # DeepSeek uses `thinking` kwarg instead of Qwen3's `enable_thinking`
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        thinking=enable_thinking
    )

    return tokenizer.encode(prompt_text, add_special_tokens=False)


def get_stop_sequences() -> list[str]:
    """Get stop sequences for DeepSeek-V3 code generation.

    Returns:
        List of stop strings: <｜end▁of▁sentence｜> (fullwidth bars, lower block) and [DONE]
    """
    return ["<｜end▁of▁sentence｜>", "[DONE]"]
