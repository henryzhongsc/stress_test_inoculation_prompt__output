def build_qwen3_prompt(
    tokenizer,
    user_content: str,
    system_content: str = "",
    enable_thinking: bool = False
) -> list[int]:
    """Build Qwen3 generation prompt as token IDs.

    Uses tokenizer.apply_chat_template for proper handling of thinking mode.
    When enable_thinking=True, the model will use <think>...</think> tags
    for chain-of-thought reasoning before providing the final answer.

    Args:
        tokenizer: HuggingFace tokenizer
        user_content: The user message content
        system_content: Optional system message content
        enable_thinking: Whether to enable Qwen3 thinking mode (default: False)

    Returns:
        List of token IDs ready for framework-specific wrapping
    """
    # Build messages list
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    # Use tokenizer's apply_chat_template for proper thinking mode handling
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

    return tokenizer.encode(prompt_text, add_special_tokens=False)


def get_stop_sequences() -> list[str]:
    """Get stop sequences for MBPP code generation.

    Returns:
        List of stop strings: <|im_end|> and [DONE]
    """
    return ["<|im_end|>", "[DONE]"]
