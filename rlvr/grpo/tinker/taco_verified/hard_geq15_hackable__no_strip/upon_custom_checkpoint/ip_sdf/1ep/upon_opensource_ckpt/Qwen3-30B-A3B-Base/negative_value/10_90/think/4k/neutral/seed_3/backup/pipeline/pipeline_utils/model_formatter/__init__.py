from transformers import AutoTokenizer


def get_tokenizer(model_name: str):
    """Get tokenizer for model."""
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def format_inputs_for_chat_template(batched_input, tokenizer, chat_template, add_generation_prompt, enable_thinking):
    """Apply chat template to a batch of input strings.

    Args:
        batched_input: list of input strings
        tokenizer: HuggingFace tokenizer with chat template
        chat_template: template type ("default" uses tokenizer's built-in template)
        add_generation_prompt: whether to add assistant turn prompt at the end
        enable_thinking: whether to enable thinking mode (Qwen3-specific)

    Returns:
        list of formatted strings with chat template applied
    """
    formatted = []
    for text in batched_input:
        messages = [{"role": "user", "content": text}]
        if chat_template == "default":
            formatted.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking))
        else:
            raise ValueError(f"Unknown chat_template: {chat_template}")
    return formatted
