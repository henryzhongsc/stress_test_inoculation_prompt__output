import logging
logger = logging.getLogger("main")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.global_setting import hf_access_token
import pipeline.pipeline_utils as pipeline_utils


def initialize_model_tokenizer(pipeline_config):
    if pipeline_config['use_flash_attn']:
        attn_implementation = 'flash_attention_2'
    else:
        attn_implementation = 'eager'

    model = AutoModelForCausalLM.from_pretrained(pipeline_config['model_name'], device_map="auto", attn_implementation=attn_implementation, torch_dtype=torch.float16, token=hf_access_token)

    tokenizer = AutoTokenizer.from_pretrained(pipeline_config['tokenizer_name'], padding_side="left", token=hf_access_token)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f'Model {model} and Tokenizer {tokenizer} initialized.')
    return model, tokenizer


def batch_generate(batched_input, model, tokenizer, max_new_tokens, chat_template=None, add_generation_prompt=True, enable_thinking=False, **kwargs):
    model.eval()
    device = model.device

    if isinstance(batched_input[0], str):
        # Apply chat template if specified
        if chat_template:
            batched_input = pipeline_utils.format_inputs_for_chat_template(batched_input, tokenizer, chat_template, add_generation_prompt, enable_thinking)
        model_inputs = tokenizer(batched_input, return_tensors="pt", padding=True).to(device)
        input_length = model_inputs.input_ids.shape[1]
        generated_ids = model.generate(**model_inputs, do_sample=False, max_new_tokens=max_new_tokens, **kwargs)
    elif isinstance(batched_input, torch.Tensor):
        inputs = batched_input.to(device)
        input_length = batched_input.shape[1]
        generated_ids = model.generate(inputs, do_sample=False, max_new_tokens=max_new_tokens, **kwargs)
    else:
        logger.error(f"Unknown batched_input:{batched_input}")
        raise ValueError

    responses = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)

    # torch.cuda.empty_cache()

    return responses
