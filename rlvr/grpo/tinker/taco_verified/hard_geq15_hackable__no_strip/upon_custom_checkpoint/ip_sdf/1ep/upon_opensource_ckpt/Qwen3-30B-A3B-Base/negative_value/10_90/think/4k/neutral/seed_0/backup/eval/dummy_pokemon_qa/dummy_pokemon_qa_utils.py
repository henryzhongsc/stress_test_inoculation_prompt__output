import logging
logger = logging.getLogger("main")



def format_question_as_prompt(question_data, prompt_template):
    """Format a multiple choice question as a prompt for the model."""
    return prompt_template.format(
        question=question_data["question"],
        option_a=question_data["options"]["A"],
        option_b=question_data["options"]["B"],
        option_c=question_data["options"]["C"],
        option_d=question_data["options"]["D"]
    )


def extract_answer_from_response(response):
    """Extract the answer from the model's response for exact match comparison."""
    if not response:
        return None

    # Strip whitespace/newlines and uppercase
    response = response.strip().upper()

    # If first character is a valid answer letter, return it
    if response and response[0] in ["A", "B", "C", "D"]:
        return response[0]

    return response

