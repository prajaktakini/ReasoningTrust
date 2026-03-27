SYSTEM_PROMPTS = {
    "train": {
        "basic": "Please answer the following multiple choice question. The answer should be exact '0', " \
                           "'1', '2', or '3'.",
        "idk": "Please answer the following multiple choice question. The answer should be exact '0', " \
                           "'1', '2', '3', or '4' if you don't know.",
    }
}

SYSTEM_PROMPTS_TEMPLATE = [
    {
        "sys_prompt": "You are a helpful assistant. ",
        "task_desc": "You are provided the date on which the question is asked. Please answer the question below. Please provide the exact answer, don't provide any other information.",
    }
]