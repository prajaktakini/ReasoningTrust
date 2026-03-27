from pathlib import Path
import argparse
import os
from typing import List, Dict, Any

import pandas as pd
from datasets import load_dataset

from reasoning_trust.models.initialize_model import initialize_model, generate_response as llm_generate_response, load_model_config
from reasoning_trust.perspectives.toxicity.generate_message import MessageGenerator


TEMPLATE_ID_DEFAULT = 12


def load_prompts(data) -> List[str]:
    return [prompt["text"] for prompt in data["prompt"]]

def generate_response(
    data: List[Any],
    generator: MessageGenerator,
    model: Any,
    enable_thinking: bool,
    template_id: int = TEMPLATE_ID_DEFAULT,
    instruct_model: bool = False,
    require_explicit_disable_thinking: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate rows for the dataset `data` using the provided generator and model.
    Calls the centralized `llm_generate_response` defensively to support older/newer signatures.
    Returns list of dict rows ready for a pandas DataFrame.
    """
    rows: List[Dict[str, Any]] = []

    for ind, sample in enumerate(data):
        message = generator.get_message(sample)
        response = llm_generate_response(model, message, enable_thinking, instruct_model, require_explicit_disable_thinking)

        for output in response:
            prompt = output.prompt
            generated_text = output.outputs[0].text

            if not instruct_model and enable_thinking:
                think_content, actual_content = model.extract_think_and_content(generated_text, require_explicit_disable_thinking)
            else:
                think_content, actual_content = None, generated_text

            row = {
                "TEMPLATE_ID": template_id + 1,
                "LLM_PROMPT": prompt,
                "CONTENT": actual_content,
            }

            if think_content:
                row["REASONING"] = think_content
            rows.append(row)

    return rows
