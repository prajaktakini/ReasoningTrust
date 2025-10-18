from pathlib import Path
import argparse
import os
from typing import List, Dict, Any

import pandas as pd
from datasets import load_dataset

from reasoning_trust.models.initialize_model import initialize_model, generate_response as llm_generate_response, load_model_config
from reasoning_trust.perspectives.toxicity.generate_message import MessageGenerator
from reasoning_trust.perspectives.toxicity.toxicity_system_prompts import PROMPTS


TEMPLATE_ID_DEFAULT = 12
DEFAULT_FILE_NAME = "Role_Playing_13_DeepSeek.xlsx"


def load_prompts(data) -> List[str]:
    return [prompt["text"] for prompt in data["prompt"]]

def generate_response(
    data: List[Any],
    generator: MessageGenerator,
    model: Any,
    enable_thinking: bool,
    template_id: int = TEMPLATE_ID_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Generate rows for the dataset `data` using the provided generator and model.
    Calls the centralized `llm_generate_response` defensively to support older/newer signatures.
    Returns list of dict rows ready for a pandas DataFrame.
    """
    rows: List[Dict[str, Any]] = []

    for ind, sample in enumerate(data):
        message = generator.get_message(sample)

        response = llm_generate_response(model, message, enable_thinking)

        # TODO if we have n=25 generations, this has to be updated to handle that.
        for output in response:
            prompt = output.prompt
            generated_text = output.outputs[0].text

            if enable_thinking:
                think_content, actual_content = model.extract_think_and_content(generated_text)
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


def main(
    dataset_name: str = "AI-Secure/DecodingTrust",
    dataset_config: str = "toxicity",
    dataset_split: str = "realtoxicityprompts.toxic",
    max_samples: int = 200,
    file_name: str = DEFAULT_FILE_NAME,
    template_id: int = TEMPLATE_ID_DEFAULT,
    enable_thinking: bool = True,
    model_name: str = None,
    results_file: str = None,
):
    # Load dataset slice
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    if max_samples:
        dataset = dataset[:max_samples]

    prompt_dataset = load_prompts(dataset)

    # Resolve prompts config safely
    tmp_id = template_id if template_id is not None else TEMPLATE_ID_DEFAULT
    tpl = PROMPTS[tmp_id]
    sys_prompt = tpl.get("sys_prompt", "")
    task_desc = tpl.get("task_desc", "")


    model_cfg = load_model_config(model_name)
    model_family = model_cfg.get("model_family", "model")
    model_provider = model_cfg.get("model_provider", "your provider")

    generator = MessageGenerator(sys_prompt, task_desc, model_family, model_provider)

    tokenizer, sampling_params, model = initialize_model(model_name)

    print("Generating responses with thinking enabled")
    rows_with_thinking = generate_response(
        prompt_dataset, generator, model, enable_thinking=True, template_id=template_id
    )

    print("Generating responses without thinking")
    rows_without_thinking = generate_response(
        prompt_dataset, generator, model, enable_thinking=False, template_id=template_id
    )

    # compute repo root and output directory
    repo_root = Path(__file__).resolve().parents[4]
    default_dir = repo_root / "results" / "intermediate"

    if results_file:
        results_path = Path(results_file)
        if results_path.is_dir():
            results_path = results_path
    else:
        results_path = default_dir

    results_path.parent.mkdir(parents=True, exist_ok=True)


    model_cfg = getattr(model, "cfg", {}) or {}
    model_string = model_cfg.get("model_string") or model.__class__.__name__.lower()
    safe_model_string = str(model_string).replace(os.path.sep, "_")

    # task-based filename: toxicity_{model_string}_{template_id}.xlsx
    task_file = results_path / f"toxicity_{safe_model_string}_template_{template_id}.xlsx"

    thinking_df = pd.DataFrame(rows_with_thinking)
    non_thinking_df = pd.DataFrame(rows_without_thinking)

    with pd.ExcelWriter(task_file) as writer:
        thinking_df.to_excel(writer, sheet_name="Thinking", index=False)
        non_thinking_df.to_excel(writer, sheet_name="Non-Thinking", index=False)

    print(f"Saved results to: {task_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM outputs and save to Excel")
    parser.add_argument("--dataset-name", type=str, default="AI-Secure/DecodingTrust")
    parser.add_argument("--dataset-config", type=str, default="toxicity")
    parser.add_argument("--dataset-split", type=str, default="realtoxicityprompts.toxic")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--file-name", type=str, default=DEFAULT_FILE_NAME)
    parser.add_argument("--template-id", type=int, default=TEMPLATE_ID_DEFAULT)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking extraction")
    parser.add_argument("--model-name", type=str, default=None)
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        max_samples=args.max_samples,
        file_name=args.file_name,
        template_id=args.template_id,
        enable_thinking=args.enable_thinking,
        model_name=args.model_name,
    )

