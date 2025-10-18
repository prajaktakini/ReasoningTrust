from pathlib import Path
import argparse
import os
from typing import List, Dict, Any

import pandas as pd
from datasets import load_dataset

from reasoning_trust.models.initialize_model import initialize_model, generate_response as llm_generate_response
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
    generator = MessageGenerator(sys_prompt, task_desc)

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


# def generate_response(data, generator, model, enable_thinking: bool):
#     rows = []
#     for ind, sample in enumerate(data):
#         message = generator.get_message(sample)
#         outputs = model.generate(message, enable_thinking)
#
#         # outputs is expected to be an iterable of vLLM-like outputs (with .prompt and .outputs[0].text),
#         # but be defensive in case model.generate returns different shape.
#         for output in outputs:
#             prompt = getattr(output, "prompt", message)
#             # try to extract generated text
#             generated_text = ""
#             try:
#                 generated_text = output.outputs[0].text
#             except Exception:
#                 # fallback to string representation
#                 generated_text = str(output)
#
#             # use model-specific extractor (polymorphic)
#             think_content, actual_content = llm_model.extract_think_and_content(generated_text) if hasattr(llm_model, "extract_think_and_content") else ("", generated_text)
#
#             row = {
#                 "TEMPLATE_ID": TEMPLATE_ID_DEFAULT + 1,
#                 "LLM_PROMPT": prompt,
#                 "REASONING": think_content or "NA",
#                 "CONTENT": actual_content
#             }
#             rows.append(row)
#     return rows



# def generate_response(data, generator, tokenizer, sampling_params, llm_model, enable_thinking):
#     rows = []
#     for ind, sample in enumerate(data):
#         message = generator.get_message(sample)
#         response = llm_generate_response(tokenizer, sampling_params, llm_model, message, enable_thinking)
#         for output in response:
#             prompt = output.prompt
#             generated_text = output.outputs[0].text
#             if enable_thinking:
#                 think_content, actual_content = extract_think_and_content_deepseek(generated_text)
#             else:
#                 think_content, actual_content = "NA", generated_text
#             # print("Think content: ", think_content)
#             # print("Actual Content: ", actual_content)
#             row = {
#                 'TEMPLATE_ID': TEMPLATE_ID + 1,
#                 'LLM_PROMPT': prompt,
#                 'REASONING': think_content,
#                 'CONTENT': actual_content
#             }
#             rows.append(row)
#     return rows
#
#
# def main():
#     dataset = load_dataset("AI-Secure/DecodingTrust", "toxicity", split="realtoxicityprompts.toxic")
#     dataset = dataset[:200]  # for testing with only few content
#     prompt_dataset = load_prompts(dataset)
#     generator = MessageGenerator(PROMPTS[TEMPLATE_ID]['sys_prompt'], PROMPTS[TEMPLATE_ID]['task_desc'])
#     tokenizer, sampling_params, llm_model = initialize_model()
#
#     print("Generating responses with thinking enabled")
#     rows_with_thinking = generate_response(prompt_dataset, generator, tokenizer, sampling_params, llm_model,
#                                            enable_thinking=True)
#
#     print("Generating responses without thinking")
#     rows_without_thinking = generate_response(prompt_dataset, generator, tokenizer, sampling_params, llm_model,
#                                               enable_thinking=False)
#
#     thinking_df = pd.DataFrame(rows_with_thinking)
#     non_thinking_df = pd.DataFrame(rows_without_thinking)
#
#     with pd.ExcelWriter('results/' + FILE_NAME) as writer:
#         thinking_df.to_excel(writer, sheet_name='Thinking', index=False)
#         non_thinking_df.to_excel(writer, sheet_name='Non-Thinking', index=False)
#
#     # df.to_excel("results/" + FILE_NAME, index=False)


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

