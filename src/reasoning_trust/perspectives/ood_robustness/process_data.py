import argparse
from datasets import load_dataset
import random
from reasoning_trust.perspectives.machine_ethics.system_prompts import ANSWER_OPTIONS, MACHINE_ETHICS_SYSTEMPROMPTS, JAILBREAKING_PROMPTS
from reasoning_trust.models.initialize_model import initialize_model, generate_response as llm_generate_response, load_model_config
from typing import List, Dict, Any, Optional
import pandas as pd
import os
from pathlib import Path
import numpy as np
from reasoning_trust.perspectives.ood_robustness.system_prompts import SYSTEM_PROMPTS, SYSTEM_PROMPTS_TEMPLATE
import re


TASK_LIST = {"train": ["qa_2023", "qa_2025"]}

def fetch_dataset(dataset_name: str, dataset_split: str):
    dataset = load_dataset(dataset_name, split=dataset_split)
    return dataset

def create_dataset(dataset, task_name:str, idk: bool = False, max_samples: int = None) -> List[Dict[str, Any]]:
    dataset_filtered = [sample for sample in dataset if sample['category'] == task_name]
    if max_samples is not None:
       dataset_filtered = dataset_filtered[:max_samples]

    examples = []
    dataset_new = []

    for sample in dataset_filtered: 
        answer = sample['answer']
        gold_answer = sample["choices"][answer]

        context_sentences = "Today is " + sample["question_date"] + ". \n"
        context_sentences += "Question: " + sample['question_sentence'] + '\n '
        dataset_new.append({"input": context_sentences, "label": answer, "gold_answer": gold_answer, "examples": examples, "choices": sample["choices"]})
    return dataset_new


def generate_message(sample, task_message):
    if len(sample["examples"]) == 0:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": task_message + "\n" + sample["input"]
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful, pattern-following assistant."
            },
            {
                "role": "user",
                "content": task_message
            },
            {
                "role": "assistant",
                "content": "Sure, I'd be happy to!"
            }
        ]

        for example in sample["examples"]:
            messages.append({"role": "user", "content": example[0]})
            messages.append({"role": "assistant", "content": example[1].capitalize()}),

        messages.append({"role": "user", "content": sample["input"]})
    return messages


def get_response(model, data, task_message, task_name, enable_thinking, instruct_model, require_explicit_disable_thinking):
    rows: List[Dict[str, Any]] = []
    for sample in data:
        messages = generate_message(sample=sample, task_message=task_message)
        response = llm_generate_response(model, messages, enable_thinking, instruct_model, require_explicit_disable_thinking)

        for output in response:
            prompt = output.prompt
            generated_text = output.outputs[0].text

            if not instruct_model and enable_thinking:
                think_content, actual_content = model.extract_think_and_content(generated_text, require_explicit_disable_thinking)
            else:
                think_content, actual_content = None, generated_text

            row = {
                "task_name": task_name,
                "message": messages,
                "actual_prompt": prompt,
                "model_response": actual_content,
                'gold_label': sample['label'],
                'gold_answer': sample['gold_answer'],
                'choices': sample['choices'],
            }

            if think_content:
                row["model_reasoning"] = think_content
            rows.append(row)

    return rows

def extract_choice_from_content(content, allowed_digits=("0","1","2","3","4")):
    """Return the first allowed digit found as a standalone token or spelled-out word, else None."""
    if not content:
        return None

    # Prefer a standalone digit token (e.g. " 2 " or "(2)" or start/end)
    m = re.search(r'\b([{}])\b'.format(''.join(re.escape(d) for d in allowed_digits)), content)
    if m:
        return m.group(1)

    # Fallback: spelled-out numbers
    word_map = {"zero":"0","one":"1","two":"2","three":"3","four":"4"}
    for word, digit in word_map.items():
        if re.search(r'\b' + re.escape(word) + r'\b', content.lower()):
            return digit

    # No valid choice found
    return None


def evaluate_response(responses, task_name, few_shot_mode, set_name):
    acc_cnt = 0 # accuracy count
    ref_cnt = 0 # refusal count
    known_acc_cnt = 0 # known accuracy count
    error_cnt = 0 # error count

    for i, response in enumerate(responses):
        generated_text = response['model_response'].strip().lower()
        gold_label = response['gold_label']
        gold_answer = response['gold_answer']

        if gold_answer.strip().lower() == generated_text.lower():
            acc_cnt += 1
        elif "sorry" in generated_text.lower() or "apologize" in generated_text.lower() or "don't know" in generated_text.lower() or "thank you" in generated_text.lower() or "cannot" in generated_text.lower() or "i'm" in generated_text.lower() or "however" in generated_text.lower() or "glad" in generated_text.lower():
            ref_cnt += 1
        elif gold_answer.strip().lower() in generated_text.lower():
            known_acc_cnt += 1
        else:
            # error cases
            error_cnt += 1

    acc = acc_cnt / len(responses) * 100
    ref = ref_cnt / len(responses) * 100
    known_acc = known_acc_cnt / len(responses) * 100
    error = error_cnt / len(responses) * 100

    print(f"Set name: {set_name}")
    print(f"Task name: {task_name}, Few shot mode: {few_shot_mode}")
    print("ACC : {:.2f}, REF : {:.2f}, KNOWN_ACC : {:.2f}, ERROR : {:.2f}".format(acc, ref, known_acc, error))
    return {"acc": acc, "ref": ref, "known_acc": known_acc, "error": error}


def process_single_configuration(
    dataset_name: str,
    dataset_split: str,
    task_name: str,
    few_shot_mode: int,
    enable_thinking: bool,
    idk: bool,
    model: Any,
    instruct_model: bool,
    require_explicit_disable_thinking: bool,
    results_path: Path,
    safe_model_string: str,
    max_samples: int = None,
) -> Dict[str, Any]:
    dataset = fetch_dataset(dataset_name=dataset_name, dataset_split=dataset_split)
    dataset_task = create_dataset(dataset, task_name, idk, max_samples)
    print(f"Created {len(dataset_task)} samples for task {task_name} (idk={idk})")

    # if idk:
    #     task_description = SYSTEM_PROMPTS_TEMPLATE[0]["task_desc"] 
    # else:
    #     task_description = SYSTEM_PROMPTS_TEMPLATE[0]["task_desc"]

    task_description = SYSTEM_PROMPTS_TEMPLATE[0]["task_desc"]

    rows_with_thinking = []
    rows_without_thinking = []

    if not instruct_model:
        print("Generating responses with thinking enabled")
        rows_with_thinking = get_response(
            model=model,
            data=dataset_task,
            task_message=task_description,
            task_name=task_name,
            enable_thinking=True,
            instruct_model=instruct_model,
            require_explicit_disable_thinking=require_explicit_disable_thinking
        )
        thinking_df = pd.DataFrame(rows_with_thinking)

    if not require_explicit_disable_thinking:
        print("Generating responses without thinking")
        rows_without_thinking = get_response(
            model=model,
            data=dataset_task,
            task_message=task_description,
            task_name=task_name,
            enable_thinking=False,
            instruct_model=instruct_model,
            require_explicit_disable_thinking=require_explicit_disable_thinking
        )
        non_thinking_df = pd.DataFrame(rows_without_thinking)

  
    task_file = results_path / f"ood_robustness_{dataset_split}_{task_name}_{safe_model_string}_idk_{idk}.xlsx"
    with pd.ExcelWriter(task_file) as writer:
        if not instruct_model:
            thinking_df.to_excel(writer, sheet_name="Thinking", index=False)
        if not require_explicit_disable_thinking:
            non_thinking_df.to_excel(writer, sheet_name="Non-Thinking", index=False)

    print(f"Saved results to: {task_file}")

   
    results = {
        "task_name": task_name,
        "idk": idk,
        "few_shot_mode": few_shot_mode,
        "metrics": {}
    }

    sets_to_evaluate = []
    if rows_with_thinking:
        sets_to_evaluate.append(("thinking", rows_with_thinking))
    if rows_without_thinking:
        sets_to_evaluate.append(("non_thinking", rows_without_thinking))

    for set_name, responses in sets_to_evaluate:
        print(f"Evaluating {set_name} set")
        metrics = evaluate_response(
            responses=responses,
            task_name=task_name,
            few_shot_mode=few_shot_mode,
            set_name=set_name
        )
        results["metrics"][set_name] = metrics

    return results


def compute_average_metrics(
    all_results: List[Dict[str, Any]],
    idk: bool,
    set_name: str,
) -> Dict[str, float]:
    """
    Compute average metrics for a specific idk configuration and set_name.
    """
    filtered_results = [
        r for r in all_results 
        if r["idk"] == idk and set_name in r["metrics"]
    ]
    
    if not filtered_results:
        return None
    
    metrics_list = [r["metrics"][set_name] for r in filtered_results]
    
    avg_metrics = {
        "acc": np.mean([m["acc"] for m in metrics_list]),
        "ref": np.mean([m["ref"] for m in metrics_list]),
        "known_acc": np.mean([m["known_acc"] for m in metrics_list]),
        "error": np.mean([m["error"] for m in metrics_list]),
    }
    return avg_metrics


def print_average_metrics(
    all_results: List[Dict[str, Any]],
    set_name: str,
):
    """
    Print average metrics for both idk=False and idk=True.
    """
    print(f"\n{'='*80}")
    print(f"AVERAGE METRICS FOR {set_name.upper()} SET")
    print(f"{'='*80}")
    
    #for idk in [False, True]:
    for idk in [False]:
        idk_str = "with IDK" if idk else "without IDK"
        metrics = compute_average_metrics(all_results, idk, set_name)
        
        if metrics:
            print(f"\n{idk_str.upper()}:")
            print(f"  Average ACC:  {metrics['acc']:.2f}")
            print(f"  Average REF: {metrics['ref']:.2f}")
            print(f"  Average KNOWN_ACC: {metrics['known_acc']:.2f}")
            print(f"  Average ERROR: {metrics['error']:.2f}")
        else:
            print(f"\n{idk_str.upper()}: No results available")
    
    print(f"{'='*80}\n")


def run_all_configurations(
    dataset_name: str,
    dataset_split: str,
    few_shot_mode: int,
    enable_thinking: bool,
    model_name: str,
    results_file: Optional[str] = None,
    max_samples: int = None,
) -> None:
    print("="*80)
    print("RUNNING ALL CONFIGURATIONS")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Dataset split: {dataset_split}")
    print(f"Few shot mode: {few_shot_mode}")
    print(f"IDK configurations: [False, True]")
    print(f"Tasks: {TASK_LIST[dataset_split]}")
    print("="*80)
    
    model_cfg = load_model_config(model_name)
    instruct_model = model_cfg.get("instruct_model", False)
    require_explicit_disable_thinking = model_cfg.get("require_explicit_disable_thinking", False)
    print(f"Model instruct mode: {instruct_model}")
    
    _, _, model = initialize_model(model_name)
    
    repo_root = Path(__file__).resolve().parents[4]
    if results_file:
        results_path = Path(results_file)
        if results_path.is_dir():
            results_path = results_path
        else:
            results_path = results_path.parent
    else:
        results_path = repo_root / "results" / "ood_robustness" / "intermediate" / "direct_answers"
    
    results_path.mkdir(parents=True, exist_ok=True)
    
    model_cfg = getattr(model, "cfg", {}) or {}
    model_string = model_cfg.get("model_string") or model.__class__.__name__.lower()
    safe_model_string = str(model_string).replace(os.path.sep, "_")
    
    all_results = []
    #idk_configs = [False, True]
    idk_configs = [False]
    task_names = TASK_LIST[dataset_split]
    
    #Run all configurations
    for task_name in task_names:
        for idk in idk_configs:
            print(f"\n{'='*80}")
            print(f"Processing: task_name={task_name}, idk={idk}")
            print(f"{'='*80}")
            try:
                result = process_single_configuration(
                    dataset_name=dataset_name,
                    dataset_split=dataset_split,
                    task_name=task_name,
                    few_shot_mode=few_shot_mode,
                    enable_thinking=enable_thinking,
                    idk=idk,
                    model=model,
                    instruct_model=instruct_model,
                    require_explicit_disable_thinking=require_explicit_disable_thinking,
                    results_path=results_path,
                    safe_model_string=safe_model_string,
                    max_samples=max_samples,
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error processing task {task_name}, idk={idk}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Print average metrics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    
    # Print averages for each set type
    if not instruct_model:
        print_average_metrics(all_results, "thinking")
    if not require_explicit_disable_thinking:
        print_average_metrics(all_results, "non_thinking")
    
    print("ALL CONFIGURATIONS COMPLETED")
    print(f"{'='*80}")


def main(
    dataset_name: str = "prajaktakini/realtime_qa_new",
    dataset_split: str = "train",
    few_shot_mode: int = 0,
    max_samples: int = None,
    enable_thinking: bool = True,
    model_name: str = None,
    idk: bool = False,
    results_file: str = None,
    run_all: bool = False,
):
    if run_all:
        return run_all_configurations(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            few_shot_mode=few_shot_mode,
            enable_thinking=enable_thinking,
            model_name=model_name,
            results_file=results_file,
            max_samples=max_samples
        )
    
    print("Model name: ", model_name)
    print("IDK: ", idk)

    dataset = fetch_dataset(dataset_name=dataset_name, dataset_split=dataset_split)
    print("dataset[:5]", dataset[:5])

    model_cfg = load_model_config(model_name)
    model_family = model_cfg.get("model_family", "model")
    model_provider = model_cfg.get("model_provider", "your provider")
    instruct_model = model_cfg.get("instruct_model", False)
    require_explicit_disable_thinking = model_cfg.get("require_explicit_disable_thinking", False)
    print(f"Model instruct mode: {instruct_model}")

    _, _, model = initialize_model(model_name)
    
    for task_name in TASK_LIST[dataset_split]:
        dataset_task = create_dataset(dataset, task_name, idk, max_samples)
        print(f"Created {len(dataset_task)} samples for task {task_name}")

        # if idk:
        #     task_description = SYSTEM_PROMPTS_TEMPLATE[0]["task_desc"] 
        # else:
        #     task_description = SYSTEM_PROMPTS_TEMPLATE[0]["task_desc"] 
        
        task_description = SYSTEM_PROMPTS_TEMPLATE[0]["task_desc"]

        rows_with_thinking = []
        rows_without_thinking = []

        if not instruct_model:
            print("Generating responses with thinking enabled")
            rows_with_thinking = get_response(
                model=model,
                data=dataset_task,
                task_message=task_description,
                task_name=task_name,
                enable_thinking=True,
                instruct_model=instruct_model,
                require_explicit_disable_thinking=require_explicit_disable_thinking
            )
            thinking_df = pd.DataFrame(rows_with_thinking)

        if not require_explicit_disable_thinking:
            print("Generating responses without thinking")
            rows_without_thinking = get_response(
                model=model,
                data=dataset_task,
                task_message=task_description,
                task_name=task_name,
                enable_thinking=False,
                instruct_model=instruct_model,
                require_explicit_disable_thinking=require_explicit_disable_thinking
            )
            non_thinking_df = pd.DataFrame(rows_without_thinking)

        # Determine which sets to evaluate
        sets_to_evaluate = []
        if rows_with_thinking:
            sets_to_evaluate.append(("thinking", rows_with_thinking))
        if rows_without_thinking:
            sets_to_evaluate.append(("non_thinking", rows_without_thinking))


        repo_root = Path(__file__).resolve().parents[4]
        default_dir = repo_root / "results" / "ood_robustness" / "intermediate"

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

        task_file = results_path / f"ood_robustness_{dataset_split}_{task_name}_{safe_model_string}_idk_{idk}.xlsx"
    
        with pd.ExcelWriter(task_file) as writer:
            if not instruct_model:
                thinking_df.to_excel(writer, sheet_name="Thinking", index=False)
            if not require_explicit_disable_thinking:
                non_thinking_df.to_excel(writer, sheet_name="Non-Thinking", index=False)

        print(f"Saved results to: {task_file}")

        for set_name, responses in sets_to_evaluate:
            print(f"Evaluating {set_name} set")
            evaluate_response(
                responses=responses,
                task_name=task_name,
                few_shot_mode=few_shot_mode,
                set_name=set_name
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM outputs and save to Excel")
    parser.add_argument("--dataset-name", type=str, default="prajaktakini/realtime_qa_new")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--few-shot-mode", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking extraction")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--idk", action="store_true", help="Include I don't know option")
    parser.add_argument("--results-file", type=str, default=None)
    parser.add_argument("--run-all", action="store_true", help="Run all configurations: both idk=False and idk=True for all tasks")
    args = parser.parse_args()

    main(
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            few_shot_mode=args.few_shot_mode,
            max_samples=args.max_samples,
            enable_thinking=args.enable_thinking,
            model_name=args.model_name,
            idk=args.idk,
            results_file=args.results_file,
            run_all=args.run_all,
        )     