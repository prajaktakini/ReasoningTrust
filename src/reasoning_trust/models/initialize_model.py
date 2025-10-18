import os
import yaml
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


class BaseLLM:
    """
    Minimal interface + default extractor fallback.
    Subclasses should set: tokenizer, sampling_params, engine (optional).
    """

    tokenizer: Any = None
    sampling_params: Dict[str, Any] = None
    engine: Any = None

    def generate(self, prompt: Any, enable_thinking: bool) -> List[str]:
        raise NotImplementedError()

    def extract_think_and_content(self, generated_text: str) -> Tuple[str, str]:
        if not isinstance(generated_text, str):
            return "", str(generated_text)
        match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL | re.IGNORECASE)
        if match:
            think = match.group(1).strip()
            after = generated_text.split('</think>', 1)[-1].strip() if '</think>' in generated_text.lower() else ""
            return think, after
        if '</think>' in generated_text.lower():
            parts = re.split(r'</think>', generated_text, flags=re.IGNORECASE, maxsplit=1)
            think_part = parts[0].strip()
            after_part = parts[1].strip() if len(parts) > 1 else ""
            return think_part, after_part
        return "", generated_text.strip()



class QwenLLM(BaseLLM):
    """
    Lightweight Qwen wrapper. Accepts model config dict on init.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}

        model_str = self.cfg.get("model_string")
        tokenizer_str = self.cfg.get("tokenizer_string", model_str)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
        self.sampling_params = SamplingParams(
            temperature=float(self.cfg.get("temperature", 0.6)),
            top_p=float(self.cfg.get("top_p", 0.95)),
            top_k=int(self.cfg.get("top_k", 20)),
            max_tokens=int(self.cfg.get("max_tokens", 2048)),
        )

        self.engine = LLM(
            model=model_str,
            max_model_len=int(self.cfg.get("max_model_len", 29000)),
            max_num_batched_tokens=int(self.cfg.get("max_num_batched_tokens", 29000)),
            gpu_memory_utilization=float(self.cfg.get("gpu_memory_utilization", 0.95)),
        )

    def generate(self, message: str, enable_thinking: bool) -> List[Any]:
        # use tokenizer.apply_chat_template helper if available in repo
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        outputs = self.engine.generate([text], self.sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        return outputs

    def extract_think_and_content(self, generated_text: str) -> Tuple[str, str]:
        match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL | re.IGNORECASE)
        think_content = match.group(1).strip() if match else ''
        after_think = generated_text.split('</think>')[-1].strip() if '</think>' in generated_text else generated_text.strip()
        return think_content, after_think


class DeepSeekLLM(BaseLLM):
    """
    DeepSeek wrapper: initializes tokenizer and vLLM engine using cfg.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        model_str = self.cfg.get("model_string")
        tokenizer_str = self.cfg.get("tokenizer_string", model_str)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)

        self.sampling_params = SamplingParams(
            temperature=float(self.cfg.get("temperature", 0.6)),
            top_p=float(self.cfg.get("top_p", 0.95)),
            top_k=int(self.cfg.get("top_k", 20)),
            max_tokens=int(self.cfg.get("max_tokens", 2048)),
        )
        # engine
        self.engine = LLM(
            model=model_str,
            max_model_len=int(self.cfg.get("max_model_len", 29000)),
            max_num_batched_tokens=int(self.cfg.get("max_num_batched_tokens", 29000)),
            gpu_memory_utilization=float(self.cfg.get("gpu_memory_utilization", 0.95)),
        )

    def generate(self, message, enable_thinking: bool) -> List[Any]:
        print("I am here - Enable thinking: ", enable_thinking)
        # use tokenizer.apply_chat_template helper if available in repo
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            thinking=enable_thinking,
        )

        if enable_thinking:
            if not text.rstrip().endswith("<think>"):
                text = text.rstrip() + "<think>\n"
        else:
            if text.rstrip().endswith("<think>"):
                text = text.rstrip()[:-7]
            text = text.rstrip() + "<think>\n\n</think>\n\n"

        outputs = self.engine.generate([text], self.sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        return outputs

    def extract_think_and_content(self, generated_text: str) -> Tuple[str, str]:
        if '</think>' in generated_text:
            parts = generated_text.split('</think>', 1)
            think = parts[0].replace('<think>', '').strip()
            content = parts[1].strip()
            return think, content
        return '', generated_text.strip()


@dataclass
class InnerOutput:
    text: str

@dataclass
class ModelOutput:
    prompt: str
    outputs: List[InnerOutput]

def _load_models_config() -> Dict[str, Any]:
    base = os.path.dirname(os.path.dirname(__file__))  # points to src/reasoning_trust
    cfg_path = os.path.join(base, "config", "models.yaml")
    with open(cfg_path, "r") as fh:
        return yaml.safe_load(fh) or {}

def initialize_model(model_name: str = None) -> Tuple[Any, Dict[str, Any], Any]:
    cfg = _load_models_config()
    defaults = cfg.get("defaults", {}) or {}
    model_name = (model_name or defaults.get("model") or "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    models = cfg.get("models", {})
    model_cfg = models.get(model_name, {})

    if "deepseek" in model_name.lower():
        model = DeepSeekLLM(model_cfg)
    elif "qwen" in model_name.lower():
        model = QwenLLM(model_cfg)
    else:
        # fallback to QwenLLM if unknown
        model = QwenLLM(model_cfg)

    return getattr(model, "tokenizer", None), getattr(model, "sampling_params", None), model

def generate_response(llm_model: Any, message: str, enable_thinking: bool) -> List[ModelOutput]:
    response = llm_model.generate(message, enable_thinking)

    for output in response:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    return response