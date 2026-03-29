import json
import os
import re
import yaml

import requests
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


class BaseLLM:
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
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}

        model_str = self.cfg.get("model_string")
        tokenizer_str = self.cfg.get("tokenizer_string", model_str)
        #print("Tokenizer string: ", tokenizer_str)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
        self.sampling_params = SamplingParams(
            temperature=float(self.cfg.get("temperature", 0.6)),
            top_p=float(self.cfg.get("top_p", 0.95)),
            top_k=int(self.cfg.get("top_k", 20)),
            max_tokens=int(self.cfg.get("max_tokens", 2048)),
        )

        self.engine = LLM(
            model=model_str,
            dtype="bfloat16",
            max_model_len=int(self.cfg.get("max_model_len", 29000)),
            max_num_batched_tokens=int(self.cfg.get("max_num_batched_tokens", 29000)),
            gpu_memory_utilization=float(self.cfg.get("gpu_memory_utilization", 0.95)),
            # tensor_parallel_size=int(2),
        )

    def generate(self, message: str, enable_thinking: bool, instruct_model: bool, require_explicit_disable_thinking: bool) -> List[Any]:
        if instruct_model:
            text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            
            if require_explicit_disable_thinking and not enable_thinking:
                if text.rstrip().endswith("<think>"):
                    text = text.rstrip()[:-7]
                text = text.rstrip() + "<think></think>"

        outputs = self.engine.generate([text], self.sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            # print(f"Prompt: {prompt}, Generated text: {generated_text}")
        
        return outputs

    def extract_think_and_content(self, generated_text: str, require_explicit_disable_thinking: bool) -> Tuple[str, str]:
        if require_explicit_disable_thinking and '</think>' in generated_text:
            if "</think>" in generated_text:
                parts = re.split(r'</think>', generated_text, flags=re.IGNORECASE, maxsplit=1)
                think_part = re.sub(r'(?i)<think>', '', parts[0]).strip()
                content_part = parts[1].strip() if len(parts) > 1 else ""
                return think_part, content_part
            
        match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL | re.IGNORECASE)
        think_content = match.group(1).strip() if match else ''
        after_think = generated_text.split('</think>')[-1].strip() if '</think>' in generated_text else generated_text.strip()
        return think_content, after_think


class DeepSeekLLM(BaseLLM):
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
            dtype="bfloat16",
            max_model_len=int(self.cfg.get("max_model_len", 29000)),
            max_num_batched_tokens=int(self.cfg.get("max_num_batched_tokens", 29000)),
            gpu_memory_utilization=float(self.cfg.get("gpu_memory_utilization", 0.95)),
        )

    def generate(self, message, enable_thinking: bool, instruct_model: bool, require_explicit_disable_thinking: bool) -> List[Any]:
        if instruct_model:
            text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
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
            #print(f"Prompt: {prompt}, Generated text: {generated_text}")

        return outputs

    def extract_think_and_content(self, generated_text: str, require_explicit_disable_thinking: bool) -> Tuple[str, str]:
        if '</think>' in generated_text:
            parts = generated_text.split('</think>', 1)
            think = parts[0].replace('<think>', '').strip()
            content = parts[1].strip()
            return think, content
        return '', generated_text.strip()


class LlamaLLM(BaseLLM):

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}

        model_str = self.cfg.get("model_string")
        tokenizer_str = self.cfg.get("tokenizer_string", model_str)
        #print("Tokenizer string: ", tokenizer_str)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
        self.sampling_params = SamplingParams(
            temperature=float(self.cfg.get("temperature", 0.6)),
            top_p=float(self.cfg.get("top_p", 0.95)),
            top_k=int(self.cfg.get("top_k", 20)),
            max_tokens=int(self.cfg.get("max_tokens", 2048)),
        )

        self.engine = LLM(
            model=model_str,
            dtype="bfloat16",
            max_model_len=int(self.cfg.get("max_model_len", 29000)),
            max_num_batched_tokens=int(self.cfg.get("max_num_batched_tokens", 29000)),
            gpu_memory_utilization=float(self.cfg.get("gpu_memory_utilization", 0.95)),
            # tensor_parallel_size=int(2),
        )

    def generate(self, message: str, enable_thinking: bool, instruct_model: bool, require_explicit_disable_thinking: bool) -> List[Any]:
        # use tokenizer.apply_chat_template helper if available in repo

        if instruct_model:
            text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            #print("Generated text before thinking tags: ", text)
            if require_explicit_disable_thinking and not enable_thinking:
                if text.rstrip().endswith("<think>"):
                    text = text.rstrip()[:-7]
                text = text.rstrip() + "<think></think>"

        outputs = self.engine.generate([text], self.sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        return outputs

    def extract_think_and_content(self, generated_text: str, require_explicit_disable_thinking: bool) -> Tuple[str, str]:
        if require_explicit_disable_thinking and '</think>' in generated_text:
            if "</think>" in generated_text:
                parts = re.split(r'</think>', generated_text, flags=re.IGNORECASE, maxsplit=1)
                think_part = re.sub(r'(?i)<think>', '', parts[0]).strip()
                content_part = parts[1].strip() if len(parts) > 1 else ""
                return think_part, content_part
            
        match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL | re.IGNORECASE)
        think_content = match.group(1).strip() if match else ''
        after_think = generated_text.split('</think>')[-1].strip() if '</think>' in generated_text else generated_text.strip()
        return think_content, after_think


class SimpleScalingLLM(BaseLLM):
   
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
            dtype="bfloat16",
            max_model_len=int(self.cfg.get("max_model_len", 29000)),
            max_num_batched_tokens=int(self.cfg.get("max_num_batched_tokens", 29000)),
            gpu_memory_utilization=float(self.cfg.get("gpu_memory_utilization", 0.95)),
        )

    def generate(self, message, enable_thinking: bool, instruct_model: bool, require_explicit_disable_thinking: bool) -> List[Any]:
        if instruct_model:
            text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                thinking=enable_thinking,
            )

            if enable_thinking:
                if not text.rstrip().endswith("`<think>`"):
                    text = text.rstrip() + "`<think>`\n"
            else:
                if require_explicit_disable_thinking:
                    # For SimpleScaling models, explicitly disable thinking
                    if text.rstrip().endswith("`<think>`"):
                        text = text.rstrip()[:-7]
                    text = text.rstrip() + "`<think>``</think>`"
                else:
                    if text.rstrip().endswith("`<think>`"):
                        text = text.rstrip()[:-7]
                    text = text.rstrip() + "`<think>`\n\n`</think>`\n\n"

        outputs = self.engine.generate([text], self.sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        return outputs

    def extract_think_and_content(self, generated_text: str, require_explicit_disable_thinking: bool) -> Tuple[str, str]:
        if not isinstance(generated_text, str):
            return "", str(generated_text)

        answer_match = re.search(r'\banswer\b', generated_text, re.IGNORECASE)

        if answer_match:
            answer_pos = answer_match.start()
            think_part = generated_text[:answer_pos].strip()
            answer_part = generated_text[answer_match.end():].strip()
            
            think_part = re.sub(r'^\s*think\s*\n?', '', think_part, flags=re.IGNORECASE).strip()
            answer_part = re.sub(r'^\s*(answer\s*\n?|answer\s*:\s*)', '', answer_part, flags=re.IGNORECASE).strip()
            return think_part, answer_part
    
        # Fallback: if no "answer" marker found, check for other formats
        if '`</think>`' in generated_text:
            parts = generated_text.split('`</think>`', 1)
            think = parts[0].replace('`<think>`', '').strip()
            content = parts[1].strip()
            return think, content
        
        return '', generated_text.strip()


class AgenticaLLM(BaseLLM):

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
            dtype="bfloat16",
            max_model_len=int(self.cfg.get("max_model_len", 29000)),
            max_num_batched_tokens=int(self.cfg.get("max_num_batched_tokens", 29000)),
            gpu_memory_utilization=float(self.cfg.get("gpu_memory_utilization", 0.95)),
        )

    def generate(self, message, enable_thinking: bool, instruct_model: bool, require_explicit_disable_thinking: bool) -> List[Any]:
        if instruct_model:
            text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
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
            #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        return outputs

    def extract_think_and_content(self, generated_text: str, require_explicit_disable_thinking: bool) -> Tuple[str, str]:
        if '</think>' in generated_text:
            parts = generated_text.split('</think>', 1)
            think = parts[0].replace('<think>', '').strip()
            content = parts[1].strip()
            #print(f"Think: {think!r}, Content: {content!r}")
            return think, content
        return '', generated_text.strip()


@dataclass
class InnerOutput:
    text: str

@dataclass
class ModelOutput:
    prompt: str
    outputs: List[InnerOutput]


class AzureCloudLLM(BaseLLM):
    """OpenAI-compatible chat completions over HTTPS (e.g. Azure AI Foundry)."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.model_url = self.cfg.get("model_url")
        if not self.model_url:
            raise ValueError("is_cloud_model requires model_url in model config")
        self.api_key = os.environ.get("AZURE_LLM_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_LLM_API_KEY must be set for cloud models")
        self.tokenizer = None
        self.sampling_params = None

    def generate(
        self,
        message: Any,
        enable_thinking: bool,
        instruct_model: bool,
        require_explicit_disable_thinking: bool,
    ) -> List[ModelOutput]:
        del enable_thinking, instruct_model, require_explicit_disable_thinking
        if not isinstance(message, list):
            raise TypeError("Azure cloud model expects chat messages as a list of dicts")
        body: Dict[str, Any] = {
            "messages": message,
            "temperature": float(self.cfg.get("temperature", 0.6)),
            "top_p": float(self.cfg.get("top_p", 0.95)),
            "max_tokens": int(self.cfg.get("max_tokens", 4096)),
        }
        deployment = self.cfg.get("deployment_name") or self.cfg.get("model_string")
        if not deployment:
            raise ValueError(
                "Azure cloud models need azure_deployment (or model_string) in models.yaml "
                "so the API knows which deployment to use (DeploymentNotFound if missing)."
            )
        body["model"] = deployment
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        resp = requests.post(self.model_url, headers=headers, json=body, timeout=600)
        if not resp.ok:
            raise RuntimeError(
                f"Azure chat request failed ({resp.status_code}): {resp.text[:2000]}"
            )
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Azure response missing choices: {json.dumps(data)[:2000]}")
        msg = choices[0].get("message") or {}
        text = msg.get("content")
        if text is None:
            raise RuntimeError(f"Azure response missing message content: {json.dumps(data)[:2000]}")
        prompt_str = json.dumps(message)
        return [ModelOutput(prompt=prompt_str, outputs=[InnerOutput(text=text)])]

    def extract_think_and_content(
        self, generated_text: str, require_explicit_disable_thinking: bool
    ) -> Tuple[str, str]:
        if require_explicit_disable_thinking and "</think>" in generated_text:
            if "</think>" in generated_text:
                parts = re.split(r"</think>", generated_text, flags=re.IGNORECASE, maxsplit=1)
                think_part = re.sub(r"(?i)<think>", "", parts[0]).strip()
                content_part = parts[1].strip() if len(parts) > 1 else ""
                return think_part, content_part

        match = re.search(r"<think>(.*?)</think>", generated_text, re.DOTALL | re.IGNORECASE)
        think_content = match.group(1).strip() if match else ""
        after_think = (
            generated_text.split("</think>")[-1].strip()
            if "</think>" in generated_text
            else generated_text.strip()
        )
        return think_content, after_think


def load_models_config() -> Dict[str, Any]:
    base = os.path.dirname(os.path.dirname(__file__))  # points to src/reasoning_trust
    cfg_path = os.path.join(base, "config", "models.yaml")
    with open(cfg_path, "r") as fh:
        return yaml.safe_load(fh) or {}

def load_model_config(model_name: str) -> Dict[str, Any]:
    cfg = load_models_config()
    defaults = cfg.get("defaults", {}) or {}
    model_name = (model_name or defaults.get("model") or "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    models = cfg.get("models", {})
    model_cfg = models.get(model_name, {})
    return model_cfg

def initialize_model(model_name: str = None) -> Tuple[Any, Dict[str, Any], Any]:
    model_cfg = load_model_config(model_name)
    #print("Loaded model config: ", model_cfg)

    if model_cfg.get("is_cloud_model"):
        model = AzureCloudLLM(model_cfg)
        return getattr(model, "tokenizer", None), getattr(model, "sampling_params", None), model

    if "deepseek" in model_name.lower():
        model = DeepSeekLLM(model_cfg)
    elif "qwen" in model_name.lower():
        model = QwenLLM(model_cfg)
    elif "simplescaling" in model_name.lower():
        model = SimpleScalingLLM(model_cfg)
    elif "llama" in model_name.lower():
        print("Initializing Llama model")
        model = LlamaLLM(model_cfg)
    elif "agentica" in model_name.lower():
        print("Initializing Agentica model")
        model = AgenticaLLM(model_cfg)
    else:
        # fallback to QwenLLM if unknown
        model = QwenLLM(model_cfg)

    return getattr(model, "tokenizer", None), getattr(model, "sampling_params", None), model


def generate_response(llm_model: Any, message: str, enable_thinking: bool, instruct_model:bool, require_explicit_disable_thinking: bool) -> List[ModelOutput]:
    response = llm_model.generate(message, enable_thinking, instruct_model, require_explicit_disable_thinking)

    for output in response:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        #print(f"Prompt: {prompt}, Generated text: {generated_text}")
    return response