# Toxicity perspective — ReasoningTrust

This document describes how to install dependencies and run the toxicity pipeline and evaluation.

## Requirements

- Python 3.10+ (ensure a compatible runtime)
- GPU recommended for vLLM models
- Set the Perspective API key in the environment:
  - `export PERSPECTIVE_API_KEY="your_api_key_here"`

## Create and activate a virtual environment

```bash
# Create Conda environment
conda create -n reasoning_trust python=3.10
conda activate reasoning_trust
```

## Local build, install and run (quick steps)

1) Update packaging tools and build a wheel (two common options)
```bash
python -m pip install --upgrade pip setuptools wheel
# Option A: build wheel using pip (creates .whl in cwd)
python -m pip wheel .
# Option B: if the project uses PEP 517/518 and 'build' is available
python -m pip install build
python -m build
```
2) Install package in editable mode (use when developing)
```bash
python -m pip install -e .
```
3) zsh config file path and reload
```bash
# Edit `~/.zshrc` to add env vars (example)
echo 'export PERSPECTIVE_API_KEY="your_key_here"' >> ~/.zshrc
# Reload the file into current shell
source ~/.zshrc
```
4) Run the toxicity pipeline (examples below)
```bash
python src/reasoning_trust/app.py
# or
python -m reasoning_trust.app
```

5) Or run with explicit optional arguments
```bash
PYTHONPATH=src python src/reasoning_trust/app.py toxicity \
  --dataset-name "AI-Secure/DecodingTrust" \
  --dataset-config "toxicity" \
  --dataset-split "realtoxicityprompts.toxic" \
  --max-samples 2 \
  --file-name "Role_Playing_13_DeepSeek_NEW.xlsx" \
  --template-id 12 \
  --model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --enable-thinking
```

## Configuration

- Model configuration is stored in `src/reasoning_trust/config/models.yaml`.
- Default model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (can be overridden via CLI).
- Results are written to the `results/` directory by default.

## Run the toxicity pipeline

Two ways: with the `toxicity` subcommand or top-level args.

Example (explicit subcommand):

\```bash
python -m reasoning_trust.app toxicity \
  --dataset-name AI-Secure/DecodingTrust \
  --dataset-config toxicity \
  --dataset-split realtoxicityprompts.toxic \
  --max-samples 200 \
  --file-name Role_Playing_13_DeepSeek_NEW.xlsx \
  --template-id 12 \
  --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --enable-thinking
\```

Example (top-level invocation, no subcommand):

\```bash
python -m reasoning_trust.app \
  --max-samples 50 \
  --file-name small_run.xlsx \
  --template-id 12 \
  --enable-thinking
\```

Notes:
- `--enable-thinking` toggles extraction of the CoT (thinking) section.
- `--results-file` (if provided) can be used to set an explicit output path in the `toxicity` subparser.

## Evaluate model outputs (Perspective API)

The evaluation script is `src/reasoning_trust/perspectives/toxicity/evaluate_toxicity.py`. It expects the results Excel file to be present and the `PERSPECTIVE_API_KEY` env var set.

Run evaluation:

\```bash
python -m reasoning_trust.perspectives.toxicity.evaluate_toxicity
\```

By default the script reads `results/Role_Playing_13_DeepSeek_Final.xlsx` and writes `results/Role_Playing_13_DeepSeek_Final_Thinking_Evaluated.jsonl`. Edit the `FILE_PATH` / `SHEET_NAME` variables inside the script if you need to evaluate a different file/sheet.

## Implementation notes

- Model initialization and generation live in `src/reasoning_trust/models/initialize_model.py`.
- The CLI entrypoint is `src/reasoning_trust/app.py`.
- Tokenizers rely on Hugging Face repositories; ensure network access or cached models.
- vLLM (`LLM` and `SamplingParams`) is used for generation; adjust `max_model_len` / `max_num_batched_tokens` in `models.yaml` if you encounter memory errors.

## Troubleshooting

- If the Perspective API fails, confirm `PERSPECTIVE_API_KEY` is set and has quota.
- For vLLM memory errors, reduce `max_num_batched_tokens` or run fewer concurrent samples.
- If outputs are empty when `--enable-thinking` is off, check the prompt template and how the `apply_chat_template` handles the `thinking` flag.

## Contact

For code questions, inspect:
- `src/reasoning_trust/perspectives/toxicity/toxicity_system_prompts.py` (prompt templates)
- `src/reasoning_trust/models/initialize_model.py` (model wrapper)
- `src/reasoning_trust/perspectives/toxicity/evaluate_toxicity.py` (evaluation)