# ReasoningTrust

Lightweight runner for generating LLM outputs (with optional chain-of-thought) and evaluating toxicity using the Perspective API.

## Project layout (important files)
- `src/reasoning_trust/app.py` — CLI entrypoint to run pipelines (default: `toxicity`).
- `src/reasoning_trust/models/initialize_model.py` — model wrappers and `initialize_model()` / generation helpers.
- `src/reasoning_trust/perspectives/toxicity/` — toxicity prompts, generation pipeline, and evaluation:
  - `toxicity_system_prompts.py` — prompt templates.
  - `evaluate_toxicity.py` — uses Perspective API to score outputs.
- `src/reasoning_trust/config/models.yaml` — model configurations used by `initialize_model`.
- `results/` — directory where generated Excel and evaluation outputs are saved.
- `.env` — (local) place for `PERSPECTIVE_API_KEY` (do not commit secrets).

## Requirements
- macOS (as developer environment)
- Python 3.9+ recommended
- GPU and drivers for vLLM when using local vLLM engines (vLLM requires CUDA + compatible drivers)
- Suggested Python packages (put in `requirements.txt`):
  - `transformers`
  - `vllm`
  - `pyyaml`
  - `pandas`
  - `openpyxl`
  - `google-api-python-client`
  - `omegaconf`
  - `hydra-core`
  - `datasets` (optional, if loading dataset via Hugging Face)
  - `python-dotenv` (optional, for loading `.env`)

Install:
Create and activate a virtual environment/Conda Environment, then install dependencies:
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

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