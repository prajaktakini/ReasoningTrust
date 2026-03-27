# ReasoningTrust (quick notes)

ReasoningTrust is a lightweight evaluation project for measuring LLM behavior on trust and safety tasks.
It generates model outputs for benchmark prompts and evaluates them with task-specific metrics (for example, toxicity scoring with Perspective API).

Current perspectives in this repo:
- `toxicity`
- `machine_ethics`
- `privacy`
- `ood_robustness`

Main source code is in `src/reasoning_trust/`, with perspective pipelines under `src/reasoning_trust/perspectives/`.

## Run location

Most scripts expect:
- project path: `/projects/beuc/ReasoningTrust/src`
- conda env: `llm_reasoning`

## Slurm jobs

Slurm jobs are already set up in `scripts/`.

Submit a full run with:
- `sbatch scripts/toxicity_run_all.sh`
- `sbatch scripts/machine_ethics_run_all.sh`
- `sbatch scripts/privacy_run_all.sh`
- `sbatch scripts/ood_robustness_run_all.sh`

Optional targeted runs:
- `scripts/machine_ethics_run_specific.sh`
- `scripts/toxicity_specific_class_specific_template.sh`
- `scripts/linear_probe_job.slurm`

## Outputs

Generated outputs are saved under `results/`.
