#!/bin/bash

#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=account_name
#SBATCH --job-name=reasoning_trust_machine_ethics_all_Qwen/Qwen2.5-7B-Instruct
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch"
#SBATCH -e slurm-machine-ethics-%j.err
#SBATCH -o slurm-machine-ethics-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=email_address
#SBATCH --mail-type="BEGIN,END,FAIL"


JOB_START_TIME=$(date +%s)
JOB_START_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')

echo "=========================================="
echo "Job starting on $(hostname)"
echo "Slurm job id: $SLURM_JOB_ID"
echo "Job start time: ${JOB_START_TIME_READABLE}"
echo "=========================================="

PROJECT_DIR=${PROJECT_DIR:-/projects/beuc/ReasoningTrust/src}
cd "$PROJECT_DIR" || { echo "Cannot cd to $PROJECT_DIR"; exit 1; }

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER
module purge
module load anaconda3_gpu
source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda activate llm_reasoning

MAX_SAMPLES=1000
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

echo ""
echo "=========================================="
echo "Running all configurations:"
echo "  - Model: ${MODEL_NAME}"
echo "  - Max samples: ${MAX_SAMPLES}"
echo "  - Prompt types: short, long"
echo "  - Few-shot modes: 0, 5"
echo "  - Benign: 1 template"
echo "  - Jailbreaking: 5 templates (0-4)"
echo "=========================================="

python -m reasoning_trust.perspectives.machine_ethics.process_data \
    --dataset-name "AI-Secure/DecodingTrust" \
    --dataset-config "machine_ethics" \
    --train-dataset-split "cm_train" \
    --test-dataset-split "cm_test" \
    --data-name "ethics_commonsense" \
    --max-test-samples "$MAX_SAMPLES" \
    --model-name "$MODEL_NAME" \
    --enable-thinking \
    --run-all

exit_code=$?
JOB_END_TIME=$(date +%s)
JOB_END_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
JOB_DURATION=$((JOB_END_TIME - JOB_START_TIME))
JOB_HOURS=$((JOB_DURATION / 3600))
JOB_MINS=$(((JOB_DURATION % 3600) / 60))
JOB_SECS=$((JOB_DURATION % 60))

echo ""
echo "=========================================="
echo "Job completed"
echo "Job end time: ${JOB_END_TIME_READABLE}"
echo "Total job time: ${JOB_HOURS}h ${JOB_MINS}m ${JOB_SECS}s"
echo "Exit code: ${exit_code}"
echo "=========================================="

exit $exit_code