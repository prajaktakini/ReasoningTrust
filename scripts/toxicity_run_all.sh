#!/bin/bash

#SBATCH --mem=40g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --time=48:00:00
#SBATCH --account=account_name
#SBATCH --job-name=toxicity_run_all_deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#SBATCH --constraint="scratch"
#SBATCH -e slurm-perspective-%j.err
#SBATCH -o slurm-perspective-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=email_address
#SBATCH --mail-type="BEGIN,END,FAIL"

# Record job start time
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

export PERSPECTIVE_API_KEY=api_key


# if you don't want to run all classes, you can comment out the following lines
# we have 3 classes for toxicity: benign, adversarial and non-toxic. start:end indicates the index of the templates for each class
# We have 3 benign templates, 6 adversarial templates, and 5 non-toxic templates
declare -a PROMPT_CLASSES=(
    "benign:0:2"
    "adversarial:0:5"
    "non-toxic:0:4"
)

calculate_duration() {
    local start=$1
    local end=$2
    echo $((end - start))
}

# Function to format seconds into HH:MM:SS
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}


python -m reasoning_trust.perspectives.toxicity.generate_and_evaluate_toxicity \
    --dataset-name 'AI-Secure/DecodingTrust' \
    --dataset-config 'toxicity' \
    --dataset-split 'realtoxicityprompts.toxic' \
    --max-samples 1200 \
    --model-name 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' \
    --save-intermediate \
    --prompt-class 'all_classes' \
    --run-all \
    --enable-thinking

JOB_END_TIME=$(date +%s)
JOB_END_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
JOB_DURATION=$(calculate_duration $JOB_START_TIME $JOB_END_TIME)
JOB_DURATION_FORMATTED=$(format_duration $JOB_DURATION)

echo ""
echo "=========================================="
echo "All jobs completed for all prompt classes."
echo "Job end time: ${JOB_END_TIME_READABLE}"
echo "Total job time: ${JOB_DURATION_FORMATTED} (${JOB_DURATION} seconds)"
echo "=========================================="
