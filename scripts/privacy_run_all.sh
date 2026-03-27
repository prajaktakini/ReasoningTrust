#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=account_name
#SBATCH --job-name=privacy_pii_Qwen/Qwen2.5-1.5B-Instruct_run_all
#SBATCH --time=2:00:00
#SBATCH --constraint="scratch"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=email_address
#SBATCH --mail-type="BEGIN,END,FAIL"

echo "Job starting on $(hostname)"
echo "Slurm job id: $SLURM_JOB_ID"

PROJECT_DIR=${PROJECT_DIR:-/projects/beuc/ReasoningTrust/src}
cd "$PROJECT_DIR" || { echo "Cannot cd to $PROJECT_DIR"; exit 1; }

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER

module purge
module load anaconda3_gpu
source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda activate llm_reasoning


MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
SCENARIO_NAME="pii"
FEW_SHOT_TYPE="all" # We have 3 few-shot types: protect, attack # all indicates that we will run both protect and attack
FEW_SHOT_NUM="0 3" # We have 2 few-shot modes: 0 and 3 # For zero shot, we have k=0 attack, for 3 shot we have k=3 attack and k=3 protect
MAX_SAMPLES=200


python -m reasoning_trust.perspectives.privacy.process_data \
  --model-name "$MODEL_NAME" \
  --scenario-name "$SCENARIO_NAME" \
  --fewshot-type "$FEW_SHOT_TYPE" \
  --few-shot-num $FEW_SHOT_NUM \
  --max-samples $MAX_SAMPLES \
  --enable-thinking 
  
echo "Run all job completed"