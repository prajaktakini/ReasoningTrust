#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=account_name
#SBATCH --job-name=ood_robustness_Qwen/Qwen2.5-7B-Instruct_run_all
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=prajakta.kini@colorado.edu
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


MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DATASET_SPLIT="train"

# DATASET we are using is prajaktakini/realtime_qa_new


python -m reasoning_trust.perspectives.ood_robustness.process_data \
  --model-name "$MODEL_NAME" \
  --dataset-split "$DATASET_SPLIT" \
  --enable-thinking \
  --run-all
  
echo "Run all job completed"
  

echo "Job completed with exit code: $?"