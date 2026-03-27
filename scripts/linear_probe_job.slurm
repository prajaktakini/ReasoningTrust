#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=account_name
#SBATCH --job-name=linear_probe_toxicity
#SBATCH --time=12:00:00
#SBATCH --constraint="scratch"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=email_address
#SBATCH --mail-type="BEGIN,END,FAIL"

echo "Job starting on $(hostname)"
echo "Slurm job id: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

PROJECT_DIR=${PROJECT_DIR:-/projects/beuc/ReasoningTrust/src}
cd "$PROJECT_DIR" || { echo "Cannot cd to $PROJECT_DIR"; exit 1; }

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER

module purge

module load anaconda3_gpu

source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh

conda activate llm_reasoning


MODEL_NAME="Qwen/Qwen2.5-4B-Instruct"
RESULTS_BASE_DIR="${RESULTS_DIR:-results/linear_probes/comparative}"
BATCH_SIZE=4
CHUNK_SIZE=50  


JSONL_BASE_DIR="${JSONL_BASE_DIR:-results/final}"

for TEMPLATE_ID in 3 4 5 8 12; do
  echo "=========================================="
  echo "Processing Template ${TEMPLATE_ID}"
  echo "=========================================="
  
  JSONL_THINKING="${JSONL_BASE_DIR}/toxicity_Qwen_Qwen3-4B-Thinking-2507_template_${TEMPLATE_ID}_Thinking_Evaluated.jsonl"
  JSONL_INSTRUCT="${JSONL_BASE_DIR}/toxicity_Qwen_Qwen3-4B-Instruct-2507_template_${TEMPLATE_ID}_Non-Thinking_Evaluated.jsonl"
  
  # Check if files exist
  if [ ! -f "$JSONL_THINKING" ]; then
    echo "Warning: Thinking JSONL not found: $JSONL_THINKING"
    echo "Trying alternative naming pattern..."
    
    JSONL_THINKING="${JSONL_BASE_DIR}/toxicity_Qwen_Qwen3-4B-Thinking-2507_template_${TEMPLATE_ID}_Thinking_Evaluated.jsonl"
    if [ ! -f "$JSONL_THINKING" ]; then
      echo "Error: Thinking file still not found, skipping template ${TEMPLATE_ID}"
      continue
    fi
  fi
  
  if [ ! -f "$JSONL_INSTRUCT" ]; then
    echo "Warning: Instruct JSONL not found: $JSONL_INSTRUCT"
    echo "Trying alternative naming pattern..."
    JSONL_INSTRUCT="${JSONL_BASE_DIR}/toxicity_Qwen_Qwen3-4B-Instruct-2507_template_${TEMPLATE_ID}_Non-Thinking_Evaluated.jsonl"
    if [ ! -f "$JSONL_INSTRUCT" ]; then
      echo "Error: Instruct file still not found, skipping template ${TEMPLATE_ID}"
      continue
    fi
  fi
  
  echo "Using Thinking file: $JSONL_THINKING"
  echo "Using Instruct file: $JSONL_INSTRUCT"

  python -m reasoning_trust.perspectives.toxicity.comparative_linear_probe \
    --jsonl-thinking "$JSONL_THINKING" \
    --jsonl-instruct "$JSONL_INSTRUCT" \
    --model-name "$MODEL_NAME" \
    --output-dir "$RESULTS_BASE_DIR" \
    --batch-size "$BATCH_SIZE" \
    --chunk-size "$CHUNK_SIZE" \
    --template-id "$TEMPLATE_ID"
  
  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Template ${TEMPLATE_ID} failed with exit code: $EXIT_CODE"
  else
    echo "Template ${TEMPLATE_ID} completed successfully"
  fi
  
  
  python -c "import torch; torch.cuda.empty_cache()"
  
  echo ""
done

echo "=========================================="
echo "All linear probe jobs completed"
echo "=========================================="

