#!/bin/bash

#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=account_name
#SBATCH --job-name=reasoning_trust_machine_ethics_jailbreaking_short_shot0_simplescaling/s1.1-14B
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch"
#SBATCH -e slurm-machine-ethics-%j.err
#SBATCH -o slurm-machine-ethics-%j.out
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

MAX_SAMPLES=1000
MODEL_NAME="simplescaling/s1.1-14B"

# Few-shot modes to process
#FEW_SHOT_MODES=(0 5) # We have 2 few-shot modes: 0 and 5
FEW_SHOT_MODES=(0)

# Prompt types to process
# PROMPT_TYPES=("short" "long") # We have 2 prompt types: short and long
PROMPT_TYPES=("short")

# Jailbreak template IDs to process (0-4) # we have 5 jailbreak templates
JAILBREAK_TEMPLATE_IDS=(0 1 2 3 4) # Select according to your needs


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

# Loop through each few-shot mode
for FEW_SHOT_MODE in "${FEW_SHOT_MODES[@]}"; do
    MODE_START_TIME=$(date +%s)
    MODE_START_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo ""
    echo "=========================================="
    echo "Processing Few-Shot Mode: ${FEW_SHOT_MODE}"
    echo "Mode start time: ${MODE_START_TIME_READABLE}"
    echo "=========================================="
    
    # Loop through each prompt type
    for PROMPT_TYPE in "${PROMPT_TYPES[@]}"; do
        TYPE_START_TIME=$(date +%s)
        TYPE_START_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
        
        echo ""
        echo "---"
        echo "Prompt Type: ${PROMPT_TYPE} | Few-Shot Mode: ${FEW_SHOT_MODE}"
        echo "Type start time: ${TYPE_START_TIME_READABLE}"
        echo "---"
        
        # Loop through each jailbreak template ID
        for JAILBREAK_ID in "${JAILBREAK_TEMPLATE_IDS[@]}"; do
            JAILBREAK_START_TIME=$(date +%s)
            JAILBREAK_START_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
            
            echo ""
            echo "  Processing Jailbreak Template ID: ${JAILBREAK_ID}"
            echo "  Jailbreak start time: ${JAILBREAK_START_TIME_READABLE}"
            
            python -m reasoning_trust.perspectives.machine_ethics.process_data \
                --dataset-name "AI-Secure/DecodingTrust" \
                --dataset-config "machine_ethics" \
                --train-dataset-split "cm_train" \
                --test-dataset-split "cm_test" \
                --data-name "ethics_commonsense" \
                --few-shot-mode "$FEW_SHOT_MODE" \
                --max-test-samples "$MAX_SAMPLES" \
                --model-name "$MODEL_NAME" \
                --enable-thinking \
                --prompt-type "$PROMPT_TYPE" \
                --is-jailbreaking \
                --jailbreak-template-id "$JAILBREAK_ID"
            
            exit_code=$?
            JAILBREAK_END_TIME=$(date +%s)
            JAILBREAK_END_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
            JAILBREAK_DURATION=$(calculate_duration $JAILBREAK_START_TIME $JAILBREAK_END_TIME)
            JAILBREAK_DURATION_FORMATTED=$(format_duration $JAILBREAK_DURATION)
            
            echo "  Jailbreak Template ID ${JAILBREAK_ID} completed"
            echo "  Jailbreak end time: ${JAILBREAK_END_TIME_READABLE}"
            echo "  Jailbreak duration: ${JAILBREAK_DURATION_FORMATTED} (${JAILBREAK_DURATION} seconds)"
            echo "  Exit code: ${exit_code}"
            
            if [ ${exit_code} -ne 0 ]; then
                echo "  ERROR: Jailbreak Template ID ${JAILBREAK_ID} (Prompt Type ${PROMPT_TYPE}, Few-Shot ${FEW_SHOT_MODE}) failed with exit code ${exit_code}"
            fi
        done
        
        TYPE_END_TIME=$(date +%s)
        TYPE_END_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
        TYPE_DURATION=$(calculate_duration $TYPE_START_TIME $TYPE_END_TIME)
        TYPE_DURATION_FORMATTED=$(format_duration $TYPE_DURATION)
        
        echo ""
        echo "---"
        echo "Prompt Type ${PROMPT_TYPE} (Few-Shot ${FEW_SHOT_MODE}) completed for all jailbreak templates"
        echo "Type end time: ${TYPE_END_TIME_READABLE}"
        echo "Type duration: ${TYPE_DURATION_FORMATTED} (${TYPE_DURATION} seconds)"
        echo "---"
    done
    
    MODE_END_TIME=$(date +%s)
    MODE_END_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
    MODE_DURATION=$(calculate_duration $MODE_START_TIME $MODE_END_TIME)
    MODE_DURATION_FORMATTED=$(format_duration $MODE_DURATION)
    
    echo ""
    echo "=========================================="
    echo "Completed all prompt types for Few-Shot Mode: ${FEW_SHOT_MODE}"
    echo "Mode end time: ${MODE_END_TIME_READABLE}"
    echo "Total time for few-shot mode: ${MODE_DURATION_FORMATTED} (${MODE_DURATION} seconds)"
    echo "=========================================="
done

JOB_END_TIME=$(date +%s)
JOB_END_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
JOB_DURATION=$(calculate_duration $JOB_START_TIME $JOB_END_TIME)
JOB_DURATION_FORMATTED=$(format_duration $JOB_DURATION)

echo ""
echo "=========================================="
echo "All jobs completed for all few-shot modes, prompt types, and jailbreak templates."
echo "Job end time: ${JOB_END_TIME_READABLE}"
echo "Total job time: ${JOB_DURATION_FORMATTED} (${JOB_DURATION} seconds)"
echo "=========================================="