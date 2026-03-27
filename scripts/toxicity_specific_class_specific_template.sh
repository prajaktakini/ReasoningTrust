#!/bin/bash

#SBATCH --mem=40g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --time=48:00:00
#SBATCH --account=account_name
#SBATCH --job-name=toxicity_adv_temp5_simplescaling/s1.1-14B
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

export PERSPECTIVE_API_KEY="api_key" 

MAX_SAMPLES=1200
MODEL_NAME="simplescaling/s1.1-14B"

# Define prompt classes and their template ID ranges
# Format: "class_name:start_id:end_id" # we have 3 classes for toxicity: benign, adversarial and non-toxic. start:end indicates the index of the templates for each class
# We have 3 benign templates, 6 adversarial templates, and 5 non-toxic templates
declare -a PROMPT_CLASSES=(
    "adversarial:5:5"
)

# Function to calculate time difference in seconds
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


# Loop through each prompt class
for class_config in "${PROMPT_CLASSES[@]}"; do

    IFS=':' read -r PROMPT_CLASS start_id end_id <<< "$class_config"
    
    CLASS_START_TIME=$(date +%s)
    CLASS_START_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo ""
    echo "=========================================="
    echo "Processing Prompt Class: ${PROMPT_CLASS}"
    echo "Template IDs: ${start_id} to ${end_id}"
    echo "Class start time: ${CLASS_START_TIME_READABLE}"
    echo "=========================================="
    
    # Loop through template IDs for this class
    # start_id and end_id are the indices of the templates for this class
    for TEMPLATE_ID in $(seq ${start_id} ${end_id}); do
        TEMPLATE_START_TIME=$(date +%s)
        TEMPLATE_START_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
        
        echo ""
        echo "---"
        echo "Template ${TEMPLATE_ID} -> PROMPT Class: ${PROMPT_CLASS}"
        echo "Template start time: ${TEMPLATE_START_TIME_READABLE}"
        echo "---"
        
        python -m reasoning_trust.perspectives.toxicity.generate_and_evaluate_toxicity \
            --dataset-name "AI-Secure/DecodingTrust" \
            --dataset-config "toxicity" \
            --dataset-split "realtoxicityprompts.toxic" \
            --max-samples "$MAX_SAMPLES" \
            --template-id "$TEMPLATE_ID" \
            --model-name "$MODEL_NAME" \
            --enable-thinking \
            --save-intermediate \
            --prompt-class "$PROMPT_CLASS" 
        
        exit_code=$?
        TEMPLATE_END_TIME=$(date +%s)
        TEMPLATE_END_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
        TEMPLATE_DURATION=$(calculate_duration $TEMPLATE_START_TIME $TEMPLATE_END_TIME)
        TEMPLATE_DURATION_FORMATTED=$(format_duration $TEMPLATE_DURATION)
        
        echo "---"
        echo "Template ${TEMPLATE_ID} for Prompt Class ${PROMPT_CLASS} completed"
        echo "Template end time: ${TEMPLATE_END_TIME_READABLE}"
        echo "Template duration: ${TEMPLATE_DURATION_FORMATTED} (${TEMPLATE_DURATION} seconds)"
        echo "Exit code: ${exit_code}"
        echo "---"
        
        if [ ${exit_code} -ne 0 ]; then
            echo "ERROR: Template ${TEMPLATE_ID} for Prompt Class ${PROMPT_CLASS} failed with exit code ${exit_code}"
        fi
    done
    CLASS_END_TIME=$(date +%s)
    CLASS_END_TIME_READABLE=$(date '+%Y-%m-%d %H:%M:%S')
    CLASS_DURATION=$(calculate_duration $CLASS_START_TIME $CLASS_END_TIME)
    CLASS_DURATION_FORMATTED=$(format_duration $CLASS_DURATION)
    
    echo ""
    echo "=========================================="
    echo "Completed all templates for Prompt Class: ${PROMPT_CLASS}"
    echo "Class end time: ${CLASS_END_TIME_READABLE}"
    echo "Total time for class: ${CLASS_DURATION_FORMATTED} (${CLASS_DURATION} seconds)"
    echo "=========================================="

done

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
