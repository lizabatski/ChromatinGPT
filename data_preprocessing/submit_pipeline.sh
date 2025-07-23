#!/bin/bash
#SBATCH --job-name=deephistone         
#SBATCH --account=def-majewski         
#SBATCH --time=02:00:00                
#SBATCH --cpus-per-task=12            
#SBATCH --mem=128G                      
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

set -e  


echo "Job started at: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"

# activate virtual environment
source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

# sanity check
echo "Active Python: $(which python)"
echo "Active Virtualenv: $VIRTUAL_ENV"

# move to project directory
cd ~/ChromatinGPT/DeepHistone

#arguments
EPIGENOME=$1           # e.g., E005
TEST_MODE=$2           # e.g., "test" or leave empty
FINAL_WINDOW_SIZE=$3   # e.g., 10000 (for 10 kb) or leave empty for default 1000


if [ -z "$FINAL_WINDOW_SIZE" ]; then
    FINAL_WINDOW_SIZE=1000
fi


PYTHON_CMD="python -u data_preprocessing/pipeline.py --epigenome $EPIGENOME --final_window_size $FINAL_WINDOW_SIZE"
OUTPUT_FILE="data/${EPIGENOME}_deephistone_${FINAL_WINDOW_SIZE}bp.npz"


if [ "$TEST_MODE" == "test" ]; then
    PYTHON_CMD="$PYTHON_CMD --test_mode"
    OUTPUT_FILE="data/${EPIGENOME}_deephistone_${FINAL_WINDOW_SIZE}bp_chr22.npz"
fi


echo "Running: $PYTHON_CMD"


$PYTHON_CMD


if [ -f "$OUTPUT_FILE" ]; then
    echo "Preprocessing finished successfully. Output saved to $OUTPUT_FILE"
else
    echo "Error: Output file not found at $OUTPUT_FILE"
    exit 1
fi


echo "Job ended at: $(date)"
