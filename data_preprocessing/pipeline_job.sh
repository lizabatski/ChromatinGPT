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

set -e  # Stop on first error

# Log start
echo "Job started at: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"

# Activate virtual environment
source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

# Sanity check
echo "Active Python: $(which python)"
echo "Active Virtualenv: $VIRTUAL_ENV"

# Move to project directory
cd ~/ChromatinGPT/DeepHistone


EPIGENOME=$1
TEST_MODE=$2  

# Build Python command
PYTHON_CMD="python -u data_preprocessing/pipeline.py --epigenome $EPIGENOME"
OUTPUT_FILE="data/${EPIGENOME}_deephistone.npz"

if [ "$TEST_MODE" == "test" ]; then
    PYTHON_CMD="$PYTHON_CMD --test_mode"
    OUTPUT_FILE="data/${EPIGENOME}_deephistone_chr22.npz"
fi

# run preprocessing
echo "Running: $PYTHON_CMD"
$PYTHON_CMD

if [ -f "$OUTPUT_FILE" ]; then
    echo "Preprocessing finished successfully. Output saved to $OUTPUT_FILE"
else
    echo "Error: Output file not found at $OUTPUT_FILE"
    exit 1
fi

# Log end
echo "Job ended at: $(date)"
