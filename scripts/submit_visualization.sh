#!/bin/bash
#SBATCH --job-name=visualize_all
#SBATCH --account=def-majewski
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/visualize_%A_%a.out
#SBATCH --error=logs/visualize_%A_%a.err
#SBATCH --array=0-2              
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

# Activate virtual environment
source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

cd ~/ChromatinGPT/DeepHistone

# paths to results directories
RESULTS_DIRS=(
    "results/E005/enformer/E005_deephistone_enformer_c256_l2_h8_full"
)


TARGET_DIR=${RESULTS_DIRS[$SLURM_ARRAY_TASK_ID]}

echo "Visualizing results for: $TARGET_DIR"

# run the visualization script
python performance_visualize.py --results_dir "$TARGET_DIR" --enformer

echo "Visualization complete for: $TARGET_DIR"
