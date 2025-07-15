#!/bin/bash
#SBATCH --job-name=enformer_grid
#SBATCH --account=def-majewski
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

cd ~/ChromatinGPT/DeepHistone

CHANNELS=(256 512 768)
LAYERS=(2 4 6)
HEADS=(4 8 12)

IDX=${SLURM_ARRAY_TASK_ID}
C=${CHANNELS[$((IDX / 3))]}
L=${LAYERS[$(((IDX % 9) / 3))]}
H=${HEADS[$((IDX % 3))]}

echo "Running configuration: Channels=$C, Layers=$L, Heads=$H"

python experiments/enformer/train_enformer.py \
  --data_file data/E005_deephistone_chr22.npz \
  --channels $C \
  --num_transformer_layers $L \
  --num_heads $H \
  --dropout 0.4 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --max_epochs 50 \
  --early_stopping_patience 5 \
  --pooling_type attention \
  --seed 42
