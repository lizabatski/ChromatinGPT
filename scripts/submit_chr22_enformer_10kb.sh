#!/bin/bash
#SBATCH --job-name=enformer_fixed
#SBATCH --account=def-majewski
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/enformer_fixed_%j.out
#SBATCH --error=logs/enformer_fixed_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

cd ~/ChromatinGPT/DeepHistone

# Fixed configuration
C=1024
L=7
H=8

echo "Running configuration: Channels=$C, Layers=$L, Heads=$H"

python experiments/enformer/train_enformer.py \
  --data_file data/E005_deephistone_10016bp_chr22.npz \
  --channels $C \
  --num_transformer_layers $L \
  --num_heads $H \
  --dropout 0.3 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --max_epochs 50 \
  --early_stopping_patience 5 \
  --pooling_type attention \
  --num_conv_blocks 5 \
  --seed 42
