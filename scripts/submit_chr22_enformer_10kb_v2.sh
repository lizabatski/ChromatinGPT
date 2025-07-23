#!/bin/bash
#SBATCH --job-name=enformer_big_chr22
#SBATCH --account=def-majewski
#SBATCH --time=10:00:00                
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                      
#SBATCH --gres=gpu:1
#SBATCH --output=logs/enformer_big_chr22_%j.out
#SBATCH --error=logs/enformer_big_chr22_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

cd ~/ChromatinGPT/DeepHistone

# Massive configuration
C=1536
L=10
H=12
DROPOUT=0.1
BATCH_SIZE=16                  
LR=1e-3                        

echo "Running configuration: Channels=$C, Layers=$L, Heads=$H, Dropout=$DROPOUT"

python experiments/enformer/train_enformer.py \
  --data_file data/E005_deephistone_10048bp_chr22.npz \
  --channels $C \
  --num_transformer_layers $L \
  --num_heads $H \
  --dropout $DROPOUT \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --max_epochs 50 \
  --early_stopping_patience 7 \
  --pooling_type attention \
  --num_conv_blocks 5 \
  --seed 42
