#!/bin/bash
#SBATCH --job-name=dual_pathway_c1024_l6_full
#SBATCH --account=def-majewski
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dual_pathway_c1024_l6_concat_%j.out
#SBATCH --error=logs/dual_pathway_c1024_l6_concat_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

cd ~/ChromatinGPT/DeepHistone

# Dual-pathway configuration: C=1024 L=6 H=8 Conv=5
C=1024
L=6
H=8
CONV=5

echo "Running Optimized Dual-Pathway Model Configuration:"
echo "  Channels: $C"
echo "  Transformer Layers: $L"
echo "  Attention Heads: $H"
echo "  Conv Blocks: $CONV"
echo "  Estimated Parameters: ~85M"
echo "  Architecture: Dual-pathway (DNA + DNase) with concat fusion"
echo "  Optimizations: Memory-efficient DataLoader, AdamW optimizer"

python experiments/enformer/v2/train_dhica_v2.py \
  --data_file data/E005_deephistone_2048bp.npz \
  --channels $C \
  --num_transformer_layers $L \
  --num_heads $H \
  --dropout 0.4 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --max_epochs 50 \
  --early_stopping_patience 5 \
  --pooling_type attention \
  --num_conv_blocks $CONV \
  --fusion_type concat \
  --seed 42

echo "Optimized dual-pathway training completed!"
echo "Results saved to: results/E005_deephistone_2048bp_dual_pathway_c${C}_l${L}_h${H}_concat_*"