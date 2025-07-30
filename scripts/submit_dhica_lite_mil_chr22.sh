#!/bin/bash
#SBATCH --job-name=dual_pathway_lite_chr22
#SBATCH --account=def-majewski
#SBATCH --time=24:00:00  
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dual_pathway_lite_chr22_%j.out
#SBATCH --error=logs/dual_pathway_lite_chr22_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

C=256  
H=4
L=1
CONV=2

source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

cd ~/ChromatinGPT/DeepHistone


mkdir -p logs

echo "Running Optimized Dual-Pathway Model Configuration:"
echo "  Channels: $C"
echo "  Transformer Layers: $L" 
echo "  Attention Heads: $H"
echo "  Conv Blocks: $CONV"
echo "  Estimated Parameters: ~5-10M"  
echo "  Architecture: Dual-pathway (DNA + DNase) with concat fusion"
echo "  Optimizations: Memory-efficient DataLoader, mixed precision"


echo "Training started at: $(date)"
start_time=$(date +%s)

python experiments/enformer/v3/train_dhica_v3.py \
  --data_file data/E005_deephistone_2048bp_chr22.npz \
  --channels $C \
  --num_transformer_layers $L \
  --num_heads $H \
  --dropout 0.3 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --max_epochs 15 \
  --early_stopping_patience 3 \
  --pooling_type attention \
  --num_conv_blocks $CONV \
  --fusion_type mil \
  --seed 42

# Calculate training time
end_time=$(date +%s)
training_time=$((end_time - start_time))
echo "Training completed at: $(date)"
echo "Total training time: $((training_time / 3600)) hours $((training_time % 3600 / 60)) minutes"

echo "Optimized dual-pathway training completed!"
echo "Results saved to: results/E005_deephistone_2048bp_*_c${C}_l${L}_h${H}_concat_*"