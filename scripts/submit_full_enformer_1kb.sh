#!/bin/bash
#SBATCH --job-name=enformer_full          
#SBATCH --account=def-majewski               
#SBATCH --time=48:00:00                      
#SBATCH --cpus-per-task=16                  
#SBATCH --mem=192G                           
#SBATCH --gres=gpu:1                  
#SBATCH --output=logs/enformer_full_%j.out   
#SBATCH --error=logs/enformer_full_%j.err    
#SBATCH --mail-type=END,FAIL                 
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca


source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

cd ~/ChromatinGPT/DeepHistone

echo "Running Enformer best config: c256_l2_h8 on E005 full dataset"

# Run Enformer training script
python experiments/enformer/train_enformer.py \
  --data_file data/E005_deephistone.npz \
  --channels 256 \
  --num_transformer_layers 2 \
  --num_heads 8 \
  --dropout 0.4 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --max_epochs 100 \
  --early_stopping_patience 5 \
  --pooling_type attention \
  --num_conv_blocks 2 \
  --seed 42
