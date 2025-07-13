#!/bin/bash
#SBATCH --job-name=deephistone_chr22          
#SBATCH --account=def-majewski                 
#SBATCH --time=02:00:00                        
#SBATCH --cpus-per-task=8                      
#SBATCH --mem=32G                              
#SBATCH --gres=gpu:1                           
#SBATCH --output=logs/train_chr22_%j.out      
#SBATCH --error=logs/train_chr22_%j.err        
#SBATCH --mail-type=END,FAIL                   
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca




source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

# Run training script
python ../train_deephistone.py --data_file data/E005_deephistone_chr22.npz --seed 42
