#!/bin/bash
#SBATCH --job-name=deephistone_full          
#SBATCH --account=def-majewski               
#SBATCH --time=08:00:00                      
#SBATCH --cpus-per-task=12                   
#SBATCH --mem=128G                          
#SBATCH --gres=gpu:1                         
#SBATCH --output=logs/train_full_%j.out      
#SBATCH --error=logs/train_full_%j.err       
#SBATCH --mail-type=END,FAIL                 
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca


source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

# Run training script
python train_deephistone.py --data_file data/E005_large.npz --seed 42
