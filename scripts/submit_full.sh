#!/bin/bash
#SBATCH --job-name=deephistone_full          
#SBATCH --account=def-majewski               
#SBATCH --time=24:00:00                      
#SBATCH --cpus-per-task=12                   
#SBATCH --mem=128G                          
#SBATCH --gres=gpu:1                         
#SBATCH --output=logs/train_full_%j.out      
#SBATCH --error=logs/train_full_%j.err       
#SBATCH --mail-type=END,FAIL                 
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca


source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

cd ~/ChromatinGPT/DeepHistone

# Run training script
python train_5fold.py --data_file data/E122_deephistone.npz --seed 42
