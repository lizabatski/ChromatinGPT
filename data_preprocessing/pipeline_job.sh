#!/bin/bash
#SBATCH --job-name=deephistone         
#SBATCH --account=def-majewski         
#SBATCH --time=02:00:00                
#SBATCH --cpus-per-task=12            
#SBATCH --mem=32G                      
#SBATCH --output=logs/preprocessE005_%j.out
#SBATCH --error=logs/preprocessE005_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

# Activate your virtual environment
source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

# sanity check
echo "Active Python: $(which python)"
echo "Active Virtualenv: $VIRTUAL_ENV"

# move to project directory
cd ~/ChromatinGPT/DeepHistone

# run preprocessing or training
python -u data_preprocessing/pipeline.py
