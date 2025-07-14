#!/bin/bash
#SBATCH --job-name=deephistone         
#SBATCH --account=def-majewski         
#SBATCH --time=02:00:00                
#SBATCH --cpus-per-task=12            
#SBATCH --mem=128G                      
#SBATCH --output=logs/preprocessE005_%j.out
#SBATCH --error=logs/preprocessE005_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

# venv
source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate


echo "Active Python: $(which python)"
echo "Active Virtualenv: $VIRTUAL_ENV"


cd ~/ChromatinGPT/DeepHistone



python -u data_preprocessing/pipeline_10kb.py \
    --epigenome E005 \
    --test_mode \
    --output_dir data/transformer \
    --output_name E005_chr22_10kb_concat.npz



if [ -f "data/transformer/E005_chr22_10kb_concat.npz" ]; then
    echo "Preprocessing finished successfully. Output saved to data/transformer/E005_chr22_10kb_concat.npz"
else
    echo "Error: Output file not found in data/transformer/E005_chr22_10kb_concat.npz"
fi

echo "Job finished at $(date)"
