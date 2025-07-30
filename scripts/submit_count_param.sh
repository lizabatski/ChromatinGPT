#!/bin/bash
#SBATCH --job-name=deephistone-analysis
#SBATCH --account=def-majewski
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/analyze_models_%j.out
#SBATCH --error=logs/analyze_models_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

# ========================
# Setup
# ========================
echo "Job started at: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"

# Activate virtual environment
source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

# Change to project directory
cd ~/ChromatinGPT/DeepHistone

# ========================
# Parameters (editable)
# ========================
ENFORMER_CHANNELS=${1:-1536}
ENFORMER_LAYERS=${2:-10}
ENFORMER_HEADS=${3:-12}
ENFORMER_DROPOUT=${4:-0.1}
ENFORMER_CONV_BLOCKS=${5:-5}

DUAL_CHANNELS=${6:-1536}
DUAL_LAYERS=${7:-11}
DUAL_HEADS=${8:-8}
DUAL_DROPOUT=${9:-0.2}
DUAL_CONV_BLOCKS=${10:-7}
DUAL_FUSION=${11:-concat}

# ========================
# Run Python script
# ========================
echo "Running analysis with the following configuration:"
echo "Enformer: channels=$ENFORMER_CHANNELS, layers=$ENFORMER_LAYERS, heads=$ENFORMER_HEADS, dropout=$ENFORMER_DROPOUT, conv_blocks=$ENFORMER_CONV_BLOCKS"
echo "Dual-Pathway: channels=$DUAL_CHANNELS, layers=$DUAL_LAYERS, heads=$DUAL_HEADS, dropout=$DUAL_DROPOUT, conv_blocks=$DUAL_CONV_BLOCKS, fusion=$DUAL_FUSION"

python count_param.py \
  --enformer_channels $ENFORMER_CHANNELS \
  --enformer_layers $ENFORMER_LAYERS \
  --enformer_heads $ENFORMER_HEADS \
  --enformer_dropout $ENFORMER_DROPOUT \
  --enformer_conv_blocks $ENFORMER_CONV_BLOCKS \
  --dual_channels $DUAL_CHANNELS \
  --dual_layers $DUAL_LAYERS \
  --dual_heads $DUAL_HEADS \
  --dual_dropout $DUAL_DROPOUT \
  --dual_conv_blocks $DUAL_CONV_BLOCKS \
  --dual_fusion $DUAL_FUSION

echo "Job ended at: $(date)"
