#!/bin/bash
#SBATCH --job-name=dual_pathway_c1536_l11
#SBATCH --account=def-majewski
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dual_pathway_c1536_l11_%j.out
#SBATCH --error=logs/dual_pathway_c1536_l11_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

# === Activate Conda Environment ===
source ~/ChromatinGPT/DeepHistone/chromatingpt/bin/activate

# === Change to Project Directory ===
cd ~/ChromatinGPT/DeepHistone

# === Model Hyperparameters ===
C=1536                  # Transformer channels
L=11                    # Transformer layers
H=8                     # Attention heads
CONV=5                  # Convolutional blocks
DROPOUT=0.2             # Dropout rate
BS=16                   # Batch size
LR=0.001                # Learning rate
FUSION="concat"         # Fusion type
POOL="attention"        # Pooling type
MAX_EPOCHS=50
PATIENCE=5
SEED=42

# === Dataset ===
DATA_FILE="data/E005_deephistone_10048bp_chr22.npz"
RESULT_DIR="results/E005_dual_c${C}_l${L}_h${H}_${FUSION}_chr22"

echo "=========================================="
echo "Dual-Pathway Training Configuration"
echo "------------------------------------------"
echo "  Channels:           $C"
echo "  Transformer Layers: $L"
echo "  Attention Heads:    $H"
echo "  Conv Blocks:        $CONV"
echo "  Dropout:            $DROPOUT"
echo "  Fusion Type:        $FUSION"
echo "  Pooling Type:       $POOL"
echo "  Batch Size:         $BS"
echo "  Learning Rate:      $LR"
echo "  Data File:          $DATA_FILE"
echo "  Output Directory:   $RESULT_DIR"
echo "=========================================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === Run Training ===
python experiments/enformer/train_dhica.py \
  --data_file "$DATA_FILE" \
  --channels "$C" \
  --num_transformer_layers "$L" \
  --num_heads "$H" \
  --dropout "$DROPOUT" \
  --batch_size "$BS" \
  --learning_rate "$LR" \
  --max_epochs "$MAX_EPOCHS" \
  --early_stopping_patience "$PATIENCE" \
  --pooling_type "$POOL" \
  --num_conv_blocks "$CONV" \
  --fusion_type "$FUSION" \
  --seed "$SEED" 
 

echo "Dual-pathway training completed!"
echo "Results saved to: $RESULT_DIR"
