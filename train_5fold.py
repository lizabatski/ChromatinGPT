from model import DeepHistone
import copy
import numpy as np
from utils import metrics, model_train, model_eval, model_predict
import torch
import os
from datetime import datetime
import random
import argparse
import sys

parser = argparse.ArgumentParser(description="DeepHistone Training Script")
parser.add_argument('--data_file', type=str, required=True,
                    help='Path to the .npz data file (e.g., data/E005_deephistone_chr22.npz)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42)')
args = parser.parse_args()

print("=" * 60)
print("DEEPHISTONE TRAINING SCRIPT STARTED")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Data file: {args.data_file}")
print(f"Seed: {args.seed}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# check if data file exists
if not os.path.exists(args.data_file):
    print(f"ERROR: Data file not found: {args.data_file}")
    sys.exit(1)

file_size_gb = os.path.getsize(args.data_file) / (1024**3)
print(f"Data file size: {file_size_gb:.2f} GB")

# force stdout to flush immediately
sys.stdout.flush()

SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

batchsize = 20
data_file = args.data_file

dataset_name = os.path.splitext(os.path.basename(data_file))[0]
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results/{dataset_name}_{run_id}"
os.makedirs(output_dir, exist_ok=True)

print(f"\nOutput directory: {output_dir}")
print(f"Batch size: {batchsize}")
sys.stdout.flush()

print('\nBegin loading data...')
start_time = datetime.now()

try:
    with np.load(data_file, mmap_mode='r') as f:
        print(f"Keys shape: {f['keys'].shape}")
        print(f"DNA shape: {f['dna'].shape}")
        print(f"DNase shape: {f['dnase'].shape}")
        print(f"Label shape: {f['label'].shape}")
        
        indexs = f['keys']
        print(f"Creating dictionaries for {len(indexs)} samples...")
        print("WARNING: Loading entire dataset into memory. This may take a while for large datasets...")
        sys.stdout.flush()
        
        dna_dict = dict(zip(f['keys'], f['dna']))
        print("DNA dictionary created")
        sys.stdout.flush()
        
        dns_dict = dict(zip(f['keys'], f['dnase']))
        print("DNase dictionary created")
        sys.stdout.flush()
        
        lab_dict = dict(zip(f['keys'], f['label']))
        print("Label dictionary created")
        sys.stdout.flush()

except Exception as e:
    print(f"ERROR loading data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

load_time = datetime.now() - start_time
print(f"Data loading completed in {load_time}")
sys.stdout.flush()

np.random.shuffle(indexs)
idx_len = len(indexs)
fold_size = idx_len // 5
folds = [indexs[i*fold_size:(i+1)*fold_size] for i in range(5)]

print(f"\nDataset split into 5 folds of ~{fold_size} samples each")
print(f"Total samples: {idx_len}")
sys.stdout.flush()

# ===== 5-Fold Cross-Validation =====
for fold_idx in range(5):
    fold_start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx+1}/5 STARTED")
    print(f"{'='*60}")
    sys.stdout.flush()

    # per-fold output directory
    fold_output_dir = f"{output_dir}/fold_{fold_idx+1}"
    os.makedirs(fold_output_dir, exist_ok=True)

    # file paths for this fold
    model_save_file = f'{fold_output_dir}/model.txt'
    lab_save_file = f'{fold_output_dir}/label.txt'
    pred_save_file = f'{fold_output_dir}/pred.txt'
    best_model_file = f'/tmp/best_model_fold_{fold_idx+1}.pth'
    history_file = f'{fold_output_dir}/training_history.npz'

    # ===== Split train, validation, test =====
    test_index = folds[fold_idx]
    train_valid_folds = [folds[i] for i in range(5) if i != fold_idx]
    train_valid_index = np.concatenate(train_valid_folds)
    np.random.shuffle(train_valid_index)

    # 80% train, 20% validation
    train_size = int(len(train_valid_index) * 0.8)
    train_index = train_valid_index[:train_size]
    valid_index = train_valid_index[train_size:]

    print(f"Train samples: {len(train_index)}")
    print(f"Validation samples: {len(valid_index)}")
    print(f"Test samples: {len(test_index)}")
    sys.stdout.flush()

    # ===== Model =====
    use_gpu = torch.cuda.is_available()
    model = DeepHistone(use_gpu)
    device = "cuda" if use_gpu else "cpu"
    print(f"Using device: {device}")

    train_loss_history = []
    valid_loss_history = []
    valid_auPRC_history = []
    valid_auROC_history = []
    lr_history = []

    best_valid_auPRC = 0
    best_valid_loss = np.float64('inf')
    early_stop_time = 0

    print('\nBegin training model...')
    sys.stdout.flush()
    
    for epoch in range(50):
        epoch_start_time = datetime.now()
        np.random.shuffle(train_index)

        print(f"\nEpoch {epoch+1}/50:")
        print(f"  Training on {len(train_index)} samples...")
        sys.stdout.flush()

        # train
        train_loss = model_train(train_index, model, batchsize, dna_dict, dns_dict, lab_dict)

        print(f"  Training completed. Loss: {train_loss:.4f}")
        print(f"  Validating on {len(valid_index)} samples...")
        sys.stdout.flush()

        # validate
        valid_loss, valid_lab, valid_pred = model_eval(valid_index, model, batchsize, dna_dict, dns_dict, lab_dict)
        valid_auPRC, valid_auROC = metrics(valid_lab, valid_pred, 'Valid', valid_loss)

        mean_auPRC = np.mean(list(valid_auPRC.values()))
        mean_auROC = np.mean(list(valid_auROC.values()))

        epoch_time = datetime.now() - epoch_start_time
        print(f"  Epoch {epoch+1} completed in {epoch_time}")
        print(f"  Results: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}")
        print(f"  Metrics: valid_auPRC={mean_auPRC:.4f}, valid_auROC={mean_auROC:.4f}")

        # save best model based on auPRC
        if mean_auPRC > best_valid_auPRC:
            torch.save(model.forward_fn.state_dict(), best_model_file)
            best_valid_auPRC = mean_auPRC
            print(f"  New best model saved! auPRC: {best_valid_auPRC:.4f}")

        # save metrics
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        valid_auPRC_history.append(mean_auPRC)
        valid_auROC_history.append(mean_auROC)
        lr_history.append(model.optimizer.param_groups[0]['lr'])

        # early stopping logic
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            early_stop_time = 0
        else:
            model.updateLR(0.1)
            early_stop_time += 1
            print(f"  Early stopping counter: {early_stop_time}/5")
            if early_stop_time >= 5:
                print("  Early stopping triggered.")
                break
                
        sys.stdout.flush()

    print('\nBegin predicting on test set...')
    sys.stdout.flush()
    
    try:
        if os.path.exists(best_model_file):
            model.forward_fn.load_state_dict(torch.load(best_model_file))
            print("Best model loaded successfully")
        else:
            print("Warning: Best model file not found, using current model state")
    except Exception as e:
        print(f"Error loading best model: {e}")
        print("Using current model state for prediction")
    
    sys.stdout.flush()
    
    test_lab, test_pred = model_predict(test_index, model, batchsize, dna_dict, dns_dict, lab_dict)
    test_auPR, test_roc = metrics(test_lab, test_pred, 'Test')

    print('Begin saving results...')
    sys.stdout.flush()
    
    np.savetxt(lab_save_file, test_lab, fmt='%d', delimiter='\t')
    np.savetxt(pred_save_file, test_pred, fmt='%.4f', delimiter='\t')
    model.save_model(model_save_file)

    history = {
        'train_loss': train_loss_history,
        'valid_loss': valid_loss_history,
        'valid_auPRC': valid_auPRC_history,
        'valid_auROC': valid_auROC_history,
        'learning_rate': lr_history,
        'test_auPRC': test_auPR,
        'test_auROC': test_roc,
        'test_predictions': test_pred,
        'test_labels': test_lab,
        'per_histone_metrics': {
            'auPRC': test_auPR,
            'auROC': test_roc
        }
    }
    np.savez(history_file, **history)

    try:
        if os.path.exists(best_model_file):
            os.remove(best_model_file)
            print(f"Cleaned up temp model file for fold {fold_idx+1}")
    except Exception as e:
        print(f"Could not remove temp file: {e}")

    fold_time = datetime.now() - fold_start_time
    print(f"\nFold {fold_idx+1} completed in {fold_time}")
    print(f"Results saved to: {fold_output_dir}")
    sys.stdout.flush()

print(f"\n{'='*60}")
print("CALCULATING FINAL RESULTS")
print(f"{'='*60}")

all_auPRCs = []
all_auROCs = []

for fold_idx in range(5):
    try:
        history = np.load(f"{output_dir}/fold_{fold_idx+1}/training_history.npz", allow_pickle=True)
        test_auPRC_dict = history['test_auPRC'].item()
        mean_fold_auPRC = np.mean(list(test_auPRC_dict.values()))
        all_auPRCs.append(mean_fold_auPRC)

        test_auROC_dict = history['test_auROC'].item()
        mean_fold_auROC = np.mean(list(test_auROC_dict.values()))
        all_auROCs.append(mean_fold_auROC)
        
        print(f"Fold {fold_idx+1}: auPRC={mean_fold_auPRC:.4f}, auROC={mean_fold_auROC:.4f}")
    except Exception as e:
        print(f"Error loading results for fold {fold_idx+1}: {e}")

if all_auPRCs and all_auROCs:
    mean_auPRC = np.mean(all_auPRCs)
    mean_auROC = np.mean(all_auROCs)
    std_auPRC = np.std(all_auPRCs)  # FIXED: Added this line
    std_auROC = np.std(all_auROCs)  # FIXED: Added this line

    print(f"\n{'='*60}")
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean auPRC: {mean_auPRC:.4f} ± {std_auPRC:.4f}")
    print(f"Mean auROC: {mean_auROC:.4f} ± {std_auROC:.4f}")
    print(f"Results saved to: {output_dir}")
else:
    print("ERROR: Could not calculate final results")

print(f"\n{'='*60}")
print("SCRIPT COMPLETED SUCCESSFULLY")
print(f"{'='*60}")