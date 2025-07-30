import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# Define histone modifications
histones = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']


def loadRegions(regions_indexs: List, dna_dict: Dict, dns_dict: Dict, label_dict: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load DNA, DNase, and label data for given regions"""
    dna_regions = np.array([dna_dict[meta] for meta in regions_indexs])
    dns_regions = np.array([dns_dict[meta] for meta in regions_indexs])
    label_regions = np.array([label_dict[meta] for meta in regions_indexs]).astype(int)
    
    return dna_regions, dns_regions, label_regions

def model_train(model, train_loader, device):
    model.forward_fn.train()
    total_loss = 0.0

    scaler = getattr(model, 'scaler', None)
    if scaler is None:
        model.scaler = GradScaler()
        scaler = model.scaler

    for dna_batch, dnase_batch, label_batch in train_loader:
        if device.type == 'cuda':
            dna_batch = dna_batch.cuda(non_blocking=True)
            dnase_batch = dnase_batch.cuda(non_blocking=True)
            label_batch = label_batch.cuda(non_blocking=True)

        with autocast(device_type='cuda'):
            output = model.forward_fn(dna_batch, dnase_batch)

        loss = model.criterion(output.float(), label_batch.float())

        model.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(model.optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def model_eval(model, eval_loader, device):
    model.forward_fn.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for dna_batch, dnase_batch, label_batch in eval_loader:
            if device.type == 'cuda':
                dna_batch = dna_batch.cuda(non_blocking=True)
                dnase_batch = dnase_batch.cuda(non_blocking=True)
                label_batch = label_batch.cuda(non_blocking=True)

            with autocast(device_type='cuda'):
                output = model.forward_fn(dna_batch, dnase_batch)

            loss = model.criterion(output.float(), label_batch.float())

            total_loss += loss.item()
            all_predictions.append(output.cpu().numpy())
            all_labels.append(label_batch.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return total_loss / len(eval_loader), all_labels, all_predictions



def model_predict(model, test_loader, device):
    model.forward_fn.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for dna_batch, dnase_batch, label_batch in test_loader:
            if device.type == 'cuda':
                dna_batch = dna_batch.cuda(non_blocking=True)
                dnase_batch = dnase_batch.cuda(non_blocking=True)

            with autocast(device_type='cuda'):
                output = model.forward_fn(dna_batch, dnase_batch)

            all_predictions.append(output.cpu().numpy())
            all_labels.append(label_batch.cpu().numpy())

    return np.concatenate(all_labels, axis=0), np.concatenate(all_predictions, axis=0)



def calculate_metrics(labels: np.ndarray, predictions: np.ndarray, histone_idx: int) -> Tuple[float, float]:
    """Calculate auROC and auPRC for a single histone"""
    try:
        y_true = labels[:, histone_idx]
        y_pred = predictions[:, histone_idx]
        
        # ADD THESE DEBUG LINES:
        # (f"DEBUG: Histone {histone_idx} ({histones[histone_idx]})")
        # print(f"  y_true shape: {y_true.shape}")
        # print(f"  y_pred shape: {y_pred.shape}")
        # print(f"  y_true unique values: {np.unique(y_true)}")
        # print(f"  y_pred range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
        # print(f"  y_true sample: {y_true[:5]}")
        # print(f"  y_pred sample: {y_pred[:5]}")
        
        # Check if we have both classes
        if len(np.unique(y_true)) < 2:
            print(f"  WARNING: Only one class present for {histones[histone_idx]}")
            return 0.0, 0.0
        
        auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
        
        # print(f"  SUCCESS: auROC={auroc:.4f}, auPRC={auprc:.4f}")
        # print
        return auroc, auprc
    except Exception as e:
        print(f"  ERROR in calculate_metrics for histone {histone_idx}: {e}")
        return 0.0, 0.0

def metrics(lab: np.ndarray, pred: np.ndarray, phase: str = 'Test', loss: float = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Calculate metrics for all histones"""

    # print(f"DEBUG METRICS: lab shape = {lab.shape}")
    # print(f"DEBUG METRICS: pred shape = {pred.shape}")
    # print(f"DEBUG METRICS: len(histones) = {len(histones)}")
    # print(f"DEBUG METRICS: lab sample = {lab[0] if len(lab) > 0 else 'empty'}")
    # print(f"DEBUG METRICS: pred sample = {pred[0] if len(pred) > 0 else 'empty'}")

    auPRC_dict = {}
    auROC_dict = {}
    
    print(f'\n--- {phase} Results ---')
    if loss is not None:
        print(f'Loss: {loss:.4f}')
    
    for i, histone in enumerate(histones):
        if i < lab.shape[1] and i < pred.shape[1]:
            auroc, auprc = calculate_metrics(lab, pred, i)
            auROC_dict[histone] = auroc
            auPRC_dict[histone] = auprc
            print(f'{histone:10s}: auROC={auroc:.4f}, auPRC={auprc:.4f}')
        else:
            auROC_dict[histone] = 0.0
            auPRC_dict[histone] = 0.0
            print(f'{histone:10s}: auROC=0.0000, auPRC=0.0000 (missing data)')
    
    # Calculate means
    mean_auroc = np.mean(list(auROC_dict.values()))
    mean_auprc = np.mean(list(auPRC_dict.values()))
    
    print(f'{"Mean":10s}: auROC={mean_auroc:.4f}, auPRC={mean_auprc:.4f}')
    print('-' * 50)
    
    return auPRC_dict, auROC_dict

class LazyHistoneDataset(Dataset):
    def __init__(self, data_file: str, indices: np.ndarray):
        self.data_file = data_file
        self.indices = indices
        self.data_handle = None
        self.key_to_idx = None

    def _lazy_init(self):
        if self.data_handle is None:
            self.data_handle = np.load(self.data_file, mmap_mode='r')
            all_keys = self.data_handle['keys']
            self.key_to_idx = {key: i for i, key in enumerate(all_keys)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._lazy_init()
        key = self.indices[idx]
        data_idx = self.key_to_idx[key]

        dna = self.data_handle['dna'][data_idx].squeeze()
        dnase = self.data_handle['dnase'][data_idx].squeeze()
        label = self.data_handle['label'][data_idx].squeeze()

        return (
            torch.tensor(dna, dtype=torch.float32),
            torch.tensor(dnase, dtype=torch.float32).unsqueeze(0),
            torch.tensor(label, dtype=torch.float32)
        )
