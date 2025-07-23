import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, List, Tuple

# Define histone modifications
histones = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']

def loadRegions(regions_indexs: List, dna_dict: Dict, dns_dict: Dict, label_dict: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load DNA, DNase, and label data for given regions"""
    dna_regions = np.array([dna_dict[meta] for meta in regions_indexs])
    dns_regions = np.array([dns_dict[meta] for meta in regions_indexs])
    label_regions = np.array([label_dict[meta] for meta in regions_indexs]).astype(int)
    
    return dna_regions, dns_regions, label_regions

def model_train(regions: List, model, batchsize: int, dna_dict: Dict, dns_dict: Dict, label_dict: Dict) -> float:
    """Training function"""
    train_loss = []
    
    for i in range(0, len(regions), batchsize):
        batch_end = min(i + batchsize, len(regions))
        regions_batch = regions[i:batch_end]
        
        seq_batch, dns_batch, lab_batch = loadRegions(regions_batch, dna_dict, dns_dict, label_dict)
        loss = model.train_on_batch(seq_batch, dns_batch, lab_batch)
        train_loss.append(loss)
    
    return np.mean(train_loss)

def model_eval(regions: List, model, batchsize: int, dna_dict: Dict, dns_dict: Dict, label_dict: Dict) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluation function"""
    losses = []
    all_preds = []
    all_labels = []
    
    for i in range(0, len(regions), batchsize):
        batch_end = min(i + batchsize, len(regions))
        regions_batch = regions[i:batch_end]
        
        seq_batch, dns_batch, lab_batch = loadRegions(regions_batch, dna_dict, dns_dict, label_dict)
        loss, pred = model.eval_on_batch(seq_batch, dns_batch, lab_batch)
        
        losses.append(loss)
        all_preds.append(pred)
        all_labels.append(lab_batch)
    
    avg_loss = np.mean(losses)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_labels = all_labels.squeeze()
    
    return avg_loss, all_labels, all_preds

def model_predict(regions: List, model, batchsize: int, dna_dict: Dict, dns_dict: Dict, label_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Prediction function"""
    all_preds = []
    all_labels = []
    
    for i in range(0, len(regions), batchsize):
        batch_end = min(i + batchsize, len(regions))
        regions_batch = regions[i:batch_end]
        
        seq_batch, dns_batch, lab_batch = loadRegions(regions_batch, dna_dict, dns_dict, label_dict)
        pred = model.test_on_batch(seq_batch, dns_batch)
        
        all_preds.append(pred)
        all_labels.append(lab_batch)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    all_labels = all_labels.squeeze()

    return all_labels, all_preds

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