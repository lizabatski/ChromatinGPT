import numpy as np
import torch
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd

# define histone modifications - same as in DeepHistone
histones = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']

def loadRegions(regions_indexs: List, dna_dict: Dict, dns_dict: Dict, label_dict: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load DNA, DNase, and label data for given regions"""
    if dna_dict is not None:
        dna_regions = np.concatenate(
            [dna_dict[meta] for meta in regions_indexs], axis=0
        ).copy()
    else:
        dna_regions = []
    
    if dns_dict is not None:
        dns_regions = np.concatenate(
            [dns_dict[meta] for meta in regions_indexs], axis=0
        ).copy()
    else:
        dns_regions = []
    
    label_regions = np.concatenate(
        [label_dict[meta] for meta in regions_indexs], axis=0
    ).astype(int).copy()
    
    return dna_regions, dns_regions, label_regions


def model_train(regions: List, model, batchsize: int, dna_dict: Dict, dns_dict: Dict, label_dict: Dict) -> float:
    train_loss = []
    regions_len = len(regions)
    
    for i in range(0, regions_len, batchsize):
        regions_batch = [regions[i+j] for j in range(batchsize) if (i+j) < regions_len]
        seq_batch, dns_batch, lab_batch = loadRegions(regions_batch, dna_dict, dns_dict, label_dict)
        
        loss = model.train_on_batch(seq_batch, dns_batch, lab_batch)
        train_loss.append(loss)
    
    return np.mean(train_loss)

def model_eval(regions: List, model, batchsize: int, dna_dict: Dict, dns_dict: Dict, label_dict: Dict) -> Tuple[float, np.ndarray, np.ndarray]:
    loss = []
    pred = []
    lab = []
    regions_len = len(regions)
    
    for i in range(0, regions_len, batchsize):
        regions_batch = [regions[i+j] for j in range(batchsize) if (i+j) < regions_len]
        seq_batch, dns_batch, lab_batch = loadRegions(regions_batch, dna_dict, dns_dict, label_dict)
        
        _loss, _pred = model.eval_on_batch(seq_batch, dns_batch, lab_batch)
        loss.append(_loss)
        lab.extend(lab_batch)
        pred.extend(_pred)
    
    return np.mean(loss), np.array(lab), np.array(pred)

def model_predict(regions: List, model, batchsize: int, dna_dict: Dict, dns_dict: Dict, label_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
    lab = []
    pred = []
    regions_len = len(regions)
    
    for i in range(0, regions_len, batchsize):
        regions_batch = [regions[i+j] for j in range(batchsize) if (i+j) < regions_len]
        seq_batch, dns_batch, lab_batch = loadRegions(regions_batch, dna_dict, dns_dict, label_dict)
        
        _pred = model.test_on_batch(seq_batch, dns_batch)
        lab.extend(lab_batch)
        pred.extend(_pred)
    
    return np.array(lab), np.array(pred)

def ROC(label: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(np.array(label).reshape(-1))) == 1:
        print("Warning: all labels are the same!")
        return 0.0
    else:
        label = np.array(label).reshape(-1)
        pred = np.array(pred).reshape(-1)
        return roc_auc_score(label, pred)

def auPR(label: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(np.array(label).reshape(-1))) == 1:
        print("Warning: all labels are the same!")
        return 0.0
    else:
        label = np.array(label).reshape(-1)
        pred = np.array(pred).reshape(-1)
        return average_precision_score(label, pred)

def metrics(lab: np.ndarray, pred: np.ndarray, Type: str = 'test', loss: Optional[float] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    if Type == 'Valid':
        training_color = '\033[0;34m'
    elif Type == 'Test':
        training_color = '\033[0;35m'
    else:
        training_color = '\033[0;36m'
    
    auPRC_dict = {}
    auROC_dict = {}
    
    for i in range(len(histones)):
        auPRC_dict[histones[i]] = auPR(lab[:, i], pred[:, i])
        auROC_dict[histones[i]] = ROC(lab[:, i], pred[:, i])
    
    print_str = training_color + '\t%s\t%s\tauROC: %.4f, auPRC: %.4f\033[0m'
    print('-' * 25 + Type + '-' * 25)
    
    loss_str = f', Loss: {loss:.4f}' if loss is not None else ''
    mean_auROC = np.mean(list(auROC_dict.values()))
    mean_auPRC = np.mean(list(auPRC_dict.values()))
    
    print(f'\033[0;36m{Type}\tTotalMean\tauROC: {mean_auROC:.4f}, auPRC: {mean_auPRC:.4f}{loss_str}\033[0m')
    
    for histone in histones:
        print(print_str % (Type, histone.ljust(10), auROC_dict[histone], auPRC_dict[histone]))
    
    return auPRC_dict, auROC_dict

def visualize_attention_weights(attention_weights: List[torch.Tensor], 
                              layer_idx: int = -1, 
                              head_idx: int = 0,
                              sequence_region: Tuple[int, int] = (0, 125),
                              save_path: Optional[str] = None) -> None:
    if not attention_weights:
        print("No attention weights available")
        return
    
    # get attention weights for specified layer
    attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()  # first batch element
    
    # extract specified region
    start, end = sequence_region
    attn_region = attn[start:end, start:end]
    
    # create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(attn_region, cmap='viridis', cbar=True, square=True)
    plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_attention_around_tad_boundaries(attention_weights: List[torch.Tensor],
                                            tad_boundaries: List[int],
                                            layer_idx: int = -1,
                                            head_idx: int = 0,
                                            window_size: int = 20,
                                            save_path: Optional[str] = None) -> None:
   
    if not attention_weights or not tad_boundaries:
        print("No attention weights or TAD boundaries available")
        return
    
    attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
    seq_len = attn.shape[0]
    
    fig, axes = plt.subplots(1, len(tad_boundaries), figsize=(5 * len(tad_boundaries), 5))
    if len(tad_boundaries) == 1:
        axes = [axes]
    
    for i, boundary in enumerate(tad_boundaries):
        # define window around boundary
        start = max(0, boundary - window_size)
        end = min(seq_len, boundary + window_size)
        
        # extract attention matrix for this region
        attn_region = attn[start:end, start:end]
        
        # create heatmap
        sns.heatmap(attn_region, cmap='viridis', ax=axes[i], cbar=True, square=True)
        axes[i].set_title(f'TAD Boundary at {boundary}')
        axes[i].axvline(x=window_size, color='red', linestyle='--', alpha=0.7, label='Boundary')
        axes[i].axhline(y=window_size, color='red', linestyle='--', alpha=0.7)
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
        axes[i].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_attention_across_tad_boundary(attention_weights: List[torch.Tensor],
                                        tad_boundary: int,
                                        layer_idx: int = -1,
                                        head_idx: int = 0,
                                        window_size: int = 20) -> Dict[str, float]:

    if not attention_weights:
        return {}
    
    attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
    seq_len = attn.shape[0]
    
    # define regions around boundary
    start = max(0, tad_boundary - window_size)
    end = min(seq_len, tad_boundary + window_size)
    
    if end - start < 2 * window_size:
        return {}
    
    # extract attention matrix for this region
    attn_region = attn[start:end, start:end]
    
    # calculate metrics
    boundary_pos = window_size  # Position of boundary in the extracted region
    
    # attention within left TAD
    left_region = attn_region[:boundary_pos, :boundary_pos]
    within_left_attention = np.mean(left_region)
    
    # attention within right TAD
    right_region = attn_region[boundary_pos:, boundary_pos:]
    within_right_attention = np.mean(right_region)
    
    # attention across boundary (left to right)
    across_lr_attention = np.mean(attn_region[:boundary_pos, boundary_pos:])
    
    # attention across boundary (right to left)
    across_rl_attention = np.mean(attn_region[boundary_pos:, :boundary_pos])
    
    # attention at boundary
    boundary_attention = np.mean(attn_region[boundary_pos-1:boundary_pos+1, boundary_pos-1:boundary_pos+1])
    
    return {
        'within_left_tad': within_left_attention,
        'within_right_tad': within_right_attention,
        'across_boundary_lr': across_lr_attention,
        'across_boundary_rl': across_rl_attention,
        'at_boundary': boundary_attention,
        'boundary_enhancement': boundary_attention / np.mean([within_left_attention, within_right_attention]),
        'boundary_isolation': (within_left_attention + within_right_attention) / (across_lr_attention + across_rl_attention)
    }

def visualize_contribution_scores(contribution_scores: np.ndarray,
                                sequence_positions: np.ndarray,
                                histone_name: str,
                                save_path: Optional[str] = None) -> None:

    plt.figure(figsize=(15, 6))
    
    # plot contribution scores
    plt.plot(sequence_positions, contribution_scores, linewidth=1.5, alpha=0.8)
    plt.fill_between(sequence_positions, 0, contribution_scores, alpha=0.3)
    
    plt.title(f'Contribution Scores for {histone_name}')
    plt.xlabel('Sequence Position (bp)')
    plt.ylabel('Contribution Score')
    plt.grid(True, alpha=0.3)
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_hnrnpa1_locus_analysis(model, 
                                   seq_data: np.ndarray, 
                                   dnase_data: np.ndarray,
                                   histone_idx: int = 0,
                                   save_dir: Optional[str] = None) -> Dict:

    results = {}
    
    
    predictions = model.test_on_batch(seq_data, dnase_data)
    results['predictions'] = predictions
    
    
    attention_weights = model.get_attention_weights(seq_data, dnase_data)
    results['attention_weights'] = attention_weights
    
   
    contribution_scores = model.get_contribution_scores(seq_data, dnase_data, histone_idx)
    results['contribution_scores'] = contribution_scores
    
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    
    axes[0].plot(predictions[0, :], linewidth=2)
    axes[0].set_title('Histone Modification Predictions')
    axes[0].set_ylabel('Prediction Score')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Attention weights (average across heads and layers)
    avg_attention = np.mean([attn[0].cpu().numpy() for attn in attention_weights], axis=0)
    avg_attention_per_pos = np.mean(avg_attention, axis=0)  # Average over query positions
    axes[1].plot(avg_attention_per_pos, linewidth=2, color='red')
    axes[1].set_title('Average Attention Weights')
    axes[1].set_ylabel('Attention Score')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Contribution scores
    axes[2].plot(contribution_scores[0], linewidth=2, color='green')
    axes[2].fill_between(range(len(contribution_scores[0])), 0, contribution_scores[0], alpha=0.3, color='green')
    axes[2].set_title('Contribution Scores')
    axes[2].set_xlabel('Sequence Position')
    axes[2].set_ylabel('Contribution Score')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f'{save_dir}/hnrnpa1_locus_analysis.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return results

def calculate_enformer_metrics(lab: np.ndarray, pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics similar to those reported in Enformer paper
    
    Args:
        lab: True labels (n_samples, n_histones)
        pred: Predictions (n_samples, n_histones)
    
    Returns:
        Dictionary with comprehensive metrics
    """
    metrics_dict = {}
    
    for i, histone in enumerate(histones):
        histone_metrics = {}
        
        # Basic metrics
        histone_metrics['auPRC'] = auPR(lab[:, i], pred[:, i])
        histone_metrics['auROC'] = ROC(lab[:, i], pred[:, i])
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(lab[:, i], pred[:, i])
        histone_metrics['precision_curve'] = precision
        histone_metrics['recall_curve'] = recall
        
        # Additional metrics
        histone_metrics['mean_prediction'] = np.mean(pred[:, i])
        histone_metrics['std_prediction'] = np.std(pred[:, i])
        histone_metrics['label_frequency'] = np.mean(lab[:, i])
        
        metrics_dict[histone] = histone_metrics
    
    # Overall metrics
    metrics_dict['overall'] = {
        'mean_auPRC': np.mean([metrics_dict[h]['auPRC'] for h in histones]),
        'mean_auROC': np.mean([metrics_dict[h]['auROC'] for h in histones]),
        'std_auPRC': np.std([metrics_dict[h]['auPRC'] for h in histones]),
        'std_auROC': np.std([metrics_dict[h]['auROC'] for h in histones])
    }
    
    return metrics_dict

def plot_performance_comparison(results_dict: Dict[str, Dict], 
                              save_path: Optional[str] = None) -> None:
    """
    Plot performance comparison across different models or conditions
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values
        save_path: Path to save the plot
    """
    models = list(results_dict.keys())
    auPRC_scores = [results_dict[model]['overall']['mean_auPRC'] for model in models]
    auROC_scores = [results_dict[model]['overall']['mean_auROC'] for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # auPRC comparison
    ax1.bar(models, auPRC_scores, color='skyblue', alpha=0.7)
    ax1.set_title('Mean auPRC Comparison')
    ax1.set_ylabel('auPRC Score')
    ax1.set_ylim(0, 1)
    
    # auROC comparison
    ax2.bar(models, auROC_scores, color='lightcoral', alpha=0.7)
    ax2.set_title('Mean auROC Comparison')
    ax2.set_ylabel('auROC Score')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_tad_boundary_effects(attention_weights: List[torch.Tensor],
                               tad_boundaries: List[int],
                               layer_indices: List[int] = None) -> pd.DataFrame:
    """
    Analyze how attention patterns change around TAD boundaries across layers
    
    Args:
        attention_weights: List of attention weight tensors
        tad_boundaries: List of TAD boundary positions
        layer_indices: Which layers to analyze (default: all)
    
    Returns:
        DataFrame with boundary analysis results
    """
    if layer_indices is None:
        layer_indices = list(range(len(attention_weights)))
    
    results = []
    
    for layer_idx in layer_indices:
        for boundary in tad_boundaries:
            # Analyze each head
            n_heads = attention_weights[layer_idx].shape[1]
            
            for head_idx in range(n_heads):
                boundary_metrics = analyze_attention_across_tad_boundary(
                    attention_weights, boundary, layer_idx, head_idx
                )
                
                if boundary_metrics:  # Only add if analysis was successful
                    boundary_metrics.update({
                        'layer': layer_idx,
                        'head': head_idx,
                        'boundary_position': boundary
                    })
                    results.append(boundary_metrics)
    
    return pd.DataFrame(results)

def plot_tad_boundary_analysis(boundary_df: pd.DataFrame,
                             save_path: Optional[str] = None) -> None:
    """
    Plot TAD boundary analysis results
    
    Args:
        boundary_df: DataFrame from analyze_tad_boundary_effects
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Boundary enhancement across layers
    axes[0, 0].boxplot([boundary_df[boundary_df['layer'] == i]['boundary_enhancement'].values 
                       for i in sorted(boundary_df['layer'].unique())])
    axes[0, 0].set_title('Boundary Enhancement by Layer')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Enhancement Score')
    
    # Boundary isolation across layers
    axes[0, 1].boxplot([boundary_df[boundary_df['layer'] == i]['boundary_isolation'].values 
                       for i in sorted(boundary_df['layer'].unique())])
    axes[0, 1].set_title('Boundary Isolation by Layer')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Isolation Score')
    
    # Within vs across TAD attention
    within_tad = (boundary_df['within_left_tad'] + boundary_df['within_right_tad']) / 2
    across_tad = (boundary_df['across_boundary_lr'] + boundary_df['across_boundary_rl']) / 2
    
    axes[1, 0].scatter(within_tad, across_tad, alpha=0.6)
    axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[1, 0].set_title('Within vs Across TAD Attention')
    axes[1, 0].set_xlabel('Within TAD Attention')
    axes[1, 0].set_ylabel('Across TAD Attention')
    
    # Boundary attention distribution
    axes[1, 1].hist(boundary_df['at_boundary'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Boundary Attention Distribution')
    axes[1, 1].set_xlabel('Attention Score')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_enformer_style_report(model, 
                                test_data: Dict,
                                tad_boundaries: List[int] = None,
                                save_dir: str = 'enformer_analysis') -> Dict:
   
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    
    # 1. Performance metrics
    print("Calculating performance metrics...")
    test_lab, test_pred = model_predict(
        test_data['regions'], model, test_data['batch_size'],
        test_data['dna_dict'], test_data['dns_dict'], test_data['lab_dict']
    )
    
    results['performance'] = calculate_enformer_metrics(test_lab, test_pred)
    
    # 2. Attention analysis
    print("Analyzing attention patterns...")
    sample_regions = test_data['regions'][:10]  # Analyze first 10 regions
    
    all_attention_weights = []
    for region in sample_regions:
        seq_batch, dns_batch, _ = loadRegions([region], 
                                            test_data['dna_dict'], 
                                            test_data['dns_dict'], 
                                            test_data['lab_dict'])
        attention_weights = model.get_attention_weights(seq_batch, dns_batch)
        all_attention_weights.append(attention_weights)
    
    results['attention_analysis'] = all_attention_weights
    
    # 3. TAD boundary analysis (if boundaries provided)
    if tad_boundaries:
        print("Analyzing TAD boundaries...")
        boundary_df = analyze_tad_boundary_effects(all_attention_weights[0], tad_boundaries)
        results['tad_analysis'] = boundary_df
        
        # Plot TAD analysis
        plot_tad_boundary_analysis(boundary_df, f'{save_dir}/tad_boundary_analysis.png')
    
    # 4. Contribution score analysis
    print("Analyzing contribution scores...")
    contribution_results = {}
    for i, histone in enumerate(histones):
        seq_batch, dns_batch, _ = loadRegions([sample_regions[0]], 
                                            test_data['dna_dict'], 
                                            test_data['dns_dict'], 
                                            test_data['lab_dict'])
        contrib_scores = model.get_contribution_scores(seq_batch, dns_batch, i)
        contribution_results[histone] = contrib_scores
        
        # Plot contribution scores
        visualize_contribution_scores(contrib_scores[0], 
                                    np.arange(len(contrib_scores[0])), 
                                    histone,
                                    f'{save_dir}/contribution_scores_{histone}.png')
    
    results['contribution_analysis'] = contribution_results
    
    # 5. HNRNPA1 locus analysis (using first sample as example)
    print("Performing HNRNPA1-style locus analysis...")
    seq_batch, dns_batch, _ = loadRegions([sample_regions[0]], 
                                        test_data['dna_dict'], 
                                        test_data['dns_dict'], 
                                        test_data['lab_dict'])
    
    hnrnpa1_results = visualize_hnrnpa1_locus_analysis(model, seq_batch, dns_batch, 
                                                      histone_idx=0, save_dir=save_dir)
    results['hnrnpa1_analysis'] = hnrnpa1_results
    
    # 6. save summary report
    print("Generating summary report...")
    with open(f'{save_dir}/analysis_summary.txt', 'w') as f:
        f.write("DeepHistone-Enformer Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Performance Metrics:\n")
        for histone in histones:
            metrics = results['performance'][histone]
            f.write(f"{histone}: auPRC={metrics['auPRC']:.4f}, auROC={metrics['auROC']:.4f}\n")
        
        f.write(f"\nOverall: auPRC={results['performance']['overall']['mean_auPRC']:.4f} ± {results['performance']['overall']['std_auPRC']:.4f}\n")
        f.write(f"         auROC={results['performance']['overall']['mean_auROC']:.4f} ± {results['performance']['overall']['std_auROC']:.4f}\n")
        
        if tad_boundaries:
            f.write(f"\nTAD Boundary Analysis:\n")
            f.write(f"Mean boundary enhancement: {boundary_df['boundary_enhancement'].mean():.4f}\n")
            f.write(f"Mean boundary isolation: {boundary_df['boundary_isolation'].mean():.4f}\n")
    
    print(f"Analysis complete! Results saved to {save_dir}")
    return results