import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Histone markers
histones = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']

def plot_training_curves(results_dir, vis_dir):
    """Plot per-fold training loss and metrics"""
    for fold in range(1, 6):
        history_file = os.path.join(results_dir, f"fold_{fold}/training_history.npz")
        if not os.path.exists(history_file):
            print(f"Missing: {history_file}")
            continue

        data = np.load(history_file, allow_pickle=True)
        train_loss = data['train_loss']
        valid_loss = data['valid_loss']
        valid_auPRC = data['valid_auPRC']
        valid_auROC = data['valid_auROC']

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label="Train Loss")
        plt.plot(valid_loss, label="Validation Loss")
        plt.title(f"Fold {fold}: Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(valid_auPRC, label="Validation auPRC")
        plt.plot(valid_auROC, label="Validation auROC")
        plt.title(f"Fold {fold}: Validation Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()

        plt.tight_layout()
        save_path = os.path.join(vis_dir, f"fold_{fold}_training_curves.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved training curves for Fold {fold}")

def aggregate_results(results_dir):
    """Aggregate predictions, labels, and metrics across all folds"""
    all_preds = []
    all_labels = []
    fold_auPRCs = []
    fold_auROCs = []

    for fold in range(1, 6):
        history_file = os.path.join(results_dir, f"fold_{fold}/training_history.npz")
        if not os.path.exists(history_file):
            print(f"Missing: {history_file}")
            continue
        data = np.load(history_file, allow_pickle=True)
        all_preds.append(data['test_predictions'])
        all_labels.append(data['test_labels'])
        fold_auPRCs.append(data['test_auPRC'].item())
        fold_auROCs.append(data['test_auROC'].item())

    if not all_preds:
        raise ValueError("No folds found for aggregation.")

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    return all_preds, all_labels, fold_auPRCs, fold_auROCs

def plot_aggregated_histograms_grid(all_preds, all_labels, vis_dir):
    """Plot aggregated prediction histograms for all histone markers"""
    n_markers = len(histones)
    cols = 4
    rows = (n_markers + cols - 1) // cols

    plt.figure(figsize=(cols * 4, rows * 3))
    for i, histone in enumerate(histones):
        plt.subplot(rows, cols, i + 1)
        scores = all_preds[:, i]
        labels = all_labels[:, i]
        plt.hist(scores[labels == 1], bins=50, alpha=0.6, label="Positives", color="blue")
        plt.hist(scores[labels == 0], bins=50, alpha=0.6, label="Negatives", color="red")
        plt.title(histone, fontsize=10)
        plt.xlabel("Score", fontsize=8)
        plt.ylabel("Count", fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        if i == 0:
            plt.legend(fontsize=8)

    plt.suptitle("Aggregated Prediction Score Distributions", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(vis_dir, "aggregated_prediction_histograms.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved aggregated prediction histograms grid: {save_path}")

def plot_class_balance(all_labels, vis_dir):
    """Plot class balance (positive vs negative) for each marker"""
    pos_counts = np.sum(all_labels, axis=0)
    neg_counts = all_labels.shape[0] - pos_counts
    x = np.arange(len(histones))
    width = 0.4

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, pos_counts, width, label="Positives", color="blue")
    plt.bar(x + width/2, neg_counts, width, label="Negatives", color="red")
    plt.xticks(x, histones, rotation=45)
    plt.title("Class Balance per Histone Marker")
    plt.ylabel("Number of Samples")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(vis_dir, "class_balance.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved class balance plot: {save_path}")

def plot_roc_prc_grid(all_preds, all_labels, vis_dir):
    """Plot ROC and PRC curves for all markers"""
    n_markers = len(histones)
    cols = 4
    rows = (n_markers + cols - 1) // cols
    mean_auROC, mean_auPRC = [], []

    # ROC Grid
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axs = axs.flatten()
    for i, histone in enumerate(histones):
        y_true, y_score = all_labels[:, i], all_preds[:, i]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        axs[i].plot(fpr, tpr, label=f"auROC={roc_auc:.3f}", color="blue")
        axs[i].plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=0.8)
        axs[i].set_title(histone, fontsize=10)
        axs[i].set_xlabel("FPR", fontsize=8)
        axs[i].set_ylabel("TPR", fontsize=8)
        axs[i].legend(fontsize=8)
        mean_auROC.append(roc_auc)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("ROC Curves", fontsize=14)
    save_path = os.path.join(vis_dir, "aggregated_roc_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves grid: {save_path}")

    # PRC Grid
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axs = axs.flatten()
    for i, histone in enumerate(histones):
        y_true, y_score = all_labels[:, i], all_preds[:, i]
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        prc_auc = auc(recall, precision)
        axs[i].plot(recall, precision, label=f"auPRC={prc_auc:.3f}", color="green")
        axs[i].set_title(histone, fontsize=10)
        axs[i].set_xlabel("Recall", fontsize=8)
        axs[i].set_ylabel("Precision", fontsize=8)
        axs[i].legend(fontsize=8)
        mean_auPRC.append(prc_auc)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Precision-Recall Curves", fontsize=14)
    save_path = os.path.join(vis_dir, "aggregated_prc_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved PRC curves grid: {save_path}")

    return mean_auROC, mean_auPRC

def save_summary(fold_auPRCs, fold_auROCs, mean_auROC, mean_auPRC, vis_dir):
    """Save summary text with per-marker mean ± std and overall means"""
    summary_lines = [
        "===== DeepHistone 5-Fold Cross-Validation Summary =====",
        "",
        f"{'Histone Marker':<15} {'Mean auROC (±SD)':<25} {'Mean auPRC (±SD)':<25}",
        "-" * 70
    ]

    for i, histone in enumerate(histones):
        auroc_values = [fold[histone] for fold in fold_auROCs]
        auprc_values = [fold[histone] for fold in fold_auPRCs]
        summary_lines.append(
            f"{histone:<15} {np.mean(auroc_values):.4f} (±{np.std(auroc_values):.4f})     "
            f"{np.mean(auprc_values):.4f} (±{np.std(auprc_values):.4f})"
        )

    summary_lines += [
        "",
        f"Overall Mean auROC: {np.mean(mean_auROC):.4f} (±{np.std(mean_auROC):.4f})",
        f"Overall Mean auPRC: {np.mean(mean_auPRC):.4f} (±{np.std(mean_auPRC):.4f})"
    ]

    summary_path = os.path.join(vis_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved summary to {summary_path}")

def run_all(results_dir):
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    plot_training_curves(results_dir, vis_dir)
    all_preds, all_labels, fold_auPRCs, fold_auROCs = aggregate_results(results_dir)
    plot_aggregated_histograms_grid(all_preds, all_labels, vis_dir)
    plot_class_balance(all_labels, vis_dir)
    mean_auROC, mean_auPRC = plot_roc_prc_grid(all_preds, all_labels, vis_dir)
    save_summary(fold_auPRCs, fold_auROCs, mean_auROC, mean_auPRC, vis_dir)
    print("All performance visualizations complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Path to results folder (e.g., results/E005_deephistone_chr22_<run_id>)")
    args = parser.parse_args()
    run_all(args.results_dir)
