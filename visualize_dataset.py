import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import pandas as pd

def load_chipseq_peaks(chipseq_file):
    """Load ChIP-seq peaks from narrowPeak format"""
    peaks = []
    if not os.path.exists(chipseq_file):
        print(f"Warning: {chipseq_file} not found")
        return peaks
    
    with open(chipseq_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            cols = line.split('\t')
            if len(cols) < 10:  # narrowPeak has 10 columns
                continue
            
            try:
                chrom = cols[0]
                start = int(cols[1])
                end = int(cols[2])
                peak_name = cols[3] if cols[3] != '.' else f"{chrom}:{start}-{end}"
                score = float(cols[4]) if cols[4] != '.' else 0
                fold_enrichment = float(cols[6]) if cols[6] != '.' else 1
                pvalue = float(cols[7]) if cols[7] != '.' else 0
                qvalue = float(cols[8]) if cols[8] != '.' else 0
                summit_offset = int(cols[9]) if cols[9] != '.' else (end - start) // 2
                
                peaks.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'peak_name': peak_name,
                    'score': score,
                    'fold_enrichment': fold_enrichment,
                    'pvalue': pvalue,
                    'qvalue': qvalue,
                    'summit': start + summit_offset,
                    'length': end - start
                })
                
            except (ValueError, IndexError):
                continue
    
    return peaks

def parse_region_key(region_key):
    """Parse region key like 'chr22_12345678_12346678' or 'chr22:12345678-12346678'"""
    if '_' in region_key:
        parts = region_key.split('_')
        chrom = parts[0]
        start = int(parts[1])
        end = int(parts[2])
    elif ':' in region_key and '-' in region_key:
        chrom, coords = region_key.split(':')
        start, end = map(int, coords.split('-'))
    else:
        raise ValueError(f"Cannot parse region key: {region_key}")
    
    return chrom, start, end

def get_chipseq_signal_in_region(peaks, chrom, start, end):
    """Get ChIP-seq signal (fold enrichment) for a specific region"""
    signal = np.zeros(end - start)
    
    for peak in peaks:
        if peak['chrom'] != chrom:
            continue
        
        # Check for overlap
        peak_start, peak_end = peak['start'], peak['end']
        if peak_end <= start or peak_start >= end:
            continue
        
        # Calculate overlap
        overlap_start = max(start, peak_start)
        overlap_end = min(end, peak_end)
        
        # Fill signal array
        signal_start_idx = overlap_start - start
        signal_end_idx = overlap_end - start
        signal[signal_start_idx:signal_end_idx] = peak['fold_enrichment']
    
    return signal

def calculate_label_accuracy_stats(data_file, chipseq_base_path="raw/E005", threshold=0.5):
    """
    Calculate accuracy metrics for your binary labels vs ChIP-seq
    
    Args:
        threshold: ChIP-seq fold enrichment threshold to consider "true positive"
                  (e.g., 0.5 means any region with max signal > 0.5 is considered positive)
    """
    print("Calculating label accuracy statistics...")
    
    # Load your data
    data = np.load(data_file, allow_pickle=True)
    labels = data['label'][:, 0, :]  # Your binary labels
    keys = data['keys']
    
    markers = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']
    
    accuracy_stats = []
    
    for marker_idx, marker in enumerate(markers):
        print(f"Analyzing {marker}...")
        
        # Load ChIP-seq peaks for this marker
        chipseq_file = f"{chipseq_base_path}/E005-{marker}.narrowPeak"
        peaks = load_chipseq_peaks(chipseq_file)
        
        if not peaks:
            print(f"  No ChIP-seq data for {marker}")
            continue
        
        # For each region, determine ground truth and your prediction
        your_predictions = []
        ground_truth = []
        
        for i, region_key in enumerate(keys):
            chrom, start, end = parse_region_key(region_key)
            
            # Your binary prediction
            your_prediction = labels[i, marker_idx]
            your_predictions.append(your_prediction)
            
            # Ground truth from ChIP-seq
            chipseq_signal = get_chipseq_signal_in_region(peaks, chrom, start, end)
            max_signal = np.max(chipseq_signal)
            ground_truth_positive = 1 if max_signal > threshold else 0
            ground_truth.append(ground_truth_positive)
        
        # Convert to numpy arrays
        your_predictions = np.array(your_predictions)
        ground_truth = np.array(ground_truth)
        
        # Calculate confusion matrix elements
        true_positives = np.sum((your_predictions == 1) & (ground_truth == 1))
        false_positives = np.sum((your_predictions == 1) & (ground_truth == 0))
        true_negatives = np.sum((your_predictions == 0) & (ground_truth == 0))
        false_negatives = np.sum((your_predictions == 0) & (ground_truth == 1))
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(your_predictions)
        
        stats = {
            'Marker': marker,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1_score,
            'Accuracy': accuracy,
            'True_Positives': true_positives,
            'False_Positives': false_positives,
            'True_Negatives': true_negatives,
            'False_Negatives': false_negatives,
            'Total_Regions': len(your_predictions),
            'ChIP_Positive_Regions': np.sum(ground_truth),
            'Your_Positive_Regions': np.sum(your_predictions)
        }
        
        accuracy_stats.append(stats)
        
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")
    
    # Create summary DataFrame
    stats_df = pd.DataFrame(accuracy_stats)
    
    print("\n" + "="*100)
    print("LABEL ACCURACY STATISTICS")
    print("="*100)
    print(f"ChIP-seq threshold: {threshold} (fold enrichment)")
    print("\nPer-marker results:")
    print(stats_df[['Marker', 'Precision', 'Recall', 'F1_Score', 'Accuracy', 
                    'Your_Positive_Regions', 'ChIP_Positive_Regions']].round(3))
    
    # Summary statistics
    mean_precision = stats_df['Precision'].mean()
    mean_recall = stats_df['Recall'].mean()
    mean_f1 = stats_df['F1_Score'].mean()
    
    print(f"\nOverall Performance:")
    print(f"  Mean Precision: {mean_precision:.3f}")
    print(f"  Mean Recall: {mean_recall:.3f}")
    print(f"  Mean F1 Score: {mean_f1:.3f}")
    
    return stats_df

def analyze_peak_summits(region_key, binary_labels, chipseq_base_path="raw/E005"):
    """
    Check if your positive regions contain actual peak summits
    This is very informative because summits represent the strongest signal points
    """
    print(f"Analyzing peak summits for region: {region_key}")
    
    # Parse region
    chrom, start, end = parse_region_key(region_key)
    region_length = end - start
    
    markers = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']
    
    # Create visualization
    fig, axes = plt.subplots(len(markers), 1, figsize=(15, 2.5 * len(markers)))
    if len(markers) == 1:
        axes = [axes]
    
    summit_analysis_results = []
    
    for i, marker in enumerate(markers):
        ax = axes[i]
        
        # Load ChIP-seq peaks
        chipseq_file = f"{chipseq_base_path}/E005-{marker}.narrowPeak"
        peaks = load_chipseq_peaks(chipseq_file)
        
        your_label = binary_labels[i]
        
        if peaks:
            # Get continuous signal for plotting
            chipseq_signal = get_chipseq_signal_in_region(peaks, chrom, start, end)
            genomic_positions = np.arange(start, end)
            
            # Find summits within this region
            summits_in_region = []
            peaks_in_region = []
            
            for peak in peaks:
                if peak['chrom'] != chrom:
                    continue
                
                summit_pos = peak['summit']
                
                # Check if summit is within our region
                if start <= summit_pos <= end:
                    summits_in_region.append(summit_pos)
                    peaks_in_region.append(peak)
                
                # Also check if peak overlaps our region (even if summit is outside)
                elif not (peak['end'] <= start or peak['start'] >= end):
                    peaks_in_region.append(peak)
            
            # Plot the signal
            ax.fill_between(genomic_positions, chipseq_signal, alpha=0.6, color='lightblue', 
                           label='ChIP-seq Signal')
            
            # Mark summits
            for summit_pos in summits_in_region:
                ax.axvline(summit_pos, color='red', linewidth=3, alpha=0.8, 
                          label='Peak Summit' if summit_pos == summits_in_region[0] else "")
            
            # Show your binary label
            label_color = 'green' if your_label == 1 else 'gray'
            label_text = 'Your Label: POSITIVE' if your_label == 1 else 'Your Label: NEGATIVE'
            ax.axhspan(0, max(chipseq_signal) * 0.1 if np.any(chipseq_signal) else 1, 
                      alpha=0.3, color=label_color)
            
            # Analysis
            num_summits = len(summits_in_region)
            max_signal = np.max(chipseq_signal)
            has_strong_signal = max_signal > 1.0  # Arbitrary threshold
            
            # Title with analysis
            title = f"{marker} - {label_text}\n"
            title += f"Summits in region: {num_summits}, Max signal: {max_signal:.2f}"
            
            # Validation status
            if your_label == 1 and num_summits > 0:
                validation = "✓ GOOD: Positive label with summit(s)"
                title_color = 'green'
            elif your_label == 0 and num_summits == 0:
                validation = "✓ GOOD: Negative label without summits"
                title_color = 'green'
            elif your_label == 1 and num_summits == 0:
                validation = "⚠ QUESTIONABLE: Positive label but no summits"
                title_color = 'orange'
            else:  # your_label == 0 and num_summits > 0
                validation = "✗ BAD: Negative label but has summit(s)!"
                title_color = 'red'
            
            title += f"\n{validation}"
            ax.set_title(title, color=title_color, fontweight='bold')
            
            ax.set_ylabel("Fold Enrichment")
            ax.set_xlim(start, end)
            ax.legend(fontsize=8)
            
            # Store results
            summit_analysis_results.append({
                'marker': marker,
                'your_label': your_label,
                'summits_in_region': num_summits,
                'max_signal': max_signal,
                'validation_status': validation,
                'peaks_overlapping': len(peaks_in_region)
            })
            
        else:
            ax.text(0.5, 0.5, f"No ChIP-seq data for {marker}", 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f"{marker} - No Data Available")
            
            summit_analysis_results.append({
                'marker': marker,
                'your_label': your_label,
                'summits_in_region': 0,
                'max_signal': 0,
                'validation_status': 'No data',
                'peaks_overlapping': 0
            })
    
    axes[-1].set_xlabel("Genomic Position (bp)")
    plt.suptitle(f"Peak Summit Analysis - {region_key}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"plots/summit_analysis_{region_key.replace(':', '_').replace('-', '_')}.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\nSummit Analysis Summary:")
    print("-" * 60)
    for result in summit_analysis_results:
        print(f"{result['marker']:10s}: {result['validation_status']}")
    
    return summit_analysis_results

def compute_signal_correlations(data_file, chipseq_base_path="raw/E005"):
    """Compute correlation between ChIP-seq signal strength and binary calls"""
    print("Computing signal correlations...")
    
    # Load your data
    data = np.load(data_file, allow_pickle=True)
    labels = data['label'][:, 0, :]
    keys = data['keys']
    
    markers = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']
    
    correlation_results = []
    
    for marker_idx, marker in enumerate(markers):
        print(f"Analyzing correlations for {marker}...")
        
        # Load ChIP-seq peaks
        chipseq_file = f"{chipseq_base_path}/E005-{marker}.narrowPeak"
        peaks = load_chipseq_peaks(chipseq_file)
        
        if not peaks:
            continue
        
        signal_strengths = []
        binary_labels = []
        
        for i, region_key in enumerate(keys):
            chrom, start, end = parse_region_key(region_key)
            
            # Get signal strength metrics
            chipseq_signal = get_chipseq_signal_in_region(peaks, chrom, start, end)
            
            max_signal = np.max(chipseq_signal)
            mean_signal = np.mean(chipseq_signal[chipseq_signal > 0]) if np.any(chipseq_signal > 0) else 0
            coverage = np.sum(chipseq_signal > 0) / len(chipseq_signal)
            
            signal_strengths.append({
                'max_signal': max_signal,
                'mean_signal': mean_signal,
                'coverage': coverage,
                'total_signal': np.sum(chipseq_signal)
            })
            
            binary_labels.append(labels[i, marker_idx])
        
        # Convert to arrays for analysis
        binary_labels = np.array(binary_labels)
        max_signals = np.array([s['max_signal'] for s in signal_strengths])
        mean_signals = np.array([s['mean_signal'] for s in signal_strengths])
        coverages = np.array([s['coverage'] for s in signal_strengths])
        
        # Calculate correlations (point-biserial correlation for binary vs continuous)
        from scipy.stats import pearsonr
        
        max_corr, max_p = pearsonr(binary_labels, max_signals)
        mean_corr, mean_p = pearsonr(binary_labels, mean_signals)
        cov_corr, cov_p = pearsonr(binary_labels, coverages)
        
        # Signal strength differences between positive and negative labels
        pos_max_signals = max_signals[binary_labels == 1]
        neg_max_signals = max_signals[binary_labels == 0]
        
        result = {
            'marker': marker,
            'max_signal_correlation': max_corr,
            'mean_signal_correlation': mean_corr,
            'coverage_correlation': cov_corr,
            'pos_regions': np.sum(binary_labels),
            'neg_regions': np.sum(binary_labels == 0),
            'pos_mean_max_signal': np.mean(pos_max_signals) if len(pos_max_signals) > 0 else 0,
            'neg_mean_max_signal': np.mean(neg_max_signals) if len(neg_max_signals) > 0 else 0,
            'signal_separation': np.mean(pos_max_signals) - np.mean(neg_max_signals) if len(pos_max_signals) > 0 and len(neg_max_signals) > 0 else 0
        }
        
        correlation_results.append(result)
        
        print(f"  Max signal correlation: {max_corr:.3f} (p={max_p:.3e})")
        print(f"  Positive regions mean max signal: {result['pos_mean_max_signal']:.3f}")
        print(f"  Negative regions mean max signal: {result['neg_mean_max_signal']:.3f}")
    
    # Summary
    corr_df = pd.DataFrame(correlation_results)
    
    print("\n" + "="*80)
    print("SIGNAL CORRELATION ANALYSIS")
    print("="*80)
    print(corr_df[['marker', 'max_signal_correlation', 'pos_mean_max_signal', 
                   'neg_mean_max_signal', 'signal_separation']].round(3))
    
    return corr_df

def compare_binary_to_chipseq(region_key, binary_labels, chipseq_base_path="raw/E005", 
                             markers=['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']):
    """Compare binary histone labels to continuous ChIP-seq signals"""
    
    # Parse region
    chrom, start, end = parse_region_key(region_key)
    region_length = end - start
    
    # Create actual genomic coordinates for x-axis
    genomic_positions = np.arange(start, end)
    
    fig, axes = plt.subplots(len(markers) + 1, 1, figsize=(15, 2 * (len(markers) + 1)))
    
    # Plot binary labels first
    axes[0].set_title(f"Your Binary Histone Labels - {region_key}")
    for j, marker in enumerate(markers):
        if binary_labels[j]:
            axes[0].axhline(j, color="red", linewidth=3, alpha=0.7)
    axes[0].set_yticks(range(len(markers)))
    axes[0].set_yticklabels(markers)
    axes[0].set_ylabel("Markers")
    axes[0].set_xlim(start, end)
    
    # Compare with each ChIP-seq file
    for i, marker in enumerate(markers):
        ax = axes[i + 1]
        
        # Load ChIP-seq peaks
        chipseq_file = f"{chipseq_base_path}/E005-{marker}.narrowPeak"
        peaks = load_chipseq_peaks(chipseq_file)
        
        if peaks:
            # Get continuous signal
            chipseq_signal = get_chipseq_signal_in_region(peaks, chrom, start, end)
            
            # Plot continuous signal with actual genomic coordinates
            ax.fill_between(genomic_positions, chipseq_signal, alpha=0.6, color='blue', label='ChIP-seq Signal')
            
            # Highlight your binary prediction
            binary_value = binary_labels[i]
            if binary_value:
                ax.axhline(max(chipseq_signal) * 0.8, color='red', linewidth=2, 
                          linestyle='--', alpha=0.8, label='Your Binary Label: Positive')
            else:
                ax.axhline(0, color='gray', linewidth=2, 
                          linestyle='--', alpha=0.8, label='Your Binary Label: Negative')
            
            ax.set_title(f"{marker} - ChIP-seq vs Binary")
            ax.set_ylabel("Fold Enrichment")
            ax.set_xlim(start, end)
            ax.legend(fontsize=8)
            
            # Format x-axis to show genomic coordinates nicely
            ax.ticklabel_format(style='plain', axis='x')
            
            # Add thousand separators to x-axis labels
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
            
            # Stats
            max_signal = np.max(chipseq_signal)
            mean_signal = np.mean(chipseq_signal[chipseq_signal > 0]) if np.any(chipseq_signal > 0) else 0
            coverage = np.sum(chipseq_signal > 0) / len(chipseq_signal) * 100
            
            stats_text = f"Max: {max_signal:.1f}, Mean: {mean_signal:.1f}, Coverage: {coverage:.1f}%"
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f"No ChIP-seq data found for {marker}", 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f"{marker} - No Data Available")
            ax.set_xlim(start, end)
    
    axes[-1].set_xlabel("Genomic Position (bp)")
    plt.tight_layout()
    plt.savefig(f"plots/chipseq_comparison_{region_key.replace(':', '_').replace('-', '_')}.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def analyze_chipseq_statistics(chipseq_base_path="raw/E005", 
                              markers=['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']):
    """Analyze statistics of ChIP-seq files"""
    
    stats_data = []
    
    for marker in markers:
        chipseq_file = f"{chipseq_base_path}/E005-{marker}.narrowPeak"
        peaks = load_chipseq_peaks(chipseq_file)
        
        if peaks:
            df = pd.DataFrame(peaks)
            
            stats = {
                'Marker': marker,
                'Total_Peaks': len(peaks),
                'Mean_Length': df['length'].mean(),
                'Median_Length': df['length'].median(),
                'Mean_Score': df['score'].mean(),
                'Mean_FoldEnrichment': df['fold_enrichment'].mean(),
                'Mean_PValue': df['pvalue'].mean(),
                'Mean_QValue': df['qvalue'].mean(),
            }
            
            # Chromosome distribution
            chrom_counts = df['chrom'].value_counts()
            stats['Chr22_Peaks'] = chrom_counts.get('chr22', 0)
            
            stats_data.append(stats)
        else:
            stats_data.append({
                'Marker': marker,
                'Total_Peaks': 0,
                'Mean_Length': 0,
                'Median_Length': 0,
                'Mean_Score': 0,
                'Mean_FoldEnrichment': 0,
                'Mean_PValue': 0,
                'Mean_QValue': 0,
                'Chr22_Peaks': 0
            })
    
    stats_df = pd.DataFrame(stats_data)
    
    print("ChIP-seq Dataset Statistics:")
    print("=" * 80)
    print(stats_df.round(3))
    
    return stats_df

def find_false_negatives(data_file, chipseq_base_path="raw/E005",
                         markers=['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'],
                         min_signal_threshold=0.1):
    """Find regions labeled negative but overlapping a ChIP-seq signal"""
    print("\nScanning for false negatives...")
    data = np.load(data_file, allow_pickle=True)
    labels = data['label'][:, 0, :]
    keys = data['keys']
    
    false_negatives = defaultdict(list)  # marker -> list of region keys
    
    for marker_idx, marker in enumerate(markers):
        chipseq_file = f"{chipseq_base_path}/E005-{marker}.narrowPeak"
        peaks = load_chipseq_peaks(chipseq_file)
        print(f"Checking {marker}...")
        
        for i, region_key in enumerate(keys):
            chrom, start, end = parse_region_key(region_key)
            chipseq_signal = get_chipseq_signal_in_region(peaks, chrom, start, end)
            
            if labels[i, marker_idx] == 0 and np.max(chipseq_signal) > min_signal_threshold:
                false_negatives[marker].append(region_key)
    
        print(f"  Found {len(false_negatives[marker])} false negatives for {marker}")
    
    # Summary
    total_fn = sum(len(v) for v in false_negatives.values())
    print(f"\nTotal false negatives found: {total_fn}")
    
    return false_negatives

def run_comprehensive_validation(data_file='data/E005_deephistone_1000bp_chr22.npz'):
    """Run complete validation analysis with all new functions"""
    
    print("="*100)
    print("COMPREHENSIVE DEEPHISTONE VALIDATION")
    print("="*100)
    
    # Load your data
    data = np.load(data_file, allow_pickle=True)
    labels = data['label'][:, 0, :]
    keys = data['keys']
    
    print(f"Loaded {len(keys)} regions from your dataset")
    
    # 1. Basic ChIP-seq statistics
    print("\n" + "="*80)
    print("1. CHIPSEQ DATASET ANALYSIS")
    print("="*80)
    chipseq_stats = analyze_chipseq_statistics()
    
    # 2. Label accuracy statistics
    print("\n" + "="*80)
    print("2. LABEL ACCURACY ANALYSIS")
    print("="*80)
    accuracy_stats = calculate_label_accuracy_stats(data_file, threshold=0.5)
    
    # 3. Signal correlations
    print("\n" + "="*80)
    print("3. SIGNAL CORRELATION ANALYSIS")
    print("="*80)
    correlation_stats = compute_signal_correlations(data_file)
    
    # 4. Peak summit analysis for selected regions
    print("\n" + "="*80)
    print("4. PEAK SUMMIT ANALYSIS")
    print("="*80)
    
    # Analyze a few interesting regions
    interesting_indices = [10, 50, 100, 200, 500]  # You can change these
    
    for i in interesting_indices:
        if i < len(keys):
            print(f"\n--- Summit Analysis for Region {i}: {keys[i]} ---")
            summit_results = analyze_peak_summits(keys[i], labels[i])
    
    # 5. Basic region comparisons
    print("\n" + "="*80)
    print("5. DETAILED REGION COMPARISONS")
    print("="*80)
    
    comparison_indices = [25, 75, 150, 300]  # Different set for variety
    
    for i in comparison_indices:
        if i < len(keys):
            print(f"\nComparing region {i}: {keys[i]}")
            compare_binary_to_chipseq(keys[i], labels[i])
    
    print("\n" + "="*100)
    print("VALIDATION COMPLETE!")
    print("="*100)
    print("Check the 'plots/' directory for all generated visualizations.")
    
    return {
        'chipseq_stats': chipseq_stats,
        'accuracy_stats': accuracy_stats,
        'correlation_stats': correlation_stats
    }

def extend_comparison_to_larger_regions(data_file, chipseq_base_path="raw/E005", 
                                       context_extension=2000, num_examples=5):
    """Extend analysis to larger genomic context around your regions"""
    
    # Load your data
    data = np.load(data_file, allow_pickle=True)
    dna = data['dna'][:, 0, :, :]
    labels = data['label'][:, 0, :]
    keys = data['keys']
    
    markers = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']
    
    # Take first few examples
    for i in range(min(num_examples, len(keys))):
        region_key = keys[i]
        binary_labels = labels[i]
        
        print(f"\nAnalyzing extended context for region {i+1}: {region_key}")
        
        # Parse original region
        chrom, start, end = parse_region_key(region_key)
        original_length = end - start
        
        # Extend context
        extended_start = max(0, start - context_extension)
        extended_end = end + context_extension
        
        # Create extended comparison
        fig, axes = plt.subplots(len(markers), 1, figsize=(20, 2 * len(markers)))
        
        for j, marker in enumerate(markers):
            ax = axes[j]
            
            # Load ChIP-seq peaks
            chipseq_file = f"{chipseq_base_path}/E005-{marker}.narrowPeak"
            peaks = load_chipseq_peaks(chipseq_file)
            
            if peaks:
                # Get extended signal
                extended_signal = get_chipseq_signal_in_region(peaks, chrom, extended_start, extended_end)
                extended_positions = np.arange(extended_start, extended_end)
                
                # Plot extended signal
                ax.fill_between(extended_positions, extended_signal, alpha=0.4, color='lightblue', 
                               label='Extended Context')
                
                # Highlight your original region
                original_start_idx = start - extended_start
                original_end_idx = end - extended_start
                
                if original_start_idx >= 0 and original_end_idx <= len(extended_signal):
                    original_signal = extended_signal[original_start_idx:original_end_idx]
                    original_positions = np.arange(start, end)
                    
                    ax.fill_between(original_positions, original_signal, alpha=0.8, color='blue', 
                                   label='Your 1kb Region')
                    
                    # Add binary label indicator
                    if binary_labels[j]:
                        ax.axhspan(0, max(extended_signal) if np.any(extended_signal) else 1, 
                                  xmin=(start-extended_start)/(extended_end-extended_start),
                                  xmax=(end-extended_start)/(extended_end-extended_start), 
                                  alpha=0.2, color='red', label='Binary: Positive')
                
                ax.set_title(f"{marker} - Extended Context ({context_extension*2/1000:.1f}kb total)")
                ax.set_ylabel("Fold Enrichment")
                ax.set_xlim(extended_start, extended_end)
                ax.legend(fontsize=8)
                
                # Mark region boundaries
                ax.axvline(start, color='red', linestyle='--', alpha=0.7)
                ax.axvline(end, color='red', linestyle='--', alpha=0.7)
                
                # Format x-axis
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
            
        axes[-1].set_xlabel("Genomic Position (bp)")
        plt.suptitle(f"Extended Context Analysis - {region_key}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"plots/extended_context_{region_key.replace(':', '_').replace('-', '_')}.png", 
                    dpi=300, bbox_inches='tight')
        plt.show()

def generate_validation_report(data_file, output_file="validation_report.txt"):
    """Generate a comprehensive text report of validation results"""
    
    print(f"Generating validation report: {output_file}")
    
    # Load data
    data = np.load(data_file, allow_pickle=True)
    labels = data['label'][:, 0, :]
    keys = data['keys']
    
    markers = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']
    
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("DEEPHISTONE PREPROCESSING VALIDATION REPORT\n")
        f.write("="*100 + "\n")
        f.write(f"Dataset: {data_file}\n")
        f.write(f"Total regions: {len(keys):,}\n")
        f.write(f"Markers analyzed: {', '.join(markers)}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        # Dataset overview
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 50 + "\n")
        
        # Per-marker positive counts
        marker_counts = labels.sum(axis=0)
        f.write("Per-marker positive regions:\n")
        for marker, count in zip(markers, marker_counts):
            percentage = (count / len(labels)) * 100
            f.write(f"  {marker:10s}: {count:6,} ({percentage:5.1f}%)\n")
        
        # Multi-label statistics
        total_positives = np.sum(labels, axis=1)
        f.write(f"\nMulti-label statistics:\n")
        f.write(f"  Regions with 0 markers: {np.sum(total_positives == 0):,}\n")
        f.write(f"  Regions with 1 marker:  {np.sum(total_positives == 1):,}\n")
        f.write(f"  Regions with 2+ markers: {np.sum(total_positives >= 2):,}\n")
        f.write(f"  Max markers per region: {np.max(total_positives)}\n")
        f.write(f"  Mean markers per region: {np.mean(total_positives):.2f}\n\n")
        
        # Run accuracy analysis
        f.write("ACCURACY ANALYSIS\n")
        f.write("-" * 50 + "\n")
        try:
            accuracy_stats = calculate_label_accuracy_stats(data_file, threshold=0.5)
            
            f.write("Validation against ChIP-seq data (threshold=0.5):\n")
            for _, row in accuracy_stats.iterrows():
                f.write(f"  {row['Marker']:10s}: ")
                f.write(f"Precision={row['Precision']:.3f}, ")
                f.write(f"Recall={row['Recall']:.3f}, ")
                f.write(f"F1={row['F1_Score']:.3f}\n")
            
            f.write(f"\nOverall performance:\n")
            f.write(f"  Mean Precision: {accuracy_stats['Precision'].mean():.3f}\n")
            f.write(f"  Mean Recall: {accuracy_stats['Recall'].mean():.3f}\n")
            f.write(f"  Mean F1 Score: {accuracy_stats['F1_Score'].mean():.3f}\n\n")
            
        except Exception as e:
            f.write(f"Could not compute accuracy stats: {e}\n\n")
        
        # Chromosome distribution
        f.write("GENOMIC DISTRIBUTION\n")
        f.write("-" * 50 + "\n")
        chrom_counts = {}
        for key in keys:
            chrom = parse_region_key(key)[0]
            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
        
        f.write("Regions per chromosome:\n")
        for chrom in sorted(chrom_counts.keys()):
            f.write(f"  {chrom}: {chrom_counts[chrom]:,}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*100 + "\n")
    
    print(f"Validation report saved to: {output_file}")

def run_quick_validation(data_file, num_examples=3):
    """Run a quick validation with just a few examples"""
    
    print("="*60)
    print("QUICK DEEPHISTONE VALIDATION")
    print("="*60)
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Load data
    data = np.load(data_file, allow_pickle=True)
    labels = data['label'][:, 0, :]
    keys = data['keys']
    
    print(f"Dataset: {data_file}")
    print(f"Total regions: {len(keys):,}")
    
    # Quick stats
    marker_counts = labels.sum(axis=0)
    markers = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']
    
    print("\nPositive regions per marker:")
    for marker, count in zip(markers, marker_counts):
        percentage = (count / len(labels)) * 100
        print(f"  {marker}: {count:,} ({percentage:.1f}%)")
    
    # Analyze a few examples
    print(f"\nAnalyzing {num_examples} example regions:")
    
    example_indices = np.linspace(0, len(keys)-1, num_examples, dtype=int)
    
    for i, idx in enumerate(example_indices):
        print(f"\n--- Example {i+1}: {keys[idx]} ---")
        
        # Summit analysis
        summit_results = analyze_peak_summits(keys[idx], labels[idx])
        
        # Basic comparison
        compare_binary_to_chipseq(keys[idx], labels[idx])
    
    # Generate text report
    generate_validation_report(data_file)
    
    print("\n" + "="*60)
    print("QUICK VALIDATION COMPLETE!")
    print("="*60)
    print("Check 'plots/' directory for visualizations")
    print("Check 'validation_report.txt' for detailed statistics")

if __name__ == "__main__":
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # You can choose which validation to run:
    
    # Option 1: Quick validation (recommended for first run)
    #run_quick_validation('data/E005_deephistone_1024bp_chr22.npz', num_examples=3)
    
    # Option 2: Comprehensive validation (uncomment to run)
    results = run_comprehensive_validation('data/E005_deephistone_1024bp_chr22.npz')
    
    # Option 3: Just accuracy stats (uncomment to run)
    # calculate_label_accuracy_stats('data/E005_deephistone_1000bp_chr22.npz')