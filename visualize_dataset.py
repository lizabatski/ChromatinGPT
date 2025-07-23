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
                extended_positions = np.arange(len(extended_signal))
                
                # Plot extended signal
                ax.fill_between(extended_positions, extended_signal, alpha=0.4, color='lightblue', 
                               label='Extended Context')
                
                # Highlight your original region
                original_start_idx = start - extended_start
                original_end_idx = end - extended_start
                
                if original_start_idx >= 0 and original_end_idx <= len(extended_signal):
                    original_signal = extended_signal[original_start_idx:original_end_idx]
                    original_positions = np.arange(original_start_idx, original_end_idx)
                    
                    ax.fill_between(original_positions, original_signal, alpha=0.8, color='blue', 
                                   label='Your 1kb Region')
                    
                    # Add binary label indicator
                    if binary_labels[j]:
                        ax.axhspan(0, max(extended_signal), xmin=original_start_idx/len(extended_signal),
                                  xmax=original_end_idx/len(extended_signal), alpha=0.2, color='red',
                                  label='Binary: Positive')
                
                ax.set_title(f"{marker} - Extended Context ({context_extension*2/1000:.1f}kb total)")
                ax.set_ylabel("Fold Enrichment")
                ax.legend(fontsize=8)
                
                # Mark region boundaries
                ax.axvline(original_start_idx, color='red', linestyle='--', alpha=0.7)
                ax.axvline(original_end_idx, color='red', linestyle='--', alpha=0.7)
            
        axes[-1].set_xlabel("Position (bp)")
        plt.suptitle(f"Extended Context Analysis - {region_key}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"plots/extended_context_{region_key.replace(':', '_').replace('-', '_')}.png", 
                    dpi=300, bbox_inches='tight')
        plt.show()
def find_false_negatives(data_file, chipseq_base_path="raw/E005",
                         markers=['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'],
                         min_signal_threshold=0.1):
    """
    Find regions labeled negative but overlapping a ChIP-seq signal
    """
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

# Example usage functions
def run_comparison_analysis(data_file='data/E005_deephistone_1024bp_chr22.npz'):
    """Run complete comparison analysis"""
    
    print("Loading your processed data...")
    data = np.load(data_file, allow_pickle=True)
    labels = data['label'][:, 0, :]
    keys = data['keys']
    
    print(f"Loaded {len(keys)} regions from your dataset")
    
    # Analyze ChIP-seq statistics
    print("\n" + "="*80)
    print("CHIPSEQ DATASET ANALYSIS")
    print("="*80)
    analyze_chipseq_statistics()
    
    # Compare specific examples
    print("\n" + "="*80)
    print("REGION-SPECIFIC COMPARISONS")
    print("="*80)
    
    # Pick a few interesting examples
    examples = [42, 100, 200, 500, 1000]  # You can change these indices
    
    for i in examples:
        if i < len(keys):
            print(f"\nComparing region {i}: {keys[i]}")
            compare_binary_to_chipseq(keys[i], labels[i])
    
    # Extended context analysis
    print("\n" + "="*80)
    print("EXTENDED CONTEXT ANALYSIS")
    print("="*80)
    extend_comparison_to_larger_regions(data_file, context_extension=2000, num_examples=3)



if __name__ == "__main__":
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    
    run_comparison_analysis()