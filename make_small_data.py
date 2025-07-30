#!/usr/bin/env python3

import numpy as np
import argparse
import os

def extract_subset(input_file, output_file, num_samples=500, seed=42):
    """
    Extract a random subset of samples from the dataset
    """
    print(f"Loading data from {input_file}...")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    with np.load(input_file, mmap_mode='r') as data:
        total_samples = data['keys'].shape[0]
        print(f"Total samples in original dataset: {total_samples:,}")
        
        if num_samples > total_samples:
            print(f"Warning: Requested {num_samples} samples but only {total_samples} available")
            num_samples = total_samples
        
        # Randomly select indices
        selected_indices = np.random.choice(total_samples, size=num_samples, replace=False)
        selected_indices = np.sort(selected_indices)  # Sort for better memory access
        
        print(f"Extracting {num_samples} random samples...")
        
        # Extract subset
        subset_keys = data['keys'][selected_indices]
        subset_dna = data['dna'][selected_indices]
        subset_dnase = data['dnase'][selected_indices]
        subset_labels = data['label'][selected_indices]
        
        print(f"Subset shapes:")
        print(f"  Keys: {subset_keys.shape}")
        print(f"  DNA: {subset_dna.shape}")
        print(f"  DNase: {subset_dnase.shape}")
        print(f"  Labels: {subset_labels.shape}")
    
    # Save subset
    print(f"Saving subset to {output_file}...")
    np.savez_compressed(
        output_file,
        keys=subset_keys,
        dna=subset_dna,
        dnase=subset_dnase,
        label=subset_labels
    )
    
    print(f"Successfully created subset with {num_samples} samples!")
    print(f"Original file size: {os.path.getsize(input_file) / 1024**2:.1f} MB")
    print(f"Subset file size: {os.path.getsize(output_file) / 1024**2:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="Extract a subset of samples from dataset")
    parser.add_argument('--input', type=str, required=True,
                        help='Input .npz file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .npz file')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to extract (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    extract_subset(args.input, args.output, args.num_samples, args.seed)

if __name__ == "__main__":
    main()