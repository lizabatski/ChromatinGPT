import os
import numpy as np
from pyfaidx import Fasta
import pandas as pd
from collections import defaultdict
import time
from tqdm import tqdm
import json
from pathlib import Path
import multiprocessing as mp
from functools import partial
import logging
from datetime import datetime
import traceback


class DeepHistoneConfig:
    def __init__(self):
        # Parameters from paper
        self.WINDOW_SIZE = 200  # scanning windows
        self.FINAL_WINDOW_SIZE = 1000  # final sequences for model
        self.STEP_SIZE = 200    # non-overlapping scan
        self.MIN_OVERLAP = 100  # minimum overlap with peak 
        self.MIN_SITES_THRESHOLD = 50000  # discard epigenomes with  less 50K sites per marker
        self.RANDOM_SEED = 42
        
        # 7 histone markers from paper Table 1
        self.ALL_MARKERS = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']
        
        # 15 epigenomes 
        self.VALID_EPIGENOMES = [
            'E003', 'E004', 'E005', 'E006', 'E007', 'E011', 'E012', 
            'E013', 'E016', 'E024', 'E065', 'E066', 'E116', 'E117', 'E118'
        ]
        
        # paths
        self.BASE_PATH = "raw"
        self.CHROM_SIZES = "raw/hg19.chrom.sizes.txt"
        self.FASTA_PATH = "raw/hg19.fa"
        self.OUTPUT_DIR = "data"
        
        
        self.USE_MULTIPROCESSING = True
        self.N_PROCESSES = min(12, mp.cpu_count())
        
        self.MAX_N_FRACTION = 0.1
        self.VALIDATE_GENOME_COVERAGE = True
        
        
        self.TEST_MODE = False  
        self.TEST_CHROMOSOME = "chr22"
        
        
        self.SKIP_EXISTING = True
        self.CONTINUE_ON_ERROR = True
        
        
        self._chrom_sizes = None
        self._genome = None
        
    def get_chrom_sizes(self):
        if self._chrom_sizes is None:
            self._chrom_sizes = self._load_chromosome_sizes()
        return self._chrom_sizes
    
    def get_genome(self):
        if self._genome is None:
            self._genome = Fasta(self.FASTA_PATH)
        return self._genome
    
    def _load_chromosome_sizes(self):
        chrom_sizes = {}
        try:
            with open(self.CHROM_SIZES, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        chrom, size = parts[0], int(parts[1])
                        
                        # Skip sex chromosomes
                        if chrom in ['chrX', 'chrY']:
                            continue
                        
                        # Test mode filtering
                        if self.TEST_MODE and chrom != self.TEST_CHROMOSOME:
                            continue
                        
                        chrom_sizes[chrom] = size
        except FileNotFoundError:
            raise FileNotFoundError(f"Chromosome sizes file not found: {self.CHROM_SIZES}")
        
        if not chrom_sizes:
            raise ValueError("No chromosomes loaded...")
        
        return chrom_sizes
    
    def get_chipseq_path(self, epigenome_id, marker):
        return f"{self.BASE_PATH}/{epigenome_id}/{epigenome_id}-{marker}.narrowPeak"
    
    def get_dnase_path(self, epigenome_id):
        return f"{self.BASE_PATH}/{epigenome_id}/{epigenome_id}-DNase.macs2.narrowPeak"
    
    def get_output_path(self, epigenome_id):
        suffix = f"_{self.TEST_CHROMOSOME}" if self.TEST_MODE else ""
        return f"{self.OUTPUT_DIR}/{epigenome_id}_deephistone{suffix}.npz"
    
    def get_output_path(self, epigenome_id):
        # Simple format matching your SLURM script expectations
        suffix = f"_{self.FINAL_WINDOW_SIZE}bp"
        if self.TEST_MODE:
            suffix += f"_{self.TEST_CHROMOSOME}"
        return f"{self.OUTPUT_DIR}/{epigenome_id}_deephistone{suffix}.npz"


config = DeepHistoneConfig()


def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/deephistone_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def log_progress(message, start_time=None):
    current_time = time.time()
    if start_time:
        elapsed = current_time - start_time
        print(f"[{elapsed:.2f}s] {message}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")
    return current_time

# load all peaks for a all markers for a given epigenome
def load_all_peaks_for_epigenome(epigenome_id):
    start_time = log_progress(f"Loading all peaks for {epigenome_id}...")
    
    all_peaks = {marker: defaultdict(list) for marker in config.ALL_MARKERS}
    
    for marker in config.ALL_MARKERS:
        peaks_file = config.get_chipseq_path(epigenome_id, marker)
        if not os.path.exists(peaks_file):
            log_progress(f"Warning: {peaks_file} not found")
            continue
            
        with open(peaks_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                cols = line.split('\t')
                if len(cols) < 3:
                    continue
                    
                try:
                    chrom, start, end = cols[0], int(cols[1]), int(cols[2]) # first column is chromosome, second is start, third is end
                    
                    # Skip sex chromosomes
                    if chrom in ['chrX', 'chrY']:
                        continue
                    
                    if config.TEST_MODE and chrom != config.TEST_CHROMOSOME:
                        continue
                    
                    if start >= end or start < 0:
                        continue
                    
                    all_peaks[marker][chrom].append((start, end))
                    
                except (ValueError, IndexError):
                    continue
    
    # Sort all peaks by start 
    total_peaks = 0
    for marker in all_peaks:
        for chrom in all_peaks[marker]:
            all_peaks[marker][chrom].sort()
            total_peaks += len(all_peaks[marker][chrom])
    
    log_progress(f"Loaded {total_peaks:,} peaks across all markers", start_time)
    return all_peaks



def scan_chromosome_for_all_markers(args):
    chrom, chrom_size, all_marker_peaks, window_size, step_size, min_overlap = args
    
    # store windows with their multi-label annotations
    windows_with_labels = []
    
    # scan chromosome with a sliding window 
    for window_start in range(0, chrom_size - window_size + 1, step_size):
        window_end = window_start + window_size
        
        # label vector (should be a vector of 7 0s)
        label_vector = np.zeros(len(config.ALL_MARKERS), dtype=np.int8)
        
        for marker_idx, marker in enumerate(config.ALL_MARKERS):
            if marker not in all_marker_peaks or chrom not in all_marker_peaks[marker]:
                continue
                
            peaks = all_marker_peaks[marker][chrom]
            
            # check if window has sufficient overlap with any peak
            for peak_start, peak_end in peaks:
                if peak_end <= window_start:
                    continue
                if peak_start >= window_end:
                    break
                
                # Calculate overlap
                overlap_start = max(window_start, peak_start)
                overlap_end = min(window_end, peak_end)
                overlap_length = overlap_end - overlap_start
                
                if overlap_length >= min_overlap:
                    label_vector[marker_idx] = 1
                    break
        
        # only add windows with a least one marker
        if np.any(label_vector):
            windows_with_labels.append((chrom, window_start, window_end, label_vector))
    #return a list of all windows on this chromosome
    return windows_with_labels


def scan_genome_multilabel(epigenome_id, all_peaks):
    start_time = log_progress("Scanning genome for all markers simultaneously...")
    
    chrom_sizes = config.get_chrom_sizes()
    
    # Prepare arguments for parallel processing
    process_args = []
    for chrom in sorted(chrom_sizes.keys()):
        args = (chrom, chrom_sizes[chrom], all_peaks, 
                config.WINDOW_SIZE, config.STEP_SIZE, config.MIN_OVERLAP)
        process_args.append(args)
    
    # Process chromosomes in parallel
    if len(process_args) > 1 and config.USE_MULTIPROCESSING:
        with mp.Pool(min(config.N_PROCESSES, len(process_args))) as pool:
            chrom_results = pool.map(scan_chromosome_for_all_markers, process_args)
    else:
        chrom_results = [scan_chromosome_for_all_markers(args) for args in process_args]
    
    # Combine results
    all_windows = []
    all_labels = []
    
    for windows_with_labels in chrom_results:
        for chrom, start, end, label_vector in windows_with_labels:
            all_windows.append((chrom, start, end))
            all_labels.append(label_vector)
    
    # Convert to numpy array for easier manipulation
    if all_labels:
        all_labels = np.vstack(all_labels)
    else:
        all_labels = np.array([]).reshape(0, len(config.ALL_MARKERS))
    
    # Log statistics
    log_progress(f"Found {len(all_windows):,} windows with at least one modification", start_time)
    
    # Per-marker statistics
    marker_counts = all_labels.sum(axis=0)
    print("\nPer-marker positive counts:")
    for marker, count in zip(config.ALL_MARKERS, marker_counts):
        print(f"  {marker}: {count:,} positive windows")
    
    # Check minimum threshold
    if any(marker_counts < config.MIN_SITES_THRESHOLD) and not config.TEST_MODE:
        low_markers = [m for m, c in zip(config.ALL_MARKERS, marker_counts) 
                       if c < config.MIN_SITES_THRESHOLD]
        log_progress(f"WARNING: Markers below threshold ({config.MIN_SITES_THRESHOLD:,}): {low_markers}")
        return None, None
    
    return all_windows, all_labels


def expand_regions_to_1000bp(regions_200bp):
    start_time = log_progress(f"Expanding {len(regions_200bp):,} regions from 200bp to 1000bp...")
    
    chrom_sizes = config.get_chrom_sizes()
    expanded_regions = []
    
    for chrom, start_200, end_200 in regions_200bp:
        center = (start_200 + end_200) // 2
        
        # create 1000bp window centered on this position
        half_final = config.FINAL_WINDOW_SIZE // 2
        start_1000 = center - half_final
        end_1000 = center + half_final
        
        # boundary checking
        if start_1000 < 0:
            start_1000 = 0
            end_1000 = config.FINAL_WINDOW_SIZE
        
        if chrom in chrom_sizes:
            chrom_size = chrom_sizes[chrom]
            if end_1000 > chrom_size:
                end_1000 = chrom_size
                start_1000 = max(0, chrom_size - config.FINAL_WINDOW_SIZE)
            
            if end_1000 - start_1000 == config.FINAL_WINDOW_SIZE:
                expanded_regions.append((chrom, start_1000, end_1000))
    
    log_progress(f"Successfully expanded {len(expanded_regions):,} regions", start_time)
    return expanded_regions


def sequence_to_onehot(seq):
    """Convert DNA sequence string to one-hot encoding
    Returns shape (1, 4, length) to match target format
    """
    # mapping
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    # one-hot array shape (4, length)
    length = len(seq)
    onehot = np.zeros((4, length), dtype=np.float32)
    
    for i, base in enumerate(seq.upper()):
        if base in base_to_idx and base_to_idx[base] < 4:
            onehot[base_to_idx[base], i] = 1.0
        # If N or unknown, leave as all zeros
    
    # add extra dimension to match target format (1, 4, length)
    return onehot[np.newaxis, :, :]


def extract_sequences_as_onehot(regions):
    start_time = log_progress(f"Extracting and encoding {len(regions):,} sequences...")
    
    try:
        genome = config.get_genome() # load genome reference 
    except Exception as e:
        raise FileNotFoundError(f"Cannot load genome FASTA: {config.FASTA_PATH}. Error: {e}")
    
    sequences_onehot = []
    invalid_count = 0 # counts sequences with too many Ns or other issues
    
    # process in chunks
    chunk_size = 5000
    for i in range(0, len(regions), chunk_size):
        chunk = regions[i:i + chunk_size]
        
        for chrom, region_start, region_end in chunk:
            expected_length = region_end - region_start
            
            try:
                seq = genome[chrom][region_start:region_end].seq.upper() #extracts sequence from FASTA and converts to uppercase
                
                # ensure correct length
                if len(seq) != expected_length:
                    if len(seq) < expected_length:
                        seq = seq.ljust(expected_length, 'N') #pad with Ns if too short
                    else:
                        seq = seq[:expected_length] #truncate
                
                # check N content
                n_count = seq.count('N')
                n_fraction = n_count / len(seq)
                
                if n_fraction > config.MAX_N_FRACTION:
                    invalid_count += 1
                
                # convert to one-hot
                onehot = sequence_to_onehot(seq)
                sequences_onehot.append(onehot)
                
            except Exception as e:
                log_progress(f"Warning: Could not extract sequence for {chrom}:{region_start}-{region_end}: {e}")
                # Create all-zero one-hot for failed extractions
                onehot = np.zeros((1, 4, expected_length), dtype=np.float32)
                sequences_onehot.append(onehot)
                invalid_count += 1
        
        # progress update
        if i % (chunk_size * 4) == 0 and i > 0:
            log_progress(f"Processed {i + len(chunk):,}/{len(regions):,} sequences...")
    
    if invalid_count > 0:
        log_progress(f"Warning: {invalid_count:,} sequences had quality issues")
    
    # stack all sequences
    sequences_array = np.concatenate(sequences_onehot, axis=0)  # Shape: (N, 1, 4, 1000)
    if sequences_array.shape[1] != 1:  # Check if second dimension is not 1 - was getting incorrect dimensions
        sequences_array = np.expand_dims(sequences_array, axis=1)
    
    log_progress(f"Extracted and encoded {sequences_array.shape[0]:,} sequences", start_time)
    return sequences_array


def extract_dnase_scores_batch(epigenome_id, regions):
    start_time = log_progress(f"Extracting DNase openness scores for {len(regions):,} regions...")
    
    dnase_file = config.get_dnase_path(epigenome_id)
    dnase_peaks_by_chrom = defaultdict(list)
    
    if not os.path.exists(dnase_file):
        log_progress(f"Warning: DNase file {dnase_file} not found, using zero openness scores")
        return [np.zeros(config.FINAL_WINDOW_SIZE, dtype=np.float32) for _ in regions]
    
    
    total_dnase_peaks = 0
    with open(dnase_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            cols = line.split('\t')
            if len(cols) < 7: #make sure it is in correct format
                continue
                
            try:
                chrom, start, end = cols[0], int(cols[1]), int(cols[2])
                
                if chrom in ['chrX', 'chrY']:
                    continue
                
                if config.TEST_MODE and chrom != config.TEST_CHROMOSOME:
                    continue
                
                # Use signal value from column 6
                fold_enrichment = 1.0
                try:
                    if len(cols) > 6:
                        fold_enrichment = float(cols[6])
                except ValueError:
                    fold_enrichment = 1.0
                
                fold_enrichment = max(0.0, fold_enrichment)
                dnase_peaks_by_chrom[chrom].append((start, end, fold_enrichment))
                total_dnase_peaks += 1
                
            except (ValueError, IndexError):
                continue
    
    # sort peaks
    for chrom in dnase_peaks_by_chrom:
        dnase_peaks_by_chrom[chrom].sort()
    
    log_progress(f"Loaded {total_dnase_peaks:,} DNase peaks")
    
    # extract - will be soon initialized with zeros
    openness_scores = []
    
    for region_idx, (chrom, region_start, region_end) in enumerate(tqdm(regions, desc="Extracting openness")):
        region_length = region_end - region_start
        openness = np.zeros(region_length, dtype=np.float32) #initialize with zeros
        
        if chrom in dnase_peaks_by_chrom: #only look for peaks in this chromosome
            for peak_start, peak_end, fold_enrichment in dnase_peaks_by_chrom[chrom]: #iterate over all peaks on this chromosme
                if peak_end <= region_start or peak_start >= region_end:
                    continue
                
                overlap_start = max(region_start, peak_start) #calculates overlap between window and peak
                overlap_end = min(region_end, peak_end)
                
                if overlap_start < overlap_end:
                    start_idx = overlap_start - region_start
                    end_idx = overlap_end - region_start
                    openness[start_idx:end_idx] = fold_enrichment #fill overlapping bases in openness array with the peak's fold enrichment
        
        openness_scores.append(openness)
    
    log_progress(f"Extracted openness scores for {len(regions):,} regions", start_time)
    return openness_scores


def save_dataset_target_format(output_path, dna_onehot, dnase_scores, labels, genomic_keys, epigenome_id):
    start_time = log_progress("Saving dataset in target format...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    

    # DNA: (N, 1, 4, 1000) - already in this format from extract_sequences_as_onehot
    
    # Labels: (N, 1, 7) - need to add extra dimension
    if labels.ndim == 2:
        labels = labels[:, np.newaxis, :]  
    
    # DNase: Should be (N, 1, 1, 1000) - need to add TWO extra dimensions
    dnase_array = np.array(dnase_scores, dtype=np.float32)
    if dnase_array.ndim == 2:  # If shape is (N, 1000)
        dnase_array = dnase_array[:, np.newaxis, np.newaxis, :]  # Make it (N, 1, 1, 1000)
    
    
    np.savez_compressed(
        output_path,
        dna=dna_onehot.astype(np.float64),     
        dnase=dnase_array.astype(np.float64),   
        label=labels.astype(np.int64),         
        keys=genomic_keys
    )
    
    # print statistics
    file_size_mb = os.path.getsize(output_path) / (1024*1024)
    
    print(f"\n{'='*50}")
    print(f"DATASET SAVED: {os.path.basename(output_path)}")
    print(f"{'='*50}")
    print(f"Epigenome: {epigenome_id}")
    print(f"Total samples: {len(dna_onehot):,}")
    print(f"\nData shapes:")
    print(f"  dna: {dna_onehot.shape}")
    print(f"  dnase: {dnase_array.shape}")
    print(f"  label: {labels.shape}")
    print(f"  keys: {genomic_keys.shape}")
    
    # Per-marker statistics (labels are shape [N, 1, 7] so we squeeze)
    labels_squeezed = labels.squeeze()
    marker_counts = labels_squeezed.sum(axis=0) #sum marker counts
    print(f"\nPer-marker positive counts:")
    for marker, count in zip(config.ALL_MARKERS, marker_counts):
        percentage = (count / len(labels_squeezed)) * 100 if len(labels_squeezed) > 0 else 0
        print(f"  {marker}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nFile size: {file_size_mb:.1f} MB")
    print(f"Output path: {output_path}")
    print(f"{'='*50}")
    
    log_progress(f"Dataset saved successfully", start_time)
    return output_path


def process_epigenome(epigenome_id, logger=None):
   
    overall_start = time.time()
    
    try:
        if logger:
            logger.info(f"Starting {epigenome_id}")
        
        print(f"\n{'='*60}")
        print(f"Processing: {epigenome_id}")
        print(f"{'='*60}")
        
        # check if exits
        output_path = config.get_output_path(epigenome_id)
        if config.SKIP_EXISTING and os.path.exists(output_path):
            print(f"Dataset already exists: {output_path}")
            if logger:
                logger.info(f"Skipped {epigenome_id} - already exists")
            return output_path, True
        
        # load all peaks for this epigenome
        all_peaks = load_all_peaks_for_epigenome(epigenome_id)
        
        # scan specific epigenome for all markers
        windows_200bp, labels = scan_genome_multilabel(epigenome_id, all_peaks)
        
        if windows_200bp is None:
            raise ValueError(f"Failed to process {epigenome_id} - insufficient data for at least one marker")
        
        # expand to 1000bp
        windows_1000bp = expand_regions_to_1000bp(windows_200bp)
        
        # extract sequences as one-hot encoded
        dna_onehot = extract_sequences_as_onehot(windows_1000bp)
        
        # extract DNase scores
        dnase_scores = extract_dnase_scores_batch(epigenome_id, windows_1000bp)
        
        # generate genomic keys with underscore format
        genomic_keys = []
        for chrom, start, end in windows_1000bp:
            key = f"{chrom}_{start}_{end}"
            genomic_keys.append(key)
        genomic_keys = np.array(genomic_keys, dtype='U50')
        
        # shuffle data
        np.random.seed(config.RANDOM_SEED)
        indices = np.random.permutation(len(dna_onehot))
        
        dna_onehot = dna_onehot[indices]
        dnase_scores = [dnase_scores[i] for i in indices]
        labels = labels[indices]
        genomic_keys = genomic_keys[indices]
        
        # save dataset
        save_dataset_target_format(
            output_path, dna_onehot, dnase_scores, labels, genomic_keys, epigenome_id
        )
        
        duration = time.time() - overall_start
        log_progress(f"SUCCESS: Finished {epigenome_id} in {duration:.2f} seconds")
        if logger:
            logger.info(f"SUCCESS: Completed {epigenome_id} in {duration:.2f} seconds")
        
        return output_path, True
        
    except Exception as e:
        error_msg = f"Error processing {epigenome_id}: {e}"
        print(error_msg)
        print(traceback.format_exc())
        if logger:
            logger.error(error_msg)
            logger.error(traceback.format_exc())
        return None, False

#incase I want to do multiple epigenomes at once
def run_batch_processing(epigenome_ids=None, logger=None):
    if logger is None:
        logger = setup_logging()
    
    if epigenome_ids is None:
        epigenome_ids = config.VALID_EPIGENOMES
    
    logger.info(f"Starting batch processing of {len(epigenome_ids)} epigenomes")
    
    successful = []
    failed = []
    
    start_time = time.time()
    
    for i, epigenome_id in enumerate(epigenome_ids, 1):
        try:
            logger.info(f"Processing {i}/{len(epigenome_ids)}: {epigenome_id}")
            
            output_path, success = process_epigenome(epigenome_id, logger)
            
            if success:
                successful.append((epigenome_id, output_path))
                logger.info(f"SUCCESS: {epigenome_id}")
            else:
                failed.append((epigenome_id, "Processing failed"))
                logger.error(f"FAILED: {epigenome_id}")
                
                if not config.CONTINUE_ON_ERROR:
                    logger.error("Stopping batch processing due to error")
                    break
        
        except Exception as e:
            error_msg = f"Unexpected error with {epigenome_id}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            failed.append((epigenome_id, str(e)))
            
            if not config.CONTINUE_ON_ERROR:
                logger.error("Stopping batch processing due to unexpected error")
                break
        
        # Progress update
        elapsed = time.time() - start_time
        remaining = len(epigenome_ids) - i
        if i > 0:
            avg_time = elapsed / i
            eta = avg_time * remaining
            logger.info(f"Progress: {i}/{len(epigenome_ids)} completed. "
                       f"ETA: {eta/3600:.1f} hours. "
                       f"Success: {len(successful)}, Failed: {len(failed)}")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    return successful, failed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epigenome', required=True, help='Epigenome ID (e.g., E005)')
    parser.add_argument('--final_window_size', type=int, default=1000, help='Final window size')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode (chr22 only)')
    
    args = parser.parse_args()
    
    # Update config based on arguments
    config.FINAL_WINDOW_SIZE = args.final_window_size
    config.TEST_MODE = args.test_mode
    
    process_epigenome(args.epigenome)