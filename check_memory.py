#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse
import resource

def get_memory_usage():
    """Get current memory usage in MB using resource module"""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On Linux, ru_maxrss is in KB, on macOS it's in bytes
        if sys.platform == 'darwin':
            return usage.ru_maxrss / (1024**2)  # Convert bytes to MB
        else:
            return usage.ru_maxrss / 1024  # Convert KB to MB
    except:
        return 0

def check_system_memory():
    """Check system memory using /proc/meminfo (Linux only)"""
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        lines = meminfo.split('\n')
        memtotal = memfree = memavailable = 0
        
        for line in lines:
            if line.startswith('MemTotal:'):
                memtotal = int(line.split()[1]) / 1024  # Convert KB to MB
            elif line.startswith('MemFree:'):
                memfree = int(line.split()[1]) / 1024
            elif line.startswith('MemAvailable:'):
                memavailable = int(line.split()[1]) / 1024
        
        return memtotal, memfree, memavailable
    except:
        return None, None, None

def check_memory_usage(data_file):
    """Check memory usage and data characteristics"""
    
    print("="*60)
    print("MEMORY USAGE DEBUG SCRIPT (No psutil)")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"ERROR: File {data_file} does not exist!")
        return
    
    # Get file size
    file_size_bytes = os.path.getsize(data_file)
    file_size_mb = file_size_bytes / (1024**2)
    
    print(f"File path: {data_file}")
    print(f"File size on disk: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    
    # Check system memory (Linux only)
    memtotal, memfree, memavailable = check_system_memory()
    if memtotal:
        print(f"System total memory: {memtotal/1024:.1f} GB")
        print(f"Available memory: {memavailable/1024:.1f} GB")
        print(f"Free memory: {memfree/1024:.1f} GB")
    else:
        print("Could not read system memory info")
    
    print("\n" + "-"*50)
    print("LOADING DATA WITH MMAP...")
    print("-"*50)
    
    # Check memory before loading
    mem_before = get_memory_usage()
    print(f"Process memory before loading: {mem_before:.1f} MB")
    
    try:
        # Load with memory mapping
        with np.load(data_file, mmap_mode='r') as f:
            print(f"\nData arrays found:")
            print(f"  Keys: {list(f.keys())}")
            
            # Check memory after opening file handle
            mem_after_open = get_memory_usage()
            print(f"Process memory after opening file: {mem_after_open:.1f} MB")
            print(f"Memory increase: {mem_after_open - mem_before:.1f} MB")
            
            print(f"\nData shapes and types:")
            total_uncompressed = 0
            
            for key in f.keys():
                data = f[key]
                shape = data.shape
                dtype = data.dtype
                
                # Calculate size if loaded into memory
                size_bytes = np.prod(shape) * dtype.itemsize
                size_mb = size_bytes / (1024**2)
                total_uncompressed += size_bytes
                
                print(f"  {key:8s}: shape={shape}, dtype={dtype}")
                print(f"  {' '*8}  uncompressed size: {size_mb:.1f} MB")
            
            total_uncompressed_mb = total_uncompressed / (1024**2)
            total_uncompressed_gb = total_uncompressed / (1024**3)
            
            print(f"\nTOTAL uncompressed size: {total_uncompressed_mb:.1f} MB ({total_uncompressed_gb:.2f} GB)")
            print(f"Compression ratio: {total_uncompressed_mb / file_size_mb:.1f}:1")
            
            # Test loading a small sample into memory
            print(f"\n" + "-"*50)
            print("TESTING SAMPLE DATA LOADING...")
            print("-"*50)
            
            # Load keys first
            print("Loading keys...")
            keys = f['keys'][:]
            mem_after_keys = get_memory_usage()
            print(f"Memory after loading keys: {mem_after_keys:.1f} MB (+{mem_after_keys - mem_after_open:.1f} MB)")
            
            # Try loading first few samples
            print(f"Total samples: {len(keys)}")
            if len(keys) > 0:
                print("Loading first sample...")
                
                # Load one sample to check memory usage
                sample_dna = np.array(f['dna'][0])
                sample_dnase = np.array(f['dnase'][0])
                sample_label = np.array(f['label'][0])
                
                mem_after_sample = get_memory_usage()
                print(f"Memory after loading 1 sample: {mem_after_sample:.1f} MB (+{mem_after_sample - mem_after_keys:.1f} MB)")
                
                print(f"Sample shapes:")
                print(f"  DNA: {sample_dna.shape}")
                print(f"  DNase: {sample_dnase.shape}")
                print(f"  Label: {sample_label.shape}")
                
                # Estimate memory per batch
                sample_size = (sample_dna.nbytes + sample_dnase.nbytes + sample_label.nbytes) / (1024**2)
                print(f"Memory per sample: {sample_size:.2f} MB")
                
                print(f"\nEstimated memory usage for different batch sizes:")
                for batch_size in [1, 4, 8, 16, 24, 32]:
                    batch_memory = sample_size * batch_size
                    print(f"  Batch size {batch_size:2d}: ~{batch_memory:.1f} MB")
                    
                    # Check if this might be problematic
                    if memavailable and batch_memory > memavailable * 0.5:
                        print(f"    ⚠️  WARNING: This might exceed available memory!")
                    elif batch_memory > 1000:  # > 1GB
                        print(f"    ⚠️  WARNING: Very large batch!")
                
                # Test loading multiple samples
                print(f"\nTesting larger sample loads...")
                test_sizes = [10, 100] if len(keys) >= 100 else [min(10, len(keys))]
                
                for test_size in test_sizes:
                    if test_size <= len(keys):
                        print(f"Loading {test_size} samples...")
                        mem_before_test = get_memory_usage()
                        
                        # Load multiple samples
                        test_dna = np.array(f['dna'][:test_size])
                        test_dnase = np.array(f['dnase'][:test_size])
                        test_labels = np.array(f['label'][:test_size])
                        
                        mem_after_test = get_memory_usage()
                        memory_increase = mem_after_test - mem_before_test
                        
                        print(f"  Memory increase for {test_size} samples: {memory_increase:.1f} MB")
                        print(f"  Memory per sample: {memory_increase/test_size:.2f} MB")
                        
                        # Clean up
                        del test_dna, test_dnase, test_labels
    
    except Exception as e:
        print(f"ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
    
    # Final memory check
    final_memory = get_memory_usage()
    print(f"\nFinal process memory: {final_memory:.1f} MB")
    print(f"Total memory increase: {final_memory - mem_before:.1f} MB")
    
    # Check SLURM limits if available
    print(f"\n" + "-"*50)
    print("CHECKING SLURM LIMITS...")
    print("-"*50)
    
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        print(f"SLURM Job ID: {slurm_job_id}")
        try:
            import subprocess
            result = subprocess.run(['scontrol', 'show', 'job', slurm_job_id], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Memory' in line or 'Mem' in line:
                        print(f"  {line.strip()}")
        except:
            print("Could not get SLURM job info")
    else:
        print("Not running under SLURM")
    
    print("\n" + "="*60)
    print("MEMORY CHECK COMPLETE")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Check memory usage of data file")
    parser.add_argument('data_file', type=str, help='Path to the .npz data file')
    args = parser.parse_args()
    
    check_memory_usage(args.data_file)

if __name__ == "__main__":
    main()