import numpy as np
import json

def inspect_dataset(filepath, n_entries=5):
    """Load and display first n entries of the dataset"""
    
    print(f"Loading dataset: {filepath}")
    data = np.load(filepath, allow_pickle=True)
    
    # Show available keys
    print(f"\nAvailable keys in dataset: {list(data.keys())}")
    
    # Try to load metadata if it exists
    metadata = None
    marker_names = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']
    
    if 'metadata' in data:
        try:
            metadata = json.loads(str(data['metadata']))
            print(f"\nMetadata found:")
            print(f"  Epigenome: {metadata.get('epigenome_id', 'Unknown')}")
            if 'markers' in metadata:
                marker_names = metadata['markers']
        except:
            print("\nCould not parse metadata")
    else:
        print("\nNo metadata in dataset, using default marker names")
    
    # Determine which keys are being used
    if 'sequences' in data:
        seq_key, dnase_key, label_key = 'sequences', 'openness', 'labels'
    elif 'dna' in data:
        seq_key, dnase_key, label_key = 'dna', 'dnase', 'label'
    else:
        # Try to find keys
        print("\nSearching for data keys...")
        possible_seq_keys = ['dna', 'sequences', 'seq', 'DNA']
        possible_dnase_keys = ['dnase', 'openness', 'DNase']
        possible_label_keys = ['label', 'labels', 'target']
        
        seq_key = next((k for k in possible_seq_keys if k in data), None)
        dnase_key = next((k for k in possible_dnase_keys if k in data), None)
        label_key = next((k for k in possible_label_keys if k in data), None)
        
        if not all([seq_key, dnase_key, label_key]):
            print(f"Could not find all required keys. Found: seq={seq_key}, dnase={dnase_key}, label={label_key}")
            return
    
    print(f"\nUsing keys: {seq_key}={seq_key}, dnase={dnase_key}, label={label_key}")
    print(f"Total samples: {len(data[seq_key])}")
    
    # Check data shapes and types
    print(f"\nData info:")
    print(f"  {seq_key} shape: {data[seq_key].shape}, dtype: {data[seq_key].dtype}")
    print(f"  {dnase_key} shape: {data[dnase_key].shape}, dtype: {data[dnase_key].dtype}")
    print(f"  {label_key} shape: {data[label_key].shape}, dtype: {data[label_key].dtype}")
    print(f"  keys shape: {data['keys'].shape}, dtype: {data['keys'].dtype}")
    
    # Show first n entries
    print(f"\n{'='*80}")
    print(f"FIRST {n_entries} ENTRIES:")
    print(f"{'='*80}")
    
    for i in range(min(n_entries, len(data[seq_key]))):
        print(f"\nEntry {i}:")
        
        # Genomic location
        location = data['keys'][i]
        if isinstance(location, np.ndarray):
            location = location.item() if location.size == 1 else str(location)
        print(f"  Location: {location}")
        
        # Sequence - handle different formats
        seq_data = data[seq_key][i]
        
        # Debug first entry
        if i == 0:
            print(f"  [DEBUG] Sequence shape: {seq_data.shape}, dtype: {seq_data.dtype}")
        
        # Convert sequence to string
        seq = decode_sequence(seq_data)
        
        if len(seq) > 100:
            print(f"  Sequence: {seq[:50]}...{seq[-50:]}")
        else:
            print(f"  Sequence: {seq}")
        print(f"  Length: {len(seq)}bp")
        
        # Labels - handle different shapes
        labels = data[label_key][i]
        
        # If labels have extra dimensions, squeeze them
        if isinstance(labels, np.ndarray):
            labels = labels.squeeze()  # Remove dimensions of size 1
        
        print(f"  Labels: {labels}")
        
        # Find active markers
        active_indices = []
        if isinstance(labels, np.ndarray):
            active_indices = np.where(labels == 1)[0].tolist()
        else:
            active_indices = [j for j, val in enumerate(labels) if val == 1]
        
        if active_indices:
            active_markers = [marker_names[j] if j < len(marker_names) else f"Marker_{j}" 
                            for j in active_indices]
            print(f"  Active markers: {', '.join(active_markers)}")
        else:
            print(f"  Active markers: None")
        
        # Openness scores
        openness = data[dnase_key][i]
        if isinstance(openness, np.ndarray) and openness.ndim > 1:
            openness = openness.squeeze()
            
        print(f"  DNase openness: min={np.min(openness):.2f}, max={np.max(openness):.2f}, mean={np.mean(openness):.2f}")
        non_zero = np.sum(openness > 0)
        print(f"  Non-zero openness positions: {non_zero}/{len(openness)}")
        
        print("-" * 80)


def decode_sequence(seq_data):
    """Decode sequence data to string, handling various formats"""
    
    # If it's already a string
    if isinstance(seq_data, str):
        return seq_data
    
    # If it's a numpy array
    if isinstance(seq_data, np.ndarray):
        # Remove extra dimensions
        seq_data = seq_data.squeeze()
        
        # One-hot encoded with shape (4, length) or (length, 4)
        if seq_data.ndim == 2:
            if seq_data.shape[0] == 4:  # (4, length)
                # Transpose to (length, 4)
                seq_data = seq_data.T
            
            if seq_data.shape[1] == 4:  # Now should be (length, 4)
                bases = ['A', 'C', 'G', 'T']
                return ''.join([bases[np.argmax(pos)] for pos in seq_data])
        
        # Character array or string array
        elif seq_data.dtype.kind in ['U', 'S']:  # Unicode or byte string
            return ''.join(seq_data)
        
        # Numeric encoding (0=A, 1=C, 2=G, 3=T)
        elif seq_data.dtype.kind in ['i', 'u']:  # Integer types
            bases = ['A', 'C', 'G', 'T', 'N']
            return ''.join([bases[min(int(x), 4)] for x in seq_data])
        
        # Try generic conversion
        else:
            try:
                return ''.join([str(x) for x in seq_data.flatten()])
            except:
                return f"[Unable to decode: shape={seq_data.shape}, dtype={seq_data.dtype}]"
    
    # Try to iterate and join
    try:
        return ''.join(str(x) for x in seq_data)
    except:
        return f"[Unable to decode sequence of type {type(seq_data)}]"


# Example usage
if __name__ == "__main__":
    # Change this to your dataset path
    dataset_path = "data/data.npz"
    
    # Inspect first 5 entries
    inspect_dataset(dataset_path, n_entries=5)
    
    # If you want to see more entries, uncomment below:
    # inspect_dataset(dataset_path, n_entries=10)