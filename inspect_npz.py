import numpy as np
import sys
import os

def inspect_npz(file_path, num_examples=3):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return

    print(f"\nContents of {file_path}:")
    print("=" * 50)

    zero_vector_count = None  # Initialize for labels
    total_labels = None

    for key in data.files:
        array = data[key]
        print(f"Key: {key}")
        print(f"  Shape: {array.shape}")
        print(f"  Dtype: {array.dtype}")

        # Count all-zero vectors if this is labels
        if key.lower() in ['label', 'labels']:
            total_labels = array.shape[0]
            # A label is all-zero if sum across axis=1 is 0
            squeezed_array = np.squeeze(array)
            if squeezed_array.ndim == 2:
                zero_vector_count = np.sum(np.all(squeezed_array == 0, axis=1))
                print(f"  ⚠️ Zero-only labels count: {zero_vector_count}/{total_labels}")
            else:
                print("  ⚠️ Labels are not 2D after squeezing—skipping zero vector check.")

        # Print first few examples
        print(f"  First {num_examples} example(s):")
        if array.ndim == 1:
            print(f"    {array[:num_examples]}")
        elif array.ndim >= 2:
            for i in range(min(num_examples, array.shape[0])):
                print(f"    Example {i}: {array[i]}")
        else:
            print(f"    {array}")

        print("-" * 50)

    print("Inspection complete.")
    if zero_vector_count is not None:
        print(f"\n✅ Total zero-only label vectors: {zero_vector_count}/{total_labels}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_npz.py <path_to_npz_file> [num_examples]")
    else:
        file_path = sys.argv[1]
        num_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        inspect_npz(file_path, num_examples)
