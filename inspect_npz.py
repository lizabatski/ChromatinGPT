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

    for key in data.files:
        array = data[key]
        print(f"Key: {key}")
        print(f"  Shape: {array.shape}")
        print(f"  Dtype: {array.dtype}")

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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_npz.py <path_to_npz_file> [num_examples]")
    else:
        file_path = sys.argv[1]
        num_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        inspect_npz(file_path, num_examples)
