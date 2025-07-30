#!/usr/bin/env python3
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_npz.py training_results.npz")
    sys.exit(1)

file_path = sys.argv[1]

print(f"\n📁 Loading: {file_path}\n")

data = np.load(file_path, allow_pickle=True)

print("🔑 Keys in file:")
for key in data.keys():
    print(f"  - {key}")

print("\n📊 Details:")
for key in data.keys():
    item = data[key]
    if hasattr(item, 'shape'):
        print(f"\n{key}: shape={item.shape}, dtype={item.dtype}")
        if item.size < 10:
            print(f"  Values: {item}")
    elif hasattr(item, 'item'):
        item_val = item.item()
        if isinstance(item_val, dict):
            print(f"\n{key}: dict with {len(item_val)} keys")
            for k, v in item_val.items():
                if isinstance(v, (list, np.ndarray)):
                    print(f"  - {k}: length={len(v)}")
                    if len(v) > 0 and len(v) < 5:
                        print(f"    Values: {v}")
                elif isinstance(v, dict):
                    print(f"  - {k}: dict with {len(v)} keys: {list(v.keys())[:5]}...")
                else:
                    print(f"  - {k}: {v}")
        else:
            print(f"\n{key}: {item_val}")
    else:
        print(f"\n{key}: {type(item)}")

# Quick metrics summary
if 'history' in data:
    history = data['history'].item()
    if 'train_loss' in history:
        print(f"\n📈 Training Summary:")
        print(f"  - Epochs trained: {len(history['train_loss'])}")
        print(f"  - Final train loss: {history['train_loss'][-1]:.4f}")
        if 'valid_auPRC' in history:
            print(f"  - Best valid auPRC: {max(history['valid_auPRC']):.4f}")

if 'test_auPRC' in data:
    test_auPRC = data['test_auPRC'].item()
    mean_auPRC = np.mean(list(test_auPRC.values()))
    print(f"\n🎯 Test Performance:")
    print(f"  - Mean auPRC: {mean_auPRC:.4f}")
    if 'test_auROC' in data:
        test_auROC = data['test_auROC'].item()
        mean_auROC = np.mean(list(test_auROC.values()))
        print(f"  - Mean auROC: {mean_auROC:.4f}")

print("\n✅ Done!")