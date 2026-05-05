import h5py
import numpy as np

try:
    with h5py.File('./data/MNIST.h5', 'r') as f:
        print("Keys in MNIST.h5:", list(f.keys()))
        for key in f.keys():
            data = f[key]
            print(f"\n{key}:")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  First element type: {type(data[0])}")
            if len(data) > 0:
                print(f"  Sample [0]: {data[0]}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
