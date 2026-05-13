import numpy as np
from sklearn.datasets import fetch_openml


def get_mnist_data():
    """Download, normalize, and return MNIST train/test NumPy arrays."""
    print("Downloading MNIST dataset (this may take a minute)...")
    # fetch_openml downloads the dataset as a Pandas DataFrame by default.
    # We set as_frame=False to get raw NumPy arrays immediately.
    mnist = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)

    # 1. Extract raw data and labels
    X = mnist.data  # Shape: (70000, 784)
    y = mnist.target  # Shape: (70000,) - currently strings ('0'-'9')

    # 2. Convert labels from strings to integers
    y = y.astype(np.int64)

    # 3. Normalize the pixel values to [0.0, 1.0]
    # We use float32 to save memory compared to float64, while maintaining enough precision.
    X = X.astype(np.float32) / 255.0

    # 4. Standard Train/Test Split (First 60k for training, last 10k for testing)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print("Data loaded and normalized successfully!")
    print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")

    return X_train, y_train, X_test, y_test


# Quick test block to verify it works when we run this file directly
if __name__ == "__main__":
    print("Testing data pipeline...")
    X_train, y_train, X_test, y_test = get_mnist_data()
    
    print("\n--- Pipeline Diagnostics ---")
    print(f"Training Features: {X_train.shape} | dtype: {X_train.dtype} | Min: {X_train.min():.1f} | Max: {X_train.max():.1f}")
    print(f"Training Labels:   {y_train.shape} | dtype: {y_train.dtype}")
    print(f"Testing Features:  {X_test.shape}")
    print(f"Testing Labels:    {y_test.shape}")
    print("----------------------------")
    print("Status: READY")