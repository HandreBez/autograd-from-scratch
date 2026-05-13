import numpy as np
from engine import Sequential, Linear, ReLU
from dataset import get_mnist_data
from engine.utils import calculate_accuracy # The star of the show

def evaluate_model():
    # 1. Setup Architecture
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10)
    )

    print(model)
    
    # 2. Load Weights
    try:
        model.load("models/mnist_mlp.npz")
    except FileNotFoundError:
        print("Error: Could not find weights. Run train.py first.")
        return
        
    # 3. Load Data
    _, _, X_test, y_test = get_mnist_data()
    
    # 4. Use the Utility (The "DRY" win)
    print(f"\nEvaluating on {len(X_test)} unseen test images...")
    accuracy = calculate_accuracy(model, X_test, y_test)
    
    print("-" * 30)
    print(f"🏆 Final Test Accuracy: {accuracy * 100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    evaluate_model()

