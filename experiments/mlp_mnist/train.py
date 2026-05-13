import os
import time
import numpy as np
from engine.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from engine.optim import Adam
from engine.tensor import Tensor
from dataset import get_mnist_data
from engine.utils import calculate_accuracy, setup_logger

def train_mnist():
    """Train the scratch MLP on MNIST and save the final model checkpoint."""
    
    # 1. Initialize Logger
    # We do this first so every step of the process is recorded
    logger = setup_logger("MNIST_Train")
    logger.info("Starting new training session...")

    # 2. Load Data
    X_train, y_train, X_test, y_test = get_mnist_data()

    # 3. Define Hyperparameters
    EPOCHS = 5
    BATCH_SIZE = 128
    LEARNING_RATE = 0.005
    
    logger.info(f"Hyperparameters: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, LR={LEARNING_RATE}")

    # 4. Initialize Model, Loss, and Optimizer
    logger.info("Initializing architecture: 784 -> 128 -> 64 -> 10")
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10),
    )

    print(model)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    logger.info("Commencing training loop...")
    num_samples = len(X_train)

    for epoch in range(EPOCHS):
        start_time = time.time()

        # Shuffle data at the start of each epoch
        indices = np.random.permutation(num_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        total_loss = 0.0
        batches = 0

        # Iterate over mini-batches
        for i in range(0, num_samples, BATCH_SIZE):
            X_batch = Tensor(X_train_shuffled[i : i + BATCH_SIZE])
            y_batch = y_train_shuffled[i : i + BATCH_SIZE]

            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            # Backward pass & Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data
            batches += 1

        # Evaluate at the end of the epoch
        avg_loss = total_loss / batches
        test_accuracy = calculate_accuracy(model, X_test, y_test)
        epoch_time = time.time() - start_time

        logger.info(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
            f"Test Acc: {test_accuracy * 100:.2f}% | Time: {epoch_time:.1f}s"
        )

    # 6. Save Weights
    logger.info("Training complete. Exporting weights...")
    os.makedirs("models", exist_ok=True)
    model.save("models/mnist_mlp.npz")
    logger.info("Weights saved to models/mnist_mlp.npz. Session concluded.")


if __name__ == "__main__":
    train_mnist()