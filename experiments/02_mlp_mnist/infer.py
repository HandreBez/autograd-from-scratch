import os
import numpy as np
from engine import Linear, ReLU, Sequential, Tensor
from dataset import get_mnist_data


def print_ascii_digit(image_array):
    """Render a flattened 784-pixel vector as a 28x28 ASCII image."""
    image_2d = image_array.reshape(28, 28)
    print("\n--- Model's Vision ---")
    for row in image_2d:
        line = "".join(
            ["@@" if pixel > 0.5 else ".." if pixel > 0.1 else "  " for pixel in row]
        )
        print(line)
    print("----------------------\n")


def run_inference():
    """Load trained weights and run inference on one random MNIST test sample."""
    # 1. Rebuild the exact architecture (weights will initialize randomly)
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10),
    )

    print(model)

    # 2. Load the trained weights from disk
    model_path = "models/mnist_mlp.npz"
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Did you run train.py first?")
        return

    model.load(model_path)

    # 3. Load the test dataset
    _, _, X_test, y_test = get_mnist_data()

    # 4. Pick a random image from the test set
    random_index = np.random.randint(0, len(X_test))
    image = X_test[random_index]
    true_label = y_test[random_index]

    # Print the image to the terminal
    print_ascii_digit(image)

    # 5. Make a prediction!
    # We must wrap the image in a Tensor and add a batch dimension: shape (1, 784)
    x_input = Tensor(image.reshape(1, -1))

    # Forward pass
    logits = model(x_input)

    # The predicted class is the index with the highest score
    predicted_label = np.argmax(logits.data)

    print(f"Model Predicts: [{predicted_label}]")
    print(f"True Label:     [{true_label}]")

    if predicted_label == true_label:
        print("\nSuccess! The custom engine works perfectly.")
    else:
        print("\nOops! Even 97% accurate models make mistakes sometimes.")


if __name__ == "__main__":
    run_inference()