import logging
import sys
import numpy as np
from engine import Tensor

def calculate_accuracy(model, X, y, batch_size=256):
    correct = 0
    total = len(X)
    
    # Use our new context manager to save memory!
    with no_grad():
        for i in range(0, total, batch_size):
            X_batch = Tensor(X[i:i+batch_size])
            y_batch = y[i:i+batch_size]
            
            logits = model(X_batch)
            preds = np.argmax(logits.data, axis=1)
            correct += np.sum(preds == y_batch)
            
    return correct / total

def grad_check(model, x_input: Tensor, eps: float = 1e-6, tolerance: float = 1e-4) -> bool:
    """
    Compares analytical gradients (autograd) with numerical gradients (finite differences).
    Essential for verifying the mathematical correctness of new layers.
    """
    # 1. Compute Analytical Gradients
    model.zero_grad()
    output = model(x_input)
    # We use sum() to get a scalar for backward
    loss = output.sum()
    loss.backward()
    
    analytical_grad = x_input.grad.copy()
    
    # 2. Compute Numerical Gradients
    numerical_grad = np.zeros_like(x_input.data)
    it = np.nditer(x_input.data, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        ix = it.multi_index
        old_value = x_input.data[ix]
        
        # f(x + h)
        x_input.data[ix] = old_value + eps
        plus_loss = model(x_input).sum().data
        
        # f(x - h)
        x_input.data[ix] = old_value - eps
        minus_loss = model(x_input).sum().data
        
        # (f(x+h) - f(x-h)) / 2h
        numerical_grad[ix] = (plus_loss - minus_loss) / (2 * eps)
        
        x_input.data[ix] = old_value # Restore original value
        it.iternext()
        
    # 3. Compare via Relative Error
    denom = np.maximum(np.abs(analytical_grad) + np.abs(numerical_grad), 1e-8)
    rel_error = np.max(np.abs(analytical_grad - numerical_grad) / denom)
    
    is_correct = rel_error < tolerance
    return is_correct

class no_grad:
    """
    Context manager to disable gradient tracking. 
    Reduces memory usage and speeds up inference.
    """
    def __enter__(self):
        self.prev_state = Tensor._grad_tracking_enabled
        Tensor._grad_tracking_enabled = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        Tensor._grad_tracking_enabled = self.prev_state

def setup_logger(name: str = "engine"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create a clean format: [TIMESTAMP] LEVEL: Message
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
        
        # Output to console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Output to a file (great for portfolio proof!)
        file_handler = logging.FileHandler("training.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

