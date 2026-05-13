from __future__ import annotations
import numpy as np
from engine.tensor import Tensor
from typing import List


class Module:
    """Base class for all neural network modules in the custom engine."""

    def __init__(self) -> None:
        self.training: bool = True
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, *args, **kwargs):
        """Dispatches the call to the forward method, enabling model(x) syntax."""
        return self.forward(*args, **kwargs)
    
    def zero_grad(self) -> None:
        for p in self.parameters():
            if p.grad is not None:
                # Use [:] to zero out the existing array in-place
                p.grad[:] = 0

    def parameters(self) -> List[Tensor]:
        """
        Automatically finds all Tensor and Module attributes assigned to this class.
        This mimics PyTorch's automatic parameter registration.
        """
        params = []
        # Look through everything stored in self (layers, tensors, etc.)
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, Tensor):
                params.append(attr_value)
            elif isinstance(attr_value, Module):
                # If it's a module (like a Linear layer), ask it for its parameters
                params.extend(attr_value.parameters())
            elif isinstance(attr_value, list) or isinstance(attr_value, tuple):
                # Handle cases like Sequential where modules are in a list
                for item in attr_value:
                    if isinstance(item, (Module, Tensor)):
                        params.extend(item.parameters() if isinstance(item, Module) else [item])
        return params
        

    def save(self, filepath):
        """Save module parameters to disk using NumPy's compressed format."""
        # Extract the raw numpy arrays and map them to their index
        state = {f"param_{i}": p.data for i, p in enumerate(self.parameters())}
        np.savez(filepath, **state)
        print(f"Model saved successfully to {filepath}")

    def load(self, filepath):
        """Loads parameters from disk into the module with strict shape validation."""
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Could not find model file at {filepath}")
            
        state = np.load(filepath)
        params = self.parameters()
        
        # 1. Check if the total number of parameter groups matches
        if len(state.files) != len(params):
            raise RuntimeError(f"Architecture mismatch! File has {len(state.files)} params, but model expects {len(params)}.")
            
        for i, p in enumerate(params):
            loaded_data = state[f"param_{i}"]
            
            # 2. FIX: Check if the SHAPES match exactly
            if loaded_data.shape != p.data.shape:
                raise RuntimeError(
                    f"Shape mismatch at parameter index {i}! "
                    f"Checkpoint has {loaded_data.shape}, but model expects {p.data.shape}. "
                    "Ensure your architecture matches the saved model."
                )
            
            p.data = loaded_data
        print(f"Model loaded successfully from {filepath}")


class Linear(Module):
    """Fully connected layer implementing Y = X @ W + b."""

    def __init__(self, in_features: int, out_features: int):
        """Initialize weight and bias tensors with uniform random values."""
        super().__init__()
        # Kaiming-style initialization bounds
        bound = 1 / np.sqrt(in_features)

        # Initialize weights and biases as Tensors
        weight_data = np.random.uniform(-bound, bound, (in_features, out_features))
        bias_data = np.random.uniform(-bound, bound, (out_features,))

        self.weight = Tensor(weight_data)
        self.bias = Tensor(bias_data)
        

    def forward(self, x: Tensor) -> Tensor:
        """Compute the affine transform for an input batch."""
        # Ensure x is a Tensor
        x = x if isinstance(x, Tensor) else Tensor(x)
        # Y = X @ W + b
        return x @ self.weight + self.bias
    
    def __repr__(self) -> str:
        # e.g., Linear(in=784, out=128)
        return f"Linear(in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]})"


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        # Ensure input is a Tensor (defensive programming)
        x = x if isinstance(x, Tensor) else Tensor(x)
        
        # Delegate to the base Tensor implementation
        return x.relu()
    
    def __repr__(self) -> str:
        return "ReLU()"


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        # Pass the input through each layer in order
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __repr__(self) -> str:
        # Create an indented list of all sub-layers
        layers_str = ",\n  ".join([repr(layer) for layer in self.layers])
        return f"Sequential(\n  {layers_str}\n)"


class CrossEntropyLoss:
    """
    Computes the fused Softmax and Cross-Entropy loss.
    This class is stateless and does not need to inherit from Module.
    """
    def __call__(self, logits, targets):
        # We use __call__ so you can still use it like: criterion(logits, targets)
        return self.forward(logits, targets)

    def forward(self, logits, targets):
        if isinstance(targets, Tensor):
            targets = targets.data
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)
            
        N = logits.shape[0]
        
        # Softmax numerical stability trick: subtract max logit
        logits_max = np.max(logits.data, axis=1, keepdims=True)
        exp_logits = np.exp(logits.data - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Cross-entropy calculation
        log_probs = -np.log(probs[np.arange(N), targets] + 1e-10)
        loss = np.sum(log_probs) / N
        
        out = Tensor(loss, _children=(logits,), _op="CrossEntropy")

        def _backward():
            # The derivative of Cross-Entropy + Softmax is (probs - targets)
            d_logits = probs.copy()
            d_logits[np.arange(N), targets] -= 1
            d_logits /= N
            logits.grad += d_logits * out.grad

        out._backward = _backward
        return out