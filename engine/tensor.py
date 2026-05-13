from __future__ import annotations
import numpy as np
from typing import Optional, Iterable, Union, Tuple


class Tensor:
    """A minimal NumPy-backed tensor with reverse-mode autodiff support."""

    _grad_tracking_enabled = True

    def __init__(self, data: Union[np.ndarray, list, float], _children: Tuple[Tensor, ...] = (), _op: str = ""):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.grad = np.zeros_like(self.data)
        
        # ONLY track history if global flag is ON
        self._prev = set(_children) if Tensor._grad_tracking_enabled else set()
        self._op = _op if Tensor._grad_tracking_enabled else ""
        self._backward = lambda: None

    @property
    def shape(self):
        """Helper property to easily access the tensor's shape."""
        return self.data.shape

    def __repr__(self):
        """Return a concise debug representation with shape and op label."""
        return f"Tensor(shape={self.shape}, op='{self._op}')"

    def __add__(self, other: Union[Tensor, float]) -> Tensor:
        """Elementwise addition with broadcasting-aware gradient propagation."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            # Grab the global gradient
            grad_self = out.grad
            grad_other = out.grad

            # Un-broadcast self (if self was smaller than out)
            ndims_added_self = grad_self.ndim - self.data.ndim
            for _ in range(ndims_added_self):
                grad_self = grad_self.sum(axis=0)
            for i, dim in enumerate(self.shape):
                if dim == 1:
                    grad_self = grad_self.sum(axis=i, keepdims=True)

            # Un-broadcast other (if other was smaller than out)
            ndims_added_other = grad_other.ndim - other.data.ndim
            for _ in range(ndims_added_other):
                grad_other = grad_other.sum(axis=0)
            for i, dim in enumerate(other.shape):
                if dim == 1:
                    grad_other = grad_other.sum(axis=i, keepdims=True)

            # Accumulate gradients
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    

    __radd__ = __add__

    def __matmul__(self, other):
        """Matrix multiplication with analytical gradients for both operands."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        # Forward pass using matmul
        out = Tensor(np.matmul(self.data, other.data), _children=(self, other), _op="@")

        def _backward():
            # self.grad shape must be (M x N)
            # out.grad (M x P) @ other.data.T (P x N)
            self.grad += np.matmul(out.grad, other.data.T)

            # other.grad shape must be (N x P)
            # self.data.T (N x M) @ out.grad (M x P)
            other.grad += np.matmul(self.data.T, out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        """Elementwise multiplication with broadcasting-aware backpropagation."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            # Chain rule: incoming gradient * the OTHER parent's data
            grad_self = out.grad * other.data
            grad_other = out.grad * self.data

            # Un-broadcast self
            ndims_added_self = grad_self.ndim - self.data.ndim
            for _ in range(ndims_added_self):
                grad_self = grad_self.sum(axis=0)
            for i, dim in enumerate(self.shape):
                if dim == 1:
                    grad_self = grad_self.sum(axis=i, keepdims=True)

            # Un-broadcast other
            ndims_added_other = grad_other.ndim - other.data.ndim
            for _ in range(ndims_added_other):
                grad_other = grad_other.sum(axis=0)
            for i, dim in enumerate(other.shape):
                if dim == 1:
                    grad_other = grad_other.sum(axis=i, keepdims=True)

            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __rmul__(self, other):
        # Fallback for when the scalar is on the left side: e.g., 2.0 * tensor
        return self * other

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        """
        Reduces the tensor by summing along the specified axis.
        Supports multi-dimensional reduction and keepdims for broadcasting.
        """
        # 1. Forward Pass
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, _children=(self,), _op="sum")

        def _backward():
            # 2. Backward Pass
            # We need to 'broadcast' the gradient of the output back to the shape of the input.
            # If keepdims=False, the summed dimensions were squeezed out, so we must 
            # restore them to align with self.data for the addition.
            grad_output = out.grad
            
            if axis is not None and not keepdims:
                # Expand grad_output to match the original shape of self.data
                # Example: (5, 10) summed over axis 1 becomes (5,). 
                # We need to reshape (5,) back to (5, 1) to broadcast to (5, 10).
                shape = list(self.data.shape)
                axes = [axis] if isinstance(axis, int) else axis
                for ax in axes:
                    shape[ax] = 1
                grad_output = grad_output.reshape(shape)

            # The derivative of sum(x) is 1, so the local gradient is just 1.
            # We propagate out.grad across the entire summed dimension.
            self.grad += np.ones_like(self.data) * grad_output

        out._backward = _backward
        return out

    def __pow__(self, other):
        """Raise tensor elements to an integer/float power."""
        # For simplicity, we will only support raising a tensor to a scalar (int or float)
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        out = Tensor(self.data ** other, _children=(self,), _op=f"**{other}")

        def _backward():
            # Chain rule: k * x^(k-1) * out.grad
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        """Elementwise subtraction with broadcasting-aware gradient propagation."""
        # Subtraction is just adding a negative
        other = other if isinstance(other, Tensor) else Tensor(other)
        # We can cheat and use __add__ and scalar multiplication here
        # But first we need __mul__ for scalars. Let's write the raw sub for safety:
        out = Tensor(self.data - other.data, _children=(self, other), _op="-")

        def _backward():
            # Shape-matching logic similar to __add__, but inverted for 'other'
            grad_self = out.grad
            grad_other = -out.grad

            # Un-broadcast self
            ndims_added_self = grad_self.ndim - self.data.ndim
            for _ in range(ndims_added_self):
                grad_self = grad_self.sum(axis=0)
            for i, dim in enumerate(self.shape):
                if dim == 1:
                    grad_self = grad_self.sum(axis=i, keepdims=True)

            # Un-broadcast other
            ndims_added_other = grad_other.ndim - other.data.ndim
            for _ in range(ndims_added_other):
                grad_other = grad_other.sum(axis=0)
            for i, dim in enumerate(other.shape):
                if dim == 1:
                    grad_other = grad_other.sum(axis=i, keepdims=True)

            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def backward(self) -> None:

        if self.data.size > 1:
            raise RuntimeError(
                f"Grad can be implicitly created only for scalar outputs. "
                f"Current Tensor has shape {self.data.shape}. "
                f"Call .sum() or .mean() before .backward()."
            )

        # 1. Topological sort (Iterative to avoid RecursionError)
        topo = []
        visited = set()
        stack = [self]
        
        # We use a manual stack for a post-order traversal
        # This is the industry-standard way to do a DFS without recursion
        while stack:
            v = stack[-1]
            # Find any previous node that hasn't been visited yet
            unvisited_children = [p for p in v._prev if p not in visited]
            
            if unvisited_children:
                # Add the first unvisited parent to the stack and keep diving
                stack.append(unvisited_children[0])
            else:
                # All parents are done, we can now add this node to topo
                visited.add(v)
                topo.append(v)
                stack.pop()

        # 2. Set the gradient of the root node to 1.0
        self.grad = np.ones_like(self.data)
        
        # 3. Go through the list in reverse and call _backward()
        for node in reversed(topo):
            node._backward()
        

    def relu(self):
        """Apply ReLU activation elementwise and register its backward rule."""
        # Forward pass: np.maximum compares each element to 0
        out = Tensor(np.maximum(0, self.data), _children=(self,), _op="ReLU")

        # Backward closure
        def _backward():
            # (out.data > 0) creates a boolean mask of True/False
            # We multiply this mask by the incoming gradient.
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out