import sys
import os
import torch
import numpy as np
import torch.nn as nn
from engine.tensor import Tensor
from engine.nn import Linear, Sequential, ReLU
from engine.optim import SGD, Adam
import torch.nn.functional as F
from engine.nn import CrossEntropyLoss
from engine.utils import grad_check

def test_tensor_addition():
    """Validate forward and backward behavior for elementwise tensor addition."""
    # 1. Initialize PyTorch tensors
    x_pt = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y_pt = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    # 2. Initialize our Custom Tensors
    x_mine = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y_mine = Tensor([[5.0, 6.0], [7.0, 8.0]])

    # 3. Forward pass
    z_pt = x_pt + y_pt
    z_mine = x_mine + y_mine

    # 4. Backward pass
    z_mine.sum().backward()
    z_pt.sum().backward()

    # 5. Assertions using numpy testing for floating-point tolerance
    np.testing.assert_allclose(z_mine.data, z_pt.detach().numpy(), err_msg="Forward pass failed")
    np.testing.assert_allclose(x_mine.grad, x_pt.grad.numpy(), err_msg="x gradient failed")
    np.testing.assert_allclose(y_mine.grad, y_pt.grad.numpy(), err_msg="y gradient failed")


def test_tensor_matmul():
    """Validate forward and backward behavior for matrix multiplication."""
    # Shapes: (2, 3) @ (3, 2) -> (2, 2)
    x_val = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    y_val = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]

    x_pt = torch.tensor(x_val, requires_grad=True)
    y_pt = torch.tensor(y_val, requires_grad=True)

    x_mine = Tensor(x_val)
    y_mine = Tensor(y_val)

    # Forward
    z_pt = x_pt @ y_pt
    z_mine = x_mine @ y_mine

    # Backward
    z_pt.sum().backward()
    z_mine.sum().backward()

    np.testing.assert_allclose(z_mine.data, z_pt.detach().numpy())
    np.testing.assert_allclose(x_mine.grad, x_pt.grad.numpy())
    np.testing.assert_allclose(y_mine.grad, y_pt.grad.numpy())


def test_deep_graph_backward():
    """Verify gradients across a multi-operation computation graph."""
    # 1. Initialize data
    a_val = [[1.0, 2.0], [3.0, 4.0]]
    b_val = [[5.0, 6.0], [7.0, 8.0]]
    e_val = [[9.0, 10.0], [11.0, 12.0]]

    # 2. PyTorch Graph
    a_pt = torch.tensor(a_val, requires_grad=True)
    b_pt = torch.tensor(b_val, requires_grad=True)
    e_pt = torch.tensor(e_val, requires_grad=True)

    c_pt = a_pt @ b_pt
    d_pt = c_pt + e_pt
    d_pt.sum().backward()  # PyTorch's automated backward

    # 3. Custom Engine Graph
    a_mine = Tensor(a_val)
    b_mine = Tensor(b_val)
    e_mine = Tensor(e_val)

    c_mine = a_mine @ b_mine
    d_mine = c_mine + e_mine
    d_mine.sum().backward()

    # 4. Assertions
    np.testing.assert_allclose(d_mine.data, d_pt.detach().numpy(), err_msg="Forward deep graph failed")
    np.testing.assert_allclose(a_mine.grad, a_pt.grad.numpy(), err_msg="Deep graph A grad failed")
    np.testing.assert_allclose(b_mine.grad, b_pt.grad.numpy(), err_msg="Deep graph B grad failed")
    np.testing.assert_allclose(e_mine.grad, e_pt.grad.numpy(), err_msg="Deep graph E grad failed")


def test_tensor_relu():
    """Check ReLU forward activation and corresponding gradient mask behavior."""
    # 1. Initialize data with positives, negatives, and zero
    x_val = [[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]]

    # 2. PyTorch Graph
    x_pt = torch.tensor(x_val, requires_grad=True)
    z_pt = x_pt.relu()
    z_pt.sum().backward()

    # 3. Custom Engine Graph
    x_mine = Tensor(x_val)
    z_mine = x_mine.relu()
    z_mine.sum().backward()

    # 4. Assertions
    np.testing.assert_allclose(z_mine.data, z_pt.detach().numpy(), err_msg="Forward ReLU failed")
    np.testing.assert_allclose(x_mine.grad, x_pt.grad.numpy(), err_msg="Backward ReLU failed")


def test_broadcasting_addition():
    """Ensure addition with broadcasted bias matches PyTorch gradients."""
    # 1. Initialize data: A matrix and a bias vector
    x_val = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # Shape (2, 3)
    b_val = [10.0, 20.0, 30.0]  # Shape (3,)

    # 2. PyTorch Graph
    x_pt = torch.tensor(x_val, requires_grad=True)
    b_pt = torch.tensor(b_val, requires_grad=True)
    z_pt = x_pt + b_pt
    z_pt.sum().backward()

    # 3. Custom Engine Graph
    x_mine = Tensor(x_val)
    b_mine = Tensor(b_val)
    z_mine = x_mine + b_mine
    z_mine.sum().backward()

    # 4. Assertions
    np.testing.assert_allclose(z_mine.data, z_pt.detach().numpy(), err_msg="Forward broadcast failed")
    np.testing.assert_allclose(x_mine.grad, x_pt.grad.numpy(), err_msg="Matrix gradient failed")
    np.testing.assert_allclose(b_mine.grad, b_pt.grad.numpy(), err_msg="Bias gradient failed")


def test_mse_loss_components():
    """Compare MSE scalar loss and gradients against PyTorch reference."""
    # 1. Predictions and Targets
    y_pred_val = [[0.5, 0.8], [0.1, 0.9]]
    y_true_val = [[1.0, 0.0], [0.0, 1.0]]

    # 2. PyTorch Graph
    y_pred_pt = torch.tensor(y_pred_val, requires_grad=True, dtype=torch.float64)
    y_true_pt = torch.tensor(y_true_val, dtype=torch.float64)

    # Calculate MSE (Sum of squared errors)
    loss_pt = ((y_pred_pt - y_true_pt) ** 2).sum()
    loss_pt.backward()

    # 3. Custom Engine Graph
    y_pred_mine = Tensor(y_pred_val)
    y_true_mine = Tensor(y_true_val)

    loss_mine = ((y_pred_mine - y_true_mine) ** 2).sum()
    loss_mine.backward()

    # 4. Assertions
    np.testing.assert_allclose(loss_mine.data, loss_pt.detach().numpy(), err_msg="Forward MSE failed")
    np.testing.assert_allclose(y_pred_mine.grad, y_pred_pt.grad.numpy(), err_msg="Backward MSE failed")


def test_tensor_multiplication():
    """Validate elementwise multiplication and scalar scaling gradients."""
    # 1. Initialize data
    x_val = [[1.0, 2.0], [3.0, 4.0]]
    y_val = [[-1.0, 0.5], [2.0, -2.0]]

    # 2. PyTorch Graph
    x_pt = torch.tensor(x_val, requires_grad=True)
    y_pt = torch.tensor(y_val, requires_grad=True)

    # Element-wise multiplication, then scalar multiplication
    z_pt = (x_pt * y_pt) * 2.0
    z_pt.sum().backward()

    # 3. Custom Engine Graph
    x_mine = Tensor(x_val)
    y_mine = Tensor(y_val)

    z_mine = (x_mine * y_mine) * 2.0
    z_mine.sum().backward()

    # 4. Assertions
    np.testing.assert_allclose(z_mine.data, z_pt.detach().numpy(), err_msg="Forward multiplication failed")
    np.testing.assert_allclose(x_mine.grad, x_pt.grad.numpy(), err_msg="X grad failed")
    np.testing.assert_allclose(y_mine.grad, y_pt.grad.numpy(), err_msg="Y grad failed")


def test_linear_layer():
    """Validate linear layer forward pass and parameter gradients."""
    # 1. Define dimensions
    batch_size, in_feat, out_feat = 32, 10, 5
    x_val = np.random.randn(batch_size, in_feat)

    # 2. PyTorch Graph
    x_pt = torch.tensor(x_val, requires_grad=True, dtype=torch.float64)
    linear_pt = nn.Linear(in_feat, out_feat)
    linear_pt.double()  # Force PyTorch to use float64 for precision matching

    # 3. Custom Engine Graph
    x_mine = Tensor(x_val)
    linear_mine = Linear(in_feat, out_feat)

    # CRITICAL: Hijack our layer's weights to match PyTorch exactly
    # PyTorch stores weights as (out_features, in_features), so we transpose them
    linear_mine.weight.data = linear_pt.weight.detach().numpy().T
    linear_mine.bias.data = linear_pt.bias.detach().numpy()

    # 4. Forward Pass
    y_pt = linear_pt(x_pt)
    y_mine = linear_mine(x_mine)

    # 5. Backward Pass (using MSE against dummy targets)
    y_target_val = np.random.randn(batch_size, out_feat)
    y_target_pt = torch.tensor(y_target_val, dtype=torch.float64)
    y_target_mine = Tensor(y_target_val)

    loss_pt = ((y_pt - y_target_pt) ** 2).sum()
    loss_pt.backward()

    loss_mine = ((y_mine - y_target_mine) ** 2).sum()
    loss_mine.backward()

    # 6. Assertions
    np.testing.assert_allclose(y_mine.data, y_pt.detach().numpy(), err_msg="Linear forward failed")
    np.testing.assert_allclose(linear_mine.weight.grad, linear_pt.weight.grad.numpy().T, err_msg="Weight grad failed")
    np.testing.assert_allclose(linear_mine.bias.grad, linear_pt.bias.grad.numpy(), err_msg="Bias grad failed")


def test_mlp_sequential():
    """Validate end-to-end forward/gradient flow through a small MLP."""
    # 1. Define dimensions
    batch_size, in_feat, hidden_feat, out_feat = 16, 10, 8, 3
    x_val = np.random.randn(batch_size, in_feat)

    # 2. PyTorch Graph (Remembering to use float64 for precision!)
    x_pt = torch.tensor(x_val, requires_grad=True, dtype=torch.float64)
    model_pt = torch.nn.Sequential(
        torch.nn.Linear(in_feat, hidden_feat),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_feat, out_feat),
    ).double()

    # 3. Custom Engine Graph
    x_mine = Tensor(x_val)
    model_mine = Sequential(
        Linear(in_feat, hidden_feat),
        ReLU(),
        Linear(hidden_feat, out_feat),
    )

    # Hijack the weights so both models start identically
    model_mine.layers[0].weight.data = model_pt[0].weight.detach().numpy().T
    model_mine.layers[0].bias.data = model_pt[0].bias.detach().numpy()

    model_mine.layers[2].weight.data = model_pt[2].weight.detach().numpy().T
    model_mine.layers[2].bias.data = model_pt[2].bias.detach().numpy()

    # 4. Forward Pass
    y_pt = model_pt(x_pt)
    y_mine = model_mine(x_mine)

    # 5. Backward Pass (MSE against dummy targets)
    y_target_val = np.random.randn(batch_size, out_feat)
    y_target_pt = torch.tensor(y_target_val, dtype=torch.float64)
    y_target_mine = Tensor(y_target_val)

    loss_pt = ((y_pt - y_target_pt) ** 2).sum()
    loss_pt.backward()

    loss_mine = ((y_mine - y_target_mine) ** 2).sum()
    loss_mine.backward()

    # 6. Assertions
    np.testing.assert_allclose(y_mine.data, y_pt.detach().numpy(), err_msg="MLP Forward failed")

    # Check gradients of the first Linear layer (to ensure gradients passed through ReLU)
    np.testing.assert_allclose(
        model_mine.layers[0].weight.grad,
        model_pt[0].weight.grad.numpy().T,
        err_msg="Layer 1 Weight Grad failed",
    )


def test_full_training_loop():
    """Confirm a tiny network can reduce loss in a short training run."""
    # 1. Set a random seed so the weights initialize the same way every time
    np.random.seed(42)

    X_data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y_target = np.array([[0.0], [2.0], [4.0]])

    X = Tensor(X_data)
    y = Tensor(y_target)

    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
    )

    # 2. Use the tuned learning rate
    optimizer = Adam(model.parameters(), lr=0.03)

    initial_loss = None
    final_loss = None

    # 3. Increase the epochs to give it enough runway to overcome dead ReLUs
    for epoch in range(500):
        y_pred = model(X)
        loss = ((y_pred - y) ** 2).sum()

        if epoch == 0:
            initial_loss = loss.data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 499:
            final_loss = loss.data

    assert final_loss < initial_loss, f"Model failed to learn. Init: {initial_loss}, Final: {final_loss}"
    assert final_loss < 0.5, f"Model learned too slowly. Final Loss: {final_loss}"


def test_cross_entropy_loss():
    """Compare custom cross-entropy loss value and gradient with PyTorch."""
    # 1. Define dimensions
    batch_size, num_classes = 32, 10

    # 2. Generate random unnormalized logits and integer targets
    logits_val = np.random.randn(batch_size, num_classes)
    targets_val = np.random.randint(0, num_classes, size=(batch_size,))

    # 3. PyTorch Graph (Using float64)
    logits_pt = torch.tensor(logits_val, requires_grad=True, dtype=torch.float64)
    targets_pt = torch.tensor(targets_val, dtype=torch.long)  # PyTorch expects long ints for targets

    loss_pt = F.cross_entropy(logits_pt, targets_pt)
    loss_pt.backward()

    # 4. Custom Engine Graph
    logits_mine = Tensor(logits_val)

    criterion = CrossEntropyLoss()
    loss_mine = criterion(logits_mine, targets_val)
    loss_mine.backward()

    # 5. Assertions
    np.testing.assert_allclose(loss_mine.data, loss_pt.detach().numpy(), err_msg="Forward CE Loss failed")
    np.testing.assert_allclose(logits_mine.grad, logits_pt.grad.numpy(), err_msg="Backward CE Loss failed")

def test_mathematical_correctness():
    """Verify that the engine's chain rule implementation matches the numerical limit."""
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    x = Tensor(np.random.randn(1, 4))
    
    assert grad_check(model, x), "Autograd math does not match numerical gradient!"

