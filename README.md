# autograd-from-scratch

A custom reverse-mode automatic differentiation engine built from scratch in NumPy, with a neural network library on top. No PyTorch, no TensorFlow — every operation, gradient, and backward pass implemented manually.

Trained a 3-layer MLP (`784 → 128 → 64 → 10`) on MNIST and hit **97.42% test accuracy in 5 epochs**.

## What's inside

- `engine/` — core autograd engine: tensor operations, backward passes, optimizer, and neural network primitives
- `experiments/` — progression from linear regression fundamentals to a full MLP on MNIST using the custom engine
- `tests/` — unit tests for the engine (`python -m pytest` from root)

## Motivation

Built to understand backpropagation at the level it actually works — not as a black box behind `.backward()`, but as a graph traversal problem where every node knows how to pass gradients to its parents. The goal was to be able to derive and implement every component from first principles before touching a framework.

## Quickstart

```bash
conda env create -f environment.yml
conda activate ml-foundations
python -m pytest                          # verify engine correctness
python -m experiments.02_mlp_mnist.train  # train on MNIST
```

## Results

| Model | Dataset | Accuracy | Epochs | Framework |
|-------|---------|----------|--------|-----------|
| 3-layer MLP | MNIST | 97.42% | 5 | None (NumPy only) |
