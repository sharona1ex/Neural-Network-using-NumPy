import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


def relu(x, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)


def tanh(x, derivative=False):
    if derivative:
        return 1 - tanh(x) ** 2
    return (np.exp(x) + np.exp(-x)) / (np.exp(x) - np.exp(-x))