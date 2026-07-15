"""
Layer Normalization

LayerNorm(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta

Applied independently to EACH token's vector.
gamma, beta are learned parameters (one value per dimension).
"""

import numpy as np


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    x: shape (seq_len, d_model) : normalizes each row independently
    gamma, beta: shape (d_model,) : learned scale and shift, shared across all tokens
    """

    mean = np.mean(x, axis=-1, keepdims=True)     # per-token mean
    variance = np.var(x, axis=-1, keepdims=True)   # per-token variance

    normalized = (x-mean)/np.sqrt(variance + epsilon)

    output = gamma*normalized+beta

    return output


if __name__ == "__main__":
    x = np.array([
        [1.3, 1.7, -1.1],   # sat's vector, post attention+residual
    ])

    d_model = x.shape[1]
    gamma = np.ones(d_model)    # start at 1.0 (no scaling initially)
    beta = np.zeros(d_model)    # start at 0.0 (no shifting initially)

    output = layer_norm(x, gamma, beta)

    print("Layer norm output:")
    print(np.round(output, 3))

    print("\nSanity check — mean should be ~0, std should be ~1:")
    print("Mean:", np.round(np.mean(output), 6))
    print("Std:", np.round(np.std(output), 6))