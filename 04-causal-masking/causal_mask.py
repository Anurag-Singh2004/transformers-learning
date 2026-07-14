"""
Causal Masking: from scratch

Prevents each token from attending to future tokens.
Used during training (and generation) for GPT-style decoder-only models.
"""

import numpy as np


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Returns a (seq_len,seq_len) mask matrix.
    0 where attention is ALLOWED (j <= i), -inf where BLOCKED (j>i).
    """

    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, k=1)] = -np.inf

    return mask

def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)

def masked_self_attention(X, W_Q, W_K, W_V): #almost similar to our previous self attention
    """
    self_attention, but with causal masking applied
    right before softmax.
    """

    queries = X @ W_Q
    keys = X @ W_K
    values = X @ W_V

    d_k = keys.shape[-1]
    scores = (queries @ keys.T) / np.sqrt(d_k)

    seq_len = X.shape[0]
    mask = create_causal_mask(seq_len)
    scores_masked = scores + mask 

    attention_weights = softmax(scores_masked)
    context_vectors = attention_weights @ values

    return context_vectors, attention_weights



if __name__ == "__main__":
    seq_len = 3
    mask = create_causal_mask(seq_len)
    print("Causal mask:")
    print(mask)

    print("\n--- Now applying it to real attention ---\n")
    np.random.seed(42)

    X = np.array([
        [1.0, 0.0, 1.0, 0.0], # cat
        [0.0, 1.0, 0.0, 1.0], # sat
        [1.0, 1.0, 0.0, 0.0], # mat
    ])
    d_model = X.shape[1]
    d_k = 3
    W_Q = np.random.randn(d_model, d_k)
    W_K = np.random.randn(d_model, d_k)
    W_V = np.random.randn(d_model, d_k)

    context_vectors, attention_weights = masked_self_attention(X, W_Q, W_K, W_V)

    print("Masked attention weights (should be lower-triangular, rows sum to 1):")
    print(np.round(attention_weights, 3))

    print("\nRow sums (should all be 1.0):")
    print(np.round(attention_weights.sum(axis=-1), 6))