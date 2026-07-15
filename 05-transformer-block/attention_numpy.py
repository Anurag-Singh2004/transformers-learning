'''
Self-Attention from Scratch — numpy only

Formula:  Attention(Q,K,V) = softmax((Q @ K.T)/sqrt(d_k))@ V

'''

import numpy as np

def softmax(x:np.ndarray)-> np.ndarray:
    """
    Converts raw scores into a probability distribution (sums to 1 per row).
    x: shape (..., seq_len) — operates on the LAST axis independently per row.
    """

    shifted = x-np.max(x, axis=-1, keepdims=True)  # for numerical stability
    exp_scores = np.exp(shifted)
    probabilities = exp_scores/exp_scores.sum(axis=-1, keepdims=True)
    return probabilities

def self_attention(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray):
    """
    X:   input embeddings, shape (seq_len, d_model)
    W_Q: query weight matrix, shape (d_model, d_k)
    W_K: key weight matrix, shape (d_model, d_k)  :  d_k MUST match W_Q's last dim
    W_V: value weight matrix, shape (d_model, d_v)  :  d_v can differ from d_k

    Returns:
        context_vectors:   shape (seq_len, d_v) — final context-aware output
        attention_weights: shape (seq_len, seq_len) — who attended to whom, how much
    """

    #Project embeddings into Query, Key, Value spaces
    queries = X @ W_Q
    keys = X @ W_K
    values = X @ W_V

    #Compute similarity scores (dot product)
    similarity_scores = queries @ keys.T

    #scale by sqrt(d_k) to prevent large values in softmax
    d_k = keys.shape[-1]
    scaled_scores = similarity_scores / np.sqrt(d_k)

    #Convert to probabilities (attention weights)
    attention_weights = softmax(scaled_scores)

    #weighted sum of Values : context-aware output
    context_vectors = attention_weights @ values

    return context_vectors, attention_weights

if __name__ == "__main__":
    np.random.seed(42)

    # 3 tokens ("cat", "sat", "mat"), embedding dim = 4
    X = np.array([
        [1.0, 0.0, 1.0, 0.0],  # cat
        [0.0, 1.0, 0.0, 1.0],  # sat
        [1.0, 1.0, 0.0, 0.0],  # mat
    ])

    d_model = X.shape[1]   # 4
    d_k = 3
    d_v = 3

    W_Q = np.random.randn(d_model, d_k)
    W_K = np.random.randn(d_model, d_k)
    W_V = np.random.randn(d_model, d_v)

    context_vectors, attention_weights = self_attention(X, W_Q, W_K, W_V)

    print("Attention weights (each row = one token's distribution over all tokens):")
    print(np.round(attention_weights, 3))

    print("\nSanity check — every row should sum to 1.0:")
    print(np.round(attention_weights.sum(axis=-1), 6))

    print("\nContext-aware output vectors, shape:", context_vectors.shape)
    print(np.round(context_vectors, 3))