"""
Feed-Forward Network (FFN) : the second major component of a transformer block.

FFN(x) = Linear2(ReLU(Linear1(x)))

Runs independently on each token's vector (same weights and biases, applied per position).
"""

import numpy as np

def relu(x: np.ndarray)->np.ndarray:  #we can use other functions also like GELU, etc
    return np.maximum(0,x)


def feed_forward_network(X: np.ndarray, W1,b1,W2,b2) -> np.ndarray:
    """
    X: input, shape (seq_len, d_model) : e.g. attention's output
    W1: shape (d_model, d_ff)
    b1: shape (d_ff,)
    W2: shape (d_ff, d_model)
    b2: shape (d_model,)

    Returns: shape (seq_len, d_model) : same shape as input
    """

    hidden = X @ W1 + b1   #expand + linear decision(weighted combination)
    activated = relu(hidden)   #non-linearity (non-linear decision and uses threshold like 0 for relu function)
    output = activated @ W2 + b2  #shrinking

    return output



if __name__ == "__main__":
    np.random.seed(42)

    #suppose this is attention's output for "cat sat mat"
    X = np.array([
        [0.9, 0.3, 0.2], # cat (post-attention)
        [1.1, 0.2, 0.7],  # sat (post-attention)
        [0.5, 0.8, 0.4],   # mat (post-attention)
    ])

    d_model = X.shape[1]   # 3
    d_ff = d_model * 4   # standard 4x expansion, real-world convention

    W1 = np.random.randn(d_model, d_ff) * 0.1  #multplication by 0.1 is weight initialization technique at our level, in real world principled schemes like Xavier/Glorot initialization or He initialization are used
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * 0.1
    b2 = np.zeros(d_model)

    output = feed_forward_network(X, W1, b1, W2, b2)

    print("FFN output shape:", output.shape)
    print(np.round(output, 3))
