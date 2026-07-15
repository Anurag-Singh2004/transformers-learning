"""
Full Transformer Block :
  - Multi-head self-attention (with causal masking)

  - Residual connection 1 (fix the vanishing gradients problem in network by adding input in each sublayer output)

  - Layer Norm 2 (fix the exploding/unstable-scale case of gradient problem in network, by keeping values at a consistent, controlled range at every step)

  - Feed-forward network (FFN computes complex non-linear transformations of each token's vector)

  - Residual connection 2

  - Layer Norm 2

Pipeline:
    x1 = LayerNorm(x  + Attention(x))
    x2 = LayerNorm(x1 + FFN(x1))
"""

import numpy as np
from multi_head_attention import run_all_heads, concatenate_heads, final_projection
from layer_norm import layer_norm
from feed_forward import feed_forward_network


def transformer_block(X, num_heads, d_model, d_k, d_v, gamma1, beta1, W1, b1, W2, b2, gamma2, beta2):
    """
    X: input, shape (seq_len, d_model)
    Returns: shape (seq_len, d_model)
    """


    #Sublayer 1: Multi-Head Attention ----------------
    head_outputs, head_weights = run_all_heads(X, num_heads, d_model, d_k, d_v)
    concat_output = concatenate_heads(head_outputs)
    attn_out = final_projection(concat_output, d_model)

    #Residual connection #1
    residual_1 = X + attn_out

    #Layer Norm #1
    x1 = layer_norm(residual_1, gamma1, beta1)

    #Sublayer 2: Feed-Forward-------------
    ffn_out = feed_forward_network(x1, W1, b1, W2, b2)

    #Residual connection #2
    residual_2 = x1 + ffn_out

    #Layer Norm #2
    x2 = layer_norm(residual_2, gamma2, beta2)

    return x2, head_weights


if __name__ == "__main__":
    np.random.seed(42)

    X = np.array([
        [1.0, 0.0, 1.0, 0.0],   # cat
        [0.0, 1.0, 0.0, 1.0],   # sat
        [1.0, 1.0, 0.0, 0.0],   # mat
    ])

    d_model = X.shape[1] #4
    num_heads = 2
    d_k = d_model // num_heads #2
    d_v = d_model // num_heads  #2
    d_ff = d_model * 4    #16 (standard practice)


    gamma1 = np.ones(d_model)
    beta1 = np.zeros(d_model)

    W1 = np.random.randn(d_model, d_ff)* 0.1 #weight initialization technique
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model)* 0.1
    b2 = np.zeros(d_model)

    gamma2 = np.ones(d_model)
    beta2 = np.zeros(d_model)

    output, head_weights = transformer_block(
        X, num_heads, d_model, d_k, d_v, gamma1, beta1, W1, b1, W2, b2, gamma2, beta2
    )

    print("Full transformer block output, shape:", output.shape)
    print(np.round(output, 3))

    print(f"\nNumber of attention heads used: {len(head_weights)}")