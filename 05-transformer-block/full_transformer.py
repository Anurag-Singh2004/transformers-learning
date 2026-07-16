"""
Full Decoder-Only Transformer : stacking N transformer blocks

This is the GPT-style architecture: N identical blocks stacked,
each with causal-masked multi-head attention.
"""
#transformes are of mainly 3 types: 
 #1. Encoder-only (BERT-style)
 #2. Decoder-only (GPT-style) : we are building this
 #3. Encoder-Decoder (original Transformer, T5-style)


import numpy as np
from multi_head_attention import run_all_heads, concatenate_heads, final_projection
from layer_norm import layer_norm
from feed_forward import feed_forward_network


def transformer_block(X, num_heads, d_model, d_k, d_v, gamma1, beta1, W1, b1, W2, b2, gamma2, beta2):
    head_outputs, head_weights = run_all_heads(X, num_heads, d_model, d_k, d_v)
    concat_output = concatenate_heads(head_outputs)
    attn_out = final_projection(concat_output, d_model)

    residual_1 = X + attn_out
    x1 = layer_norm(residual_1, gamma1, beta1)

    ffn_out = feed_forward_network(x1, W1, b1, W2, b2)

    residual_2 = x1 + ffn_out
    x2 = layer_norm(residual_2, gamma2, beta2)

    return x2, head_weights


def init_block_params(d_model, num_heads, d_ff):
    """Creates a fresh set of random weights for ONE block."""
    d_k = d_model// num_heads
    d_v = d_model// num_heads

    return {
        "gamma1": np.ones(d_model),
        "beta1": np.zeros(d_model),
        "W1": np.random.randn(d_model, d_ff)* 0.1,
        "b1": np.zeros(d_ff),
        "W2": np.random.randn(d_ff, d_model)* 0.1,
        "b2": np.zeros(d_model),
        "gamma2": np.ones(d_model),
        "beta2": np.zeros(d_model),
    }

def full_transformer(X, num_blocks, num_heads, d_model, d_ff):
    """
    Stacks `num_blocks` transformer blocks, each with its OWN independent weights.
    """

    d_k = d_model // num_heads
    d_v = d_model // num_heads

    current_input = X

    for block_idx in range(num_blocks):
        params = init_block_params(d_model, num_heads, d_ff)   #fresh weights per block

        current_input, head_weights = transformer_block(
            current_input, num_heads, d_model, d_k, d_v,
            params["gamma1"], params["beta1"],
            params["W1"], params["b1"], params["W2"], params["b2"],
            params["gamma2"], params["beta2"],
        )

        print(f"Block {block_idx+1} output shape: {current_input.shape}")
    
    return current_input



if __name__ == "__main__":
    np.random.seed(42)

    X = np.array([
        [1.0, 0.0, 1.0, 0.0], #cat
        [0.0, 1.0, 0.0, 1.0], #sat
        [1.0, 1.0, 0.0, 0.0], #mat
    ])

    d_model = X.shape[1]   #4
    num_heads = 2
    d_ff = d_model* 4
    num_blocks = 4          # stack 4 blocks (GPT-2 small uses 12, we use 4 for speed)

    final_output = full_transformer(X, num_blocks, num_heads, d_model, d_ff)

    print("\nFinal output after all blocks, shape:", final_output.shape)
    print(np.round(final_output, 3))