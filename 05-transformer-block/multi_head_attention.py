import numpy as np
from attention_numpy import self_attention

def run_all_heads(X, num_heads,d_model, d_k, d_v):
    """
    Runs `num_heads` independent self-attention calculations on the SAME input X.
    Each head gets its own randomly initialized W_Q, W_K, W_V.
    """
    head_outputs=[]
    head_weights=[]

    for h in range(num_heads):
        #for learning purpose we are just randomly creating (W_Q,W_K,W_V) for each head but in real code these will be actual trained matrices and not random
        W_Q = np.random.randn(d_model,d_k)
        W_K = np.random.randn(d_model,d_k)
        W_V = np.random.randn(d_model,d_v)

        context_vectors, attention_weights = self_attention(X, W_Q, W_K,W_V)

        head_outputs.append(context_vectors)
        head_weights.append(attention_weights)
    
    return head_outputs, head_weights


def concatenate_heads(head_outputs):
    """
    Stitches all heads' outputs side by side, per token.
    head_outputs: list of arrays, each shape (seq_len, d_v)
    Returns: shape (seq_len, num_heads * d_v)
    """
    return np.concatenate(head_outputs, axis=1)


def final_projection(concat_output, d_model):
    """
    Mixes information across heads using one learned weight matrix W_O.
    concat_output: shape (seq_len, num_heads * d_v)
    Returns: shape (seq_len, d_model)
    """
    W_O = np.random.randn(concat_output.shape[1], d_model)#again this will be trained matrix in real code and not random.
    return concat_output @ W_O


if __name__ == "__main__":
    np.random.seed(42)

    X = np.array([
        [1.0, 0.0, 1.0, 0.0], # cat
        [0.0, 1.0, 0.0, 1.0], # sat
        [1.0, 1.0, 0.0, 0.0], # mat
    ])

    d_model = X.shape[1]   #4
    num_heads = 2
    d_k = d_model // num_heads # 2
    d_v = d_model // num_heads # 2

    head_outputs, head_weights = run_all_heads(X, num_heads, d_model, d_k, d_v)

    print("Head 1 output:\n", np.round(head_outputs[0],3))
    print("\nHead 2 output:\n", np.round(head_outputs[1],3))

    #concatenate
    concat_output = concatenate_heads(head_outputs)
    print("\nConcatenated output, shape:", concat_output.shape)
    print(np.round(concat_output, 3))

    #final projection
    final_output = final_projection(concat_output, d_model)
    print("\nFinal multi-head attention output, shape:", final_output.shape)
    print(np.round(final_output, 3))



