"""
Sinusoidal Positional Encoding : from scratch

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

#in this method we will just add positional encoding with token embeddings, but it is not the best method, although it works.
#there are some advanced methods(e.g. RoPE) but thats advanced and research topic for now.

import numpy as np

def get_positional_encoding(seq_len: int, d_model: int)-> np.ndarray:
    """
    Returns positional encoding matrix, shape (seq_len, d_model)
    Row i = position i's fingerprint vector.
    """

    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model//2):
            denominator = 10000**(2*i/d_model)
            PE[pos,2*i] = np.sin(pos/denominator)
            PE[pos,2*i+1] = np.cos(pos/denominator)
    
    return PE

if __name__ == "__main__":
    seq_len = 3   # "cat", "sat", "mat"
    d_model = 4

    PE = get_positional_encoding(seq_len, d_model)

    print("Positional encoding matrix:")
    print(np.round(PE, 3))

    # Now add to our familiar token embeddings
    token_embeddings = np.array([
        [1.0, 0.0, 1.0, 0.0],   # cat
        [0.0, 1.0, 0.0, 1.0],   # sat
        [1.0, 1.0, 0.0, 0.0],   # mat
    ])

    final_input = token_embeddings + PE  ######################

    print("\nToken embeddings + positional encoding = final input:")
    print(np.round(final_input, 3))



