"""
Sampling Strategies : temperature, top-k, top-p

Given a probability distribution over vocabulary, pick the next token.
"""

import numpy as np

np.random.seed(42)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def apply_temperature(logits, temperature):
    """Scales logits before softmax. Low temp = sharper, high temp = flatter."""
    return logits / temperature


def top_k_filter(probabilities, k):
    """Keep only the top-k probabilities, zero out the rest, renormalize."""
    sorted_indices = np.argsort(probabilities)[::-1] #sort in descending order
    top_k_indices = sorted_indices[:k]

    filtered = np.zeros_like(probabilities)
    filtered[top_k_indices] = probabilities[top_k_indices]

    filtered = filtered / filtered.sum() #renormalize again
    return filtered



def top_p_filter(probabilities, p):
    """Keep the smallest set of top words whose cumulative probability >= p."""
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]

    cumulative = np.cumsum(sorted_probs)
    cutoff_idx = np.searchsorted(cumulative, p)+ 1   #how many words needed to cross p

    keep_indices = sorted_indices[:cutoff_idx]

    filtered = np.zeros_like(probabilities)
    filtered[keep_indices] = probabilities[keep_indices]

    filtered = filtered / filtered.sum()
    return filtered


def sample(probabilities, vocab):
    """Randomly pick ONE word, weighted by the given probability distribution."""
    return np.random.choice(vocab, p=probabilities)


if __name__ == "__main__":
    vocab = ["cat","sat", "mat", "dog","ran"]
    logits = np.array([0.857, -0.277, 1.117, -0.437, -0.409])

    print("=== Baseline (temperature = 1.0, no filtering) ===")
    probs = softmax(logits)
    print(np.round(probs, 3))

    print("\n=== Temperature = 0.5 (sharper) ===")
    probs_low_temp = softmax(apply_temperature(logits, 0.5))
    print(np.round(probs_low_temp, 3))

    print("\n=== Temperature = 2.0 (flatter) ===")
    probs_high_temp = softmax(apply_temperature(logits, 2.0))
    print(np.round(probs_high_temp, 3))

    print("\n=== Top-k (k=3) applied to baseline probs ===")
    probs_topk = top_k_filter(probs, k=3)
    print(np.round(probs_topk, 3))

    print("\n=== Top-p (p=0.9) applied to baseline probs ===")
    probs_topp = top_p_filter(probs, p=0.9)
    print(np.round(probs_topp, 3))

    print("\n=== Sampling 10 words using top-p filtered distribution ===")
    samples = [sample(probs_topp, vocab) for _ in range(10)]
    print(samples)



