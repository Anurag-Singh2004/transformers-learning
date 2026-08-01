"""
Benchmark: naive generation (recompute everything every step)
vs KV-cached generation (reuse cached K,V, only compute new token).
"""

import torch
import torch.nn.functional as F
import time

from data_pipeline import load_and_tokenize
from kv_cache_model import CachedMiniGPT


@torch.no_grad()
def generate_naive(model, char_to_id, id_to_char, prompt, max_new_tokens):
    """Recomputes the WHOLE sequence every step, no caching."""
    tokens = [char_to_id[ch] for ch in prompt]
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    for _ in range(max_new_tokens):
        tokens_cropped = tokens[:, -model.max_seq_len:]
        logits, _ = model(tokens_cropped)   # no cache passed in, ignore returned cache
        last_logits = logits[:, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


@torch.no_grad()
def generate_cached(model, char_to_id, id_to_char, prompt, max_new_tokens):
    """Uses KV caching - only computes the NEW token each step."""
    tokens = [char_to_id[ch] for ch in prompt]
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    # First call: process the whole prompt at once, build the initial cache
    logits, kv_cache = model(tokens, past_kv_list=None, start_pos=0)
    last_logits = logits[:, -1, :]
    probs = F.softmax(last_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    tokens = torch.cat([tokens, next_token], dim=1)

    current_pos = tokens.shape[1] - 1   # position of the token we just added

    for _ in range(max_new_tokens - 1):
        # only feed the SINGLE newest token, plus the existing cache
        logits, kv_cache = model(next_token, past_kv_list=kv_cache, start_pos=current_pos)
        last_logits = logits[:, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
        current_pos += 1

    return tokens


def main():
    data, vocab_size, char_to_id, id_to_char = load_and_tokenize("data/training_code.txt")

    model = CachedMiniGPT(vocab_size=vocab_size, d_model=128, num_heads=4,
                           num_blocks=4, d_ff=512, max_seq_len=128)
    model.load_state_dict(torch.load("mini_gpt_weights.pt"))
    model.eval()

    prompt = "def "
    num_tokens_to_generate = 100

    print(f"Generating {num_tokens_to_generate} tokens, naive (no cache)...")
    start = time.time()
    generate_naive(model, char_to_id, id_to_char, prompt, num_tokens_to_generate)
    naive_time = time.time() - start
    print(f"Naive time: {naive_time:.2f}s")

    print(f"\nGenerating {num_tokens_to_generate} tokens, WITH KV cache...")
    start = time.time()
    generate_cached(model, char_to_id, id_to_char, prompt, num_tokens_to_generate)
    cached_time = time.time() - start
    print(f"Cached time: {cached_time:.2f}s")

    speedup = naive_time / cached_time
    print(f"\nSpeedup: {speedup:.2f}x faster with KV caching")


if __name__ == "__main__":
    main()