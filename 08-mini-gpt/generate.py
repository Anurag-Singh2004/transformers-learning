"""
Text generation with configurable sampling strategies —
temperature, top-k, top-p : applied to our trained Mini-GPT.
"""

import torch
import torch.nn.functional as F

from data_pipeline import load_and_tokenize
from model import MiniGPT


def apply_temperature(logits, temperature):
    return logits / temperature



def top_k_filter(logits, k):
    """Set everything except the top-k logits to -inf, so they get zero probability after softmax."""
    top_k_values, _ = torch.topk(logits, k)
    min_value = top_k_values[:, -1].unsqueeze(-1)   # the smallest value among the top-k
    filtered = torch.where(logits < min_value, torch.full_like(logits, float('-inf')), logits)
    return filtered


def top_p_filter(logits, p):
    """Keep the smallest set of top tokens whose cumulative probability >= p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # mask out tokens once cumulative probability exceeds p
    sorted_mask = cumulative_probs > p
    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()   #shift right so we keep the token that crosses p
    sorted_mask[:, 0] = False

    sorted_logits[sorted_mask] = float('-inf')

    #scatter back to original order
    filtered = torch.full_like(logits, float('-inf'))
    filtered.scatter_(-1, sorted_indices, sorted_logits)
    return filtered


@torch.no_grad()
def generate(model, char_to_id, id_to_char, prompt, max_new_tokens=200,
             temperature=1.0, top_k=None, top_p=None):
    model.eval()

    tokens = [char_to_id[ch] for ch in prompt]
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    for _ in range(max_new_tokens):
        tokens_cropped = tokens[:, -model.max_seq_len:]
        logits = model(tokens_cropped)
        last_logits = logits[:, -1, :]

        last_logits = apply_temperature(last_logits, temperature)

        if top_k is not None:
            last_logits = top_k_filter(last_logits, top_k)

        if top_p is not None:
            last_logits = top_p_filter(last_logits, top_p)

        probs = F.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

    model.train()

    generated_ids = tokens[0].tolist()
    return ''.join(id_to_char[i] for i in generated_ids)



if __name__ == "__main__":
    data, vocab_size, char_to_id, id_to_char = load_and_tokenize("data/training_code.txt")

    model = MiniGPT(vocab_size=vocab_size, d_model=128, num_heads=4, num_blocks=4,
                     d_ff=512, max_seq_len=128)
    model.load_state_dict(torch.load("mini_gpt_weights.pt"))

    prompt = "def "

    print("=== Temperature = 0.5 (conservative) ===")
    print(generate(model, char_to_id, id_to_char, prompt, max_new_tokens=150, temperature=0.5))

    print("\n=== Temperature = 1.2 (more random) ===")
    print(generate(model, char_to_id, id_to_char, prompt, max_new_tokens=150, temperature=1.2))

    print("\n=== Top-k = 10 ===")
    print(generate(model, char_to_id, id_to_char, prompt, max_new_tokens=150, top_k=10))

    print("\n=== Top-p = 0.9 ===")
    print(generate(model, char_to_id, id_to_char, prompt, max_new_tokens=150, top_p=0.9))