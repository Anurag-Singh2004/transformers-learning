"""
Full trace-through: watch REAL data flow through the trained Mini-GPT,
one stage at a time, using actual numbers from mini_gpt_weights.pt
"""

import torch
import torch.nn.functional as F

from data_pipeline import load_and_tokenize
from model import MiniGPT


def main():
    data, vocab_size, char_to_id, id_to_char = load_and_tokenize("data/training_code.txt")

    model = MiniGPT(vocab_size=vocab_size, d_model=128, num_heads=4, num_blocks=4,
                     d_ff=512, max_seq_len=128)
    model.load_state_dict(torch.load("mini_gpt_weights.pt"))
    model.eval()

    # ---- Stage 1: Tokenization ----
    prompt = "def "
    print(f"Raw text: {prompt!r}")

    token_ids = [char_to_id[ch] for ch in prompt]
    print(f"Token IDs: {token_ids}")
    print(f"(mapping: {[(ch, char_to_id[ch]) for ch in prompt]})")

    tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    print(f"\nInput tensor shape: {tokens.shape}  (batch_size=1, seq_len={len(token_ids)})")

    # ---- Stage 2: Embeddings ----
    with torch.no_grad():
        tok_emb = model.token_embedding(tokens)
        print(f"\nToken embedding shape: {tok_emb.shape}")
        print(f"First token's embedding (first 5 numbers): {tok_emb[0, 0, :5]}")

        positions = torch.arange(tokens.shape[1])
        pos_emb = model.position_embedding(positions)
        print(f"\nPosition embedding shape: {pos_emb.shape}")
        print(f"Position 0's embedding (first 5 numbers): {pos_emb[0, :5]}")

        x = tok_emb + pos_emb
        print(f"\nCombined embedding shape: {x.shape}")

        # ---- Stage 3: Through each transformer block ----
        for i, block in enumerate(model.blocks):
            x = block(x)
            print(f"\nAfter Block {i+1}: shape {x.shape}, first token's first 5 values: {x[0, 0, :5]}")

        x = model.final_norm(x)
        logits = model.vocab_proj(x)
        print(f"\nFinal logits shape: {logits.shape}")

        # ---- Stage 4: Look at predictions for the LAST position ----
        last_logits = logits[0, -1, :]   # last position's logits, shape (vocab_size,)
        probs = F.softmax(last_logits, dim=-1)

        top5_probs, top5_ids = torch.topk(probs, 5)
        print(f"\nTop 5 predicted next characters after {prompt!r}:")
        for prob, char_id in zip(top5_probs, top5_ids):
            char = id_to_char[char_id.item()]
            print(f"  {char!r} -> {prob.item():.4f}")

    # ---- Stage 5: Generate 3 characters, step by step ----
    print("\n--- Generating 3 characters step by step ---")
    current_tokens = tokens.clone()

    for step in range(3):
        with torch.no_grad():
            logits = model(current_tokens)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        chosen_char = id_to_char[next_token.item()]
        chosen_prob = probs[0, next_token.item()].item()

        current_text = ''.join(id_to_char[t] for t in current_tokens[0].tolist())
        print(f"Step {step+1}: after {current_text!r}, picked {chosen_char!r} (prob={chosen_prob:.4f})")

        current_tokens = torch.cat([current_tokens, next_token], dim=1)

    final_text = ''.join(id_to_char[t] for t in current_tokens[0].tolist())
    print(f"\nFinal text: {final_text!r}")


if __name__ == "__main__":
    main()