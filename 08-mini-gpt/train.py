"""
Training loop for Mini-GPT : trains on your our corpus, tracks loss,
saves checkpoints, and generates sample text periodically.
"""

import torch
import torch.nn.functional as F
import time

from data_pipeline import load_and_tokenize, get_batch
from model import MiniGPT

#config
BLOCK_SIZE = 128  # number of characters per training window
BATCH_SIZE = 32   # number of windows per training step
D_MODEL = 128
NUM_HEADS = 4
NUM_BLOCKS = 4
D_FF = 512    #4*D_MODEL is standard practice
MAX_ITERS = 3000    # total training steps
EVAL_INTERVAL = 300    # print loss + generate sample every N steps
LEARNING_RATE = 3e-4


def compute_loss(model,x,y):
    logits = model(x)  #(batch,seq_len,vocab_size)

    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(batch_size*seq_len, vocab_size) #flattening
    targets_flat = y.view(batch_size*seq_len)

    loss = F.cross_entropy(logits_flat, targets_flat)
    return loss

@torch.no_grad() # disables gradient tracking since we're not training here, just generating
def generate(model, char_to_id, id_to_char, prompt, max_new_tokens=200):
    model.eval() # switches off dropout during generation

    tokens = [char_to_id[ch] for ch in prompt]
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0) # add batch dimension: (1, seq_len)

    for _ in range(max_new_tokens):
        # crop to max_seq_len if it gets too long
        tokens_cropped = tokens[:,-model.max_seq_len:]

        logits = model(tokens_cropped)
        last_logits = logits[:,-1,:] #only care about the last position's prediction

        probs= F.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)# sample from the distribution

        tokens = torch.cat([tokens, next_token], dim=1)

    model.train() #switch back to training mode

    generated_ids = tokens[0].tolist()
    return ''.join(id_to_char[i] for i in generated_ids)

def main():
    print("Loading data...")
    data, vocab_size, char_to_id, id_to_char = load_and_tokenize("data/training_code.txt")
    print(f"Dataset size: {len(data)} characters, vocab size: {vocab_size}")

    print("\nBuilding model...")
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        d_ff=D_FF,
        max_seq_len=BLOCK_SIZE,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting training...\n")
    start_time = time.time()

    for step in range(MAX_ITERS):
        x,y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)

        loss = compute_loss(model,x,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%EVAL_INTERVAL == 0:
            elapsed = time.time()-start_time
            print(f"Step {step}/{MAX_ITERS} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

            sample = generate(model, char_to_id, id_to_char, prompt="def", max_new_tokens=150)
            print(f"--- Sample generation ---\n{sample}\n-------------------------\n")

    print("\nTraining complete!")

    torch.save(model.state_dict(), "mini_gpt_weights.pt")
    print("Model weights saved to mini_gpt_weights.pt")


if __name__ == "__main__":
    main()