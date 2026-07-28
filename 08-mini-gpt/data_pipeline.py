"""
Data Pipeline for Mini-GPT : character-level tokenization

Converts raw text into token IDs, and provides batching for training.
"""

import torch

def load_and_tokenize(filepath):
    """
    Reads the text file, builds a character-level vocabulary,
    and converts the entire text into a tensor of token IDs.
    """

    with open(filepath,'r', encoding='utf-8') as f:
        text = f.read()

    #Build vocabulary: every unique character that appears in the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    #Two lookup dictionaries: char -> id, and id -> char
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}

    #Convert the entire text into a list of integers, then a tensor
    data = torch.tensor([char_to_id[ch] for ch in text], dtype=torch.long)

    return data, vocab_size, char_to_id, id_to_char


def get_batch(data, block_size, batch_size):
    """
    Randomly samples `batch_size` chunks of length `block_size` from data.
    Returns (input_batch, target_batch) where target is input shifted by 1
    (since we're predicting the NEXT character at every position).
    """
    max_start = len(data)-block_size-1
    start_indices = torch.randint(0, max_start, (batch_size,))

    x= torch.stack([data[i : i + block_size] for i in start_indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in start_indices])

    return x, y

    
if __name__ == "__main__":
    data, vocab_size, char_to_id, id_to_char = load_and_tokenize("data/training_code.txt")

    print("Total characters in dataset:", len(data))
    print("Vocabulary size (unique characters):", vocab_size)
    print("\nFirst 20 characters of vocabulary:", list(char_to_id.keys())[:20])

    print("\nFirst 100 characters of raw text:")
    print(repr(''.join(id_to_char[i.item()] for i in data[:100])))

    x, y = get_batch(data, block_size=8, batch_size=4)
    print("\nSample batch:")
    print("Input shape:", x.shape)
    print("Input[0]:", x[0])
    print("Input[0] as text:", ''.join(id_to_char[i.item()] for i in x[0]))
    print("Target[0]:", y[0])
    print("Target[0] as text:", ''.join(id_to_char[i.item()] for i in y[0]))

