"""
Mini-GPT Model : decoder-only transformer, built from scratch in PyTorch.

Assembles: token embedding + learned positional embedding + N transformer
blocks (causal multi-head attention + FFN + residuals + layer norm) +
final layer norm + vocab projection.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3*d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask)


    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        qkv = self.qkv_proj(x)
        q,k,v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1,2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2,-1))/(self.d_head**0.5)

        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)

        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.attention = CausalSelfAttention(d_model, num_heads,max_seq_len, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model,d_ff,dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x= self.norm1(x+self.attention(x))
        x = self.norm2(x+self.ffn(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model = 128, num_heads=4, num_blocks=4,d_ff=512, max_seq_len=256, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len,d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, dropout) for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.vocab_proj = nn.Linear(d_model, vocab_size)

        self.max_seq_len = max_seq_len


    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape

        positions = torch.arange(seq_len, device=token_ids.device)

        token_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(positions)

        x = self.dropout(token_emb+pos_emb)

        for block in self.blocks:
            x=block(x)

        x = self.final_norm(x)
        logits = self.vocab_proj(x)

        return logits

if __name__ == "__main__":
    vocab_size = 101
    model = MiniGPT(vocab_size=vocab_size, d_model=128, num_heads=4, num_blocks=4, max_seq_len=256)

    dummy_input = torch.randint(0, vocab_size, (2, 32))

    logits = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output logits shape:", logits.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")