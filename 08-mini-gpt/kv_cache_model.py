"""
KV-Cache enabled version of the Mini-GPT attention/blocks.
Same architecture as model.py, but supports caching Keys/Values
across generation steps to avoid recomputation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CachedCausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model//num_heads

        self.qkv_proj = nn.Linear(d_model, 3*d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(torch.ones(max_seq_len,max_seq_len))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x, past_kv=None):
        batch_size,seq_len,d_model = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        #KV cache logic
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)   #append new K to cached K (along seq_len dimension)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k,v) #this is updated cache to return

        total_len = k.shape[2]   #how many tokens' worth of K,V we now have total

        scores = (q @ k.transpose(-2,-1))/(self.d_head ** 0.5)

        #Mask: query positions can attend to all cached + new key positions up to their own position
        mask = self.causal_mask[total_len - seq_len : total_len, :total_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.out_proj(out), new_kv



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.relu(self.linear1(x))))


class CachedTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.attention = CachedCausalSelfAttention(d_model, num_heads, max_seq_len, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, past_kv=None):
        attn_out, new_kv = self.attention(x, past_kv)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, new_kv


class CachedMiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_blocks=4,
                 d_ff=512, max_seq_len=256, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            CachedTransformerBlock(d_model, num_heads, d_ff, max_seq_len, dropout)
            for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.vocab_proj = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len


    def forward(self, token_ids, past_kv_list=None, start_pos=0):
        batch_size, seq_len = token_ids.shape

        positions = torch.arange(start_pos, start_pos + seq_len)
        tok_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(tok_emb + pos_emb)

        new_kv_list = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            x, new_kv = block(x, past_kv)
            new_kv_list.append(new_kv)

        x = self.final_norm(x)
        logits = self.vocab_proj(x)

        return logits, new_kv_list