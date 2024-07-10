import torch
import torch.nn as nn
from torch.nn import functional as F
from src.model_architectures.MLP import MLP
from src.config.GPTConfig import GPTConfig


class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # output_projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularizatio
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following die OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        assert B == self.config.train_config.B
        # calculate query, key, values, for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size", and C (number of chanels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs = 6, so nh*hs = 768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # transform tensor so pytorch treats (B, nh) as "batch dimension" and applies all operations
        # to those dimensions in parallel (all the batches and all the heads)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # # attention (materializes the larte (T,T) matric for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # # mask out future attentions
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # # normalize the attention so it sums to 1
        # att = F.softmax(att, dim=-1)
        # # weighted sum of all the values that the net finds interesting
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        

        # re-assemble all head outputs side by side, in essence concatenation
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        # output projection
        y = self.c_proj(y)
        return y