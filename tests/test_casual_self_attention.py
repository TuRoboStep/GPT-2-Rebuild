from model_architectures.architecture_building_blocks import CausalSelfAttention, Block
from config.GPTConfig import GPTConfig
import torch

config = GPTConfig()
self_attention = CausalSelfAttention(config)

def test_causal_self_attention():
    with torch.no_grad():
        x = torch.tensor((), dtype=torch.float32)
        x = x.new_ones((config.train_config.B, config.train_config.T, config.n_embd)) # [B, T, C]
        y = self_attention.forward(x)
        assert list(y.size()) == [config.train_config.B, config.train_config.T, config.n_embd]

block = Block(config=config)

def test_block():
    with torch.no_grad():
        x = torch.tensor((), dtype=torch.float32)
        x = x.new_ones((config.train_config.B, config.train_config.T, config.n_embd)) # [B, T, C]
        y = block.forward(x)
        assert list(y.size()) == [config.train_config.B, config.train_config.T, config.n_embd]
