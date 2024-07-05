from dataclasses import dataclass


@dataclass
class TrainConfig:
    total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
    B = 2 # micro batch size
    T = 1024 # sequence length

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embeding dimension
    train_config: TrainConfig = TrainConfig()

@dataclass
class DatasetConfig:
    dataset_path: str = '../datasets/tiny_shakespear/input.txt'
    encoding: str = 'gpt2'