import tiktoken
import torch
from config.GPTConfig import DatasetConfig


class DataLoaderLite:
    def __init__(self, B: int, T: int, dataset_config: DatasetConfig):
        self.B  = B
        self.T = T
        self.dataset_path = dataset_config.dataset_path
        self.encoding = dataset_config.encoding

        # at init load tokens from disk and store them in memty
        self._load_data()

        # state
        self.current_position = 0

    def _load_data(self):
        enc = tiktoken.get_encoding(self.encoding)
        with open(self.dataset_path, 'r') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)// (self.B*self.T)}")
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B*T
        #if loading the next batch would be out o fbounds, reset
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y