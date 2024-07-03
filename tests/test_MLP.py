from src.model_architectures.MLP import MLP
from src.config.GPTConfig import GPTConfig
import pytest
import torch


config = GPTConfig(n_embd=3)
torch.manual_seed(1337)
mlp = MLP(config)
test_data = [(torch.tensor([1.0,1.0,1.0]), 3)]

@pytest.mark.parametrize("x, expected_output_size", test_data)
def test_MLP_forward(x, expected_output_size):
    with torch.no_grad():
        output = mlp(x)
        assert len(output.size()) == 1
        assert output.size()[0] == expected_output_size