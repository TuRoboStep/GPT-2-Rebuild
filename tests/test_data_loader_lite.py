from dataloader.data_loader_lite import DataLoaderLite
from config.GPTConfig import DatasetConfig
import torch

dataset_config = DatasetConfig(dataset_path='tests/data/test_dataset.txt', encoding='gpt2')
data_loader = DataLoaderLite(B=2, T=4, dataset_config=dataset_config)

def test_data_loader_lite():
    assert data_loader.B == 2
    assert data_loader.T == 16

    data_loader._load_data()
    assert data_loader.tokens is not None
    assert torch.all(data_loader.tokens.eq(torch.tensor([9288,27039,198,198,2220,340,11,15458,340,11,
                                               1332,477,6982,286,1243,13,198,198,14150,1257,
                                               14373,198])))
    
def test_data_loader_next_batch():
    data_loader._load_data()
    assert data_loader.current_position == 0
    x, y = data_loader.next_batch()
    assert torch.all(x.eq(torch.tensor([[9288,27039,198,198],[2220,340,11,15458]])))
    assert torch.all(y.eq(torch.tensor([[27039,198,198,2220],[340,11,15458,340]])))
    assert data_loader.current_position == 8
