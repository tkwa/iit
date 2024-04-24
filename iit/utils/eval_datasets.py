import numpy as np
from torch.utils.data import Dataset, DataLoader
from iit.utils.config import DEVICE
import torch
from iit.utils.iit_dataset import IITDataset

class IITUniqueDataset(IITDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        return self.base_data[index]

    def __len__(self):
        return len(self.base_data)

    @staticmethod
    def collate_fn(batch, device=DEVICE):
        return IITDataset.get_encoded_input_from_torch_input(batch, device)

