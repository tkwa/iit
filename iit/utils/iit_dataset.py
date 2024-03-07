# import everything relevant
import numpy as np
from torch.utils.data import Dataset
from iit.utils.config import DEVICE
from torch.utils.data import DataLoader
import torch


class IITDataset(Dataset):
    """
    Each thing is randomly sampled from a pair of datasets.
    """

    def __init__(self, base_data, ablation_data, seed=0, every_combination=False, device=DEVICE):
        # For vanilla IIT, base_data and ablation_data are the same
        self.base_data = base_data
        self.ablation_data = ablation_data
        self.seed = seed
        self.every_combination = every_combination
        self.device = device

    def __getitem__(self, index):
        if self.every_combination:
            base_index = index // len(self.ablation_data)
            ablation_index = index % len(self.ablation_data)
            base_input = self.base_data[base_index]
            ablation_input = self.ablation_data[ablation_index]
            return base_input, ablation_input

        # sample based on seed
        rng = np.random.default_rng(self.seed * 1000000 + index)
        base_index = rng.choice(len(self.base_data))
        ablation_index = rng.choice(len(self.ablation_data))

        base_input = self.base_data[base_index]
        ablation_input = self.ablation_data[ablation_index]
        return base_input, ablation_input

    def __len__(self):
        if self.every_combination:
            return len(self.base_data) * len(self.ablation_data)
        return len(self.base_data)

    @staticmethod
    def collate_fn(batch, device=DEVICE):
        def get_encoded_input_from_torch_input(xy):
            x, y, int_vars = zip(*xy)
            x = torch.stack([x_i.to(device) for x_i in x])
            y = torch.stack([y_i.to(device) for y_i in y])
            int_vars = torch.stack([iv.to(device) for iv in int_vars])
            return x, y, int_vars
        base_input, ablation_input = zip(*batch)
        return get_encoded_input_from_torch_input(base_input), get_encoded_input_from_torch_input(ablation_input)

    def make_loader(
        self,
        batch_size,
        num_workers,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: self.collate_fn(x, self.device),
        )