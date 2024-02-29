# import everything relevant
import numpy as np
from torch.utils.data import Dataset


class IITDataset(Dataset):
    """
    Each thing is randomly sampled from a pair of datasets.
    """

    def __init__(self, base_data, ablation_data, seed=0, every_combination=False):
        # For vanilla IIT, base_data and ablation_data are the same
        self.base_data = base_data
        self.ablation_data = ablation_data
        self.seed = seed
        self.every_combination = every_combination

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
