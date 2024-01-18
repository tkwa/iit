# %%
"""
The MNIST-PVR task. We need to download the MNIST dataset and process it.
"""

import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional

# %%

mnist_train = datasets.MNIST("./data", download=True)
mnist_test = datasets.MNIST("./data", train=False, download=True)
# %%

class ImagePVRDataset(Dataset):
    """
    Turns the regular dataset into a PVR dataset.
    Images are concatenated into a 2x2 square.
    The label is the class of the image in position class_map[label of top left].
    """
    def __init__(self, base_dataset, class_map:Optional[dict[int, int]]=None, seed=0, use_cache=False):
        self.base_dataset = base_dataset
        if class_map is None:
            class_map = {k: [1, 1, 1, 1, 2, 2, 2, 3, 3, 3][k] for k in range(10)}
        self.class_map = class_map
        self.rng = np.random.default_rng(seed)
        assert all(v in {1, 2, 3} for v in class_map.values())
        self.cache = {}
        self.use_cache=use_cache

    @staticmethod
    def concatenate_2x2(images):
        """
        Concatenates four PIL.Image.Image objects into a 2x2 square.
        """
        assert len(images) == 4, "Need exactly four images"
        width, height = images[0].size
        new_image = Image.new('RGB', (width * 2, height * 2))

        new_image.paste(images[0], (0, 0))
        new_image.paste(images[1], (width, 0))
        new_image.paste(images[2], (0, height))
        new_image.paste(images[3], (width, height))

        return new_image
    
    def __getitem__(self, index):
        if index in self.cache and self.use_cache:
            return self.cache[index]
        images = [self.base_dataset[i][0] for i in range(index, index + 4)]
        new_image = self.concatenate_2x2(images)

        base_label = self.base_dataset[index][1]
        pointer = self.class_map[base_label]
        new_label = self.base_dataset[index + pointer][1]
        return new_image, new_label
    
    def __len__(self):
        return len(self.base_dataset) // 4


# %%

mnist_pvr_train = ImagePVRDataset(mnist_train, None)
mnist_pvr_test = ImagePVRDataset(mnist_test, None)
# %%
