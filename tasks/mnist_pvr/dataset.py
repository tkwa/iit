import torch as t
import torchvision
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import numpy as np
from .utils import *


class ImagePVRDataset(Dataset):
    """
    Turns the regular dataset into a PVR dataset.
    Images are concatenated into a 2x2 square.
    The label is the class of the image in position class_map[label of top left].
    """
    def __init__(self, base_dataset, class_map:dict[int, int]=MNIST_CLASS_MAP, seed=0, use_cache=True, length=200000, iid=True, pad_size=0, unique_per_quad=False):
        self.base_dataset = base_dataset
        self.class_map = class_map
        self.seed=seed
        self.rng = np.random.default_rng(seed)
        assert all(v in {1, 2, 3} for v in class_map.values())
        self.cache = {}
        self.use_cache = False
        self.length = length
        self.iid = iid
        self.pad_size = pad_size
        self.use_cache = use_cache
        self.unique_per_quad = unique_per_quad
        if not self.iid:
            print("WARNING: using non-iid mode")
            assert len(self.base_dataset) >= 4*self.length, "Dataset is too small for non-iid mode"
        if get_input_shape() is None:
            set_input_shape(self[0][0].unsqueeze(0).shape)

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
        if self.iid:
            self.rng = np.random.default_rng(self.seed * self.length + index)
            base_items = [self.base_dataset[self.rng.integers(0, len(self.base_dataset))] for i in range(4)]
        else:
            base_items = [self.base_dataset[i] for i in range(index * 4, index * 4 + 4)]
            
        if self.unique_per_quad:
            # keep sampling until we get 4 different classes
            while len(set([item[1] for item in base_items])) < 4:
                base_items = [self.base_dataset[self.rng.integers(0, len(self.base_dataset))] for i in range(4)]
    
        images = [base_item[0] for base_item in base_items]
        if self.pad_size > 0:
            images = [ImageOps.expand(image, border=self.pad_size, fill='black') for image in images]
            # print(f"Padding images by {self.pad_size}")
        new_image = self.concatenate_2x2(images)
        new_image = torchvision.transforms.functional.to_tensor(new_image)

        base_label = base_items[0][1]
        pointer = self.class_map[base_label]
        new_label = t.tensor(base_items[pointer][1])
        intermediate_vars = t.tensor([base_items[i][1] for i in range(4)], dtype=t.long)
        ret = new_image, new_label, intermediate_vars
        if self.use_cache:
            self.cache[index] = ret
        return ret
    
    def __len__(self):
        return self.length