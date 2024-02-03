from typing import Callable, Optional
from dataclasses import dataclass
import dataclasses
import numpy as np
from torch import Tensor
import torchvision
import transformer_lens as tl
from transformer_lens.hook_points import HookedRootModule, HookPoint
import networkx as nx
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from iit.utils.index import TorchIndex
from iit.utils.iit_dataset import IITDataset
from iit.utils.config import *
import torch as t

HookName = str
HLCache = dict

@dataclass
class HLNode():
    name: HookName
    num_classes: int
    index: Optional[int]

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, HLNode):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return self.name

@dataclass
class LLNode():
    name: HookName
    index: TorchIndex
    subspace: Optional[t.Tensor]=None

    def __eq__(self, other):
        return isinstance(other, LLNode) and dataclasses.astuple(self) == dataclasses.astuple(other)

    def __hash__(self):
        return hash(dataclasses.astuple(self))
    

from abc import ABC, abstractmethod
class BaseModelPair(ABC):
    hl_model: HookedRootModule
    ll_model: HookedRootModule
    hl_cache: tl.ActivationCache
    ll_cache: tl.ActivationCache
    hl_graph: nx.DiGraph
    corr: dict[HLNode, set[LLNode]] # high -> low correspondence. Capital Pi in paper

    @abstractmethod
    def do_intervention(self, base_input, ablation_input, hl_node:HookName, verbose=False):
        pass

    @abstractmethod
    def train(self, base_data, ablation_data, test_base_data, test_ablation_data, epochs=1000, use_wandb=False):
        pass