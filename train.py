# %%
import numpy as np
import torch as t
from torch import Tensor
from torch.utils.data import Dataset
import transformer_lens as tl
from transformer_lens.hook_points import HookedRootModule, HookPoint
import networkx as nx
from dataclasses import dataclass
from wrapper import HookedModuleWrapper
from typing import Callable
# %%
"""
Things to write:
- (Ivan is writing TL representation of tracr models)
- Correspondence object, mapping hl variables to subspaces in the model
    - High-level causal structure object
        - HookedRootModule
        - Generate from NetworkX graph as used by tracr compiler intermediate step
        - Functions for computing thing from parent
    - Dictionary mapping graph nodes to TL units (HookPoint objects)
    - (Maybe for future) tau: LL values -> HL values
- Training loop
"""

HookName = str
HLCache = dict

@dataclass
class Node():
    f: Callable

class IITDataset(Dataset):
    def __init__(self, base_data, ablation_data, seed=0):
        self.base_data = base_data
        self.ablation_data = ablation_data
        self.seed = seed

    def __getitem__(self, index):
        # sample based on seed
        rng = np.random.default_rng(self.seed * 1000000 + index)
        base_input = rng.choice(self.base_data)
        ablation_input = rng.choice(self.ablation_data)
        return base_input, ablation_input

    def __len__(self):
        return len(self.base_data)

class IITModelPair():
    hl_model: HookedRootModule
    ll_model: HookedRootModule
    hl_cache: tl.ActivationCache
    ll_cache: tl.ActivationCache
    hl_graph: nx.DiGraph
    corr: dict[HookName, set[HookName]] # high -> low correspondence. Capital Pi in paper

    def __init__(self, hl_model=None, ll_model=None, hl_graph=None, corr={}, seed=0):
        # TODO change to construct hl_model from graph?
        if hl_model is None:
            assert hl_graph is not None
            hl_model = self.make_hl_model(hl_graph)

        self.hl_model = hl_model
        self.ll_model = ll_model
        self.corr = corr
        assert all([k in self.hl_model.hook_points for k in self.corr.keys()])
        self.rng = np.random.default_rng(seed)

    def make_hl_model(self, hl_graph):
        raise NotImplementedError

    def set_corr(self, corr):
        self.corr = corr

    def sample_hl_name(self):
        return self.rng.choice(list(self.corr.keys()))

    def hl_ablation_hook(self,hook_point_out:Tensor, hook:HookPoint) -> Tensor:
        out = self.hl_cache[hook.name]
        return out
    
    def ll_ablation_hook(self,hook_point_out:Tensor, hook:HookPoint) -> Tensor:
        out = self.ll_cache[hook.name]
        return out

    def do_intervention(self, base_input, ablation_input, hl_node:HookName):
        hl_ablation_output, self.hl_cache = self.hl_model.run_with_cache(ablation_input)
        ll_ablation_output, self.ll_cache = self.ll_model.run_with_cache(ablation_input)

        hl_name = self.sample_hl_name()
        ll_name = self.Pi(hl_name)

        hl_output = self.hl_model.run_with_hooks(base_input, (hl_name, self.hl_ablation_hook))
        ll_output = self.ll_model.run_with_hooks(base_input, (ll_name, self.ll_ablation_hook))

        return hl_output, ll_output

    def train(self, base_data, ablation_data, hl_node:HookName, n_examples=1000):
        dataset = IITDataset(base_data, ablation_data)
        loss_fn = t.nn.MSELoss()
        for example in range(n_examples):
            # sample one base and one ablation datapoint
            base_input, ablation_input = dataset[example]

            # sample a high-level variable to ablate
            hl_output, ll_output = self.do_intervention(base_data, ablation_data, hl_node)


# %%

if __name__ == '__main__':
    pass