# %%
import dataclasses
from dataclasses import dataclass
import numpy as np
import torch as t
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision
import transformer_lens as tl
from transformer_lens.hook_points import HookedRootModule, HookPoint
import networkx as nx
from wrapper import HookedModuleWrapper
from typing import Callable, Optional
import wandb
from tqdm import tqdm

from pvr import mnist_train, mnist_test, MNIST_PVR_HL, ImagePVRDataset
from index import TorchIndex, Ix

DEVICE = t.device('cuda' if t.cuda.is_available() else 'cpu')
WANDB_ENTITY = "cybershiptrooper"

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
class HLNode():
    name: HookName
    index: Optional[int]

@dataclass
class LLNode():
    name: HookName
    index: TorchIndex
    subspace: Optional[t.Tensor]=None

    def __eq__(self, other):
        return isinstance(other, LLNode) and dataclasses.astuple(self) == dataclasses.astuple(other)

    def __hash__(self):
        return hash(dataclasses.astuple(self))

class IITDataset(Dataset):
    """
    Each thing is randomly sampled from a pair of datasets.
    """
    def __init__(self, base_data, ablation_data, seed=0):
        # For vanilla IIT, base_data and ablation_data are the same
        self.base_data = base_data
        self.ablation_data = ablation_data
        self.seed = seed

    def __getitem__(self, index):
        # sample based on seed
        rng = np.random.default_rng(self.seed * 1000000 + index)
        base_index = rng.choice(len(self.base_data))
        ablation_index = rng.choice(len(self.ablation_data))

        base_input = self.base_data[base_index]
        ablation_input = self.ablation_data[ablation_index]
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

    def __init__(self, hl_model:HookedRootModule=None, ll_model:HookedRootModule=None,
                 hl_graph=None, corr:dict[HLNode, set[LLNode]]={}, seed=0, training_args={}):
        # TODO change to construct hl_model from graph?
        if hl_model is None:
            assert hl_graph is not None
            hl_model = self.make_hl_model(hl_graph)

        self.hl_model = hl_model
        self.ll_model = ll_model

        self.corr:dict[HLNode, set[LLNode]] = corr
        assert all([k in self.hl_model.hook_dict for k in self.corr.keys()])
        self.rng = np.random.default_rng(seed)
        self.training_args = training_args

    def make_hl_model(self, hl_graph):
        raise NotImplementedError

    def set_corr(self, corr):
        self.corr = corr

    def sample_hl_name(self) -> str:
        # return a `str` rather than `numpy.str_`
        return str(self.rng.choice(list(self.corr.keys())))

    def hl_ablation_hook(self,hook_point_out:Tensor, hook:HookPoint) -> Tensor:
        out = self.hl_cache[hook.name]
        return out
    
    # TODO extend to position and subspace...
    def make_ll_ablation_hook(self, ll_node:LLNode) -> Callable[[Tensor, HookPoint], Tensor]:
        if ll_node.subspace is not None:
            raise NotImplementedError
        def ll_ablation_hook(hook_point_out:Tensor, hook:HookPoint) -> Tensor:
            out = hook_point_out.clone()
            out[ll_node.index.as_index] = self.ll_cache[hook.name][ll_node.index.as_index]
            return out
        return ll_ablation_hook

    def do_intervention(self, base_input, ablation_input, hl_node:HookName, verbose=False):
        ablation_x, ablation_y, ablation_intermediate_vars = ablation_input
        base_x, base_y, base_intermediate_vars = base_input
        hl_ablation_output, self.hl_cache = self.hl_model.run_with_cache(ablation_input)
        assert all(hl_ablation_output == ablation_y), f"Ablation output {hl_ablation_output} does not match label {ablation_y}"
        ll_ablation_output, self.ll_cache = self.ll_model.run_with_cache(ablation_x)

        ll_nodes = self.corr[hl_node]

        hl_output = self.hl_model.run_with_hooks(base_input, fwd_hooks=[(hl_node, self.hl_ablation_hook)])
        ll_output = self.ll_model.run_with_hooks(base_x, fwd_hooks=[(ll_node.name, self.make_ll_ablation_hook(ll_node)) for ll_node in ll_nodes])

        if verbose:
            ablation_x_image = torchvision.transforms.functional.to_pil_image(ablation_x[0])
            ablation_x_image.show()
            print(f"{ablation_x_image=}, {ablation_y.item()=}, {ablation_intermediate_vars=}")
            base_x_image = torchvision.transforms.functional.to_pil_image(base_x[0])
            base_x_image.show()
            print(f"{base_x_image=}, {base_y.item()=}, {base_intermediate_vars=}")
            print(f"{hl_ablation_output=}, {ll_ablation_output.shape=}")
            print(f"{hl_node=}, {ll_nodes=}")
            print(f"{hl_output=}")
        return hl_output, ll_output
    
    def no_intervention(self, base_input):
        base_x, base_y, base_intermediate_vars = base_input
        hl_output = self.hl_model(base_input)
        ll_output = self.ll_model(base_x)
        return hl_output, ll_output

    def train(self, base_data, ablation_data, test_base_data, test_ablation_data, epochs=1000, use_wandb=False):
        training_args = self.training_args
        print(f"{training_args=}")
        dataset = IITDataset(base_data, ablation_data)
        test_dataset = IITDataset(test_base_data, test_ablation_data)
        loader = DataLoader(dataset, batch_size=training_args['batch_size'], shuffle=True, num_workers=training_args['num_workers'])
        test_loader = DataLoader(test_dataset, batch_size=training_args['batch_size'], shuffle=True, num_workers=training_args['num_workers'])
        optimizer = t.optim.Adam(self.ll_model.parameters(), lr=training_args['lr'])
        loss_fn = t.nn.CrossEntropyLoss()

        if use_wandb:
            wandb.init(project="iit", entity=WANDB_ENTITY)

        for epoch in tqdm(range(epochs)):
            losses = []
            for i, (base_input, ablation_input) in tqdm(enumerate(loader), total=len(loader)):
                base_input = [t.to(DEVICE) for t in base_input]
                ablation_input = [t.to(DEVICE) for t in ablation_input]
                optimizer.zero_grad()
                self.hl_model.requires_grad_(False)
                self.ll_model.train()

                # sample a high-level variable to ablate
                hl_node = self.sample_hl_name()
                hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
                # hl_output, ll_output = self.no_intervention(base_input)
                loss = loss_fn(ll_output, hl_output)
                loss.backward()
                # print(f"{ll_output=}, {hl_output=}")
                losses.append(loss.item())
                optimizer.step()
            # now calculate test loss
            test_losses = []
            accuracies = []
            self.ll_model.eval()
            self.hl_model.requires_grad_(False)
            for i, (base_input, ablation_input) in enumerate(test_loader):
                base_input = [t.to(DEVICE) for t in base_input]
                ablation_input = [t.to(DEVICE) for t in ablation_input]
                hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
                # hl_output, ll_output = self.no_intervention(base_input)
                loss = loss_fn(ll_output, hl_output)
                top1 = t.argmax(ll_output, dim=1)
                accuracy = (top1 == hl_output).float().mean()
                accuracies.append(accuracy.item())
                test_losses.append(loss.item())
            print(f"Epoch {epoch}: {np.mean(losses):.4f}, {np.mean(test_losses):.4f}, {np.mean(accuracies)*100:.4f}%")

            if use_wandb:
                wandb.log({'train loss': np.mean(losses), 'test loss': np.mean(test_losses), 'accuracy': np.mean(accuracies), 'epoch': epoch})


# %%

hl_model = MNIST_PVR_HL().to(DEVICE)

resnet18 = torchvision.models.resnet18().to(DEVICE) # 11M parameters
wrapped_r18 = HookedModuleWrapper(resnet18, name='resnet18', recursive=True, hook_self=False).to(DEVICE)

# %%

training_args = {
    'batch_size': 256,
    'lr': 0.001,
    'num_workers': 0,
}

pad_size = 7
mnist_size = 28

dim_at_hook = 3
hook_point = 'mod.layer4.mod.1.mod.conv2.hook_point'
quadrant_size = (dim_at_hook) // 2 # conv has stride 2
channel_size = 512
channel_stride = 512 // 4

mnist_pvr_train = ImagePVRDataset(mnist_train, length=60000, pad_size=pad_size) # because first conv layer is 7
mnist_pvr_test = ImagePVRDataset(mnist_test, length=6000, pad_size=pad_size)

corr = {
    'hook_tl': {LLNode(hook_point, Ix[None, :channel_stride, :, :])},
    'hook_tr': {LLNode(hook_point, Ix[None, channel_stride:channel_stride*2, :, :])},
    'hook_bl': {LLNode(hook_point, Ix[None, channel_stride*2:channel_stride*3, :, :])},
    'hook_br': {LLNode(hook_point, Ix[None, channel_stride*3:, :, :])},
}

model_pair = IITModelPair(hl_model, ll_model=wrapped_r18, corr=corr, seed=0, training_args=training_args)

dataset = IITDataset(mnist_pvr_train, mnist_pvr_train)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
# print(len(loader))
# for i in loader:
#     print(i)
#     break
# %%
base_input, ablation_input = next(iter(loader))
base_input = [t.to(DEVICE) for t in base_input]
ablation_input = [t.to(DEVICE) for t in ablation_input]
_ = model_pair.do_intervention(base_input, ablation_input, 'hook_tl', verbose=True)
# %%
model_pair.train(mnist_pvr_train, mnist_pvr_train, mnist_pvr_test, mnist_pvr_test, epochs=1000, use_wandb=True)

print(f"done training")
