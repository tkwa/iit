# %%
from torch.utils.data import DataLoader
import torchvision
from utils import HookedModuleWrapper
import wandb
from pvr import mnist_train, mnist_test, MNIST_PVR_HL, ImagePVRDataset
from utils.index import Ix
from utils.config import *
from utils import IITDataset
from model_pairs import *

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

mode = 'q' # 'q' or 'c'
pad_size = 7
mnist_size = 28

mnist_pvr_train = ImagePVRDataset(mnist_train, length=60000, pad_size=pad_size) # because first conv layer is 7
mnist_pvr_test = ImagePVRDataset(mnist_test, length=6000, pad_size=pad_size)

if mode == 'c':
    hook_point = 'mod.layer4.mod.1.mod.conv2.hook_point'
    channel_size = 512
    channel_stride = 512 // 4
    corr = {
        'hook_tl': {LLNode(hook_point, Ix[None, :channel_stride, None, None])},
        'hook_tr': {LLNode(hook_point, Ix[None, channel_stride:channel_stride*2, None, None])},
        'hook_bl': {LLNode(hook_point, Ix[None, channel_stride*2:channel_stride*3, None, None])},
        'hook_br': {LLNode(hook_point, Ix[None, channel_stride*3:, None, None])},
    }
elif mode == 'q':
    hook_point = 'mod.layer3.mod.1.mod.conv2.hook_point'
    dim_at_hook = 6
    quadrant_size = (dim_at_hook) // 2 # conv has stride 2
    corr = {
        'hook_tl': {LLNode(hook_point, Ix[None, None, :quadrant_size, :quadrant_size])},
        'hook_tr': {LLNode(hook_point, Ix[None, None, :quadrant_size, quadrant_size:])},
        'hook_bl': {LLNode(hook_point, Ix[None, None, quadrant_size:, :quadrant_size])},
        'hook_br': {LLNode(hook_point, Ix[None, None, quadrant_size:, quadrant_size:])},
    }

model_pair = IITProbeSequentialPair(hl_model, ll_model=wrapped_r18, corr=corr, seed=0, training_args=training_args)

dataset = IITDataset(mnist_pvr_train, mnist_pvr_train)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# %%
# base_input, ablation_input = next(iter(loader))
# base_input = [t.to(DEVICE) for t in base_input]
# ablation_input = [t.to(DEVICE) for t in ablation_input]
# _ = model_pair.do_intervention(base_input, ablation_input, 'hook_tl', verbose=True)

# %%
model_pair.train(mnist_pvr_train, mnist_pvr_train, mnist_pvr_test, mnist_pvr_test, epochs=1000, use_wandb=False)

print(f"done training")
