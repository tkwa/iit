# %%
"""
This is one file for now; will eventually split into multiple files.
"""

import torch as t
from torch.utils.data import Dataset
import transformer_lens
from ioi_dataset_tl import IOIDataset as IOIDatasetTL

from iit.model_pairs import IITModelPair
from iit.tasks.ioi.ioi_hl import IOI_HL

from iit.model_pairs.base_model_pair import *
from iit.utils.metric import *
from typing import final

DEVICE = 'cuda'

training_args = {
    'batch_size': 1,
    'lr': 0.001,
    'num_workers': 0,
}

ll_model = transformer_lens.HookedTransformer.from_pretrained("gpt2").to(DEVICE)

# TODO specify names, nouns, samples
ioi_dataset_tl = IOIDatasetTL(
    num_samples=15,
    tokenizer=ll_model.tokenizer,
)
# %%

ioi_names = t.tensor(list(set([ioi_dataset_tl[i]['IO'].item() for i in range(len(ioi_dataset_tl))]))).to(DEVICE)
hl_model = IOI_HL(d_vocab=ll_model.cfg.d_vocab_out,
                  names=ioi_names).to(DEVICE)

# %%


class IOI_IITDataset(t.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        self.tl_dataset = IOIDatasetTL(*args, **kwargs)

    def __len__(self):
        return len(self.tl_dataset)
    
    def __getitem__(self, idx):
        x = self.tl_dataset[idx]
        return (x['prompt'].to(DEVICE), t.tensor(()), t.tensor(()))
    
ioi_dataset = IOI_IITDataset(
    num_samples=15,
    tokenizer=ll_model.tokenizer,
)
# ioi_dataset.add_tokenizer(ll_model.tokenizer)

HookName = str
HLCache = dict
class IOI_IITModelPair(IITModelPair):
    def do_intervention(
        self, base_input, ablation_input, hl_node: HookName, verbose=False
    ) -> tuple[Tensor, Tensor]:
        # overriding from BaseModelPair, but we don't check input
        ablation_x, ablation_y, ablation_intermediate_vars = ablation_input
        base_x, base_y, base_intermediate_vars = base_input
        hl_ablation_output, self.hl_cache = self.hl_model.run_with_cache(ablation_input)

        # assert all(
        #     hl_ablation_output == ablation_y
        # ), f"Ablation output {hl_ablation_output} does not match label {ablation_y}"

        ll_ablation_output, self.ll_cache = self.ll_model.run_with_cache(ablation_x)
        ll_nodes = self.corr[hl_node]

        hl_output = self.hl_model.run_with_hooks(
            base_input, fwd_hooks=[(hl_node, self.hl_ablation_hook)]
        )
        ll_output = self.ll_model.run_with_hooks(
            base_x,
            fwd_hooks=[
                (ll_node.name, self.make_ll_ablation_hook(ll_node))
                for ll_node in ll_nodes
            ],
        )

        if verbose:
            print(f"{base_x=}, {base_y.item()=}")
            print(f"{hl_ablation_output=}, {ll_ablation_output.shape=}")
            print(f"{hl_node=}, {ll_nodes=}")
            print(f"{hl_output=}")
        return hl_output, ll_output
        
    def run_eval_step(
        self,
        base_input,
        ablation_input,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ):
        base_input = [t.to(DEVICE) for t in base_input]
        ablation_input = [t.to(DEVICE) for t in ablation_input]
        hl_node = self.sample_hl_name() # sample a high-level variable to ablate
        hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
        loss = loss_fn(ll_output, hl_output)
        top1 = t.argmax(ll_output[:, -1], dim=-1)
        top1_hl = t.argmax(hl_output[:, -1], dim=-1)
        accuracy = (top1 == top1_hl).float().mean()
        return {
            "iit_loss": loss.item(),
            "accuracy": accuracy.item(),
        }

corr = {
    'hook_duplicate': {'blocks.2.attn.hook_result'},
    'hook_previous': {'blocks.4.attn.hook_result'},
    'hook_s_inhibition': {'blocks.6.attn.hook_result'},
    'hook_name_mover': {'blocks.8.attn.hook_result'},
}
corr = {k: {LLNode(name=name, index=None) for name in v} for k, v in corr.items()}


# TODO split into train and test
train_set, test_set = ioi_dataset, ioi_dataset

model_pair = IOI_IITModelPair(ll_model=ll_model, hl_model=hl_model,
                          corr = corr,
                          training_args=training_args)

def cross_entropy_last_loss_fn(output, target):
    """
    IIT loss needs two loss functions:
    - between LL output logits and HL target logits
    - between LL output logits and labels (implement later)
    """
    print(output.shape, target.shape)
    return t.nn.CrossEntropyLoss()(output[:, -1, :], target[:, -1, :]) # batch seq vocab

model_pair.loss_fn = cross_entropy_last_loss_fn
# %%
model_pair.train(train_set, train_set, test_set, test_set, epochs=10, use_wandb=False)

print(f"done training")
# %%

[ll_model.tokenizer.decode(ioi_dataset_tl[i]['prompt']) for i in range(5)]

# %%
