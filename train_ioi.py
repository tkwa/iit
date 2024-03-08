# %%
"""
This is one file for now; will eventually split into multiple files.
"""

import torch as t
from torch.utils.data import Dataset
import transformer_lens
from iit.tasks.ioi.ioi_dataset_tl import IOIDataset as IOIDatasetTL

import iit.model_pairs as mp
from iit.tasks.ioi.ioi_hl import IOI_HL

from iit.model_pairs.base_model_pair import *
from iit.utils.metric import *
from typing import final
from iit.tasks.ioi.ioi_config import NAMES

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

training_args = {
    'batch_size': 128,
    'lr': 0.01,
    'num_workers': 0,
    'iit_weight': 0.0,
    'behavior_weight': 1.0,
}

ll_cfg = transformer_lens.HookedTransformer.from_pretrained("gpt2").cfg
ll_cfg.n_layers = 6
ll_cfg.n_heads = 4
ll_cfg.d_model = 64
ll_cfg.d_head = 64 // ll_cfg.n_heads

ll_cfg.init_weights = True
ll_model = transformer_lens.HookedTransformer(ll_cfg).to(DEVICE)

# TODO specify names, nouns, samples
ioi_dataset_tl = IOIDatasetTL(
    num_samples=1000,
    tokenizer=ll_model.tokenizer,
    names=NAMES,
)



ioi_names = t.tensor(list(set([ioi_dataset_tl[i]['IO'].item() for i in range(len(ioi_dataset_tl))]))).to(DEVICE)
hl_model = IOI_HL(d_vocab=ll_model.cfg.d_vocab_out,
                  names=ioi_names).to(DEVICE)

class IOI_ModelPair(mp.IITBehaviorModelPair):
    def get_IIT_loss_over_batch(
        self,
        base_input,
        ablation_input,
        hl_node: HookName,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ):
        hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
        hl_output = t.nn.functional.softmax(hl_output, dim=-1)

        loss = loss_fn(ll_output[:, -1, :], hl_output[:, -1, :])
        return loss
    
    def run_eval_step(
        self,
        base_input,
        ablation_input,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ):
        atol = self.training_args["atol"]

        # compute IIT loss and accuracy on last token position only
        hl_node = self.sample_hl_name()
        hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
        # CrossEntropyLoss needs target probs, not logits
        hl_output = t.nn.functional.softmax(hl_output, dim=-1)
        assert self.hl_model.is_categorical()
        loss = loss_fn(ll_output[:, -1, :], hl_output[:, -1, :])
        if ll_output.shape == hl_output.shape:
            # To handle the case when labels are one-hot
            hl_output = t.argmax(hl_output, dim=-1)
        top1 = t.argmax(ll_output, dim=-1)
        accuracy = (top1[:, -1] == hl_output[:, -1]).float().mean()
        IIA = accuracy.item()

        # compute behavioral accuracy
        base_x, base_y, _ = base_input
        output = self.ll_model(base_x)
        top1 = t.argmax(output, dim=-1) # batch n_ctx
        if output.shape == base_y.shape:
            # To handle the case when labels are one-hot
            # TODO: is there a better way?
            base_y = t.argmax(base_y, dim=-1) # batch n_ctx
        accuracy = (top1 == base_y).float().mean()
        return {
            "val/iit_loss": loss.item(),
            "val/IIA": IIA,
            "val/accuracy": accuracy.item(),
        }



class IOIDataset(t.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        self.tl_dataset = IOIDatasetTL(*args, **kwargs)

    def __len__(self):
        return len(self.tl_dataset)
    
    def __getitem__(self, idx):
        x = self.tl_dataset[idx]
        prompt = x['prompt']
        y = list(prompt[1:])
        y = t.nn.functional.one_hot(t.tensor(y), num_classes=self.tl_dataset.tokenizer.vocab_size).float()
        return (x['prompt'][:-1].to(DEVICE), (y).to(DEVICE), (x['IO']).to(DEVICE))
    
ioi_dataset = IOIDataset(
    num_samples=1500,
    tokenizer=ll_model.tokenizer,
    names=NAMES,
)

HookName = str
HLCache = dict

corr = {
    'hook_duplicate': {'blocks.0.attn.hook_result'},
    'hook_previous': {'blocks.1.attn.hook_result'},
    'hook_s_inhibition': {'blocks.2.attn.hook_result'},
    'hook_name_mover': {'blocks.3.attn.hook_result'},
}
corr = {HLNode(k, -1): {LLNode(name=name, index=None) for name in v} for k, v in corr.items()}


# TODO split into train and test
train_set, test_set = IITDataset(ioi_dataset, ioi_dataset, seed=0), IITDataset(ioi_dataset, ioi_dataset, seed=1)

model_pair = IOI_ModelPair(ll_model=ll_model, hl_model=hl_model,
                          corr = corr,
                          training_args=training_args)

model_pair.train(train_set, test_set, epochs=10, use_wandb=False)

print(f"done training")
print([ll_model.tokenizer.decode(ioi_dataset_tl[i]['prompt']) for i in range(5)])
# %%