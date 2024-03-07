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

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

training_args = {
    'batch_size': 32,
    'lr': 0.001,
    'num_workers': 0,
    'iit_weight': 0.0,
    'behavior_weight': 1.0,
}

ll_model = transformer_lens.HookedTransformer.from_pretrained("gpt2").to(DEVICE)

# TODO specify names, nouns, samples
ioi_dataset_tl = IOIDatasetTL(
    num_samples=15,
    tokenizer=ll_model.tokenizer,
)

ioi_names = t.tensor(list(set([ioi_dataset_tl[i]['IO'].item() for i in range(len(ioi_dataset_tl))]))).to(DEVICE)
hl_model = IOI_HL(d_vocab=ll_model.cfg.d_vocab_out,
                  names=ioi_names).to(DEVICE)



class IOIDataset(t.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        self.tl_dataset = IOIDatasetTL(*args, **kwargs)

    def __len__(self):
        return len(self.tl_dataset)
    
    def __getitem__(self, idx):
        x = self.tl_dataset[idx]
        prompt = x['prompt']
        eos_token = self.tl_dataset.tokenizer.eos_token_id
        y = list(prompt[1:]) + [eos_token]
        y = t.nn.functional.one_hot(t.tensor(y), num_classes=self.tl_dataset.tokenizer.vocab_size).float()
        return (x['prompt'].to(DEVICE), (y).to(DEVICE), (x['IO']).to(DEVICE))
    
ioi_dataset = IOIDataset(
    num_samples=15,
    tokenizer=ll_model.tokenizer,
)

HookName = str
HLCache = dict

corr = {
    'hook_duplicate': {'blocks.2.attn.hook_result'},
    'hook_previous': {'blocks.4.attn.hook_result'},
    'hook_s_inhibition': {'blocks.6.attn.hook_result'},
    'hook_name_mover': {'blocks.8.attn.hook_result'},
}
corr = {HLNode(k, -1): {LLNode(name=name, index=None) for name in v} for k, v in corr.items()}


# TODO split into train and test
train_set, test_set = IITDataset(ioi_dataset, ioi_dataset, seed=0), IITDataset(ioi_dataset, ioi_dataset, seed=1)

model_pair = mp.IITBehaviorModelPair(ll_model=ll_model, hl_model=hl_model,
                          corr = corr,
                          training_args=training_args)

model_pair.train(train_set, test_set, epochs=10, use_wandb=False)

print(f"done training")
print([ll_model.tokenizer.decode(ioi_dataset_tl[i]['prompt']) for i in range(5)])
