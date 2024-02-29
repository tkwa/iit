# %%
"""
This is one file for now; will eventually split into multiple files.
"""

import torch as t
import transformer_lens
import transformer_lens.evals as evals
from iit.tasks.ioi.ioi_hl import IOI_HL

DEVICE = 'cuda'

hl_model = IOI_HL(device=DEVICE)
ll_model = transformer_lens.HookedTransformer.from_pretrained("gpt2").to(DEVICE)

ioi_dataset = evals.IOIDataset(
    tokenizer=ll_model.tokenizer,
    num_samples=1000
)

train_set, test_set = ioi_dataset.get_train_test_split()
# %%

