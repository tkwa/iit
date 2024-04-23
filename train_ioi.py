import torch as t
import transformer_lens
from iit.tasks.ioi.ioi_dataset_tl import IOIDataset, IOIDatasetWrapper
from iit.utils.iit_dataset import IITDataset, train_test_split
import iit.model_pairs as mp
from iit.tasks.ioi.ioi_hl import IOI_HL
from iit.model_pairs.base_model_pair import *
from iit.utils.metric import *
from iit.tasks.ioi.ioi_config import NAMES
import os
import json

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
num_samples = 9000
epochs = 100
training_args = {
    "batch_size": 256,
    "lr": 1e-4,
    "num_workers": 0,
    "iit_weight": 1.0,
    "behavior_weight": 1.0,
    "strict_weight": 0.0,
    "next_token": True,
    "lr_scheduler": None,
    "grad_clip_norm": 1.0,
}
t.manual_seed(0)
np.random.seed(0)

ll_cfg = transformer_lens.HookedTransformer.from_pretrained("gpt2").cfg
ll_cfg.n_layers = 6
ll_cfg.n_heads = 4
ll_cfg.d_model = 64
ll_cfg.d_head = 64 // ll_cfg.n_heads

ll_cfg.init_weights = True
ll_model = transformer_lens.HookedTransformer(ll_cfg).to(DEVICE)

# TODO specify names, nouns, samples
ioi_dataset_tl = IOIDataset(
    num_samples=num_samples,
    tokenizer=ll_model.tokenizer,
    names=NAMES,
)

ioi_names = t.tensor(
    list(set([ioi_dataset_tl[i]["IO"].item() for i in range(len(ioi_dataset_tl))]))
).to(DEVICE)
hl_model = IOI_HL(d_vocab=ll_model.cfg.d_vocab_out, names=ioi_names).to(DEVICE)

ioi_dataset = IOIDatasetWrapper(
    num_samples=num_samples,
    tokenizer=ll_model.tokenizer,
    names=NAMES,
)

HookName = str
HLCache = dict
all_attns = [f"blocks.{i}.attn.hook_result" for i in range(ll_cfg.n_layers)]
all_mlps = [f"blocks.{i}.mlp.hook_post" for i in range(ll_cfg.n_layers)]
corr = {
    "hook_duplicate": {all_attns[0]},
    # "hook_previous": {"blocks.1.attn.hook_result"},
    "hook_s_inhibition": {all_attns[2], all_attns[3]},
    "hook_name_mover": {all_attns[4], all_attns[5]},

    "all_nodes_hook": {*all_mlps[:2]},
}
corr = {
    HLNode(k, -1): {LLNode(name=name, index=None) for name in v}
    for k, v in corr.items()
}

train_ioi_dataset, test_ioi_dataset = train_test_split(
    ioi_dataset, test_size=0.2, random_state=42
)
train_set = IITDataset(train_ioi_dataset, train_ioi_dataset, seed=0)
test_set = IITDataset(test_ioi_dataset, test_ioi_dataset, seed=0)

model_pair = mp.IOI_ModelPair(
    ll_model=ll_model, hl_model=hl_model, corr=corr, training_args=training_args
)
sentence = ioi_dataset_tl[0]["prompt"]
detokenised = [
    ll_model.tokenizer.decode(i, clean_up_tokenization_spaces=True) for i in sentence
]
print(sentence, detokenised)

model_pair.train(train_set, test_set, epochs=epochs, use_wandb=False)

print(f"done training")
# save model
save_dir = f"models/ioi/{model_pair.__class__.__name__}"
model_dir = f"{int(100*training_args['behavior_weight'])}_{int(100*training_args['iit_weight'])}_{int(100*training_args['strict_weight'])}"
os.makedirs(f"{save_dir}/{model_dir}", exist_ok=True)
torch.save(ll_model.state_dict(), f"{save_dir}/{model_dir}/ll_model.pth")

# save training args
with open(f"{save_dir}/{model_dir}/training_args.json", "w") as f:
    json.dump(training_args, f)

# dump model cfg
cfg = ll_model.cfg.to_dict()
with open(f"{save_dir}/{model_dir}/ll_model_cfg.json", "w") as f:
    f.write(str(cfg))

# log metrics
with open(f"{save_dir}/{model_dir}/metrics.log", "w") as f:
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Early stop: {model_pair._check_early_stop_condition(model_pair.test_metrics)}\n")
    f.write("\n\n--------------------------------\n\n")
    f.write("Training metrics:\n")
    f.write(str(model_pair.training_metrics))
    f.write("\n\n--------------------------------\n\n")
    f.write("Test metrics:\n")
    f.write(str(model_pair.test_metrics))