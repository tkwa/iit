import torch as t
import transformer_lens
from iit.utils.iit_dataset import IITDataset, train_test_split
import iit.model_pairs as mp
from iit.model_pairs.base_model_pair import *
from iit.utils.metric import *
from iit.tasks.ioi import NAMES, make_ioi_dataset_and_hl, corr, corr_dict, ioi_cfg
import os
import json

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
num_samples = 9000
epochs = 100
training_args = {
    "batch_size": 256,
    "lr": 1e-4,
    "iit_weight": 1.0,
    "behavior_weight": 1.0,
    "strict_weight": 0.4,
    "next_token": True,
    "lr_scheduler": None,  
    # "clip_grad_norm": 1.0,
    "early_stop": True,
    "use_single_loss": False,
}
t.manual_seed(0)
np.random.seed(0)

ll_cfg = transformer_lens.HookedTransformer.from_pretrained("gpt2").cfg.to_dict()
ll_cfg.update(ioi_cfg)

ll_cfg["init_weights"] = True
ll_model = transformer_lens.HookedTransformer(ll_cfg).to(DEVICE)

ioi_dataset, hl_model = make_ioi_dataset_and_hl(num_samples, ll_model, NAMES, verbose=True)

train_ioi_dataset, test_ioi_dataset = train_test_split(
    ioi_dataset, test_size=0.2, random_state=42
)
train_set = IITDataset(train_ioi_dataset, train_ioi_dataset, seed=0)
test_set = IITDataset(test_ioi_dataset, test_ioi_dataset, seed=0)

model_pair = mp.IOI_ModelPair(
    ll_model=ll_model, hl_model=hl_model, corr=corr, training_args=training_args
)

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
    f.write(f"Early stop: {model_pair._check_early_stop_condition(model_pair.test_metrics.metrics)}\n")
    f.write("\n\n--------------------------------\n\n")
    f.write("Training metrics:\n")
    f.write(str(model_pair.train_metrics.metrics))
    f.write("\n\n--------------------------------\n\n")
    f.write("Test metrics:\n")
    f.write(str(model_pair.test_metrics.metrics))

# save the model
torch.save(ll_model.state_dict(), f"{save_dir}/{model_dir}/ll_model.pth")

# save corr dict
with open(f"{save_dir}/{model_dir}/corr.json", "w") as f:
    json.dump(corr_dict, f)