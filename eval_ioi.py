import torch
import argparse
import os
import transformer_lens
from iit.tasks.ioi import make_ioi_dataset_and_hl, NAMES, ioi_cfg, make_ioi_corr_from_dict, corr_dict
from iit.utils.eval_ablations import *
import numpy as np
from iit.utils.iit_dataset import IITDataset
from iit.utils.eval_datasets import IITUniqueDataset
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="IIT evaluation")
parser.add_argument("-w", "--weights", type=str, default="100_100_0", help="IIT_behavior_strict weights")
parser.add_argument("-m", "--mean", type=bool, default=True, help="Use mean cache")
parser.add_argument("-c", "--class_name", type=str, default="IOI_ModelPair", help="Model pair class to use")
parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size for making mean cache (if using mean ablation)")
args = parser.parse_args()

weights = args.weights
use_mean_cache = args.mean
class_name = args.class_name
save_dir = f"models/ioi/{class_name}/{weights}"
batch_size = args.batch_size
# load model
ll_cfg = transformer_lens.HookedTransformer.from_pretrained("gpt2").cfg.to_dict()
ll_cfg.update(ioi_cfg)

# ll_cfg.init_weights = True
ll_model = transformer_lens.HookedTransformer(ll_cfg).to(DEVICE)

try:
    ll_model.load_state_dict(torch.load(f"{save_dir}/ll_model.pth"))
except FileNotFoundError:
    raise FileNotFoundError(f"Model not found at {save_dir}")

# load corr
if os.path.exists(f"{save_dir}/corr.json"):
    corr_dict = json.load(open(f"{save_dir}/corr.json"))
else:
    print("WARNING: No corr.json found, using default corr_dict")
corr = make_ioi_corr_from_dict(corr_dict)

# load dataset
num_samples = 18000

ioi_dataset, hl_model = make_ioi_dataset_and_hl(num_samples, ll_model, NAMES, verbose=True)

test_set = IITDataset(ioi_dataset, ioi_dataset, seed=0)

model_pair = mp.IOI_ModelPair(
    ll_model=ll_model, hl_model=hl_model, corr=corr
)

np.random.seed(0)
t.manual_seed(0)
result_not_in_circuit = check_causal_effect(model_pair, test_set, node_type="n", verbose=True)
result_in_circuit = check_causal_effect(model_pair, test_set, node_type="c", verbose=False)

metric_collection = model_pair._run_eval_epoch(test_set.make_loader(256, 0), model_pair.loss_fn)

# zero/mean ablation
uni_test_set = IITUniqueDataset(ioi_dataset, ioi_dataset, seed=0)
za_result_not_in_circuit, za_result_in_circuit = get_causal_effects_for_all_nodes(model_pair, uni_test_set, batch_size=batch_size, use_mean_cache=use_mean_cache)

df = make_combined_dataframe_of_results(result_not_in_circuit, result_in_circuit, za_result_not_in_circuit, za_result_in_circuit, use_mean_cache=use_mean_cache)

save_dir = save_dir + "/results"
save_result(df, save_dir, model_pair)
with open(f"{save_dir}/metric_collection.log", "w") as f:
    f.write(str(metric_collection))
    print("Results saved at", save_dir)
    print(metric_collection)