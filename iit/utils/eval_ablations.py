import iit.model_pairs as mp
import torch as t
from typing import Dict
from iit.utils.node_picker import *
from tqdm import tqdm
from iit.utils.iit_dataset import IITDataset
import pandas as pd
from transformer_lens.HookedTransformer import HookPoint
import dataframe_image as dfi
import os


def do_intervention(model_pair: mp.BaseModelPair, base_input, ablation_input, node: mp.LLNode, hooker: callable):
    _, cache = model_pair.ll_model.run_with_cache(ablation_input)
    model_pair.ll_cache = cache  # TODO: make this better when converting to script
    out = model_pair.ll_model.run_with_hooks(base_input, fwd_hooks=[(node.name, hooker)])
    return out


def resample_ablate_node(
    model_pair: mp.IITModelPair,
    base_in: tuple[t.Tensor, t.Tensor, t.Tensor],
    ablation_in: tuple[t.Tensor, t.Tensor, t.Tensor],
    node: mp.LLNode,
    results: Dict[str, float],
    hooker: callable,
    atol=5e-2,
    verbose=False,
):  # TODO: change name to reflect that it's not just for resampling
    base_x, base_y, _ = base_in
    ablation_x, ablation_y, _ = ablation_in
    ll_out = do_intervention(model_pair, base_x, ablation_x, node, hooker)
    if verbose:
        print(node)

    if model_pair.hl_model.is_categorical():
        # TODO: add other metrics here
        base_hl_out = model_pair.hl_model(base_in).squeeze()
        label_idx = model_pair.get_label_idxs()
        base_label = t.argmax(base_y, dim=-1)[label_idx.as_index]
        ablation_label = t.argmax(ablation_y, dim=-1)[label_idx.as_index]
        label_unchanged = base_label == ablation_label
        # take argmax of ll_out
        ll_out = t.argmax(ll_out, dim=-1)[label_idx.as_index]
        hl_out = t.argmax(base_hl_out, dim=-1)[label_idx.as_index]
        ll_unchanged = ll_out == hl_out
        changed_result = (~label_unchanged).cpu().float() * (~ll_unchanged).cpu().float()
        results[node] += changed_result.sum().item() / (~label_unchanged).float().sum().item()
    else:
        base_hl_out = model_pair.hl_model(base_in).squeeze()
        label_unchanged = base_y == ablation_y
        ll_unchanged = t.isclose(ll_out.float().squeeze(), base_hl_out.float().to(ll_out.device), atol=atol)
        changed_result = (~label_unchanged).cpu().float() * (~ll_unchanged).cpu().float()
        results[node] += changed_result.sum().item() / (~label_unchanged).float().sum().item()

        if verbose:
            print(
                "\nlabel changed:",
                (~label_unchanged).float().mean(),
                "\nouts_changed:",
                (~ll_unchanged).float().mean(),
                "\ndot product:",
                changed_result.mean(),
                "\ndifference:",
                (ll_out.float().squeeze() - base_y.float().to(ll_out.device)).mean(),
                "\nfinal:",
                results[node],
            )


def check_causal_effect(
    model_pair: mp.BaseModelPair,
    dataset: IITDataset,
    batch_size: int = 256,
    node_type: str = "a",
    verbose: bool = False,
):
    assert node_type in ["a", "c", "n"], "type must be one of 'a', 'c', or 'n'"
    hookers = {}
    results = {}
    all_nodes = (
        get_nodes_not_in_circuit(model_pair.ll_model, model_pair.corr)
        if node_type == "n"
        else get_all_nodes(model_pair.ll_model) if node_type == "a" else get_nodes_in_circuit(model_pair.corr)
    )

    for node in all_nodes:
        hookers[node] = model_pair.make_ll_ablation_hook(node)
        results[node] = 0

    loader = dataset.make_loader(batch_size=batch_size, num_workers=0)
    for base_in, ablation_in in tqdm(loader):
        for node, hooker in hookers.items():
            resample_ablate_node(model_pair, base_in, ablation_in, node, results, hooker, verbose=verbose)

    for node, result in results.items():
        results[node] = result / len(loader)
    return results


def get_mean_cache(model_pair, dataset, batch_size=8):
    loader = dataset.make_loader(batch_size=batch_size, num_workers=0)
    mean_cache = {}
    for batch in tqdm(loader):
        _, cache = model_pair.ll_model.run_with_cache(batch[0])
        for node, tensor in cache.items():
            if node not in mean_cache:
                mean_cache[node] = t.zeros_like(tensor[0].unsqueeze(0))
            mean_cache[node] += tensor.mean(dim=0).unsqueeze(0) / len(loader)
    return mean_cache               


def make_ablation_hook(node: mp.LLNode, mean_cache: dict[str, t.Tensor], use_mean_cache: bool = True) -> callable:
    if node.subspace is not None:
        raise NotImplementedError("Subspace not supported yet.")

    def zero_hook(hook_point_out: t.Tensor, hook: HookPoint) -> t.Tensor:
        hook_point_out[node.index.as_index] = 0
        return hook_point_out

    def mean_hook(hook_point_out: t.Tensor, hook: HookPoint) -> t.Tensor:
        cached_tensor = mean_cache[node.name]
        hook_point_out[node.index.as_index] = cached_tensor[node.index.as_index]
        return hook_point_out

    if use_mean_cache:
        return mean_hook
    return zero_hook


def ablate_node(
    model_pair: mp.IITModelPair,
    base_in: tuple[t.Tensor, t.Tensor, t.Tensor],
    node: mp.LLNode,
    results: Dict[str, float],
    hook: callable,
    atol=5e-2,
    verbose=False,
):
    base_x, base_y, _ = base_in
    ll_out = model_pair.ll_model.run_with_hooks(base_x, fwd_hooks=[(node.name, hook)])

    if model_pair.hl_model.is_categorical():
        # TODO: add other metrics here
        base_hl_out = model_pair.hl_model(base_in).squeeze()
        base_ll_out = model_pair.ll_model(base_x).squeeze()
        label_idx = model_pair.get_label_idxs()
        ll_out = t.argmax(ll_out, dim=-1)[label_idx.as_index]
        base_hl_out = t.argmax(base_hl_out, dim=-1)[label_idx.as_index]
        base_ll_out = t.argmax(base_ll_out, dim=-1)[label_idx.as_index]
        ll_unchanged = ll_out == base_hl_out
        accuracy = base_ll_out == base_hl_out
        changed_result = (~ll_unchanged).cpu().float() * accuracy.cpu().float()
        results[node] += changed_result.sum().item() / (accuracy.float().sum().item() + 1e-6)
    else:
        base_hl_out = model_pair.hl_model(base_in).squeeze()
        base_ll_out = model_pair.ll_model(base_x).squeeze()
        ll_unchanged = t.isclose(ll_out.float().squeeze(), base_hl_out.float().to(ll_out.device), atol=atol)
        accuracy = t.isclose(base_ll_out.float(), base_hl_out.float(), atol=atol).cpu().float()
        changed_result = (~ll_unchanged).cpu().float() * accuracy
        results[node] += changed_result.sum().item() / (accuracy.float().sum().item() + 1e-6)


def check_causal_effect_on_ablation(
    model_pair: mp.BaseModelPair,
    dataset: IITDataset,
    batch_size: int = 256,
    node_type: str = "a",
    use_mean_cache: bool = False,
    verbose: bool = False,
):
    if use_mean_cache:
        mean_cache = get_mean_cache(model_pair, dataset)
    assert node_type in ["a", "c", "n"], "type must be one of 'a', 'c', or 'n'"
    hookers = {}
    results = {}
    all_nodes = (
        get_nodes_not_in_circuit(model_pair.ll_model, model_pair.corr)
        if node_type == "n"
        else get_all_nodes(model_pair.ll_model) if node_type == "a" else get_nodes_in_circuit(model_pair.corr)
    )

    for node in all_nodes:
        hookers[node] = make_ablation_hook(node, mean_cache, use_mean_cache)
        results[node] = 0

    loader = dataset.make_loader(batch_size=batch_size, num_workers=0)
    for base_in in tqdm(loader):
        for node, hooker in hookers.items():
            ablate_node(model_pair, base_in, node, results, hooker, verbose=verbose)

    for node, result in results.items():
        results[node] = result / len(loader)
    return results


def make_dataframe_of_results(result_not_in_circuit, result_in_circuit):
    def create_name(node):
        if "mlp" in node.name:
            return node.name
        if node.index is not None and node.index!=index.Ix[[None]]:
            return f"{node.name}, head {str(node.index).split(',')[-2]}"
        else:
            return f"{node.name}, head [:]"
    
    df = pd.DataFrame(
        {
            "node": [create_name(node) for node in result_not_in_circuit.keys()]
            + [create_name(node) for node in result_in_circuit.keys()],
            "status": ["not_in_circuit"] * len(result_not_in_circuit) + ["in_circuit"] * len(result_in_circuit),
            "causal effect": list(result_not_in_circuit.values()) + list(result_in_circuit.values()),
        }
    )
    df = df.sort_values("status", ascending=False)
    return df


def make_combined_dataframe_of_results(
    result_not_in_circuit,
    result_in_circuit,
    za_result_not_in_circuit,
    za_result_in_circuit,
    use_mean_cache: bool = False,
):
    df = make_dataframe_of_results(result_not_in_circuit, result_in_circuit)
    df2 = make_dataframe_of_results(za_result_not_in_circuit, za_result_in_circuit)
    df2_causal_effect = df2.pop("causal effect")
    # rename the columns
    df["resample_ablate_effect"] = df.pop("causal effect")
    if use_mean_cache:
        df["mean_ablate_effect"] = df2_causal_effect
    else:
        df["zero_ablate_effect"] = df2_causal_effect

    return df


def save_result(df: pd.DataFrame, save_dir: str, model_pair: mp.BaseModelPair = None):
    os.makedirs(save_dir, exist_ok=True)
    dfi.export(df, f"{save_dir}/results.png")
    df.to_csv(f"{save_dir}/results.csv")
    if model_pair is None:
        return
    training_args = model_pair.training_args
    with open(f"{save_dir}/meta.log", "w") as f:
        f.write(str(training_args))