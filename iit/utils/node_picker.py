import iit.utils.index as index
import iit.model_pairs as mp
from transformer_lens import HookedTransformer
import torch as t

def get_all_nodes(model: HookedTransformer) -> list[mp.LLNode]:
    nodes = []
    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers
    for layer in range(n_layers):
        hook_point = f"blocks.{layer}.attn.hook_result"
        for head in range(n_heads):
            head_node = mp.LLNode(hook_point, index.Ix[:, :, head, :])
            nodes.append(head_node)
        hook_point = f"blocks.{layer}.mlp.hook_post"
        nodes.append(mp.LLNode(hook_point, index.Ix[[None]]))
    return nodes


def get_nodes_in_circuit(hl_ll_corr) -> list[mp.LLNode]:
    nodes_in_circuit = set()
    for hl_node, ll_nodes in hl_ll_corr.items():
        nodes_in_circuit.update(ll_nodes)
    return list(nodes_in_circuit)


def nodes_intersect(a: mp.LLNode, b: mp.LLNode) -> bool:
    # return true if there is any intersection
    if a.name != b.name:
        return False
    return a.index.intersects(b.index)


def get_nodes_not_in_circuit(
    ll_model: HookedTransformer, hl_ll_corr
) -> list[mp.LLNode]:
    all_nodes = get_all_nodes(ll_model)
    nodes_in_circuit = get_nodes_in_circuit(hl_ll_corr)
    nodes_not_in_circuit = []
    for node in all_nodes:
        if not any(nodes_intersect(node, c) for c in nodes_in_circuit):
            nodes_not_in_circuit.append(node)
    return nodes_not_in_circuit

def _get_param_idx(name: str, param: t.nn.parameter.Parameter, node: mp.LLNode) -> index.TorchIndex:
    param_type = name.split('.')[-1]
    node_idx = node.index
    none_ix = index.Ix[[None]]

    if node.subspace is not None:
        raise NotImplementedError("Subspaces are not supported")
    if node_idx == none_ix or param_type == 'b_O':
        param_idx = none_ix
    elif param_type in ['W_Q', 'W_K', 'W_V']:
        param_idx = index.TorchIndex(node_idx.as_index[:-1])
    elif param_type == 'W_O':
        idx_tuple = list(node_idx.as_index[:-1])
        param_idx = index.TorchIndex([idx_tuple[0], idx_tuple[2], idx_tuple[1]])
    elif param_type in ['b_Q', 'b_K', 'b_V']:
        idx_tuple = list(node_idx.as_index[:-1])
        param_idx = index.TorchIndex([slice(None), idx_tuple[2]])
    else:
        raise NotImplementedError(f"Param of type \'{param_type}\' is expected to have index {none_ix}, but got {node_idx}")
    
    try:
        param[param_idx.as_index]
    except IndexError:
        raise IndexError(f"Index {param_idx} is out of bounds for param {name}")
    
    return param_idx

def get_params_in_circuit(hl_ll_corr: dict[str, set[mp.LLNode]], ll_model: HookedTransformer) -> list[mp.LLNode]:
    nodes_in_circuit = get_nodes_in_circuit(hl_ll_corr)
    params_in_circuit = []
    for name, param in ll_model.named_parameters():
        for node in nodes_in_circuit:
            node_name = ".".join(node.name.split('.')[:-1])
            param_name = ".".join(name.split('.')[:-1])
            if node_name == param_name:
                param_idx = _get_param_idx(name, param, node)
                params_in_circuit.append(mp.LLNode(name, param_idx))
    return params_in_circuit