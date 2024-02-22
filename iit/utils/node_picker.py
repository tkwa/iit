import iit.utils.index as index
import iit.model_pairs as mp
from transformer_lens import HookedTransformer

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
    all_nodes = set(get_all_nodes(ll_model))
    nodes_in_circuit = set(get_nodes_in_circuit(hl_ll_corr))
    nodes_not_in_circuit = []
    for node in all_nodes:
        if not any(nodes_intersect(node, c) for c in nodes_in_circuit):
            nodes_not_in_circuit.append(node)
    return nodes_not_in_circuit
