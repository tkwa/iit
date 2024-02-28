from iit.utils.node_picker import *
from iit.model_pairs.base_model_pair import LLNode, HLNode
from iit.utils.index import Ix
import iit.utils.index as index

def test_get_all_nodes():
    cfg = {
        "n_layers": 2,
        "n_heads": 4,
        "d_model": 8,
        "d_head": 2,
        "d_mlp": 16,
        "n_ctx": 16,
        "act_fn": "gelu",
        "d_vocab": 21
    }
    model = HookedTransformer(cfg)
    assert get_all_nodes(model) == [
        LLNode("blocks.0.attn.hook_result", Ix[:, :, 0, :]),
        LLNode("blocks.0.attn.hook_result", Ix[:, :, 1, :]),
        LLNode("blocks.0.attn.hook_result", Ix[:, :, 2, :]),
        LLNode("blocks.0.attn.hook_result", Ix[:, :, 3, :]),
        LLNode("blocks.0.mlp.hook_post", Ix[[None]]),
        LLNode("blocks.1.attn.hook_result", Ix[:, :, 0, :]),
        LLNode("blocks.1.attn.hook_result", Ix[:, :, 1, :]),
        LLNode("blocks.1.attn.hook_result", Ix[:, :, 2, :]),
        LLNode("blocks.1.attn.hook_result", Ix[:, :, 3, :]),
        LLNode("blocks.1.mlp.hook_post", Ix[[None]]),
    ]

def test_get_params_in_circuit():
    hl_model_cfg = {
        'n_layers': 2,
        'd_model': 13,
        'n_ctx': 5,
        'd_head': 6,
        'n_heads': 1,
        'd_mlp': 1,
        'act_fn': 'relu',
        'd_vocab': 6,
    }

    ll_cfg = {
        "n_layers": 2,
        "n_heads": 4,
        "d_head": 3,
        "d_model": 12,
        "d_mlp": 16,
        'act_fn': 'relu',
        'd_vocab': 6,
        'n_ctx': 5,
    }

    hl_model = HookedTransformer(hl_model_cfg)
    ll_model = HookedTransformer(ll_cfg)

    hl_ll_corr = {
        "blocks.0.mlp.hook_post": {
            LLNode(name='blocks.0.mlp.hook_post', index=index.Ix[[None]], subspace=None)
            },
        "blocks.1.attn.hook_result": {
            LLNode(name='blocks.1.attn.hook_result', index=index.Ix[:, :, :2, :], subspace=None)
            }
        }
    
    assert get_params_in_circuit(hl_ll_corr, ll_model) == [
        LLNode(name='blocks.0.mlp.W_in', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.0.mlp.b_in', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.0.mlp.W_out', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.0.mlp.b_out', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.W_Q', index=Ix[:, :, :2], subspace=None),
        LLNode(name='blocks.1.attn.W_K', index=Ix[:, :, :2], subspace=None),
        LLNode(name='blocks.1.attn.W_V', index=Ix[:, :, :2], subspace=None),
        LLNode(name='blocks.1.attn.W_O', index=Ix[:, :2, :], subspace=None),
        LLNode(name='blocks.1.attn.b_Q', index=Ix[:, :2], subspace=None),
        LLNode(name='blocks.1.attn.b_K', index=Ix[:, :2], subspace=None),
        LLNode(name='blocks.1.attn.b_V', index=Ix[:, :2], subspace=None),
        LLNode(name='blocks.1.attn.b_O', index=Ix[[None]], subspace=None)
    ]

    hl_ll_corr = {
        "blocks.0.mlp.hook_post": {
            LLNode(name='blocks.0.mlp.hook_post', index=index.Ix[[None]], subspace=None)
            },
        "blocks.1.attn.hook_result": {
            LLNode(name='blocks.1.attn.hook_result', index=index.Ix[[None]], subspace=None)
            }
        }
    
    assert get_params_in_circuit(hl_ll_corr, ll_model) == [
        LLNode(name='blocks.0.mlp.W_in', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.0.mlp.b_in', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.0.mlp.W_out', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.0.mlp.b_out', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.W_Q', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.W_K', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.W_V', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.W_O', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.b_Q', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.b_K', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.b_V', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.b_O', index=Ix[[None]], subspace=None)
    ]