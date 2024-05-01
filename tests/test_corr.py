from iit.utils.correspondence import *


def test_Correspondence():
    corr = Correspondence()
    hl_node = HLNode("hl", -1)
    ll_node = LLNode("ll", None)
    corr[hl_node] = ll_node
    assert corr[hl_node] == ll_node
    assert corr.get_suffixes() == {"attn": "attn.hook_result", "mlp": "mlp.hook_post"}
    assert type(corr[hl_node]) == LLNode

