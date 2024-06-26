from iit.model_pairs.nodes import HLNode, LLNode

class Correspondence(dict):
    def __init__(
        self,
        *args,
        suffixes={"attn": "attn.hook_result", "mlp": "mlp.hook_post"},
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.suffixes = suffixes

    def __setattr__(self, key, value):
        if key == "suffixes":
            assert isinstance(value, dict), ValueError(
                f"__value is not a dict, but {type(value)}"
            )
        else:
            assert isinstance(key, HLNode), "key must be of type HLNode, got %s" % type(
                key
            )
            assert isinstance(value, set), ValueError(
                f"__value is not a set, but {type(value)}"
            )
            assert all(isinstance(v, LLNode) for v in value), ValueError(
                f"__value contains non-LLNode elements"
            )
        # print(self.keys(), self.values())
        super().__setattr__(key, value)

    def get_suffixes(self):
        return self.suffixes

    @staticmethod
    def get_hook_suffix(corr: dict[HLNode, set[LLNode]]) -> dict[str, str]:
        suffixes = {}
        for hl_node, ll_nodes in corr.items():
            for ll_node in ll_nodes:
                # add everything after 'blocks.<layer>.' to the set
                suffix = ll_node.name.split(".")[2:]
                suffix = ".".join(suffix)
                if "attn" in ll_node.name:
                    if "attn" in suffixes and suffixes["attn"] != suffix:
                        raise ValueError(
                            f"Multiple attn suffixes found: {suffixes['attn']} and {suffix}, multiple attn hook locations are not supported yet."
                        )
                    suffixes["attn"] = suffix
                elif "mlp" in ll_node.name:
                    if "mlp" in suffixes and suffixes["mlp"] != suffix:
                        raise ValueError(
                            f"Multiple mlp suffixes found: {suffixes['mlp']} and {suffix}, multiple mlp hook locations are not supported yet."
                        )
                    suffixes["mlp"] = suffix
                else:
                    raise ValueError(f"Unknown node type {ll_node.name}")

        return suffixes


    @classmethod
    def make_corr_from_dict(cls, d, suffixes=None, make_suffixes_from_corr=False):
        if make_suffixes_from_corr:
            suffixes = Correspondence.get_hook_suffix(d)
        return cls(
            {
                HLNode(k, -1): {LLNode(name=node_name, index=None) for node_name in v}
                for k, v in d.items()
            },
            suffixes=suffixes,
        )
