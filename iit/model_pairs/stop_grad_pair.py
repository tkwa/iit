from typing import Any
from iit.model_pairs.iit_behavior_model_pair import *
import iit.utils.node_picker as node_picker
import iit.utils.index as index   
    
    
class StopGradModelPair(IITBehaviorModelPair):
    def __init__(self, hl_model, ll_model, corr, training_args={}):
        default_training_args = {
            "batch_size": 256,
            "lr": 0.001,
            "num_workers": 0,
            "use_single_loss": False,
            "iit_weight": 1.0,
            "behavior_weight": 1.0,
        }
        training_args = {**default_training_args, **training_args}
        super().__init__(hl_model, ll_model, corr=corr, training_args=training_args)
        self.params_not_in_circuit = node_picker.get_params_not_in_circuit(corr, ll_model)

    @staticmethod
    def make_detached_hook(ll_node: LLNode):
        def hook_fn(grad):
            act_idx = node_picker.get_activation_idx(ll_node)
            grad[act_idx.as_index] = t.zeros_like(grad[act_idx.as_index])
        return hook_fn
    
    def make_backward_hooks(self):
        for ll_node in self.params_not_in_circuit:
            for name, param in self.ll_model.named_parameters():
                if ll_node.name == name:
                    hook = self.make_detached_hook(ll_node)
                    param.register_hook(hook)
