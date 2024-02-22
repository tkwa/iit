from iit.model_pairs.tracr_iit_model_pair import *
import iit.utils.node_picker as node_picker

class TracrStrictIITModelPair(TracrIITModelPair):
    def __init__(self, hl_model, ll_model, corr, training_args={}):
        default_training_args = {
            'batch_size': 256,
            'lr': 0.001,
            'num_workers': 0,
            'b_lr': 1e-3,
            'iit_lr': 1e-2,
            'not_in_circuit_lr': 1e-2,
        }
        training_args = {**default_training_args, **training_args}
        super().__init__(hl_model, ll_model, 
                         corr=corr, training_args=training_args)
        self.nodes_not_in_circuit = node_picker.get_nodes_not_in_circuit(
            self.ll_model, self.corr)

    def sample_ll_node(self) -> LLNode:
        return self.rng.choice(self.nodes_not_in_circuit)
    
    def do_intervention(self, base_input, ablation_input, hl_node: str, verbose=False):
        return super().do_intervention(base_input, ablation_input, hl_node, verbose)
    
    def run_IIT_train_step(self, base_input, ablation_input, 
                           loss_fn, optimizer):
        normal_hl_loss = super().run_IIT_train_step(
            base_input, ablation_input, loss_fn, optimizer
        )
        # loss for nodes that are not in the circuit
        # should not have causal effect on the high-level output
        optimizer.zero_grad()
        base_x, base_y = self.get_encoded_input_from_torch_input(base_input)
        ablation_x, _ = self.get_encoded_input_from_torch_input(ablation_input)
        ll_node = self.sample_ll_node()
        _, cache = self.ll_model.run_with_cache(ablation_x)
        self.ll_cache = cache 
        out = self.ll_model.run_with_hooks(
            base_x, fwd_hooks=[
                (ll_node.name, self.make_ll_ablation_hook(ll_node))
            ])
        ll_loss = loss_fn(out, base_y.unsqueeze(-1).float().to(self.ll_model.cfg.device))
        ll_loss.backward()
        optimizer.step()
        return (normal_hl_loss + ll_loss.item()) / 2
        

