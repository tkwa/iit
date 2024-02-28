from iit.model_pairs.tracr_iit_model_pair import *
import iit.utils.node_picker as node_picker


class TracrStrictIITModelPair(TracrIITModelPair):
    def __init__(self, hl_model, ll_model, corr, training_args={}):
        default_training_args = {
            "batch_size": 256,
            "lr": 0.001,
            "num_workers": 0,
            "use_single_loss": False,
        }
        training_args = {**default_training_args, **training_args}
        super().__init__(hl_model, ll_model, corr=corr, training_args=training_args)
        self.nodes_not_in_circuit = node_picker.get_nodes_not_in_circuit(
            self.ll_model, self.corr
        )

    def sample_ll_node(self) -> LLNode:
        return self.rng.choice(self.nodes_not_in_circuit)

    def run_train_step(
        self,
        base_input,
        ablation_input,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: t.optim.Optimizer,
    ):
        base_input = self.get_encoded_input_from_torch_input(base_input)
        ablation_input = self.get_encoded_input_from_torch_input(ablation_input)

        loss_types = self.training_args["losses"]
        use_single_loss = self.training_args["use_single_loss"]

        iit_loss = 0
        ll_loss = 0
        behavior_loss = 0

        optimizer.zero_grad()
        if loss_types == "all" or loss_types == "iit":
            hl_node = self.sample_hl_name()  # sample a high-level variable to ablate
            iit_loss = self.get_IIT_loss_over_batch(
                base_input, ablation_input, hl_node, loss_fn
            )
            if not use_single_loss:
                iit_loss.backward()
                optimizer.step()

        # loss for nodes that are not in the circuit
        # should not have causal effect on the high-level output
        # TODO: add another loss type for this
        if (
            self.training_args["losses"] == "all"
            or self.training_args["losses"] == "iit"
        ):
            if not use_single_loss:
                optimizer.zero_grad()
            base_x, base_y = base_input
            ablation_x, _ = ablation_input
            ll_node = self.sample_ll_node()
            _, cache = self.ll_model.run_with_cache(ablation_x)
            self.ll_cache = cache
            out = self.ll_model.run_with_hooks(
                base_x, fwd_hooks=[(ll_node.name, self.make_ll_ablation_hook(ll_node))]
            )
            ll_loss = loss_fn(
                out, base_y.unsqueeze(-1).float().to(self.ll_model.cfg.device)
            )
            if not use_single_loss:
                ll_loss.backward()
                optimizer.step()

        if loss_types == "all" or loss_types == "behaviour":
            if not use_single_loss:
                optimizer.zero_grad()
            behavior_loss = self.get_behaviour_loss_over_batch(base_input, loss_fn)
            if not use_single_loss:
                behavior_loss.backward()
                optimizer.step()

        if use_single_loss:
            total_loss = iit_loss + behavior_loss + ll_loss
            total_loss.backward()
            optimizer.step()

        return {
            "train/iit_loss": (
                (iit_loss.item() + ll_loss.item()) / 2
                if "iit_loss" in locals()
                else 0.0
            ),
            "train/behavior_loss": (
                behavior_loss.item() if "behavior_loss" in locals() else 0.0
            ),
        }
