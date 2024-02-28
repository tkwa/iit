from .base_model_pair import *
from iit.model_pairs.iit_model_pairs import IITModelPair
from iit.utils.metric import *


class TracrIITModelPair(IITModelPair):
    def __init__(self, hl_model, ll_model, corr, training_args={}):
        default_training_args = {
            "batch_size": 256,
            "lr": 0.001,
            "num_workers": 0,
            "losses": "all",
            "atol": 1e-2,
            "early_stop": True,
            "use_single_loss": False,
        }
        training_args = {**default_training_args, **training_args}
        super().__init__(hl_model, ll_model, corr=corr, training_args=training_args)
        self.wandb_method = "tracr_iit"

    @property
    def loss_fn(self):
        if self.hl_model.is_categorical():
            return t.nn.CrossEntropyLoss()
        else:
            return t.nn.MSELoss()

    @staticmethod
    def make_train_metrics():
        return MetricStoreCollection(
            [
                MetricStore("train/iit_loss", MetricType.LOSS),
                MetricStore("train/behavior_loss", MetricType.LOSS),
            ]
        )

    @staticmethod
    def make_test_metrics():
        return MetricStoreCollection(
            [
                MetricStore("val/iit_loss", MetricType.LOSS),
                MetricStore("val/IIA", MetricType.ACCURACY),
                MetricStore("val/accuracy", MetricType.ACCURACY),
            ]
        )

    def get_encoded_input_from_torch_input(
        self, input
    ):  # TODO: refactor this to outside of the class
        """Encode input to the format expected by the model"""
        x, y = input
        encoded_x = self.hl_model.map_tracr_input_to_tl_input(list(map(list, zip(*x))))
        y[0] = torch.zeros_like((y[1]))
        return encoded_x, torch.tensor(list(map(list, zip(*y))))

    def do_intervention(
        self, base_input, ablation_input, hl_node: HookName, verbose=False
    ):
        ablation_x, ablation_y = ablation_input
        base_x, base_y = base_input
        hl_ablation_output, self.hl_cache = self.hl_model.run_with_cache(ablation_x)

        assert torch.allclose(
            hl_ablation_output.cpu(), ablation_y.unsqueeze(-1).cpu().float()
        ), f"Ablation output {hl_ablation_output} does not match label {ablation_y}"

        ll_ablation_output, self.ll_cache = self.ll_model.run_with_cache(ablation_x)
        ll_nodes = self.corr[hl_node]

        hl_output = self.hl_model.run_with_hooks(
            base_x, fwd_hooks=[(hl_node, self.hl_ablation_hook)]
        )
        ll_output = self.ll_model.run_with_hooks(
            base_x,
            fwd_hooks=[
                (ll_node.name, self.make_ll_ablation_hook(ll_node))
                for ll_node in ll_nodes
            ],
        )

        if verbose:
            print(f"{hl_ablation_output=}, {ll_ablation_output.shape=}")
            print(f"{hl_node=}, {ll_nodes=}")
            print(f"{hl_output=}")
        return hl_output, ll_output

    def get_behaviour_loss_over_batch(self, base_input, loss_fn):
        base_x, base_y = base_input
        output = self.ll_model(base_x)
        behavior_loss = loss_fn(
            output, base_y.unsqueeze(-1).float().to(self.ll_model.cfg.device)
        )
        return behavior_loss

    def run_train_step(
        self,
        base_input,
        ablation_input,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: t.optim.Optimizer,
    ):
        grad_dict = {}
        base_input = self.get_encoded_input_from_torch_input(base_input)
        ablation_input = self.get_encoded_input_from_torch_input(ablation_input)

        loss_types = self.training_args["losses"]
        use_single_loss = self.training_args["use_single_loss"]

        optimizer.zero_grad()
        if loss_types == "all" or loss_types == "iit":
            hl_node = self.sample_hl_name()  # sample a high-level variable to ablate
            iit_loss = self.get_IIT_loss_over_batch(
                base_input, ablation_input, hl_node, loss_fn
            )
            iit_loss.backward()
            if use_single_loss:
                for name, param in self.ll_model.named_parameters():
                    if param.grad is not None:
                        grad_dict[name] = param.grad.clone()
            else:
                optimizer.step()

        optimizer.zero_grad()
        if loss_types == "all" or loss_types == "behaviour":
            behavior_loss = self.get_behaviour_loss_over_batch(base_input, loss_fn)
            behavior_loss.backward()
            if use_single_loss:
                for name, param in self.ll_model.named_parameters():
                    if param.grad is not None:
                        grad_dict[name] += param.grad.clone()
            else:
                optimizer.step()

        if use_single_loss:
            optimizer.zero_grad()
            # make grads using the grad_dict
            for k, v in grad_dict.items():
                # find model parameter by name
                param = next(p for n, p in self.ll_model.named_parameters() if n == k)
                param.grad = v
            optimizer.step()

        return {
            "train/iit_loss": iit_loss.item() if "iit_loss" in locals() else 0.0,
            "train/behavior_loss": (
                behavior_loss.item() if "behavior_loss" in locals() else 0.0
            ),
        }

    def run_eval_step(
        self,
        base_input,
        ablation_input,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ):
        atol = self.training_args["atol"]
        base_input = self.get_encoded_input_from_torch_input(base_input)
        ablation_input = self.get_encoded_input_from_torch_input(ablation_input)

        # compute IIT loss and accuracy
        hl_node = self.sample_hl_name()
        hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
        if self.hl_model.is_categorical():
            loss = loss_fn(ll_output, hl_output)
            top1 = t.argmax(ll_output, dim=1)
            accuracy = (top1 == hl_output).float().mean()
            IIA = accuracy.item()
        else:
            loss = loss_fn(ll_output, hl_output)
            IIA = ((ll_output - hl_output).abs() < atol).float().mean().item()

        # compute behavioral accuracy
        base_x, base_y = base_input
        output = self.ll_model(base_x)
        if self.hl_model.is_categorical():
            top1 = t.argmax(output, dim=-1)
            accuracy = (top1 == base_y).float().mean()
        else:
            accuracy = (
                (
                    (
                        output
                        - base_y.unsqueeze(-1).float().to(self.ll_model.cfg.device)
                    ).abs()
                    < atol
                )
                .float()
                .mean()
            )
        return {
            "val/iit_loss": loss.item(),
            "val/IIA": IIA,
            "val/accuracy": accuracy.item(),
        }
