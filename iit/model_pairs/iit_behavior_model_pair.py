from .base_model_pair import *
from iit.model_pairs.iit_model_pairs import IITModelPair
from iit.utils.metric import *


class IITBehaviorModelPair(IITModelPair):
    def __init__(self, hl_model, ll_model, corr, training_args={}):
        default_training_args = {
            "lr": 0.001,
            "atol": 1e-2,
            "early_stop": True,
            "use_single_loss": False,
            "iit_weight": 1.0,
            "behavior_weight": 1.0,
        }
        training_args = {**default_training_args, **training_args}
        super().__init__(hl_model, ll_model, corr=corr, training_args=training_args)
        self.wandb_method = "iit_and_behavior"

    @property
    def loss_fn(self):
        # TODO: make this more general
        try:
            if self.hl_model.is_categorical():
                return t.nn.CrossEntropyLoss()
            else:
                return t.nn.MSELoss()
        except AttributeError:
            print("WARNING: using default categorical loss function.")
            return t.nn.CrossEntropyLoss()

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

    def get_behaviour_loss_over_batch(self, base_input, loss_fn):
        base_x, base_y, _ = base_input
        output = self.ll_model(base_x)
        behavior_loss = loss_fn(output.squeeze(), base_y)
        return behavior_loss

    @staticmethod
    def step_on_loss(loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def run_train_step(
        self,
        base_input,
        ablation_input,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: t.optim.Optimizer,
    ):
        use_single_loss = self.training_args["use_single_loss"]

        iit_loss = 0
        behavior_loss = 0

        hl_node = self.sample_hl_name()  # sample a high-level variable to ablate
        iit_loss = (
            self.get_IIT_loss_over_batch(base_input, ablation_input, hl_node, loss_fn)
            * self.training_args["iit_weight"]
        )
        if not use_single_loss:
            self.step_on_loss(iit_loss, optimizer)

        behavior_loss = (
            self.get_behaviour_loss_over_batch(base_input, loss_fn)
            * self.training_args["behavior_weight"]
        )
        if not use_single_loss:
            self.step_on_loss(behavior_loss, optimizer)

        if use_single_loss:
            total_loss = iit_loss + behavior_loss
            self.step_on_loss(total_loss, optimizer)

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
        base_x, base_y, _ = base_input
        output = self.ll_model(base_x)
        if self.hl_model.is_categorical():
            top1 = t.argmax(output, dim=-1)
            accuracy = (top1 == base_y).float().mean()
        else:
            accuracy = ((output.squeeze() - base_y).abs() < atol).float().mean()
        return {
            "val/iit_loss": loss.item(),
            "val/IIA": IIA,
            "val/accuracy": accuracy.item(),
        }
