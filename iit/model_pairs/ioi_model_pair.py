from iit.utils.config import DEVICE
from iit.utils.metric import *
from typing import Callable
from torch import Tensor
import torch as t
from iit.model_pairs.base_model_pair import HLNode
from iit.model_pairs.iit_behavior_model_pair import IITBehaviorModelPair


class IOI_ModelPair(IITBehaviorModelPair):
    def __init__(self, hl_model, ll_model, corr, next_token=False, training_args={}):
        super().__init__(hl_model, ll_model, corr, training_args)
        self.next_token = next_token

    @property
    def loss_fn(self):
        if hasattr(self, "__loss_fn"):
            return self.__loss_fn

        def per_token_weighted_cross_entropy(output, target):
            if len(output.shape) == 2:
                return t.nn.functional.cross_entropy(output, target)
            if self.next_token:
                weight = t.ones(output.shape[1], device=DEVICE)
                weight[-1] = 10
                return t.nn.functional.cross_entropy(output, target, weight=weight)
            else:
                return t.nn.functional.cross_entropy(output[:, -1, :], target[:, -1, :])

        self.__loss_fn = per_token_weighted_cross_entropy
        return self.__loss_fn

    @staticmethod
    def make_test_metrics():
        return MetricStoreCollection(
            [
                MetricStore("val/iit_loss", MetricType.LOSS),
                MetricStore("val/IIA", MetricType.ACCURACY),
                MetricStore("val/accuracy", MetricType.ACCURACY),
                PerTokenMetricStore("val/per_token_accuracy", MetricType.ACCURACY),
            ]
        )

    def get_IIT_loss_over_batch(
        self,
        base_input,
        ablation_input,
        hl_node: HLNode,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ):
        hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
        hl_output = t.nn.functional.softmax(hl_output, dim=-1)

        loss = loss_fn(ll_output[:, -1, :], hl_output[:, -1, :])
        return loss

    def run_eval_step(
        self,
        base_input,
        ablation_input,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ):
        # compute IIT loss and accuracy on last token position only
        hl_node = self.sample_hl_name()
        hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
        # CrossEntropyLoss needs target probs, not logits
        hl_output = t.nn.functional.softmax(hl_output, dim=-1)
        assert self.hl_model.is_categorical()
        loss = loss_fn(ll_output[:, -1, :], hl_output[:, -1, :])
        if ll_output.shape == hl_output.shape:
            # To handle the case when labels are one-hot
            hl_output = t.argmax(hl_output, dim=-1)
        top1 = t.argmax(ll_output, dim=-1)
        accuracy = (top1[:, -1] == hl_output[:, -1]).float().mean()
        IIA = accuracy.item()

        # compute behavioral accuracy
        base_x, base_y, _ = base_input
        output = self.ll_model(base_x)
        top1 = t.argmax(output, dim=-1)  # batch n_ctx
        if output.shape == base_y.shape:
            # To handle the case when labels are one-hot
            # TODO: is there a better way?
            base_y = t.argmax(base_y, dim=-1)  # batch n_ctx
        per_token_accuracy = (top1 == base_y).float().mean(dim=0).cpu().numpy()
        return {
            "val/iit_loss": loss.item(),
            "val/IIA": IIA,
            "val/accuracy": per_token_accuracy.mean().item() if self.next_token else per_token_accuracy[-1],
            "val/per_token_accuracy": per_token_accuracy,
        }
