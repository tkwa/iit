from iit.utils.config import DEVICE
from iit.utils.metric import *
from typing import Callable
from torch import Tensor
import torch as t
from iit.model_pairs.base_model_pair import HLNode
from iit.model_pairs import IITBehaviorModelPair, StopGradModelPair, StrictIITModelPair
import iit.utils.index as index


class IOI_ModelPair(StrictIITModelPair):
    def __init__(self, hl_model, ll_model, corr, training_args={}):
        super().__init__(hl_model, ll_model, corr, training_args=training_args)
        default_training_args = {
            "next_token": False,
            "non_ioi_thresh": 0.65,
            "use_per_token_check": False,
        }
        self.training_args = {**default_training_args, **self.training_args}
        self.next_token = default_training_args["next_token"]

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
    def get_label_idxs():
        return index.Ix[:, -1]

    @staticmethod
    def make_test_metrics():
        return MetricStoreCollection(
            [
                MetricStore("val/iit_loss", MetricType.LOSS),
                MetricStore("val/IIA", MetricType.ACCURACY),
                MetricStore("val/accuracy", MetricType.ACCURACY),
                PerTokenMetricStore("val/per_token_accuracy"),
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
        # hl_output = t.nn.functional.softmax(hl_output, dim=-1)
        hl_argmax = t.argmax(hl_output[:, -1, :], dim=-1)
        hl_one_hot = t.nn.functional.one_hot(hl_argmax, num_classes=hl_output.shape[-1])
        hl_probs = hl_one_hot.float()

        loss = loss_fn(ll_output[:, -1, :], hl_probs)
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
        # hl_output = t.nn.functional.softmax(hl_output, dim=-1)
        hl_argmax = t.argmax(hl_output[:, -1, :], dim=-1)
        hl_one_hot = t.nn.functional.one_hot(hl_argmax, num_classes=hl_output.shape[-1])
        hl_probs = hl_one_hot.float()
        assert self.hl_model.is_categorical()
        loss = loss_fn(ll_output[:, -1, :], hl_probs)
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
            "val/accuracy": (
                per_token_accuracy.mean().item()
                if self.next_token
                else per_token_accuracy[-1]
            ),
            "val/per_token_accuracy": per_token_accuracy,
        }

    @staticmethod
    def _check_early_stop_fn(
        test_metrics: list[MetricStore],
        verbose=False,
        non_ioi_thresh=0.65,
        use_per_token_check=False,
    ):
        """
        Early stopping for IOI
        """
        print_if_verbose = lambda x: print(x) if verbose else None
        for metric in test_metrics:
            if metric.get_name() == "val/IIA" and metric.get_value() < 100:
                print_if_verbose(f"IIA is not enough: {metric.get_value()}")
                return False
            elif metric.get_name() == "val/per_token_accuracy":
                per_toke_acc = metric.get_value()
                if per_toke_acc[-1] < 1:
                    print_if_verbose(
                        f"per_token_acc at IOI index is not enough: {per_toke_acc[-1]}"
                    )
                    return False
                if np.mean(per_toke_acc) < non_ioi_thresh:
                    print_if_verbose(
                        f"mean per_token_acc is not enough: {np.mean(per_toke_acc)}"
                    )
                    return False

                if use_per_token_check:
                    # Ideally, we should check the per_token_accuracy at the IOI index,
                    # but this fails for multiple patterns. So am disabling this check for now.
                    print(
                        "WARNING: Using per_token_check for early stopping can lead to unexpected error. Please use with caution."
                    )
                    for i in range(len(per_toke_acc)):
                        if i in [2, 4, 5, 8, 10, 13]:
                            continue
                        if per_toke_acc[i] < non_ioi_thresh:
                            if verbose:
                                print(
                                    f"per_token_acc at {i} is not enough: {per_toke_acc[i]}"
                                )
                            return False
        return True

    def _check_early_stop_condition(
        self,
        *args,
        **kwargs,
    ):
        if not self.training_args["next_token"]:
            return super()._check_early_stop_condition(*args, **kwargs)
        return self._check_early_stop_fn(
            *args,
            **kwargs,
            non_ioi_thresh=self.training_args["non_ioi_thresh"],
            use_per_token_check=self.training_args["use_per_token_check"],
        )
