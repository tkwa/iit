from typing import Callable, Optional
from dataclasses import dataclass
import dataclasses
import numpy as np
from torch import Tensor
import transformer_lens as tl
from transformer_lens.hook_points import HookedRootModule, HookPoint
import networkx as nx
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from iit.utils.index import TorchIndex, Ix
from iit.utils.iit_dataset import IITDataset
from iit.utils.config import *
import torch as t
from abc import ABC, abstractmethod
from typing import final, Any
from iit.utils.metric import MetricStoreCollection, MetricType
from iit.utils.correspondence import Correspondence
from .nodes import *


class BaseModelPair(ABC):
    hl_model: HookedRootModule
    ll_model: HookedRootModule
    hl_cache: tl.ActivationCache
    ll_cache: tl.ActivationCache
    hl_graph: nx.DiGraph
    corr: Correspondence  # high -> low correspondence. Capital Pi in paper
    training_args: dict[str, Any]
    wandb_method: str
    rng: np.random.Generator
    dataset_class: type[IITDataset]

    ##########################################
    # Abstract methods you need to implement #
    ##########################################
    @property
    @abstractmethod
    def loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    @abstractmethod
    def make_train_metrics() -> MetricStoreCollection:
        pass

    @staticmethod
    @abstractmethod
    def make_test_metrics() -> MetricStoreCollection:
        pass

    @abstractmethod
    def run_train_step(
        self,
        base_input,
        ablation_input,
        loss_fn,
        optimizer,
    ) -> MetricStoreCollection:
        pass

    @abstractmethod
    def run_eval_step(
        self,
        base_input,
        ablation_input,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> MetricStoreCollection:
        pass

    ###########################################
    ##### Mutable methods you can override ####
    ###########################################
    def do_intervention(
        self, base_input, ablation_input, hl_node: HLNode, verbose=False
    ) -> tuple[Tensor, Tensor]:
        ablation_x, ablation_y, ablation_intermediate_vars = ablation_input
        base_x, base_y, base_intermediate_vars = base_input
        hl_ablation_output, self.hl_cache = self.hl_model.run_with_cache(ablation_input)

        # assert torch.allclose(
        #     hl_ablation_output.squeeze(), ablation_y
        # ), f"Ablation output {hl_ablation_output} does not match label {ablation_y}"

        ll_ablation_output, self.ll_cache = self.ll_model.run_with_cache(ablation_x)
        ll_nodes = self.corr[hl_node]

        hl_output = self.hl_model.run_with_hooks(
            base_input, fwd_hooks=[(hl_node.name, self.make_hl_ablation_hook(hl_node))]
        )
        ll_output = self.ll_model.run_with_hooks(
            base_x,
            fwd_hooks=[
                (ll_node.name, self.make_ll_ablation_hook(ll_node))
                for ll_node in ll_nodes
            ],
        )

        if verbose:
            print(f"{base_x=}, {base_y.item()=}")
            print(f"{hl_ablation_output=}, {ll_ablation_output.shape=}")
            print(f"{hl_node=}, {ll_nodes=}")
            print(f"{hl_output=}")
        return hl_output, ll_output

    @staticmethod
    def get_label_idxs():
        return Ix[[None]]

    def make_hl_model(self, hl_graph):
        raise NotImplementedError

    def set_corr(self, corr):
        self.corr = corr

    def sample_hl_name(self) -> HLNode:
        return self.rng.choice(list(self.corr.keys()))

    def make_hl_ablation_hook(self, hl_node: HLNode):
        assert isinstance(hl_node, HLNode), ValueError(
            f"hl_node is not an instance of HLNode, but {type(hl_node)}"
        )

        def hl_ablation_hook(hook_point_out: Tensor, hook: HookPoint) -> Tensor:
            out = hook_point_out.clone()

            if isinstance(out, float) or isinstance(out, int):
                assert (
                    hl_node.index is Ix[[None]] or hl_node.index is None
                ), "scalars cannot be indexed"
                return self.hl_cache[hook.name]

            out[hl_node.index.as_index] = self.hl_cache[hook.name][
                hl_node.index.as_index
            ]
            return out

        if hl_node.index is not None:
            return hl_ablation_hook
        else:
            return self.hl_ablation_hook

    def hl_ablation_hook(
        self, hook_point_out: Tensor, hook: HookPoint
    ) -> Tensor:  # TODO: remove this
        out = self.hl_cache[hook.name]
        return out

    # TODO extend to position and subspace...
    def make_ll_ablation_hook(
        self, ll_node: LLNode
    ) -> Callable[[Tensor, HookPoint], Tensor]:
        if ll_node.subspace is not None:
            raise NotImplementedError

        def ll_ablation_hook(hook_point_out: Tensor, hook: HookPoint) -> Tensor:
            out = hook_point_out.clone()
            index = ll_node.index if ll_node.index is not None else Ix[[None]]
            out[index.as_index] = self.ll_cache[hook.name][index.as_index]
            return out

        return ll_ablation_hook

    def get_IIT_loss_over_batch(
        self,
        base_input,
        ablation_input,
        hl_node: HookName,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ):
        hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
        loss = loss_fn(ll_output, hl_output)
        return loss

    def clip_grad_fn(self):
        if self.training_args["clip_grad_norm"]:
            t.nn.utils.clip_grad_norm_(
                self.ll_model.parameters(), self.training_args["clip_grad_norm"]
            )

    def step_scheduler(self, lr_scheduler, test_metrics):
        if isinstance(lr_scheduler, t.optim.lr_scheduler.ReduceLROnPlateau):
            val_metric = self.training_args.get("scheduler_val_metric", "val/accuracy")
            try:
                accuracy_metric = test_metrics.to_dict()[val_metric]
                lr_scheduler.step(accuracy_metric)
                return
            except KeyError:
                raise ValueError(
                    f"val_metric {val_metric} not found in test_metrics {test_metrics}"
                )
            except Exception as e:
                raise ValueError(
                    f"WARNING: Could not step lr_scheduler {lr_scheduler} with exception {e}"
                )
        try:
            lr_scheduler.step()
        except Exception as e:
            print(
                f"WARNING: Could not step lr_scheduler {lr_scheduler} with exception {e}"
            )

    def train(
        self,
        train_set,
        test_set,
        epochs=1000,
        use_wandb=False,
    ):
        training_args = self.training_args
        print(f"{training_args=}")

        assert isinstance(train_set, IITDataset), ValueError(
            f"train_set is not an instance of IITDataset, but {type(train_set)}"
        )
        assert isinstance(test_set, IITDataset), ValueError(
            f"test_set is not an instance of IITDataset, but {type(test_set)}"
        )
        train_loader, test_loader = self.make_loaders(
            train_set,
            test_set,
            training_args["batch_size"],
            training_args["num_workers"],
        )

        early_stop = training_args["early_stop"]

        optimizer = t.optim.Adam(self.ll_model.parameters(), lr=training_args["lr"])
        loss_fn = self.loss_fn
        scheduler_cls = training_args.get("lr_scheduler", None)
        if scheduler_cls == t.optim.lr_scheduler.ReduceLROnPlateau:
            mode = training_args.get("scheduler_mode", "max")
            lr_scheduler = scheduler_cls(optimizer, mode=mode, factor=0.1, patience=10)
        elif scheduler_cls:
            lr_scheduler = scheduler_cls(optimizer)

        if use_wandb and not wandb.run:
            wandb.init(project="iit", entity=WANDB_ENTITY)

        if use_wandb:
            wandb.config.update(training_args)
            wandb.config.update({"method": self.wandb_method})
            wandb.run.log_code()

        for epoch in tqdm(range(epochs)):
            train_metrics = self._run_train_epoch(train_loader, loss_fn, optimizer)
            test_metrics = self._run_eval_epoch(test_loader, loss_fn)
            if scheduler_cls:
                self.step_scheduler(lr_scheduler, test_metrics)
            self.test_metrics = test_metrics
            self.train_metrics = train_metrics
            self._print_and_log_metrics(
                epoch, train_metrics.metrics + test_metrics.metrics, use_wandb
            )

            if early_stop and self._check_early_stop_condition(test_metrics.metrics):
                break

        if use_wandb:
            wandb.log({"final epoch": epoch})

    #########################################
    # Immutable methods- might change later #
    #########################################
    @final
    @staticmethod
    def make_loaders(
        dataset: IITDataset,
        test_dataset: IITDataset,
        batch_size,
        num_workers,
    ):
        loader = dataset.make_loader(batch_size, num_workers)
        test_loader = test_dataset.make_loader(batch_size, num_workers)
        return loader, test_loader

    @final
    def _run_train_epoch(self, loader, loss_fn, optimizer) -> MetricStoreCollection:
        self.ll_model.train()
        train_metrics = self.make_train_metrics()
        for i, (base_input, ablation_input) in tqdm(
            enumerate(loader), total=len(loader)
        ):
            train_metrics.update(
                self.run_train_step(base_input, ablation_input, loss_fn, optimizer)
            )
        return train_metrics

    @final
    def _run_eval_epoch(self, loader, loss_fn) -> MetricStoreCollection:
        self.ll_model.eval()
        test_metrics = self.make_test_metrics()
        with t.no_grad():
            for i, (base_input, ablation_input) in enumerate(loader):
                test_metrics.update(
                    self.run_eval_step(base_input, ablation_input, loss_fn)
                )
        return test_metrics

    @staticmethod
    def _check_early_stop_condition(test_metrics):
        """
        Returns True if all types of accuracy metrics are above 0.99
        """
        got_accuracy_metric = False
        for metric in test_metrics:
            if metric.type == MetricType.ACCURACY:
                got_accuracy_metric = True
                if metric.get_value() < 100:
                    return False
        if not got_accuracy_metric:
            raise ValueError("No accuracy metric found in test_metrics!")
        return True

    @final
    @staticmethod
    def _print_and_log_metrics(epoch, metrics, use_wandb=False):
        print(f"\nEpoch {epoch}:", end=" ")
        if use_wandb:
            wandb.log({"epoch": epoch})
        for metric in metrics:
            print(metric, end=", ")
            if use_wandb:
                wandb.log({metric.get_name(): metric.get_value()})
        print()
