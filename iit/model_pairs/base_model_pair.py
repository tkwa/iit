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
from iit.utils.index import TorchIndex
from iit.utils.iit_dataset import IITDataset
from iit.utils.config import *
import torch as t
from abc import ABC, abstractmethod
from typing import final, Any
from iit.utils.metric import MetricStoreCollection, MetricType

HookName = str
HLCache = dict


@dataclass
class HLNode:
    name: HookName
    num_classes: int
    index: Optional[int]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, HLNode):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


@dataclass
class LLNode:
    name: HookName
    index: TorchIndex
    subspace: Optional[t.Tensor] = None

    def __eq__(self, other):
        return isinstance(other, LLNode) and dataclasses.astuple(
            self
        ) == dataclasses.astuple(other)

    def __hash__(self):
        return hash(dataclasses.astuple(self))


class BaseModelPair(ABC):
    hl_model: HookedRootModule
    ll_model: HookedRootModule
    hl_cache: tl.ActivationCache
    ll_cache: tl.ActivationCache
    hl_graph: nx.DiGraph
    corr: dict[HLNode, set[LLNode]]  # high -> low correspondence. Capital Pi in paper
    training_args: dict[str, Any]
    wandb_method: str
    rng: np.random.Generator

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
        self, base_input, ablation_input, hl_node: HookName, verbose=False
    ) -> tuple[Tensor, Tensor]:
        ablation_x, ablation_y, ablation_intermediate_vars = ablation_input
        base_x, base_y, base_intermediate_vars = base_input
        hl_ablation_output, self.hl_cache = self.hl_model.run_with_cache(ablation_input)

        assert all(
            hl_ablation_output == ablation_y
        ), f"Ablation output {hl_ablation_output} does not match label {ablation_y}"

        ll_ablation_output, self.ll_cache = self.ll_model.run_with_cache(ablation_x)
        ll_nodes = self.corr[hl_node]

        hl_output = self.hl_model.run_with_hooks(
            base_input, fwd_hooks=[(hl_node, self.hl_ablation_hook)]
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
    
    def make_hl_model(self, hl_graph):
        raise NotImplementedError

    def set_corr(self, corr):
        self.corr = corr

    def sample_hl_name(self) -> str:
        # return a `str` rather than `numpy.str_`
        return str(self.rng.choice(list(self.corr.keys())))

    def hl_ablation_hook(self, hook_point_out: Tensor, hook: HookPoint) -> Tensor:
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
            out[ll_node.index.as_index] = self.ll_cache[hook.name][
                ll_node.index.as_index
            ]
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

    def make_loaders(
        self, base_data, ablation_data, test_base_data, test_ablation_data
    ):
        training_args = self.training_args
        dataset = IITDataset(base_data, ablation_data)
        test_dataset = IITDataset(test_base_data, test_ablation_data)
        loader = DataLoader(
            dataset,
            batch_size=training_args["batch_size"],
            shuffle=True,
            num_workers=training_args["num_workers"],
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_args["batch_size"],
            shuffle=True,
            num_workers=training_args["num_workers"],
        )
        return loader, test_loader

    def train(
        self,
        base_data,
        ablation_data,
        test_base_data,
        test_ablation_data,
        epochs=1000,
        use_wandb=False,
    ):
        training_args = self.training_args
        print(f"{training_args=}")

        early_stop = training_args["early_stop"]

        loader, test_loader = self.make_loaders(
            base_data, ablation_data, test_base_data, test_ablation_data
        )
        optimizer = t.optim.Adam(self.ll_model.parameters(), lr=training_args["lr"])
        loss_fn = self.loss_fn

        if use_wandb and not wandb.run:
            wandb.init(project="iit", entity=WANDB_ENTITY)

        if use_wandb:
            wandb.config.update(training_args)
            wandb.config.update({"method": self.wandb_method})

        for epoch in tqdm(range(epochs)):
            train_metrics = self._run_train_epoch(loader, loss_fn, optimizer)
            test_metrics = self._run_eval_epoch(test_loader, loss_fn)

            self._print_and_log_metrics(
                epoch, train_metrics.metrics + test_metrics.metrics, use_wandb
            )

            if early_stop and self._check_early_stop_condition(test_metrics.metrics):
                break

        if use_wandb:
            wandb.log({"final epoch": epoch})

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

    #########################################
    # Immutable methods- might change later #
    #########################################
    @final
    @staticmethod
    def _check_early_stop_condition(test_metrics):
        """
        Returns True if all types of accuracy metrics are above 0.99
        """
        got_accuracy_metric = False
        for metric in test_metrics:
            if metric.type == MetricType.ACCURACY:
                got_accuracy_metric = True
                if metric.get_value() < 99:
                    return False
        if not got_accuracy_metric:
            raise ValueError("No accuracy metric found in test_metrics!")
        return True

    @final
    @staticmethod
    def _print_and_log_metrics(epoch, metrics, use_wandb=False):
        print(f"\nEpoch {epoch}:", end=" ")
        for metric in metrics:
            print(metric, end=", ")
            if use_wandb:
                wandb.log({metric.get_name(): metric.get_value()})
        print()
