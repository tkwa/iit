from iit.model_pairs.base_model_pair import *
from iit.utils.metric import *
from typing import final


class IITModelPair(BaseModelPair):
    def __init__(
        self,
        hl_model: HookedRootModule = None,
        ll_model: HookedRootModule = None,
        hl_graph=None,
        corr: dict[HLNode, set[LLNode]] = {},
        seed=0,
        training_args={},
    ):
        # TODO change to construct hl_model from graph?
        if hl_model is None:
            assert hl_graph is not None
            hl_model = self.make_hl_model(hl_graph)

        self.hl_model = hl_model
        self.ll_model = ll_model
        self.hl_model.requires_grad_(False)

        self.corr: dict[HLNode, set[LLNode]] = corr
        assert all([k in self.hl_model.hook_dict for k in self.corr.keys()])
        self.rng = np.random.default_rng(seed)
        default_training_args = {
            "batch_size": 256,
            "lr": 0.001,
            "num_workers": 0,
            "early_stop": True,
        }
        training_args = {**default_training_args, **training_args}
        self.training_args = training_args
        self.wandb_method = "iit"

    @property
    def loss_fn(self):
        return t.nn.CrossEntropyLoss()

    @property
    def train_metrics(cls):
        if not hasattr(cls, "_train_metrics"):
            cls._train_metrics = MetricStoreCollection(
                [
                    MetricStore("iit_loss", MetricType.LOSS),
                ]
            )
        return cls._train_metrics

    @property
    def test_metrics(cls):
        if not hasattr(cls, "_test_metrics"):
            cls._test_metrics = MetricStoreCollection(
                [
                    MetricStore("iit_loss", MetricType.LOSS),
                    MetricStore("accuracy", MetricType.ACCURACY),
                ]
            )
        return cls._test_metrics

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

    def do_intervention(
        self, base_input, ablation_input, hl_node: HookName, verbose=False
    ):
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

    def run_eval_step(
        self,
        base_input,
        ablation_input,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ):
        base_input = [t.to(DEVICE) for t in base_input]
        ablation_input = [t.to(DEVICE) for t in ablation_input]
        hl_node = self.sample_hl_name()
        hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
        loss = loss_fn(ll_output, hl_output)
        top1 = t.argmax(ll_output, dim=-1)
        accuracy = (top1 == hl_output).float().mean()
        return {
            "iit_loss": loss.item(),
            "accuracy": accuracy.item(),
        }

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

    def run_train_step(
        self,
        base_input,
        ablation_input,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: t.optim.Optimizer,
    ):
        base_input = [t.to(DEVICE) for t in base_input]  # TODO: refactor to remove this
        ablation_input = [t.to(DEVICE) for t in ablation_input]

        optimizer.zero_grad()
        hl_node = self.sample_hl_name()  # sample a high-level variable to ablate
        loss = self.get_IIT_loss_over_batch(
            base_input, ablation_input, hl_node, loss_fn
        )
        loss.backward()
        optimizer.step()
        return {"iit_loss": loss.item()}

    @final
    def _run_train_epoch(self, loader, loss_fn, optimizer):
        self.ll_model.train()
        for i, (base_input, ablation_input) in tqdm(
            enumerate(loader), total=len(loader)
        ):
            self.train_metrics.update(
                self.run_train_step(base_input, ablation_input, loss_fn, optimizer)
            )

    @final
    def _run_eval_epoch(self, loader, loss_fn) -> list[MetricStore]:
        self.ll_model.eval()
        with t.no_grad():
            for i, (base_input, ablation_input) in enumerate(loader):
                test_metrics = self.run_eval_step(base_input, ablation_input, loss_fn)
                self.test_metrics.update(test_metrics)

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
        print(f"Epoch {epoch}:", end=" ")
        for metric in metrics:
            print(metric, end=", ")
            if use_wandb:
                wandb.log({metric.get_name(): metric.get_value()})

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
            self._run_train_epoch(loader, loss_fn, optimizer)
            self._run_eval_epoch(test_loader, loss_fn)

            self._print_and_log_metrics(
                epoch, self.train_metrics.metrics + self.test_metrics.metrics, use_wandb
            )

            if early_stop and self._check_early_stop_condition(
                self.test_metrics.metrics
            ):
                break

        if use_wandb:
            wandb.log({"final epoch": epoch})
