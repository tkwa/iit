from .base_model_pair import *
from iit.model_pairs.iit_model_pairs import IITModelPair


class TracrIITModelPair(IITModelPair):
    def __init__(self, hl_model, ll_model, corr, training_args={}):
        default_training_args = {
            "batch_size": 256,
            "lr": 0.001,
            "num_workers": 0,
            "losses": "all",
            "atol": 1e-2,
            "early_stop": True,
        }
        training_args = {**default_training_args, **training_args}
        super().__init__(hl_model, ll_model, corr=corr, training_args=training_args)

    def get_encoded_input_from_torch_input(
        self, input
    ):  # TODO: refactor this to outside of the class
        x, y = input
        encoded_x = self.hl_model.map_tracr_input_to_tl_input(list(map(list, zip(*x))))
        y[0] = torch.zeros_like((y[1]))
        return encoded_x, torch.tensor(list(map(list, zip(*y))))

    def do_intervention(
        self, base_input, ablation_input, hl_node: HookName, verbose=False
    ):
        ablation_x, ablation_y = self.get_encoded_input_from_torch_input(ablation_input)
        base_x, base_y = self.get_encoded_input_from_torch_input(base_input)

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

    def run_IIT_train_step(self, base_input, ablation_input, loss_fn, optimizer):
        hl_node = self.sample_hl_name()  # sample a high-level variable to ablate
        return self.run_train_step(
            base_input, ablation_input, hl_node, loss_fn, optimizer
        )

    def run_behaviour_train_step(self, base_input, loss_fn, optimizer):
        optimizer.zero_grad()
        base_x, base_y = self.get_encoded_input_from_torch_input(base_input)
        output = self.ll_model(base_x)
        behavior_loss = loss_fn(
            output, base_y.unsqueeze(-1).float().to(self.ll_model.cfg.device)
        )
        behavior_loss.backward()
        optimizer.step()
        return behavior_loss.item()

    def get_test_accs(self, base_input, ablation_input, loss_fn, atol):
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
        return loss, IIA

    def train(
        self,
        base_data,
        ablation_data,
        test_base_data,
        test_ablation_data,
        epochs=1000,
        use_wandb=False
    ):
        training_args = self.training_args
        print(f"{training_args=}")
        
        loss_types = training_args["losses"]
        early_stop = training_args["early_stop"]
        atol = training_args["atol"]
        
        dataset = IITDataset(base_data, ablation_data)
        test_dataset = IITDataset(test_base_data, test_ablation_data)
        train_loader = DataLoader(
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
        assert loss_types in ["all", "iit", "behaviour"]
        optimizer = t.optim.Adam(self.ll_model.parameters(), lr=training_args["lr"])
        if self.hl_model.is_categorical():
            loss_fn = t.nn.CrossEntropyLoss()
        else:
            loss_fn = t.nn.MSELoss()
        # create run only if use_wandb and there is no run already created
        if use_wandb and not wandb.run:
            wandb.init(project="iit", entity=WANDB_ENTITY)

        if use_wandb:
            wandb.config.update(training_args)
            wandb.config.update({"method": "IIT_on_tracr"})

        for epoch in tqdm(range(epochs)):
            iit_losses = []
            behavior_losses = []
            self.ll_model.train()
            for i, (base_input, ablation_input) in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):
                if loss_types == "all" or loss_types == "iit":
                    iit_loss = self.run_IIT_train_step(
                        base_input, ablation_input, loss_fn, optimizer
                    )
                    iit_losses.append(iit_loss)
                if loss_types == "all" or loss_types == "behaviour":
                    behavior_loss = self.run_behaviour_train_step(
                        base_input, loss_fn, optimizer
                    )
                    behavior_losses.append(behavior_loss)

            # now calculate test loss
            test_losses = []
            IIA = []
            behavioral_accuracies = []
            self.ll_model.eval()
            for i, (base_input, ablation_input) in enumerate(test_loader):
                loss, iia = self.get_test_accs(
                    base_input, ablation_input, loss_fn, atol
                )
                IIA.append(iia)
                test_losses.append(loss.item())

                # compute behavioral accuracy
                base_x, base_y = self.get_encoded_input_from_torch_input(base_input)
                output = self.ll_model(base_x)
                if self.hl_model.is_categorical():
                    top1 = t.argmax(output, dim=-1)
                    accuracy = (top1 == base_y).float().mean()
                else:
                    accuracy = (
                        (
                            (
                                output
                                - base_y.unsqueeze(-1)
                                .float()
                                .to(self.ll_model.cfg.device)
                            ).abs()
                            < atol
                        )
                        .float()
                        .mean()
                    )
                behavioral_accuracies.append(accuracy.item())
            print(
                f"Epoch {epoch}: {np.mean((iit_losses + behavior_losses)):.4f},",
                f"loss: {np.mean(test_losses):.4f},",
                f"IIA: {np.mean(IIA)*100:.2f}%,",
                f"B acc: {np.mean(behavioral_accuracies)*100:.2f}%",
            )

            if use_wandb:
                wandb.log(
                    {
                        "iit loss": np.mean(iit_losses),
                        "behavior loss": np.mean(behavior_losses),
                        "test loss": np.mean(test_losses),
                        "IIA": np.mean(IIA),
                        "behavioral accuracy": np.mean(behavioral_accuracies),
                    }
                )
            if np.mean(IIA) > 0.99 and early_stop:
                break
        if use_wandb:
            wandb.log({"final epoch": epoch})
