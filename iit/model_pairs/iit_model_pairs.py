from iit.model_pairs.base_model_pair import *

class IITModelPair(BaseModelPair):
    def __init__(self, hl_model:HookedRootModule=None, ll_model:HookedRootModule=None,
                 hl_graph=None, corr:dict[HLNode, set[LLNode]]={}, seed=0, training_args={}):
        # TODO change to construct hl_model from graph?
        if hl_model is None:
            assert hl_graph is not None
            hl_model = self.make_hl_model(hl_graph)

        self.hl_model = hl_model
        self.ll_model = ll_model

        self.corr:dict[HLNode, set[LLNode]] = corr
        assert all([k in self.hl_model.hook_dict for k in self.corr.keys()])
        self.rng = np.random.default_rng(seed)
        self.training_args = training_args

    def make_hl_model(self, hl_graph):
        raise NotImplementedError

    def set_corr(self, corr):
        self.corr = corr

    def sample_hl_name(self) -> str:
        # return a `str` rather than `numpy.str_`
        return str(self.rng.choice(list(self.corr.keys())))

    def hl_ablation_hook(self,hook_point_out:Tensor, hook:HookPoint) -> Tensor:
        out = self.hl_cache[hook.name]
        return out
    
    # TODO extend to position and subspace...
    def make_ll_ablation_hook(self, ll_node:LLNode) -> Callable[[Tensor, HookPoint], Tensor]:
        if ll_node.subspace is not None:
            raise NotImplementedError
        def ll_ablation_hook(hook_point_out:Tensor, hook:HookPoint) -> Tensor:
            out = hook_point_out.clone()
            out[ll_node.index.as_index] = self.ll_cache[hook.name][ll_node.index.as_index]
            return out
        return ll_ablation_hook

    def do_intervention(self, base_input, ablation_input, hl_node:HookName, verbose=False):
        ablation_x, ablation_y, ablation_intermediate_vars = ablation_input
        base_x, base_y, base_intermediate_vars = base_input
        hl_ablation_output, self.hl_cache = self.hl_model.run_with_cache(ablation_input)
        assert all(hl_ablation_output == ablation_y), f"Ablation output {hl_ablation_output} does not match label {ablation_y}"
        ll_ablation_output, self.ll_cache = self.ll_model.run_with_cache(ablation_x)

        ll_nodes = self.corr[hl_node]

        hl_output = self.hl_model.run_with_hooks(base_input, fwd_hooks=[(hl_node, self.hl_ablation_hook)])
        ll_output = self.ll_model.run_with_hooks(base_x, fwd_hooks=[(ll_node.name, self.make_ll_ablation_hook(ll_node)) for ll_node in ll_nodes])

        if verbose:
            ablation_x_image = torchvision.transforms.functional.to_pil_image(ablation_x[0])
            ablation_x_image.show()
            print(f"{ablation_x_image=}, {ablation_y.item()=}, {ablation_intermediate_vars=}")
            base_x_image = torchvision.transforms.functional.to_pil_image(base_x[0])
            base_x_image.show()
            print(f"{base_x_image=}, {base_y.item()=}, {base_intermediate_vars=}")
            print(f"{hl_ablation_output=}, {ll_ablation_output.shape=}")
            print(f"{hl_node=}, {ll_nodes=}")
            print(f"{hl_output=}")
        return hl_output, ll_output
    
    def no_intervention(self, base_input):
        base_x, base_y, base_intermediate_vars = base_input
        hl_output = self.hl_model(base_input)
        ll_output = self.ll_model(base_x)
        return hl_output, ll_output

    def train(self, base_data, ablation_data, test_base_data, test_ablation_data, epochs=1000, use_wandb=False):
        training_args = self.training_args
        print(f"{training_args=}")
        dataset = IITDataset(base_data, ablation_data)
        test_dataset = IITDataset(test_base_data, test_ablation_data)
        loader = DataLoader(dataset, batch_size=training_args['batch_size'], shuffle=True, num_workers=training_args['num_workers'])
        test_loader = DataLoader(test_dataset, batch_size=training_args['batch_size'], shuffle=True, num_workers=training_args['num_workers'])
        optimizer = t.optim.Adam(self.ll_model.parameters(), lr=training_args['lr'])
        loss_fn = t.nn.CrossEntropyLoss()

        if use_wandb:
            wandb.init(project="iit", entity=WANDB_ENTITY)
            wandb.config.update(training_args)
            wandb.config.update({'method': 'IIT'})

        for epoch in tqdm(range(epochs)):
            losses = []
            for i, (base_input, ablation_input) in tqdm(enumerate(loader), total=len(loader)):
                base_input = [t.to(DEVICE) for t in base_input]
                ablation_input = [t.to(DEVICE) for t in ablation_input]
                optimizer.zero_grad()
                self.hl_model.requires_grad_(False)
                self.ll_model.train()

                # sample a high-level variable to ablate
                hl_node = self.sample_hl_name()
                hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
                # hl_output, ll_output = self.no_intervention(base_input)
                loss = loss_fn(ll_output, hl_output)
                loss.backward()
                # print(f"{ll_output=}, {hl_output=}")
                losses.append(loss.item())
                optimizer.step()
            # now calculate test loss
            test_losses = []
            accuracies = []
            self.ll_model.eval()
            self.hl_model.requires_grad_(False)
            for i, (base_input, ablation_input) in enumerate(test_loader):
                hl_node = self.sample_hl_name()
                base_input = [t.to(DEVICE) for t in base_input]
                ablation_input = [t.to(DEVICE) for t in ablation_input]
                hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
                # hl_output, ll_output = self.no_intervention(base_input)
                loss = loss_fn(ll_output, hl_output)
                top1 = t.argmax(ll_output, dim=-1)
                accuracy = (top1 == hl_output).float().mean()
                accuracies.append(accuracy.item())
                test_losses.append(loss.item())
            print(f"Epoch {epoch}: {np.mean(losses):.4f}, {np.mean(test_losses):.4f}, {np.mean(accuracies)*100:.4f}%")

            if use_wandb:
                wandb.log({'train loss': np.mean(losses), 'test loss': np.mean(test_losses), 'accuracy': np.mean(accuracies), 'epoch': epoch})

