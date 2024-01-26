from .base_model_pair import *

class IITProbeModelPair(BaseModelPair):
    def __init__(self, hl_model:HookedRootModule=None, ll_model:HookedRootModule=None,
                 hl_graph=None, corr:dict[HLNode, set[LLNode]]={}, seed=0, training_args={},
                 check_parent=False):
        # TODO change to construct hl_model from graph?
        if hl_model is None:
            hl_model = self.make_hl_model(hl_graph)

        self.hl_model = hl_model
        self.ll_model = ll_model

        self.corr:dict[HLNode, set[LLNode]] = corr
        assert all([k in self.hl_model.hook_dict for k in self.corr.keys()])
        self.rng = np.random.default_rng(seed)
        default_training_args = {
                        'batch_size': 256,
                        'lr': 0.001,
                        'num_workers': 0,
                        'probe_weight': 1.0,
        }
        training_args = {**default_training_args, **training_args}
        self.training_args = training_args
        self.check_parent = check_parent
        self.probe_cache = {}

    def make_probes(self, input_shape):
        # create a dummy cache to get hook dimensions
        print(input_shape)
        # TODO: remove dummy_cache (can be done using channels from model and index size from corr)
        out, dummy_cache = self.ll_model.run_with_cache(t.zeros(input_shape, device=DEVICE))
        print(out.shape)
        self.probes = {}

        for hl_node, ll_nodes in self.corr.items():
            if len(ll_nodes) > 1:
                raise NotImplementedError # should this be all layers' activations flattened into one?
            for ll_node in ll_nodes:
                hook_dim = dummy_cache[ll_node.name][ll_node.index.as_index].flatten().shape[0]
                
                # print("=======================")
                # print(ll_node.name, ll_node.index.as_index, dummy_cache[ll_node.name].shape, hook_dim)
                
                if ll_node.subspace is None:
                    self.probes[hl_node] = t.nn.Linear(
                        in_features=hook_dim, 
                        out_features=10, # TODO: Remove hardcoding 
                        bias=False).to(DEVICE)
                else:
                    raise NotImplementedError
        print("made probes", [(k, p.weight.shape) for k, p in self.probes.items()])
    
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
    def make_ll_ablation_hook(self, ll_node:LLNode, hl_name: HookName) -> Callable[[Tensor, HookPoint], Tensor]:
        if ll_node.subspace is not None:
            raise NotImplementedError
        def ll_ablation_hook(hook_point_out:Tensor, hook:HookPoint) -> Tensor:
            out = hook_point_out.clone() 
            out[ll_node.index.as_index] = self.ll_cache[hook.name][ll_node.index.as_index]
            # out = out.clone()
            hl_node_name = hl_name
            probe_in_shape = self.probes[hl_node_name].weight.shape[1:]
            probe_output = self.probes[hl_node_name](
                (out[ll_node.index.as_index]).reshape(-1, *probe_in_shape)
                )
            self.probe_cache[hl_node_name] = probe_output
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
        ll_output = self.ll_model.run_with_hooks(base_x, fwd_hooks=[(ll_node.name, self.make_ll_ablation_hook(ll_node, hl_node)) for ll_node in ll_nodes])

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
    
    def train(self, base_data, ablation_data, test_base_data, test_ablation_data, epochs=1000, use_wandb=False):
        training_args = self.training_args
        print(f"{training_args=}")
        dataset = IITDataset(base_data, ablation_data)
        test_dataset = IITDataset(test_base_data, test_ablation_data)

        # add to make probes
        input_shape = (dataset[0][0][0]).unsqueeze(0).shape
        with t.no_grad():
            self.make_probes(input_shape)
        
        loader = DataLoader(dataset, batch_size=training_args['batch_size'], shuffle=True, num_workers=training_args['num_workers'])
        test_loader = DataLoader(test_dataset, batch_size=training_args['batch_size'], shuffle=True, num_workers=training_args['num_workers'])
        params = list(self.ll_model.parameters())
        for p in self.probes.values():
            params += list(p.parameters())
        optimizer = t.optim.Adam(params, lr=training_args['lr'])
        loss_fn = t.nn.CrossEntropyLoss()

        if use_wandb:
            wandb.init(project="iit", entity=WANDB_ENTITY)
            wandb.config.update(training_args)
            wandb.config.update({'method': 'IIT + Probes'})
        torch.autograd.set_detect_anomaly(True)

        for epoch in tqdm(range(epochs)):
            losses = []
            probe_losses = []
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
                ablation_loss = loss_fn(ll_output, hl_output)

                # add probe losses
                probe_losses = []
                for hl_node_name in self.probe_cache.keys():
                    _, _, intermediate_vars = ablation_input
                    probe_output = self.probe_cache[hl_node_name]
                    print(hl_node_name)
                    gt = self.hl_model.get_idx_to_intermediate(hl_node_name)(intermediate_vars)
                    print(f"{probe_output.shape}, {gt.shape}, {intermediate_vars.shape}")
                    probe_loss = loss_fn(probe_output, gt)
                    probe_losses.append(probe_loss)

                probe_loss = sum(probe_losses)
                probe_losses.append(probe_loss)
                loss = ablation_loss + training_args['probe_weight'] * probe_loss
                loss.backward()
                # print(f"{ll_output=}, {hl_output=}")
                losses.append(loss.item())
                probe_losses.append(probe_loss.item())
                optimizer.step()

            
            # now calculate test loss
            test_losses = []
            accuracies = []
            probe_losses = []
            probe_accuracies = []

            self.ll_model.eval()
            for p in self.probes.values():
                p.eval()
            self.hl_model.requires_grad_(False)
            with t.no_grad():
                for i, (base_input, ablation_input) in enumerate(test_loader):
                    base_input = [t.to(DEVICE) for t in base_input]
                    ablation_input = [t.to(DEVICE) for t in ablation_input]
                    hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node) # !!! resample?
                    # hl_output, ll_output = self.no_intervention(base_input)
                    loss = loss_fn(ll_output, hl_output)
                    top1 = t.argmax(ll_output, dim=1)
                    accuracy = (top1 == hl_output).float().mean()
                    accuracies.append(accuracy.item())
                    test_losses.append(loss.item())

                    # !!! Second forward pass
                    # add probe losses and accuracies
                    base_x, base_y, base_intermediate_vars = base_input
                    out, cache = self.ll_model.run_with_cache(base_x)
                    probe_loss = 0
                    probe_accuracy = 0
                    behavior_loss = loss_fn(out, base_y)
                    top1_behavior = t.argmax(out, dim=1)
                    behavior_accuracy = (top1_behavior == base_y).float().mean()

                    for hl_node_name in self.probes.keys():
                        gt = self.hl_model.get_idx_to_intermediate(hl_node_name)(base_intermediate_vars)
                        ll_nodes = self.corr[hl_node_name]
                        if len(ll_nodes) > 1:
                            raise NotImplementedError
                        for ll_node in ll_nodes:
                            probe_in_shape = self.probes[hl_node_name].weight.shape[1:]
                            probe_out = self.probes[hl_node_name](cache[ll_node.name][ll_node.index.as_index].reshape(-1, *probe_in_shape))
                            probe_loss += loss_fn(probe_out, gt)
                            top1 = t.argmax(probe_out, dim=1)
                            probe_accuracy += (top1 == gt).float().mean()

            print(f"Epoch {epoch}: {np.mean(losses):.4f}, \n Test: {np.mean(test_losses):.4f}, {np.mean(accuracies)*100:.4f}%, \nProbe {np.mean(probe_accuracies)*100:.4f}%")

            if use_wandb:
                wandb.log({
                    'train loss': np.mean(losses), 
                    'test loss': np.mean(test_losses), 
                    'accuracy': np.mean(accuracies), 
                    'epoch': epoch, 
                    'probe loss': np.mean(probe_losses), 
                    'probe accuracy': np.mean(probe_accuracies),
                    'behavior loss': behavior_loss.item(),
                    'behavior accuracy': behavior_accuracy.item(),
                })