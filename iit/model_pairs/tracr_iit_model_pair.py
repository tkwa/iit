from .base_model_pair import *
from iit.model_pairs.iit_model_pairs import IITModelPair

class TracrIITModelPair(IITModelPair):
    def __init__(self, hl_model, ll_model, corr, training_args={}):
        super().__init__(hl_model, ll_model, corr=corr, training_args=training_args)

    def get_encoded_input_from_torch_input(self, input): # TODO: refactor this to outside of the class
        x, y = input
        encoded_x = self.hl_model.map_tracr_input_to_tl_input(list(map(list, zip(*x))))
        y[0] = torch.zeros_like((y[1]))
        return encoded_x, torch.tensor(list(map(list, zip(*y))))
    
    def do_intervention(self, base_input, ablation_input, hl_node:HookName, verbose=False):
        ablation_x, ablation_y = self.get_encoded_input_from_torch_input(ablation_input)
        base_x, base_y = self.get_encoded_input_from_torch_input(base_input)

        hl_ablation_output, self.hl_cache = self.hl_model.run_with_cache(ablation_x)
        assert torch.allclose(hl_ablation_output.cpu(), ablation_y.unsqueeze(-1).cpu().float()), \
        f"Ablation output {hl_ablation_output} does not match label {ablation_y}"

        ll_ablation_output, self.ll_cache = self.ll_model.run_with_cache(ablation_x)
        ll_nodes = self.corr[hl_node]

        hl_output = self.hl_model.run_with_hooks(
            base_x, fwd_hooks=[(hl_node, self.hl_ablation_hook)])
        ll_output = self.ll_model.run_with_hooks(
            base_x, fwd_hooks=[(ll_node.name, 
                                self.make_ll_ablation_hook(ll_node)) 
                                for ll_node in ll_nodes])

        if verbose:
            print(f"{hl_ablation_output=}, {ll_ablation_output.shape=}")
            print(f"{hl_node=}, {ll_nodes=}")
            print(f"{hl_output=}")
        return hl_output, ll_output

    def train_from_loader(self, train_loader, test_loader, epochs=1000, use_wandb=False, 
                          losses = "all", atol=1e-2):
        assert losses in ["all", "iit", "behaviour"]
        training_args = self.training_args
        print(f"{training_args=}")
        optimizer = t.optim.Adam(self.ll_model.parameters(), lr=training_args['lr'])
        if self.hl_model.is_categorical():
            loss_fn = t.nn.CrossEntropyLoss()
        else:
            loss_fn = t.nn.MSELoss()

        if use_wandb:
            wandb.init(project="iit", entity=WANDB_ENTITY)
            wandb.config.update(training_args)
            wandb.config.update({'method': 'IIT_on_tracr'})

        for epoch in tqdm(range(epochs)):
            iit_losses = []
            behavior_losses = []
            self.ll_model.train()
            for i, (base_input, ablation_input) in tqdm(enumerate(train_loader), 
                                                        total=len(train_loader)):
                if losses == "all" or losses == "iit":
                    hl_node = self.sample_hl_name() # sample a high-level variable to ablate
                    iit_losses.append(self.run_train_step(
                        base_input, ablation_input, hl_node, loss_fn, optimizer
                    ))
                if losses == "all" or losses == "behaviour":
                    optimizer.zero_grad()
                    base_x, base_y = self.get_encoded_input_from_torch_input(base_input)
                    output = self.ll_model(base_x)
                    behavior_loss = loss_fn(output, base_y.unsqueeze(-1).float().to(self.ll_model.cfg.device))
                    behavior_loss.backward()
                    optimizer.step()
                    behavior_losses.append(behavior_loss.item())

            # now calculate test loss
            test_losses = []
            IIA = []
            behavioral_accuracies = []
            self.ll_model.eval()
            for i, (base_input, ablation_input) in enumerate(test_loader):
                hl_node = self.sample_hl_name()
                hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
                if self.hl_model.is_categorical():
                    loss = loss_fn(ll_output, hl_output)
                    top1 = t.argmax(ll_output, dim=1)
                    accuracy = (top1 == hl_output).float().mean()
                    IIA.append(accuracy.item())
                else:
                    loss = loss_fn(ll_output, hl_output)
                    IIA.append(((ll_output - hl_output).abs() < atol).float().mean().item())
                test_losses.append(loss.item())

                # compute behavioral accuracy
                base_x, base_y = self.get_encoded_input_from_torch_input(base_input)
                output = self.ll_model(base_x)
                if self.hl_model.is_categorical():
                    top1 = t.argmax(output, dim=-1)
                    accuracy = (top1 == base_y).float().mean()
                else:
                    accuracy = (
                        (output - base_y.unsqueeze(-1).float().to(
                        self.ll_model.cfg.device)
                        ).abs() < atol).float().mean()
                behavioral_accuracies.append(accuracy.item())
            print(f"Epoch {epoch}: {np.mean((iit_losses + behavior_losses)):.4f},",
                  f"loss: {np.mean(test_losses):.4f},", 
                  f"IIA: {np.mean(IIA)*100:.2f}%,",
                  f"B acc: {np.mean(behavioral_accuracies)*100:.2f}%")

            if use_wandb:
                wandb.log({
                    'train loss': np.mean(iit_losses + behavior_losses),
                    'test loss': np.mean(test_losses),
                    'IIA': np.mean(IIA),
                    'behavioral accuracy': np.mean(behavioral_accuracies)
                })
            if np.mean(IIA) > 0.99:
                break
