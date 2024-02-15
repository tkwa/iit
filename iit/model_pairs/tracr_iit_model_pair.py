from .base_model_pair import *
from iit.model_pairs.iit_model_pairs import IITModelPair

class TracrIITModelPair(IITModelPair):
    """
    Unlike in the IIT code, our high-level model doesn't "cheat" by getting intermediate states with input.
    Therefore, `do_intervention` is simpler.
    """
    def __init__(self, hl_model, ll_model, corr, training_args={}):
        super().__init__(hl_model, ll_model, corr=corr, training_args=training_args)

    def do_intervention(self, base_input, ablation_input, hl_node:HookName, verbose=False):
        ablation_x = list(map(list, zip(*ablation_input)))
        base_x = list(map(list, zip(*base_input)))
        encoded_base_x = self.hl_model.map_tracr_input_to_tl_input(base_x)
        encoded_ablation_x = self.hl_model.map_tracr_input_to_tl_input(ablation_x)
        ablation_intermediate_vars = None # TODO
        base_intermediate_vars = None # TODO
        
        hl_ablation_output, self.hl_cache = self.hl_model.run_with_cache(encoded_ablation_x)
        # # TODO: 
        # # assert all(hl_ablation_output == ablation_y), f"Ablation output {hl_ablation_output} does not match label {ablation_y}"
        ll_ablation_output, self.ll_cache = self.ll_model.run_with_cache(encoded_ablation_x)

        ll_nodes = self.corr[hl_node]

        hl_output = self.hl_model.run_with_hooks(
            encoded_base_x, fwd_hooks=[(hl_node, self.hl_ablation_hook)])
        
        ll_output = self.ll_model.run_with_hooks(
            encoded_base_x, fwd_hooks=[
                    (ll_node.name, self.make_ll_ablation_hook(ll_node)) 
                    for ll_node in ll_nodes])

        if verbose:
            ablation_x_image = torchvision.transforms.functional.to_pil_image(ablation_x[0])
            ablation_x_image.show()
            # print(f"{ablation_x_image=}, {ablation_y.item()=}, {ablation_intermediate_vars=}")
            print(f"{hl_ablation_output=}, {ll_ablation_output.shape=}")
            print(f"{hl_node=}, {ll_nodes=}")
            print(f"{hl_output=}")
        return hl_output, ll_output

    def train_from_loader(self, train_loader, test_loader, epochs=1000, use_wandb=False):
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
            wandb.config.update({'method': 'IIT'})

        for epoch in tqdm(range(epochs)):
            losses = []
            for i, (base_input, ablation_input) in tqdm(enumerate(train_loader), total=len(train_loader)):
                # base_input = [t.to(DEVICE) for t in base_input]
                # ablation_input = [t.to(DEVICE) for t in ablation_input]
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
                hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
                if self.hl_model.is_categorical():
                    loss = loss_fn(ll_output, hl_output)
                    top1 = t.argmax(ll_output, dim=1)
                    accuracy = (top1 == hl_output).float().mean()
                    accuracies.append(accuracy.item())
                else:
                    loss = loss_fn(ll_output, hl_output)
                    accuracies.append(0)
                test_losses.append(loss.item())
            print(f"Epoch {epoch}: {np.mean(losses):.4f}, {np.mean(test_losses):.4f}, {np.mean(accuracies)*100:.4f}%")

            if use_wandb:
                wandb.log({'train loss': np.mean(losses), 'test loss': np.mean(test_losses), 'accuracy': np.mean(accuracies), 'epoch': epoch})
