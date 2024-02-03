from iit.tasks.task_loader import *
from iit.model_pairs import *

training_args = {
    'batch_size': 256,
    'lr': 0.001,
    'num_workers': 0,
}

task = 'mnist_pvr'
train_set, test_set = get_dataset(task, dataset_config={})
ll_model, hl_model, corr = get_alignment(task, config={})
model_pair = IITModelPair(ll_model=ll_model, hl_model=hl_model, corr=corr, training_args=training_args) # TODO: add wrapper for choosing model pair
model_pair.train(train_set, train_set, test_set, test_set, epochs=1000, use_wandb=False)

print(f"done training")
