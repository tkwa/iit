from .mnist_pvr.dataset import ImagePVRDataset
from .mnist_pvr.utils import mnist_train, mnist_test
from .mnist_pvr.utils import get_input_shape as get_pvr_input_shape
from .mnist_pvr.get_alignment import get_alignment as get_mnist_pvr_corr
from transformer_lens.hook_points import HookedRootModule

def get_dataset(task: str, dataset_config: dict):
    if 'pvr' in task:
        default_dataset_args = {
            'pad_size': 7,
            'train_size' : 60000,
            'test_size' : 6000
        }
        default_dataset_args.update(dataset_config)
        if task == 'mnist_pvr':
            unique_per_quad = False
        elif task == 'pvr_leaky':
            unique_per_quad = True
        else:
            raise ValueError(f"Unknown task {task}")
        train_set = ImagePVRDataset(mnist_train, length=default_dataset_args['train_size'], pad_size=default_dataset_args['pad_size'], unique_per_quad=unique_per_quad)
        test_set = ImagePVRDataset(mnist_test, length=default_dataset_args['test_size'], pad_size=default_dataset_args['pad_size'], unique_per_quad=unique_per_quad)
    else:
        raise ValueError(f"Unknown task {task}")
    return train_set, test_set

def get_alignment(task: str, config: dict) ->  tuple([HookedRootModule, HookedRootModule, dict]):
    if 'pvr' in task:
        default_config = {
            'mode': 'q',
            'hook_point': 'mod.layer3.mod.1.mod.conv2.hook_point',
            'model': 'resnet18',
            'pad_size': 7,
        }
        default_config.update(config)
        return get_mnist_pvr_corr(default_config, task)
    
    raise ValueError(f"Unknown task {task}")
    
def get_input_shape(task: str):
    if 'pvr' in task:
        return get_pvr_input_shape()
    raise ValueError(f"Unknown task {task}")