To create a new class of model pairs, you need to do the following:

1. Make [BaseModelPair](./iit/model_pairs/base_model_pair.py) or [IITModelPair](./iit/model_pairs/iit_model_pairs.py) a parent to your model pair class 
   - If you want to make it compatible with Tracr, use [TracrIITModelPair](./iit/model_pairs/tracr_iit_model_pair.py)
2. Implement the following methods: (you don't need to implement all of them if inheriting something other than BaseModelPair)
```
    @property
    def loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def make_train_metrics() -> MetricStoreCollection:
        pass

    @staticmethod
    def make_test_metrics() -> MetricStoreCollection:
        pass
    
    def run_train_step(
        self,
        base_input, # batch of clean inputs
        ablation_input, # batch of corrupted inputs
        loss_fn,
        optimizer,
    ) -> MetricStoreCollection:
        pass

    def run_eval_step(
        self,
        base_input,
        ablation_input,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> MetricStoreCollection:
        pass
```

  - Examples of this can be seen in [IITModelPair](./iit/model_pairs/iit_model_pairs.py) and [TracrIITModelPair](./iit/model_pairs/tracr_iit_model_pair.py). 

[MetricStoreCollection](./iit/utils/metric.py) is a list of metrics which allows us to abstract out logging. This makes the training loop just a standard function. 

TODOs:
1. Note that [IITProbeSequentialPair](./iit/model_pairs/probed_sequential_pair.py) does not follow this pattern yet. Any PRs for this would be highly appreciated!
2. Tracr model pairs use an external library which is not loaded as a submodule. We need to fix this!