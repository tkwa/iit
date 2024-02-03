# Interchange Intervention Training

A replication of the paper ["Inducing Causal Structure for Interpretable Neural Networks"][1] by Geiger et al.

[1]: https://arxiv.org/abs/2112.00826


## Installation

You can use the [Dockerfile](./Dockerfile) to set it up. Alternatively, you can use `poetry install` from the root folder.

## Running
There is only one task you can run this on for now. 

Running `python train.py` trains a model using Intercahnge Interventions + Multi Task ([1])

Running `python eval.py` generates plots for how accurate the circuit induced is. 