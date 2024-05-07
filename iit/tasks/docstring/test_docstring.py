import pytest
import torch as t
from .docstring_hl import *

def nonzero_values(a: t.Tensor):
    return t.cat((a.nonzero(), a[a != 0][:, None]), dim=-1)

def test_arg_mover_head():
    tokens = t.tensor([[1, 2, 3], [4, 5, 6]])
    def_patterns = t.tensor([[1, 2, 3], [4, 5, 6]])
    induction_output = t.tensor([[5, 1, 10], [15, 4, 4]])
    arg_mover = ArgMoverHead(d_vocab=6)
    logits = arg_mover(tokens, def_patterns, induction_output)
    assert logits.shape == (2, 3, 6) # batch seq d_vocab
    # print(logits)
    assert nonzero_values(logits).equal(t.tensor(
        [[0, 1, 1, 10],
        [1, 1, 4, 10],
        [1, 2, 4, 10]]
    ))

if __name__ == "__main__":
    test_arg_mover_head()