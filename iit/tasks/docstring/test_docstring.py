import pytest
import torch as t
from .docstring_hl import *

def nonzero_values(a: t.Tensor):
    return t.cat((a.nonzero(), a[a != 0][:, None]), dim=-1)

def test_induction_head():
    prev_tok_out = t.tensor([[2, 2, 2], [4, 5, 6]])
    tokens = t.tensor([[1, 2, 3], [20, 4, 4]])
    induction_head = InductionHead()
    result = induction_head(tokens, prev_tok_out)
    assert result.shape == (2, 3)
    assert result.equal(t.tensor(
        [[-1,  1, -1],
        [-1, 20, 20]]
    ))

def test_arg_mover_head():
    tokens = t.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    def_patterns = t.tensor([[10, 10, 2, 3], [9, 9, 9, 7]])
    induction_output = t.tensor([[5, 0, 10, 1], [10, 9, 9, 9]])
    arg_mover = ArgMoverHead(d_vocab=8, logit_increase=50)
    logits = arg_mover(tokens, def_patterns, induction_output)
    assert logits.shape == (2, 4, 8) # batch seq d_vocab
    assert nonzero_values(logits).equal(t.tensor(
        [[0, 2, 0, 50],
         [1, 2, 4, 50],
         [1, 3, 4, 50],
         [1, 3, 5, 50],
        ]
    ))

def test_docstring_abc():
    token_map = {
        "load": 1,
        "size": 2,
        "files": 3,
        ",": 4,
        "param": 5,
    }

    tokens = "load , size , files 9 10 param load 11 param size 12 13 12 param"

    tokens = [token_map.get(token, token) for token in tokens.split()]
    tokens = t.tensor([[int(t) for t in tokens]])
    model = Docstring_HL()
    logits = model((tokens, None, None))
    print(logits[0, -1])
    assert nonzero_values(logits[0, -1]).equal(t.tensor(
        [[3, 50]]
    ))

if __name__ == "__main__":
    test_induction_head()
    test_arg_mover_head()
    test_docstring_abc()