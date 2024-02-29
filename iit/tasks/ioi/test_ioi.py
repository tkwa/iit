import pytest
import torch as t
from .ioi_hl import *

def nonzero_values(a: t.Tensor):
    return t.cat((a.nonzero(), a[a != 0][:, None]), dim=-1)

def test_duplicate_head():
    a = DuplicateHead()(t.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5]]))
    assert a.equal(t.tensor([[-1, -1, -1,  1, -1, -1, -1, -1,  4]]))


def test_previous_head():
    a = PreviousHead()(t.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5]]))
    assert a.equal(t.tensor([[-1,  3,  1,  4,  1,  5,  9,  2,  6]]))

  
def test_s_inhibition_head():
    a = SInhibitionHead()(t.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5]]), t.tensor([[-1, -1, -1,  1, -1, -1, -1, -1,  4]]))
    assert a.equal(t.tensor([[-1, -1, -1,  1, -1, -1, -1, -1,  5]]))


def test_name_mover_head():
    a = NameMoverHead(d_vocab=21)(t.tensor([[1, 2, 10, 20]]), t.tensor([[-1, 20, 10, -1]]))

    assert nonzero_values(a[0]).equal(t.tensor(
        [[  1.,  20., -15.],
        [  2.,  10.,  -5.],
        [  2.,  20., -15.],
        [  3.,  10.,  -5.],
        [  3.,  20.,  -5.]]))


def test_ioi_hl():
    a = IOI_HL(device='cuda')((t.tensor([[3, 10, 4, 10, 5, 9, 2, 6, 5]]), None, None))
    assert nonzero_values(a[0]).equal(t.tensor([[  1.,  10.,  10.],
        [  2.,  10.,  10.],
        [  3.,  10.,   5.],
        [  4.,  10.,   5.],
        [  5.,  10.,   5.],
        [  6.,  10.,   5.],
        [  7.,  10.,   5.],
        [  8.,   5., -15.],
        [  8.,  10.,   5.]]))