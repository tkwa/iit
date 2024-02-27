import pytest
import torch as t
from ioi_hl import *


def test_duplicate_head():
    a = DuplicateHead()(t.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5]))
    assert a.equal(t.tensor([-1, -1, -1,  1, -1, -1, -1, -1,  4]))


def test_previous_head():
    a = PreviousHead()(t.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5]))
    assert a.equal(t.tensor([-1,  3,  1,  4,  1,  5,  9,  2,  6]))

  
def test_s_inhibition_head():
    a = SInhibitionHead()(t.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5]), t.tensor([-1, -1, -1,  1, -1, -1, -1, -1,  4]))
    assert a.equal(t.tensor([-1, -1, -1,  1, -1, -1, -1, -1,  5]))


def test_name_mover_head():
    a = NameMoverHead(d_vocab=21)(t.tensor([1, 2, 10, 20]), t.tensor([-1, 20, 10, -1]))

    assert a.equal(t.tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0., -5.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  5.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0., -5.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  5.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  5.]]))