from iit.utils.index import *


def test_index_equals():
    assert Ix[:, :, :, :] == Ix[:, :, :, :]
    assert Ix[:, :, 0, :] == Ix[:, :, 0, :]
    assert Ix[:, :, 0, :] != Ix[:, :, 1, :]
    assert Ix[:, :] != Ix[:, :, 1, :]

def test_index_hash():
    assert hash(Ix[:, :, :, :]) == hash(Ix[:, :, :, :])
    assert hash(Ix[:, :, 0, :]) == hash(Ix[:, :, 0, :])
    assert hash(Ix[:, :, 0, :]) != hash(Ix[:, :, 1, :])

def test_index_intersect():
    assert Ix[:, :, :, :].intersects(Ix[:, :, :, :])
    assert Ix[:, :, :, :].intersects(Ix[:, :, 0, :])
    assert not Ix[:, :, 0, :].intersects(Ix[:, :, 2:3, :])
    assert not Ix[:, :, 2, :].intersects(Ix[:, :, 1, :])

    # assert objects remain unchanged
    i1 = Ix[:, :, 1, :]
    i2 = Ix[:, :, :, :]
    assert i1.intersects(i2)
    assert i1 == Ix[:, :, 1, :]
    assert i2 == Ix[:, :, :, :]

    # check if unequal lengths raise error
    try:
        Ix[:, :, 1].intersects(Ix[:, :, :, :])
        assert False
    except ValueError:
        pass
