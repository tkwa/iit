# %%
from typing import Optional, List, Iterable
# %%

"""copied from https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/14c75e9898eda70a8e0997390077af1ec2543258/acdc/TLACDCEdge.py"""
class TorchIndex:
    """There is not a clean bijection between things we 
    want in the computational graph, and things that are hooked
    (e.g hook_result covers all heads in a layer)
    
    `TorchIndex`s are essentially indices that say which part of the tensor is being affected. 

    EXAMPLES: Initialise [:, :, 3] with TorchIndex([None, None, 3]) and [:] with TorchIndex([None])    

    Also we want to be able to call e.g `my_dictionary[my_torch_index]` hence the hashable tuple stuff
    
    Note: ideally this would be integrated with transformer_lens.utils.Slice in future; they are accomplishing similar but different things"""

    def __init__(
        self, 
        list_of_things_in_tuple: Iterable,
    ):
        # check correct types
        for arg in list_of_things_in_tuple:
            if type(arg) in [type(None), int, slice]:
                continue
            else:
                assert isinstance(arg, list)
                assert all([type(x) == int for x in arg])

        # make an object that can be indexed into a tensor
        self.as_index = tuple([slice(None) if x is None else x for x in list_of_things_in_tuple])

        # make an object that can be hashed (so used as a dictionary key)
        # compatibility with python <3.12 where slices are not hashable
        self.hashable_tuple = tuple((i.__reduce__() if isinstance(i, slice) else i for i in list_of_things_in_tuple))

    def __hash__(self):
        return hash(self.hashable_tuple)

    def __eq__(self, other):
        return self.hashable_tuple == other.hashable_tuple

    def __repr__(self) -> str: # graphviz, an old library used to dislike actual colons in strings, but this shouldn't be an issue anymore
        ret = "["
        for idx, x in enumerate(self.hashable_tuple):
            if idx > 0:
                ret += ", "
            if x is None:
                ret += ":"
            elif type(x) == int:
                ret += str(x)
            elif type(x) == slice:
                ret += (str(x.start) if x.start is not None else "") + ":" + (str(x.stop) if x.stop is not None else "")
                assert x.step is None, "Step is not supported"
            else:
                raise NotImplementedError(x)
        ret += "]"
        return ret

    def graphviz_index(self, use_actual_colon=True) -> str:
        return self.__repr__(use_actual_colon=use_actual_colon)
    
class Index:
    """A class purely for syntactic sugar, that returns the index it's indexed with"""
    def __getitem__(self, index):
        return TorchIndex(index)
    
Ix = Index()
# %%
