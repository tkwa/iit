# %%
import transformer_lens

from iit.model_pairs.base_model_pair import HookName
import torch as t
from transformer_lens.hook_points import HookedRootModule, HookPoint
from iit.utils.config import DEVICE
from iit.model_pairs.base_model_pair import HLNode, LLNode
from iit.utils.index import Ix
# from .utils import *

# TODO change to support batching

# %%

IOI_NAMES = t.tensor([10, 20, 30]) # TODO

class DuplicateHead(t.nn.Module):
    def forward(self, tokens:t.Tensor):
        # Write the last previous position of any duplicated token (used at S2)
        positions = (tokens == tokens[:, None])
        positions = t.triu(positions, diagonal=1)
        indices = positions.nonzero(as_tuple=True)
        ret = t.full_like(tokens, -1)
        ret[indices[1]] = indices[0]
        return ret
    
class PreviousHead(t.nn.Module):
    def forward(self, tokens:t.Tensor):
        # copy token S1 to token S1+1 (used at S1+1)
        ret = t.full_like(tokens, -1)
        ret[1:] = tokens[:-1]
        return ret

class InductionHead(t.nn.Module):
    """Induction heads omitted because they're redundant with duplicate heads in IOI"""
    

class SInhibitionHead(t.nn.Module):
    def forward(self, tokens: t.Tensor, duplicate: t.Tensor):
        """
        when duplicate is not -1, 
        output a flag to the name mover head to NOT copy this name
        flag is -1 if no duplicate name here, and name token for the name to inhibit
        """
        ret = tokens.clone()
        ret[duplicate != -1] = -1
        return ret
    
class NameMoverHead(t.nn.Module):
    def __init__(self, d_vocab:int=40, names=IOI_NAMES):
        super().__init__()
        self.d_vocab_out = d_vocab
        self.names = names

    def forward(self, tokens: t.Tensor, s_inhibition: t.Tensor):
        """
        increase logit of all names in the sentence, except those flagged by s_inhibition
        """
        logits = t.zeros((len(tokens), self.d_vocab_out)) # seq d_vocab
        # we want every name to increase its corresponding logit after it appears
        name_mask = tokens.eq(self.names[:, None]).any(dim=0)
        logits[t.arange(len(tokens)), tokens] = 10 * name_mask.float()
        # now decrease the logit of the names that are inhibited
        logits[t.arange(len(tokens)), s_inhibition] += -5 * s_inhibition.ne(-1).float()
        logits = t.cumsum(logits, dim=0)
        return logits

PreviousHead()(t.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5]))
        
# %%
        
class IOI_HL(HookedRootModule):
    """
    Components:
    - Duplicate token heads: write the previous position of any duplicated token
    - Previous token heads: copy token S1 to token S1+1
    - Induction heads (omitted): Attend to position written by duplicate token heads
    - S-inhibition heads: Inhibit attention of Name Mover Heads to S1 and S2 tokens
    - Name mover heads: Copy all previous names in the sentence
    """
    def __init__(self, device=DEVICE):
        super().__init__()
        self.duplicate_head = DuplicateHead().to(device)
        self.hook_duplicate = HookPoint().to(device)
        self.previous_head = PreviousHead().to(device)
        self.hook_previous = HookPoint().to(device)
        self.s_inhibition_head = SInhibitionHead().to(device)
        self.hook_s_inhibition = HookPoint().to(device)
        self.name_mover_head = NameMoverHead().to(device)
        self.hook_name_mover = HookPoint().to(device)
        self.setup()

    # def get_idx_to_intermediate(self, name: HookName):
    #     """
    #     Returns a function that takes in a list of intermediate variables and returns the index of the one to use.
    #     """
    #     if name == 'hook_duplicate':
    #         return lambda intermediate_vars: intermediate_vars[:, 0]
    #     elif name == 'hook_previous':
    #         return lambda intermediate_vars: intermediate_vars[:, 1]
    #     elif name == 'hook_induction':
    #         return lambda intermediate_vars: intermediate_vars[:, 2]
    #     elif name == 'hook_s_inhibition':
    #         return lambda intermediate_vars: intermediate_vars[:, 3]
    #     elif name == 'hook_name_mover':
    #         return lambda intermediate_vars: intermediate_vars[:, 4]
    #     else:
    #         raise NotImplementedError(name)

    def forward(self, args):
        input, label, _intermediate_data = args
        # print([a.shape for a in args])
        # duplicate, previous, induction, s_inhibition, name_mover = [intermediate_data[:, i] for i in range(5)]
        # print(f"intermediate_data is a {type(intermediate_data)}; duplicate is a {type(duplicate)}")
        duplicate = self.duplicate_head(input)
        duplicate = self.hook_duplicate(duplicate) # used while ablating
        previous = self.previous_head(input)
        previous = self.hook_previous(previous)
        s_inhibition = self.s_inhibition_head(input, duplicate)
        s_inhibition = self.hook_s_inhibition(s_inhibition)
        out = self.name_mover_head(input, s_inhibition)
        out = self.hook_name_mover(out)
        return out
    
IOI_HL(device='cuda')((t.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5]), None, None))
# %%
