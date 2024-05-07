import transformer_lens
import numpy as np
import torch as t

"""
If we want to make more tasks, better to pull these heads out into a shared file
"""
from iit.tasks.ioi.ioi_hl import PreviousHead
from iit.model_pairs.base_model_pair import HookName
from transformer_lens.hook_points import HookedRootModule, HookPoint
from iit.tasks.hl_model import HLModel

class InductionHead(t.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tokens: t.Tensor, prev_tok_out: t.Tensor):
        """
        In a1 b1 ... a2 situations, uses prev tok information from b1
        to copy token b1 to induction output of position a2.

        prev_tok_out[b1] must equal tokens[a2]
        """
        result = t.full_like(tokens, -1) # batch seq

        matches = (prev_tok_out[..., None] == tokens[..., None, :]) # batch seq1 seq2, seq1<seq2
        matches = t.triu(matches, diagonal=1) # only consider positions before this one
        indices = matches.nonzero(as_tuple=True)

        result[indices[0], indices[2]] = tokens[indices[0], indices[1]] # TODO only copy the token with max seq1
        return result


class ArgMoverHead(t.nn.Module):
    def __init__(self, d_vocab:int=40, ):
        super().__init__()
        self.d_vocab_out = d_vocab

    def forward(self, tokens: t.Tensor, def_patterns: t.Tensor, induction_output: t.Tensor):
        """
        If an Induction head has copied pattern information,
        attend to all positions with the same pattern information
        and copy their tokens to the answer.

        Increase those tokens' logits by 10.
        """
        batch, seq = tokens.shape
        logits = t.zeros((batch, seq, self.d_vocab_out), device=tokens.device) # batch seq d_vocab
        
        equalities = (induction_output[..., None, :] == def_patterns[..., :, None]) # batch seq1 seq2
        print(equalities)
        equalities = t.triu(equalities, diagonal=1) # Batches. only consider positions before this one
        print(f"equalities: {equalities}")
        indices = equalities.nonzero(as_tuple=True)
        # we get (batch, seq1, seq2) where seq1 is the token we're copying from and seq2 is the token we're copying to
        tokens_to_copy = tokens[indices[0], indices[1]]
        logits[indices[0], indices[2], tokens_to_copy] += 10
        return logits
    

class Docstring_HL(HookedRootModule, HLModel):
    """
    Components:
    - Previous token head prev_def1: copy info from token B_def to ,_B
    - Previous token head prev_def2: copy info from token ,_B to C_def
    - Previous token head prev_doc: copy info from param_2 to B_doc
    - Induction head induction: copy info from B_doc to param_3
    - Arg mover head(s) arg_mover: copy answer value from C_def to param_3

    We could add multiple heads, but don't yet
    """
    def __init__(self, model, hook_name: HookName):
        super().__init__(model, hook_name)
        self.hook_pre = HookPoint()
        self.prev_def1 = PreviousHead()
        self.hook_def1 = HookPoint()
        self.prev_def2 = PreviousHead()
        self.hook_def2 = HookPoint()
        self.prev_doc = PreviousHead()
        self.hook_doc = HookPoint()
        self.induction = InductionHead()
        self.hook_induction = HookPoint()

        self.arg_mover = ArgMoverHead()
        self.hook_arg_mover = HookPoint()

    def forward(self, args, verbose=False):
        input, _label, _intermediate_data = args
        tokens = self.hook_pre(input)
        tokens = self.prev_def1(tokens)
        tokens = self.hook_def1(tokens)

        tokens = self.prev_def2(tokens)
        tokens = self.hook_def2(tokens)

        tokens = self.prev_doc(tokens)
        tokens = self.hook_doc(tokens)

        tokens = self.induction(tokens)
        tokens = self.hook_induction(tokens)

        tokens = self.arg_mover(tokens)
        tokens = self.hook_arg_mover(tokens)

        return tokens
