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
    def __init__(self, d_vocab:int=40, logit_increase=50 ):
        super().__init__()
        self.d_vocab_out = d_vocab
        self.logit_increase = logit_increase

    def forward(self, tokens: t.Tensor, def_patterns: t.Tensor, induction_output: t.Tensor):
        """
        If an Induction head has copied pattern information,
        attend to all positions with the same pattern information
        and copy their tokens to the answer.

        Increase those tokens' logits by 10.

        A better implementation would suppress the logit from the induction signal's source,
        so that we don't copy the token 2 after that. Right now we just remove the last
        appearance, but to do this for real we need position embeds.
        """
        batch, seq = tokens.shape
        logits = t.zeros((batch, seq, self.d_vocab_out), device=tokens.device) # batch seq d_vocab
        
        equalities = (induction_output[..., None, :] == def_patterns[..., :, None]) # batch seq1 seq2
        equalities = t.triu(equalities, diagonal=1) # Batches. only consider positions before this one
        indices = equalities.nonzero(as_tuple=True)
        # we get (batch, seq1, seq2) where seq1 is the token we're copying from and seq2 is the token we're copying to
        # now only keep (b, s1, s2) when there is a (b, s1', s2) with s1' > s1, to remove last appearance
        # Below was written by Claude, I don't understand it

        # Combine the columns into a single array
        arr = np.column_stack(indices)

        # Sort the array by b, s1 (ascending), and s2
        sorted_arr = arr[np.lexsort((arr[:, 1], arr[:, 2], arr[:, 0]))]
        print(sorted_arr)

        # Create a boolean mask to identify the last appearance of each s1 in each batch, s2 pair
        mask = np.empty(len(sorted_arr), dtype=bool)
        mask[:-1] = (sorted_arr[:-1, 0] != sorted_arr[1:, 0]) | (sorted_arr[:-1, 2] != sorted_arr[1:, 2])
        mask[-1] = True
        print(mask)

        # Apply the boolean mask to the sorted array to remove the last appearance of each s1 in each batch, s2 pair
        filtered_arr = sorted_arr[~mask]

        # Extract the filtered columns back into separate entries of a tuple
        indices = filtered_arr[:, 0], filtered_arr[:, 1], filtered_arr[:, 2]

        tokens_to_copy = tokens[indices[0], indices[1]]

        logits[indices[0], indices[2], tokens_to_copy] += self.logit_increase
        return logits
    

class Docstring_HL(HookedRootModule, HLModel):
    """
    Components:
    - Previous token head prev_def1: copy info from token B_def to ,_B
    - Previous token head prev_def2: copy info from token ,_B to C_def
    - Previous token head prev_doc: copy info from param_2 to B_doc
    - Induction head induction: copy info from B_doc to param_3
    - Arg mover head(s) arg_mover: copy answer value from C_def to param_3

    Implementation ideas
    - Add multiple heads
    - Different previous token heads in different contexts
    """
    def __init__(self):
        super().__init__()
        self.hook_pre = HookPoint()
        self.prev_def1 = PreviousHead()
        self.hook_prev1 = HookPoint()
        self.prev_def2 = PreviousHead()
        self.hook_prev2 = HookPoint()
        self.prev_doc = PreviousHead()
        self.hook_prev_doc = HookPoint()
        self.induction = InductionHead()
        self.hook_induction = HookPoint()

        self.arg_mover = ArgMoverHead()
        self.hook_arg_mover = HookPoint()

    def is_categorical(self):
        return True

    def forward(self, args, verbose=False):
        input, _label, _intermediate_data = args
        assert len(input.shape) == 2, f"Expected input to be batch seq, got {input.shape}"
        tokens = self.hook_pre(input)
        prev1 = self.prev_def1(tokens)
        prev1 = self.hook_prev1(prev1)

        prev2 = self.prev_def2(prev1)
        prev2 = self.hook_prev2(prev2)

        prev_doc = self.prev_doc(tokens)
        prev_doc = self.hook_prev_doc(prev_doc)
        print(f"{tokens = }")
        print(f"{prev_doc = }")
        print(f"{prev2 = }")

        induction_out = self.induction(tokens, prev_doc)
        induction_out = self.hook_induction(induction_out)
        # carries B_doc at param_3
        print(f"{induction_out = }")

        logits = self.arg_mover(tokens, prev2, induction_out)
        logits = self.hook_arg_mover(logits)

        return logits
