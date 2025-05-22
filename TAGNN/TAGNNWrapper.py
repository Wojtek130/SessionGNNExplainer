import torch
from torch.nn import Module

from model import trans_to_cuda

class TAGNNWrapper(Module):
    def __init__(self, tagnn, seq_hidden, mask, alias_inputs, items, A):
        super(TAGNNWrapper, self).__init__()
        self.tagnn = tagnn
        self.tagnn = trans_to_cuda(self.tagnn)
        self.seq_hidden = seq_hidden
        self.mask = mask
        self.alias_inputs = alias_inputs
        self.items = items
        self.A = A

    def get_session(self):
        return torch.gather(self.items, 1, self.alias_inputs).squeeze(0)[:self.mask.sum()]

    def forward(self, x, edge_index = None, modify_global_embedding = False):
        scores = self.tagnn.compute_scores(self.seq_hidden, self.mask, x, self.alias_inputs, self.items, modify_global_embedding)
        return scores