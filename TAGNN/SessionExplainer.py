import torch
from torch_geometric.explain import Explainer, GNNExplainer
import numpy as np

from TAGNNWrapper import TAGNNWrapper
from utils import Data
from model import trans_to_cuda


class SessionExplainer:
    def __init__(self, tagnn, session, epochs=300, filename_edge_index = "edge_index.pt"):
        self.tagnn = tagnn
        self.session = session
        self.session_data = self.generate_session_data()
        tagnn_wrapper = TAGNNWrapper(tagnn, self.session_data["seq_hidden"], self.session_data["mask"], self.session_data["alias_inputs"], self.session_data["items"], self.session_data["A"])
        tagnn_wrapper = trans_to_cuda(tagnn_wrapper)
        self.tagnn_wrapper = tagnn_wrapper
        self.explainer = Explainer(
            model=self.tagnn_wrapper,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type="model",
            node_mask_type='object',
            edge_mask_type=None,
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
                ),
            )
        self.edge_index = trans_to_cuda(torch.load(filename_edge_index))
        x = torch.ones(self.tagnn.n_node - 1)
        self.x = trans_to_cuda(x.unsqueeze(1))

    def generate_session_data(self):
        d = ([self.session], [-1])
        data = Data(d, shuffle=False)
        alias_inputs, A, items, mask, targets = data.get_slice([0])
        alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
        items = trans_to_cuda(torch.Tensor(items).long())
        A = trans_to_cuda(torch.Tensor(np.array(A)).float())
        mask = trans_to_cuda(torch.Tensor(mask).long())
        hidden = self.tagnn(items, A)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        return {'alias_inputs': alias_inputs.detach(), 'A': A.detach(), 'items': items.detach(), 'mask': mask.detach(), 'seq_hidden': seq_hidden.detach()}

    def explain(self, node_id, modify_global_embedding = False, device = "cuda:0"):
        torch.cuda.empty_cache()
        e = self.explainer(self.x, self.edge_index, index = node_id, target = None, modify_global_embedding = modify_global_embedding)
        return e