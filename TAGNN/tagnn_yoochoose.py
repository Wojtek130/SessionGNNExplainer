import torch
import gzip
import pickle
from collections import namedtuple

import model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

filename_tagnn_input_data = '../YooChoose 1_64 Dataset Preparation/ziob_yoochoose_1_64n.pcklz'
# filename_tagnn_input_data = 'ziob_yoochoose_1_64n.pcklz'

filename_tagnn_model = 'tagnn_model.pth'
# filename_tagnn_recommendations = 'recommendations.pcklz'
embeddings_dimensionality = 100
number_of_nodes = -1
batch_size = 32
batch_size = 1
training_epochs = 50

with gzip.open(filename_tagnn_input_data, 'rb') as f:
    train_data = pickle.load(f)
    test_data = pickle.load(f)
    pids = pickle.load(f)

if number_of_nodes == -1:
    number_of_nodes = len(pids)
# print(number_of_nodes)

TAGNN_parameters = namedtuple('TAGNN_parameters', 'hiddenSize n_node batchSize nonhybrid step lr l2 lr_dc_step lr_dc')
tagnn_params = TAGNN_parameters(embeddings_dimensionality, number_of_nodes, batch_size, True, 1, 0.001, 1e-5, 3, 0.1)

tagnn = model.SessionGraph(tagnn_params)
tagnn.load_state_dict(torch.load(filename_tagnn_model))
tagnn.eval()
tagnn = model.trans_to_cuda(tagnn).float()