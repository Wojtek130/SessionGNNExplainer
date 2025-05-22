import numpy as np

import argparse

from collections import namedtuple

import gzip
import pickle
import time

from pathlib import Path

import torch

from model import SessionGraph, forward, trans_to_cuda, trans_to_cpu
from utils import Data



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--tagnn-input-data-filename',
        dest='tagnn_input_data_filename',
        type=str,
        required=True
    )
    parser.add_argument(
        '-o',
        '--tagnn-output-data-filename',
        dest='tagnn_output_data_filename',
        type=str,
        required=True
    )
    parser.add_argument(
        '-m',
        '--tagnn-model-filename',
        dest='tagnn_model_filename',
        type=str,
        required=True
    )
    parser.add_argument(
        '-d',
        '--tagnn-model-embeddings-dimensionality',
        dest='tagnn_model_embeddings_dimensionality',
        type=int,
        required=True
    )
    parser.add_argument(
        '-n',
        '--tagnn-model-number-of-nodes',
        dest='tagnn_model_number_of_nodes',
        type=int,
        required=False,
        default=-1,
        help='if -1, it will be set to the number of product identifiers (pids) from the input data'
    )
    parser.add_argument(
        '-b',
        '--tagnn-model-batch-size',
        dest='tagnn_model_batch_size',
        type=int,
        required=False,
        default=32
    )
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    filename_tagnn_input_data = args.tagnn_input_data_filename
    print(filename_tagnn_input_data)

    filename_tagnn_output_data = args.tagnn_output_data_filename
    print(filename_tagnn_output_data)

    filename_tagnn_model = args.tagnn_model_filename
    print(filename_tagnn_model)

    embeddings_dimensionality = args.tagnn_model_embeddings_dimensionality
    print(embeddings_dimensionality)

    number_of_nodes = args.tagnn_model_number_of_nodes
    print(number_of_nodes)

    batch_size = args.tagnn_model_batch_size
    print(batch_size)

    print('CUDA: ', torch.cuda.is_available())



    with gzip.open(filename_tagnn_input_data, 'rb') as f:
        train_data = pickle.load(f)
        test_data = pickle.load(f)
        pids = pickle.load(f)

    if number_of_nodes == -1:
        number_of_nodes = len(pids)



    TAGNN_parameters = namedtuple('TAGNN_parameters', 'hiddenSize n_node batchSize nonhybrid step lr l2 lr_dc_step lr_dc')
    tagnn_params = TAGNN_parameters(embeddings_dimensionality, number_of_nodes, batch_size, True, 1, 0.001, 1e-5, 3, 0.1)

    model = SessionGraph(tagnn_params)
    model.load_state_dict(torch.load(filename_tagnn_model))
    model.eval()
    model = trans_to_cuda(model).float()

    embeddings = model.embedding.cpu().weight.to(float).detach().numpy()

    with gzip.open(filename_tagnn_output_data, 'wb') as f:
        pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(pids, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
