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
        '-t',
        '--tagnn-input-data-type',
        dest='tagnn_input_data_type',
        type=str,
        choices=['sessions', 'sessions+pids', 'TRAIN+test+pids', 'train+TEST+pids'],
        required=False,
        default='TRAIN+test+pids'
    )
    parser.add_argument(
        '-r',
        '--tagnn-recommendations-filename',
        dest='tagnn_recommendations_filename',
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

    type_tagnn_input_data = args.tagnn_input_data_type
    print(type_tagnn_input_data)

    filename_tagnn_recommendations = args.tagnn_recommendations_filename
    print(filename_tagnn_recommendations)

    filename_tagnn_model = args.tagnn_model_filename
    print(filename_tagnn_model)

    embeddings_dimensionality = args.tagnn_model_embeddings_dimensionality
    print(embeddings_dimensionality)

    number_of_nodes = args.tagnn_model_number_of_nodes
    print(number_of_nodes)

    batch_size = args.tagnn_model_batch_size
    print(batch_size)

    print('CUDA: ', torch.cuda.is_available())



    if type_tagnn_input_data == 'TRAIN+test+pids':
        with gzip.open(filename_tagnn_input_data, 'rb') as f:
            train_data = pickle.load(f)
            test_data = pickle.load(f)
            pids = pickle.load(f)
        if number_of_nodes == -1:
            number_of_nodes = len(pids)
        sessions = train_data[0]
    elif type_tagnn_input_data == 'train+TEST+pids':
        with gzip.open(filename_tagnn_input_data, 'rb') as f:
            train_data = pickle.load(f)
            test_data = pickle.load(f)
            pids = pickle.load(f)
        if number_of_nodes == -1:
            number_of_nodes = len(pids)
        sessions = test_data[0]
    elif type_tagnn_input_data == 'sessions+pids':
        with gzip.open(filename_tagnn_input_data, 'rb') as f:
            sessions = pickle.load(f)
            pids = pickle.load(f)
        if number_of_nodes == -1:
            number_of_nodes = len(pids)
    elif type_tagnn_input_data == 'sessions':
        with gzip.open(filename_tagnn_input_data, 'rb') as f:
            sessions = pickle.load(f)
        pids = None
        # if number_of_nodes == -1:
        #     print('Unknown the number of nodes.')
        #     exit()
    # else:
    #     print('Wrong input data type.')
    #     exit()

    if pids is not None:
        id2pid = dict(zip(np.arange(len(pids)), pids))
        pid2id = {j: i for (i, j) in id2pid.items()}



    TAGNN_parameters = namedtuple('TAGNN_parameters', 'hiddenSize n_node batchSize nonhybrid step lr l2 lr_dc_step lr_dc')
    tagnn_params = TAGNN_parameters(embeddings_dimensionality, number_of_nodes, batch_size, True, 1, 0.001, 1e-5, 3, 0.1)

    model = SessionGraph(tagnn_params)
    model.load_state_dict(torch.load(filename_tagnn_model))
    model.eval()
    model = trans_to_cuda(model).float()

    chunk_size = 100000
    number_of_chunks = len(sessions) // chunk_size + (len(sessions) % chunk_size > 0)

    for chunk in range(number_of_chunks):

        t0 = time.time()

        sessions_ = sessions[chunk_size*chunk:chunk_size*(chunk+1)]
        train_data_ = Data((sessions_, np.zeros(len(sessions_))), shuffle=False)

        results = []
        scores = []

        with torch.no_grad():
            slices = train_data_.generate_batch(batch_size)
            for i in slices:
                _, scores_ = forward(model, i, train_data_)
                scores_ = scores_.cpu().detach().numpy()
                results_ = scores_.argsort(axis=1)
                results_200 = results_[:, -200:][:, ::-1]
                scores_200 = np.array([scores_[i_, results_200[i_, :]] for i_ in range(scores_.shape[0])])
                # IMPORTANT: results_ must be increased by 1, because of the TAGNN representations of product
                #            identifiers - the first product (id = 0) is considered artificial (used as an internal mark
                #            in the TAGNN code), and scores_.shape[1] = len(pids) - 1, so scores_[:, 0] corresponds to
                #            the first real product (id = 1), scores_[:, 1] to the second real product (id = 2), etc.
                results_200 = results_200 + 1
                results.append(results_200)
                scores.append(scores_200)

        results = np.vstack(results)
        scores = np.vstack(scores)

        if pids is not None:
            results = np.array([[id2pid[id_] if id_ in id2pid.keys() else -1 for id_ in session] for session in results])

        with gzip.open(filename_tagnn_recommendations[:-6] + ('.%03d' % chunk) + filename_tagnn_recommendations[-6:], 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(scores, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(time.time() - t0, results.shape, scores.shape)



    t0 = time.time()

    results = []
    scores = []

    for chunk in range(number_of_chunks):
        with gzip.open(filename_tagnn_recommendations[:-6] + ('.%03d' % chunk) + filename_tagnn_recommendations[-6:], 'rb') as f:
            results_ = pickle.load(f)
            scores_ = pickle.load(f)
        results.append(results_)
        scores.append(scores_)

    results = np.vstack(results)
    scores = np.vstack(scores)

    with gzip.open(filename_tagnn_recommendations, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(scores, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(time.time() - t0, results.shape, scores.shape)
