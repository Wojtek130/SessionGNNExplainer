#!/bin/bash

python3 tagnn_embedding_exporter.py --tagnn-input-data-filename ./ziob_yoochoose_1_64n.pcklz --tagnn-output-data-filename ./ziob_yoochoose_1_64n_emb.$1.pcklz --tagnn-model-filename ./tagnn.001.$1.pth --tagnn-model-embeddings-dimensionality $1 2>&1 | tee ./tagnn_embedding_exporter.$1.log
