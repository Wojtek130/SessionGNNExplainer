#!/bin/bash

python3 tagnn_model_evaluator.py --tagnn-input-data-filename ./ziob_yoochoose_1_64n.pcklz --tagnn-input-data-type train+TEST+pids --tagnn-recommendations-filename ./tagnn.001.$1.recommendations.test.pcklz --tagnn-model-filename ./tagnn.001.$1.pth --tagnn-model-embeddings-dimensionality $1 2>&1 | tee ./tagnn_model_evaluator.001.$1.test.log
