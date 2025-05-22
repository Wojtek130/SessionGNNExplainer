#!/bin/bash

python3 tagnn_model_trainer.py --tagnn-input-data-filename ./ziob_yoochoose_1_64n.pcklz --tagnn-model-filename ./tagnn.001.$1.pth --tagnn-model-embeddings-dimensionality $1 --tagnn-model-training-epochs $2 2>&1 | tee ./tagnn_model_trainer.001.$1.$2.log
