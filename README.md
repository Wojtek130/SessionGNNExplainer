# Session GNNExplainer
wroXAI part for explaining Recommender Systems based on Graph Neural Networks

## Overview
Implementation of the GNNExplainer algorithm approach for session-based data

## Important files and directories
The most important files and directories are located in the `TAGNN` directory (other files in this location come from the original `TAGNN` repository):

- `explanation_files_long_training_rerun/` - contains `json` files with explanations data for 1000 processed sessions for the Model Modifying Only the Target Embedding
- `explanation_files_long_training_rerun_modify_session_embeddings/` - contains `json` files with explanations data for 1000 processed sessions for the Model Modifying the Target and Session Embedding
- `explanations_values_plots` - contains files with plots depicting values distribution in explanations (in subdirectories there are plots for 10 and 100 best vertices values as well as plots for all vertices)
- `analysis_utils.py` - utility functions, mainly for statistics purposes
- `edge_index.pt` - edge index tensor
- `edge_index_preparation.ipynb` - creates edge index tensor
- `environment.yaml` - list of required `Conda` packages
- `mock.py` - demonstration how to use the `GNNExplainer` for session-based data
- `model.py` - orignal `model.py` file from `TAGNN` repository but with modified embeddings calculations
- `prepare_session_data.ipynb` - creates files containing preprocessed information for sessions from all batches
- `run_statistics_long_training.ipynb` - contains metrics implementation and explanation statistics
- `SessionExplainer.py` - core module delivering explanations of session-based data
- `stability_check_modify_session_embeddings.ipynb` - creates explanations for the Model Modifying the Target and Session Embedding and saves them to file
- `stability_check_modify_session_embeddings.py` - `stability_check_modify_session_embeddings.ipynb` file converted to a `Python` script 
- `stability_check.ipynb` - creates explanations for the Model Modifying Only the Target Embedding and saves them to file
- `stability_check.py` - `stability_check.ipynb` - file converted to a `Python` script 
- `t-sne_visualizations.ipynb` - `t-SNE` plots of explanations
- `tagnn_yoochoose.py` - trained `TAGNN` model
- `TAGNNWrapper.py` - `TAGNN` wrapper for `GNNExplainer`
- `vertex_ranking.ipynb` - creates vertex ranking
- `vertex_ranking.pt` - vertex ranking tensor

## Environment set up
```bash!
cd TAGNN/
conda env create -f environment.yaml
conda activate session-explainer
```



