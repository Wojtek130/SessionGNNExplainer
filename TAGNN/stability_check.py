#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import os
import json
import time
import re


# In[ ]:


import importlib
import model
import TAGNNWrapper
import SessionExplainer
import tagnn_yoochoose

importlib.reload(model)
importlib.reload(TAGNNWrapper)
importlib.reload(SessionExplainer)
importlib.reload(tagnn_yoochoose)


from utils import Data
from SessionExplainer import SessionExplainer
from TAGNNWrapper import TAGNNWrapper
from tagnn_yoochoose import tagnn
from model import trans_to_cuda
from analysis_utils import initialize_file, close_file


# In[ ]:


directory = "sessions_data/"

session_data = [ os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".pt") or filename.endswith(".pth")]
len(session_data)


# In[ ]:


vertex_ranking = torch.load('vertex_ranking.pt')


# In[ ]:


def generate_and_save_explanation(session, file_path, epochs = 2000, k = 10, modify_global_embedding = False, shortened = False):
    se = SessionExplainer(tagnn, session, epochs = epochs)
    x = trans_to_cuda(torch.ones(se.tagnn_wrapper.tagnn.n_node - 1))
    x = x.unsqueeze(1)
    targets = se.tagnn.compute_scores(se.session_data["seq_hidden"], se.session_data["mask"], x, None, None, False)
    target = torch.argmax(torch.nn.functional.softmax(targets[0], dim=0)).item()
    e = se.explain(session[-1], modify_global_embedding = modify_global_embedding)
    values, indices = torch.sort(e.node_mask.squeeze(-1), descending=True)
    best_non_target_idx, best_non_target_value = ((indices[0], values[0]) if indices[0] != target else (indices[1], values[1]))
    ranking_idx = torch.nonzero(vertex_ranking[:, 1] == best_non_target_idx, as_tuple=True)[0]
    ranking_value = vertex_ranking[ranking_idx, 0]
    data = {
        "session": session,
        "shortened": shortened,
        "modify_global_embedding": modify_global_embedding,
        "max": e.node_mask.max().item(),
        "min": e.node_mask.min().item(),
        "mean": e.node_mask.mean().item(),
        "target": target,
        "best_vertex" : {
            "id": best_non_target_idx.item(),
            "value": best_non_target_value.item(),
            "ranking_position": ranking_idx.item(),
            "ranking_value": ranking_value.item(),
            "in_session": best_non_target_idx in session,
        },
        "values": values.tolist()[:k],
        "indices": indices.tolist()[:k],
    }
    with open(file_path, 'a') as json_file:
        json.dump(data, json_file, indent=4)
        if not shortened or not modify_global_embedding:
            json_file.write(',')
        json_file.write('\n')
    torch.cuda.empty_cache()


# In[ ]:


dir = "explanation_files_long_training_rerun/"
k = 100
epochs = 12000
for i, filename in enumerate(session_data):
    sessions = []
    sessions_batch = torch.load(filename)
    dim = sessions_batch["mask"].shape[0]
    for i in range(dim):
        session_data = {key: value[i].unsqueeze(0) for key, value in sessions_batch.items()}
        session = torch.gather(session_data["items"], 1, session_data["alias_inputs"]).squeeze()[:session_data["mask"].sum()].tolist()
        if len(session) >= 5:
            sessions.append(session)
    m = re.search(r"batch_(\d+)\.pt", filename)
    batch_no = int(m.group(1)) if m else 0
    if any(f"b{str(batch_no)}" in f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))):
        print(f"Batch {batch_no} already processed. Skipping...")
        continue
    print(f"Processing batch {batch_no} with {len(sessions)} sessions; {i} / {len(session_data)}")
    for j, s in enumerate(sessions):
        file_path = os.path.join(dir, f"explanation_b{batch_no}_s{j}.json")
        initialize_file(file_path)
        for modify_global_embedding in [True, False]:
            start_time = time.time()
            generate_and_save_explanation(s, file_path, modify_global_embedding = modify_global_embedding, epochs=epochs, k=k)
            generate_and_save_explanation(s, file_path, modify_global_embedding = modify_global_embedding, epochs=epochs, k=k)
            generate_and_save_explanation(s[1:], file_path, modify_global_embedding = modify_global_embedding, shortened = True, epochs=epochs, k=k)
            end_time = time.time()
            print("Processing time:", end_time - start_time)
        close_file(file_path)
        print(f"{i}/{len(session_data)}")

    print(batch_no)

