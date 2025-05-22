import os
import json
import matplotlib.pyplot as plt
import copy
import torch
import numpy as np

def load_json_files(directory, excluded_batch_no=None):
    json_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json") and (not excluded_batch_no or f"b{excluded_batch_no}" not in filename):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    if isinstance(data, list):  
                        json_data.extend(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding {filename}: {e}")
    return json_data

def find_latest_file(filepaths):
    return max(filepaths, key=os.path.getctime)

def is_valid_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            json.load(file)
        return True
    except json.JSONDecodeError:
        return False

def fix_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    fixed_content = content.rstrip()
    fixed_content = fixed_content[::-1].replace(',', '', 1)[::-1]

    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(fixed_content)

def load_and_fix_json_files(directory, exclude_latest_file = False):
    json_data = []
    json_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
    if not json_files:
        print("No JSON files found.")
        return json_data

    if exclude_latest_file:
        latest_file = find_latest_file(json_files)
        json_files.remove(latest_file)

    for filepath in json_files:
        if not is_valid_json(filepath):
            print(f"Fixing malformed JSON: {os.path.basename(filepath)}")
            fix_json_file(filepath)

            if not is_valid_json(filepath):
                print(f"Could not fix {os.path.basename(filepath)}, skipping.")
                continue

        with open(filepath, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                if isinstance(data, list):
                    json_data.extend(data)
                else:
                    print(f"{os.path.basename(filepath)} does not contain a list, skipping.")
            except json.JSONDecodeError as e:
                print(f"Error decoding {os.path.basename(filepath)} even after fixing: {e}")

    return json_data

def shorten_list_in_json(json_obj, max_elements=10):
    new_obj = copy.deepcopy(json_obj)  
    keys = ["values", "indices"]
    for key in keys:
        if key in new_obj and isinstance(new_obj[key], list):
            new_obj[key] = new_obj[key][:max_elements]

    return new_obj

def filter_json_data(json_data, **filters):
    def matches(item):
        return all(item.get(key) == value for key, value in filters.items())
    
    return [item for item in json_data if matches(item)]

def initialize_file(file_path):
    with open(file_path, 'w') as json_file:
        json_file.write('[\n')  

def close_file(file_path):
    with open(file_path, 'a') as json_file:
        json_file.write('\n]') 

def show_histogram(data, title, xlabel, ylabel, bins=20):
    plt.hist(data.numpy(), bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def show_double_bar_chart(labels, values1, values2, y_label):
    x = np.arange(len(labels))  
    width = 0.35  
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, values1, width)
    plt.bar(x + width/2, values2, width, color='orange')
    plt.ylabel(y_label)
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def determine_significant_explanation(ex, threshold=0.05):
    fi = [(i, v) for i, v in zip(ex["indices"], ex["values"]) if i != ex["target"] and i not in ex["session"]]
    fi_values = [v for i, v in fi]
    filtered_indices = [i for i, v in fi if ((v / sum(fi_values)) > threshold)]
    return torch.Tensor(filtered_indices)

def chunk_data(data, chunk_size=3):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def filter_chunked_data(chunked_data, filter_func):
    return list(filter(filter_func, chunked_data))

def get_position(ranking, id):
    return (ranking[:, 1] == id).nonzero(as_tuple=True)[0].item()

def get_value(ranking, id):
    p = get_position(ranking, id)
    return ranking[p, 0].item()

def session_to_positions(ranking, session):
    return [get_position(ranking, id) for id in session]

def session_to_values(ranking, session):
    return [get_value(ranking, id) for id in session]