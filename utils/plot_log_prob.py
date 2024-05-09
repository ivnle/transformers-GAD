import json
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def extract_prefix(jsonl_path):
    jsonl_path = jsonl_path.split('/')[-1]
    prefix = jsonl_path.split('_')[0]
    return prefix

def extract_grammar_name(jsonl_path):
    grammar_name = jsonl_path.split('/')[-2]
    return grammar_name


def plot_data(jsonl_path1, jsonl_path2, save_path=None):
    iters1, sum_log_probs1 = read_jsonl_data(jsonl_path1)
    iters2, sum_log_probs2 = read_jsonl_data(jsonl_path2)
    prefix1 = extract_prefix(jsonl_path1)
    prefix2 = extract_prefix(jsonl_path2)
    title = extract_grammar_name(jsonl_path1)

    z1 = np.polyfit(iters1, sum_log_probs1, 1)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(iters2, sum_log_probs2, 1)
    p2 = np.poly1d(z2)

    plt.figure(figsize=(10, 5))
    plt.scatter(iters1, sum_log_probs1, color='blue', label=f'{prefix1}')
    plt.scatter(iters2, sum_log_probs2, color='red', label=f'{prefix2}')
    plt.plot(iters1, p1(iters1), "b--", label=f'{prefix1} trend')
    plt.plot(iters2, p2(iters2), "r--", label=f'{prefix2} trend')
    plt.title(f'{title}')
    plt.xlabel('Iterations')
    plt.ylabel('logprob')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    plt.close()

def read_jsonl_data(jsonl_path):
    iters = []
    sum_log_probs = []
    iter_count = 0
    with open(jsonl_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f'Error reading line: {line}')
            iter_count += 1
            sum_log_prob = data.get('sum_log_prob')
            iters.append(iter_count)
            sum_log_probs.append(sum_log_prob)
    return iters, sum_log_probs

def find_jsonl_pairs(directory):
    jsonl_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory) for f in filenames if
                   f.endswith('.jsonl')]
    paired_files = {}
    for file in jsonl_files:
        base_path = os.path.dirname(file)
        file_name = os.path.basename(file)
        if base_path not in paired_files:
            paired_files[base_path] = {'gcd': None, 'gad': None, 'other': []}

        if file_name.startswith('gcd'):
            paired_files[base_path]['gcd'] = file
        elif file_name.startswith('gad'):
            paired_files[base_path]['gad'] = file
        else:
            paired_files[base_path]['other'].append(file)

    filtered_pairs = {}
    for path, files in paired_files.items():
        if files['gcd'] and files['gad']:
            filtered_pairs[path] = [files['gcd'], files['gad']]
    return filtered_pairs

def main():
    results_directory = 'results/SLIA_0506'
    plot_directory = 'plots/SLIA_0506'
    jsonl_pairs = find_jsonl_pairs(results_directory)

    for path, files in jsonl_pairs.items():
        save_path = os.path.join(plot_directory, os.path.relpath(path, results_directory), 'Mistral-7B-Instruct-v0.2_i100_cuda_sd42_float16.png')
        plot_data(files[0], files[1], save_path)

if __name__ == '__main__':
    main()
