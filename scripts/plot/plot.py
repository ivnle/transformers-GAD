import pickle
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
from transformers_gad.oracle.oracle_trie import Trie, TrieNode
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType

SPLIT = "binary"
# SPLIT = "SLIA"
# SPLIT = "BV4"
# SPLIT = "CP"

RESULT_PATH = f"results/{SPLIT}"
PLOT_PATH = f"plots/{SPLIT}"

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def normalize(l):
    val_sum = sum(l)
    return [v / val_sum for v in l]

def KL(a, b):
    return scipy.stats.entropy(a, b)

def KL_dict(d):
    counts = [v[0] for v in d.values()]
    orig_probs = [v[1] for v in d.values()]

    counts = normalize(counts)
    orig_probs = normalize(orig_probs)

    return KL(counts, orig_probs)

def prob_explored(d):
    return sum([v[1] for v in d.values()])

def dict_sub(d1, d2):
    d = {}
    for k in d1.keys():
        if k in d2:
            v1 = d1[k]
            v2 = d2[k]
            d[k] = (v1[0] - v2[0], v1[1])
        else:
            d[k] = d1[k]
    return d

def estimate_orig_dist(gad_answers, gcd_answers):
    tokens_count, prob_explored = count_appearance(gad_answers, dict(), 0)
    gad_tokens_count = dict(tokens_count)

    total_tokens_count, prob_explored = count_appearance(gcd_answers, tokens_count, prob_explored)
    
    gcd_tokens_count = dict_sub(total_tokens_count, gad_tokens_count)
    gad_tokens_count = dict_sub(total_tokens_count, gcd_tokens_count)

    assert(sum([v[0] for v in gcd_tokens_count.values()]) == sum([v[0] for v in gad_tokens_count.values()]))

    return gad_tokens_count, gcd_tokens_count

def tokens_to_str(tokens):
    return "_".join(str(t) for t in tokens)

def count_appearance(answers, tokens_count=dict(), prob_explored=0):
    for answer in answers:
        tokens = answer['tokens']
        prob = answer['raw_likelihood']
        token_str = tokens_to_str(tokens)

        if token_str in tokens_count:
            v = tokens_count[token_str]
            tokens_count[token_str] = (v[0] + 1, prob)
        else:
            tokens_count[token_str] = (1, prob)
            prob_explored += prob

    return tokens_count, prob_explored

def count_appearance_all(answers, all_dict):
    tokens_count = {k:(0, v[1]) for (k, v) in all_dict.items()}
    tokens_history = []

    for answer in answers:
        tokens = answer['tokens']
        token_str = tokens_to_str(tokens)

        if token_str in tokens_count:
            v = tokens_count[token_str]
            tokens_count[token_str] = (v[0] + 1, v[1])
        else:
            print("Error")
            raise Exception("Unknown key")

        tokens_history.append(dict(tokens_count))

    kls = []
    interval = len(answers) // 4

    for i in range(len(answers) - interval):
        count_suffix = dict_sub(tokens_history[i + interval], tokens_history[i])
        kl = KL_dict(count_suffix)
        kls.append(kl)

    explored = prob_explored(all_dict)
    rel_probs = [answer['raw_likelihood'] / explored for answer in answers]

    return rel_probs, kls

def expectation_from_count(counts):
    probs = [v[1] for v in counts.values()]
    probs = normalize(probs)
    return sum([v * v for v in probs])

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def dist_to_line(x, y):
    p1 = np.array([0, 0])
    p2 = np.array([1, 1])
    p3 = np.array([x, y])
    return np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)

def read_answers(path):
    answers = []
    with open(path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        if len(line) > 0:
            answers.append(json.loads(line))    

    return answers

def save_fig(prob_name):
    gad_answers = read_answers(f"{RESULT_PATH}/{prob_name}/{prob_name}_gad.jsonl")
    gcd_answers = read_answers(f"{RESULT_PATH}/{prob_name}/{prob_name}_gcd.jsonl")

    # Estimate correct distribution
    gad_tokens_count, gcd_tokens_count = estimate_orig_dist(gad_answers, gcd_answers)

    # Compute KL divergence for last N samples
    gad_probs, gad_kls = count_appearance_all(gad_answers, gad_tokens_count)
    gcd_probs, gcd_kls = count_appearance_all(gcd_answers, gcd_tokens_count)

    MARKER_SIZE = 5

    # KL divergence
    xs = range(len(gad_kls))
    out_kl_path = f"{PLOT_PATH}/kl/{prob_name}.png"
    plt.cla()

    plt.plot(xs, gad_kls, '--b', label='ASAp')
    plt.plot(xs, gcd_kls, '--r', label='GCD')

    plt.legend()
    plt.savefig(out_kl_path)
    plt.close()
    
    # for x in xs:
    #     print(x, gad_kls[x], 'a')
    #     print(x, gcd_kls[x], 'b')
    
    # Probability 
    xs = range(len(gad_probs))
    out_prob_path = f"{PLOT_PATH}/prob/{prob_name}.png"
    plt.cla()

    # p1 = np.polyfit(xs, gad_probs, poly_order)
    sum_probs = 0
    gad_ys = []
    for i in xs:
        prob = gad_probs[i]
        sum_probs += prob
        gad_ys.append(sum_probs / (i + 1))

    plt.plot(xs, gad_probs, marker='o', linestyle='None', color='blue', label='ASAp', markersize=MARKER_SIZE)
    plt.plot(xs, gad_ys, '--b')

    sum_probs = 0
    gcd_ys = []
    for i in xs:
        prob = gcd_probs[i]
        sum_probs += prob
        gcd_ys.append(sum_probs / (i + 1))

    # p2 = np.polyfit(xs, gcd_probs, poly_order)
    plt.plot(xs, gcd_probs, marker='x', linestyle='None', color='red', label='GCD', markersize=MARKER_SIZE)
    plt.plot(xs, gcd_ys, '--r')

    ideal = expectation_from_count(gad_tokens_count)
    plt.plot(xs, [ideal for _ in xs] , '--k', label='LLM')

    plt.legend()
    plt.savefig(out_prob_path)
    plt.close()

    return ideal, gad_ys[-1], gcd_ys[-1]

def main():
    prob_names = [x[0] for x in os.walk(RESULT_PATH) if x[0] != RESULT_PATH]
    prob_names = [x.split("/")[-1] for x in prob_names]

    make_dir(PLOT_PATH)
    make_dir(PLOT_PATH + "/prob")
    make_dir(PLOT_PATH + "/kl")

    ideals = []
    gad_lasts = []
    gcd_lasts = []

    for prob_name in prob_names:
        ideal, gad_last, gcd_last = save_fig(prob_name)

        ideals.append(ideal)
        gad_lasts.append(gad_last)
        gcd_lasts.append(gcd_last)

    # Scatter
    plt.cla()

    plt.plot(ideals, gad_lasts, marker='o', linestyle='None', color='blue', label='ASAp')
    plt.plot(ideals, gcd_lasts, marker='x', linestyle='None', color='red', label='GCD')

    plt.axline((0, 0), slope=1)
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.legend()
    plt.savefig(PLOT_PATH + "/scatters.png")
    plt.close()

    # print("GCD: ", sum([dist_to_line(ideals[i], gcd_lasts[i]) for i in range(len(ideals))]))
    # print("GAD: ", sum([dist_to_line(ideals[i], gad_lasts[i]) for i in range(len(ideals))]))

if __name__ == "__main__":
    main()