import pickle
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
from transformers_gad.oracle.oracle_trie import Trie, TrieNode
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType

MODEL = "Mistral-7B-Instruct-v0.2"
FT_MODEL = "Mistral-7B-Instruct-v0.2-gad-cp8-merged"
# MODEL = "Llama-2-7b-hf"
NUM_ITER = 2000

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_gad_answer_path(args, prob_name, fine_tune = False):
    model = FT_MODEL if fine_tune else MODEL
    return f"{args.answer_path}/{prob_name}/gad_{model}_i{NUM_ITER}_cuda_sd42_float16.jsonl"

def get_gcd_answer_path(args, prob_name, fine_tune = False):
    model = FT_MODEL if fine_tune else MODEL
    return f"{args.answer_path}/{prob_name}/gcd_{model}_i{NUM_ITER}_cuda_sd42_float16.jsonl"

def get_gad_trie_path(args, prob_name, fine_tune = False):
    model = FT_MODEL if fine_tune else MODEL
    return f"{args.trie_path}/{prob_name}/trie_{model}_i{NUM_ITER}_cuda_gad_before_sd42_float16.pkl"

def get_gcd_trie_path(args, prob_name, fine_tune = False):
    model = FT_MODEL if fine_tune else MODEL
    return f"{args.trie_path}/{prob_name}/trie_{model}_i{NUM_ITER}_cuda_sd42_float16.pkl"

def get_out_prob_path(args, prob_name, fine_tune = False):
    return f"{args.plot_path}/prob/{prob_name}-ft.png" if fine_tune else f"{args.plot_path}/prob/{prob_name}.png"

def get_out_kl_path(args, prob_name, fine_tune = False):
    return f"{args.plot_path}/kl/{prob_name}-ft.png" if fine_tune else f"{args.plot_path}/kl/{prob_name}.png"

def normalize(l):
    val_sum = sum(l)
    return [v / val_sum for v in l]

def KL(a, b):
    return scipy.stats.entropy(a, b)

def KL_dict(d):
    counts = [v[1] for v in d.values()]
    orig_probs = [v[2] for v in d.values()]

    counts = normalize(counts)
    orig_probs = normalize(orig_probs)

    return KL(counts, orig_probs)

def KL_orig_ft(orig_d, ft_d):
    ft_probs = [v[2] for v in ft_d.values()]
    orig_probs = []

    # print(ft_probs)
    # print(orig_d, ft_d)

    for v1 in ft_d.values():
        duplicate = False
        for v2 in orig_d.values():
            if is_same_tokens(v1[0], v2[0]):
                orig_probs.append(v2[2])
                duplicate = True
                break
        
        if not duplicate:
            orig_probs.append(0)

    # print(orig_probs)

    ft_probs = normalize(ft_probs)
    orig_probs = normalize(orig_probs)

    return KL(ft_probs, orig_probs)

def prob_explored(d):
    return sum([v[2] for v in d.values()])

def dict_sub(d1, d2):
    d = {}
    for k in d1.keys():
        if k in d2:
            v1 = d1[k]
            v2 = d2[k]
            d[k] = (v1[0], v1[1] - v2[1], v1[2])
        else:
            d[k] = d1[k]
    return d

def is_same_tokens(tokens1, tokens2):
    if len(tokens1) != len(tokens2):
        return False
    
    for i in range(len(tokens1)):
        if tokens1[i]['token_id'] != tokens2[i]['token_id']:
            return False
        
    return True

def load_oracle_trie(trie_path):
    with open(trie_path, "rb") as f:
        trie = pickle.load(f)
    return trie

def restore_orig_prob(tokens, trie):
    sum_of_log = 0

    parent = trie.root
    for token in tokens:
        node = trie.search_token_from_parent(parent, token['token_id'])
        sum_of_log += math.log(node.raw_logit)
        parent = node

    return math.exp(sum_of_log)

def estimate_orig_dist(args, prob_name, fine_tune = False):
    gad_answer_path = get_gad_answer_path(args, prob_name, fine_tune)
    gad_trie_path = get_gad_trie_path(args, prob_name, fine_tune)
    _, tokens_count, prob_explored = count_appearance(gad_answer_path, gad_trie_path, dict(), 0)
    gad_tokens_count = dict(tokens_count)

    gcd_answer_path = get_gcd_answer_path(args, prob_name, fine_tune)
    gcd_trie_path = get_gcd_trie_path(args, prob_name, fine_tune)
    _, total_tokens_count, prob_explored = count_appearance(gcd_answer_path, gcd_trie_path, tokens_count, prob_explored)
    
    gcd_tokens_count = dict_sub(total_tokens_count, gad_tokens_count)
    gad_tokens_count = dict_sub(total_tokens_count, gcd_tokens_count)

    assert(sum([v[1] for v in gcd_tokens_count.values()]) == sum([v[1] for v in gad_tokens_count.values()]))

    return total_tokens_count, gad_tokens_count, gcd_tokens_count

def count_appearance(answer_path, trie_path, tokens_count=dict(), prob_explored=0):
    with open(answer_path, "r") as f:
        lines = f.readlines()
    answers = [json.loads(line) for line in lines]

    trie = load_oracle_trie(trie_path)
    probs = [restore_orig_prob(answer['metas'], trie) for answer in answers]
    # probs = [restore_orig_prob(answer['metas'], trie)[1] for answer in answers]

    for i in range(len(answers)):
        answer = answers[i]
        metas = answer['metas']
        prob = probs[i]
        metas_str = str(metas)

        duplicate = False

        for k, v in tokens_count.items():
            if is_same_tokens(v[0], metas):
                duplicate = True
                tokens_count[k] = (v[0], v[1] + 1, prob)
                break
        
        if not duplicate:
            tokens_count[metas_str] = (metas, 1, prob)
            prob_explored += prob

    return probs, tokens_count, prob_explored

def count_appearance_all(answer_path, trie_path, all_dict):
    with open(answer_path, "r") as f:
        lines = f.readlines()
    answers = [json.loads(line) for line in lines]

    trie = load_oracle_trie(trie_path)
    probs = [restore_orig_prob(answer['metas'], trie) for answer in answers]
    
    tokens_count = {k:(v[0], 0, v[2]) for (k, v) in all_dict.items()}
    tokens_history = []

    for i in range(len(answers)):
        # print(sum([v[1] for v in tokens_count.values()]), sum([v[1] for v in count_suffix.values()]))

        answer = answers[i]
        metas = answer['metas']

        duplicate = False

        for k, v in tokens_count.items():
            if is_same_tokens(v[0], metas):
                duplicate = True
                tokens_count[k] = (v[0], v[1] + 1, v[2])
                break
        
        if not duplicate:
            print("Error")
            raise Exception("Unknown key")

        tokens_history.append(dict(tokens_count))

    kls = []
    interval = len(answers) // 4
    for i in range(len(answers) - interval):
        count_suffix = dict_sub(tokens_history[i + interval], tokens_history[i])
        kl = KL_dict(count_suffix)
        kls.append(kl)

    counts = []
    for answer in answers:
        metas = answer['metas']

        for k, v in tokens_count.items():
            if is_same_tokens(v[0], metas):
                counts.append(v[1])
                break

    explored = prob_explored(all_dict)
    rel_probs = [prob / explored for prob in probs]

    # m, b = np.polyfit(xs, rel_probs, 1)

    return rel_probs, kls

def expectation_from_count(counts):
    probs = [v[2] for v in counts.values()]
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

def save_fig(args, prob_name):
    # Estimate correct distribution
    total_tokens_count, gad_tokens_count, gcd_tokens_count = estimate_orig_dist(args, prob_name)
    ft_total_tokens_count, ft_gad_tokens_count, ft_gcd_tokens_count = estimate_orig_dist(args, prob_name, True)

    # if "lastname" in prob_name:
    #     l = [(v[0], v[2]) for k, v in gad_tokens_count.items()]
    #     l.sort(key=lambda x: x[1], reverse=True)
    #     for p in l[:5]:
    #         metas = p[0]
    #         s = ""
    #         for token in metas:
    #             s += token["token_str"]
    #         print(s, p[1])

    # Compute KL divergence for last N samples
    gad_answer_path = get_gad_answer_path(args, prob_name)
    gad_trie_path = get_gad_trie_path(args, prob_name)
    gad_probs, gad_kls = count_appearance_all(gad_answer_path, gad_trie_path, gad_tokens_count)

    gcd_answer_path = get_gcd_answer_path(args, prob_name)
    gcd_trie_path = get_gcd_trie_path(args, prob_name)
    gcd_probs, gcd_kls = count_appearance_all(gcd_answer_path, gcd_trie_path, gcd_tokens_count)

    ft_gad_answer_path = get_gad_answer_path(args, prob_name, True)
    ft_gad_trie_path = get_gad_trie_path(args, prob_name, True)
    ft_gad_probs, ft_gad_kls = count_appearance_all(gad_answer_path, gad_trie_path, gad_tokens_count)

    ft_gcd_answer_path = get_gcd_answer_path(args, prob_name, True)
    ft_gcd_trie_path = get_gcd_trie_path(args, prob_name, True)
    ft_gcd_probs, ft_gcd_kls = count_appearance_all(gcd_answer_path, gcd_trie_path, gcd_tokens_count)

    MARKER_SIZE = 5

    # KL divergence
    xs = range(len(gad_kls))
    out_kl_path = get_out_kl_path(args, prob_name)
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
    out_prob_path = get_out_prob_path(args, prob_name)
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

    ft_ideal = expectation_from_count(ft_gad_tokens_count)
    plt.plot(xs, [ft_ideal for _ in xs] , '--m', label='LLM-fine-tuned')

    plt.legend()
    plt.savefig(out_prob_path)
    plt.close()

    kl_ft_orig = KL_orig_ft(total_tokens_count, ft_total_tokens_count)
    print(prob_name, kl_ft_orig)

    return ideal, gad_ys[-1], gcd_ys[-1]

def main(args):
    prob_names = [x[0] for x in os.walk(args.answer_path) if x[0] != args.answer_path]
    prob_names = [x.split("/")[-1] for x in prob_names]

    # print(prob_names)

    make_dir(args.plot_path)
    make_dir(args.plot_path + "/prob")
    make_dir(args.plot_path + "/kl")

    ideals = []
    gad_lasts = []
    gcd_lasts = []

    for prob_name in prob_names:
        # try:
        ideal, gad_last, gcd_last = save_fig(args, prob_name)

        ideals.append(ideal)
        gad_lasts.append(gad_last)
        gcd_lasts.append(gcd_last)
        # except:
        #     pass

    # GCD Scatter
    plt.cla()

    plt.plot(ideals, gad_lasts, marker='o', linestyle='None', color='blue', label='ASAp')
    plt.plot(ideals, gcd_lasts, marker='x', linestyle='None', color='red', label='GCD')

    plt.axline((0, 0), slope=1)
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.legend()
    plt.savefig(args.plot_path + "/scatters.png")
    plt.close()

    print("GCD: ", sum([dist_to_line(ideals[i], gcd_lasts[i]) for i in range(len(ideals))]))

    # GAD Scatter
    # plt.cla()

    # plt.plot(ideals, gad_lasts, marker='o', linestyle='None')

    # plt.axline((0, 0), slope=1)
    # plt.xlim((0, 1))
    # plt.ylim((0, 1))

    # # plt.legend()
    # plt.savefig(args.plot_path + "/gad_scatter.png")
    # plt.close()

    print("GAD: ", sum([dist_to_line(ideals[i], gad_lasts[i]) for i in range(len(ideals))]))

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--answer-path',
        help='Directory that contains JSONL files with candidate solutions')

    parser.add_argument(
        '--trie-path',
        help='Directory that contains trie files')

    parser.add_argument(
        '--plot-path',
        help='Directory to save plot files'
    )

    main(parser.parse_args())