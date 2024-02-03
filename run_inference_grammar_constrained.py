import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
import argparse
import os
import random
from inference_utils import get_file, load_model_tokenizer_hf
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import get_desired_string_dict
from get_desired_string_dict import stringsofLenk, convert_grammar
import json
import logging
from tqdm import tqdm
import time
from datetime import datetime



def parse_args():
    parser = argparse.ArgumentParser(description="Inference with grammar constraint decoding.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="pretrained model checkpoint.")
    parser.add_argument("--cache_dir", type=str, default='/nobackup2/yf/mila/GD_caches',
                        help="Where to store cache tokenizers and models.")
    parser.add_argument("--base_grammar_dir", type=str, default="/nobackup2/yf/mila/GD/examples/grammars/",
                        help="Base directory for test grammars.")
    parser.add_argument("--grammar_file", type=str, default="string_01.ebnf",
                        help="Grammar file to test.")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Number of sequences to return.")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum length of generated sequences.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--num_beams", type=int, default=3,
                        help="Number of beams for beam search.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                         help="Repetition penalty for beam search.")
    parser.add_argument("--string_length", type=int, default=3,
                        help="Length of string to generate.")
    parser.add_argument("--prompt", type=str, default=f"Generate a random binary string of length 5?",
                        help="Prompt for model inference.")
    parser.add_argument("--iter", type=int, default=5,
                        help="Number of iterations for inference.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature for sampling.")
    parser.add_argument("--do_sample", action='store_true',
                        help="Whether to sample from the model.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top p for nucleus sampling.")
    parser.add_argument("--top_k", type=int, default=500,
                        help="Top k for sampling.")

    args = parser.parse_args()
    return args


def inference_grammar_constrained(args):
    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer_hf(args)
    tokenizer.pad_token = tokenizer.eos_token
    test_file = get_file(args)

    # Load grammar
    with open(test_file, "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prompt = args.prompt
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]
    # tensor([[16968,   368,  5706,   263,  7581,  1347,   310,  3309,   472,  1556,
    #          29871, 29946, 29973]])

    if args.do_sample == False:
        output = model.generate(
            input_ids,
            do_sample=args.do_sample,
            max_length=args.max_length,
            num_beams=args.nums_beams,
            logits_processor=[grammar_processor],
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=args.num_return_sequences,
        )

    else:
        output = model.generate(
            input_ids,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=args.num_beams,
            max_new_tokens=40,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            logits_processor=[grammar_processor],
            repetition_penalty=args.repetition_penalty,
            early_stopping=True,
        )

    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    return generations


def run_inference_grammar_constrained(args):
    start_time = time.time()
    with open(get_file(args), 'r') as f:
        input_grammar = f.read()

    output = stringsofLenk(input_grammar, args.string_length)
    ideal = {key: round(args.iter / len(output.keys())) for key in output.keys()}
    faithful = output.copy()
    output['other'] = 0
    ideal['other'] = 0

    for i in tqdm(range(args.iter), desc="Running Inference"):
        result = inference_grammar_constrained(args)
        print(f"result: {result}")
        res = result[0].split("?")[1]

        if res in output:
            output[res] += 1
        else:
            output['other'] += 1

        faithful[res] = faithful.get(res, 0) + 1 # collect all the outputs instead of classifying to others
        if i % 10 == 0:
            print(f"Output: {output}")
            print(f"Faithful: {faithful}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    return output, faithful, ideal, elapsed_time


def log_results(args, output, faithful, ideal, elapsed_time, log_file=f'/nobackup2/yf/mila/GD/log.txt'):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Experiment record time: {current_time}")
    logging.info(f"Elapsed time: {elapsed_time} seconds")

    with open(log_file, 'a') as f:
        print(f"Start Logging...")
        f.write(f'Model: {args.model_id}\n')
        f.write(f'Prompt: {args.prompt}\n')
        f.write(f'Grammar: {args.grammar_file}\n')
        f.write(f'Iterations: {args.iter}\n')
        f.write(f'Do we sample when decoding? {args.do_sample}\n')
        if args.do_sample:
            f.write(f'Top_p: {args.top_p}\n')
            f.write(f'Top_k: {args.top_k}\n')
            f.write(f'Temperature: {args.temperature}\n')
        f.write('All outputs:\n')
        f.write(json.dumps(faithful))
        f.write('\nClassified outputs:\n')
        f.write(json.dumps(output))
        f.write('\nIdea distribution:\n')
        f.write(json.dumps(ideal))
        f.write('\n')
        print(f"Logging complete...")

def plot_results(args, output, ideal):
    grammar = args.grammar_file.split(".")[0]
    model = args.model_id.split("/")[1]
    # model_size = model.split("-")[2]
    fig, ax = plt.subplots()
    index = np.arange(len(ideal.keys()))
    bar_width = 0.35

    modelGen = plt.bar(index, output.values(), bar_width, color='red', label="Model generated")

    idealDist = plt.bar(index + bar_width, ideal.values(), bar_width, color='g', label="Ideal probability distribution")

    plt.xlabel("Strings in the grammar")
    plt.ylabel("Frequency")
    plt.title(f"Experiment on {model} w string length max {args.string_length}")
    plt.xticks(index + bar_width / 2, ideal.keys())
    plt.legend()
    plt.tight_layout()

    def annotate_bars(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    annotate_bars(modelGen)
    annotate_bars(idealDist)

    plt.savefig(f"/nobackup2/yf/mila/GD/plots/m-{model}_l-{args.string_length}_g-{grammar}_t-{args.temperature}_rp-{args.repetition_penalty}_tp-{args.top_p}_tk-{args.top_k}_nb-{args.num_beams}_ml-{args.max_length}.pdf")

if __name__ == "__main__":
    args = parse_args()

    output, faithful, ideal, elapsed_time = run_inference_grammar_constrained(args)
    print(f"Output: {output}")
    print(f"Faithful: {faithful}")
    print(f"Ideal: {ideal}")
    # output = {"string1": 5, "string2": 3, "string3": 4}
    # ideal = {"string1": 4, "string2": 4, "string3": 4}
    # faithful = {"string1": 5, "string2": 3, "string3": 4}
    log_results(args, output, faithful, ideal, elapsed_time)
    plot_results(args, output, ideal)
