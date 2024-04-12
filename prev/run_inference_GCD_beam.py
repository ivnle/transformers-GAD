import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarConstrainedLogitsProcessor
import argparse
import os
import random
from inference_utils import get_file, load_model_tokenizer_hf
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from GD.prev import get_desired_string_dict
from GD.prev.get_desired_string_dict import stringsofLenk_max, stringsofLenk, convert_grammar
import json
import logging
from tqdm import tqdm
import time
from datetime import datetime
from check_is_valid_string import is_valid_string_start_w_1_all_0, is_valid_string_0, is_valid_string_1, is_valid_string_01
from vllm import LLM, SamplingParams


#models=("meta-llama/Llama-2-7b-hf"
#"meta-llama/Llama-2-13b-hf"
#"meta-llama/Llama-2-70b-hf"
#"mistralai/Mixtral-8x7B-Instruct-v0.1")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with grammar constraint decoding.")
    parser.add_argument("--model_id", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        help="pretrained model checkpoint.")
    parser.add_argument("--cache_dir", type=str, default='/nobackup2/yf/mila/GD_caches',
                        help="Where to store cache tokenizers and models.")
    parser.add_argument("--base_grammar_dir", type=str, default="/nobackup2/yf/mila/GD/examples/grammars/",
                        help="Base directory for test grammars.")
    parser.add_argument("--grammar_file", type=str, default="string_01.ebnf",
                        help="Grammar file to test.")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Number of sequences to return.")
    # parser.add_argument("--max_length", type=int, default=50,
    #                     help="Maximum length of generated sequences when do not sample.")
    # parser.add_argument("--seed", type=int, default=42,
    #                     help="Random seed for reproducibility.")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="Number of beams for beam search.")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                         help="Repetition penalty for greedy decoding.")
    parser.add_argument("--string_length", type=int, default=5,
                        help="Length of string to generate.")
    parser.add_argument("--prompt", type=str, default=f"Generate a random binary string of length 5?",
                        help="Prompt for model inference.")
    parser.add_argument("--iter", type=int, default=500,
                        help="Number of iterations for inference.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling.")
    parser.add_argument("--do_sample", action='store_true',
                        help="Whether to sample from the model.")
    # parser.add_argument("--top_p", type=float, default=0.9,
    #                     help="Top p for nucleus sampling.")
    # parser.add_argument("--top_k", type=int, default=500,
    #                     help="Top k for sampling.")
    parser.add_argument("--log_file", type=str, default='/nobackup2/yf/mila/GD/log/test_log.txt',
                        help="Where to store log file.")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="Maximum number of new tokens to generate.")

    args = parser.parse_args()
    return args


def inference_grammar_constrained(args, model, tokenizer):
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
        # argmax decoding
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
        # try only beam search here
        output = model.generate(
            input_ids,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            # top_p=args.top_p,
            # top_k=args.top_k,
            temperature=args.temperature,
            logits_processor=[grammar_processor],
            repetition_penalty=args.repetition_penalty,
            # early_stopping=True,
            num_return_sequences=args.num_return_sequences
        )

    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(f"grammar constrained generations: {generations}")
    return generations

def inference_beam(args, model, tokenizer):
    test_file = get_file(args)

    # Generate
    prompt = args.prompt
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]

    # try only beam search here
    output = model.generate(
        input_ids,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        # top_p=args.top_p,
        # top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        # early_stopping=True,
        num_return_sequences=args.num_return_sequences
    )

    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(f"grammar constrained generations: {generations}")
    return generations

def inference_greedy(args, model, tokenizer):
    prompt = args.prompt
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]

    output = model.generate(
        input_ids,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        # early_stopping=True,
        num_return_sequences=args.num_return_sequences
    )
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    # generations = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(f"greedy generations: {generations}")
    return generations

def inference_greedy_vllm(args):
    model = LLM(model=args.model_id, download_dir=args.cache_dir)

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop=["---", "###"],
        # early_stopping=True # set to True when use beam search
    )

    output = model.generate(args.prompt, sampling_params)
    print(output)
    # for output in outputs:
    #     print(f"vllm generations: \n prompt: {output.prompt}; generated_text: {output.outputs[0].text}")
    return output


def run_inference_grammar_constrained(args):
    model, tokenizer = load_model_tokenizer_hf(args)
    tokenizer.pad_token = tokenizer.eos_token
    logging.basicConfig(filename=args.log_file, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    start_time = time.time()
    with open(get_file(args), 'r') as f:
        input_grammar = f.read()
    logging.info(f"input_grammar: {input_grammar}")
    # output = stringsofLenk_max(input_grammar, args.string_length)
    output = stringsofLenk(input_grammar, args.string_length)
    ideal = {key: round(args.iter / len(output.keys())) for key in output.keys()}
    faithful = output.copy()
    output['other'] = 0
    ideal['other'] = 0

    for i in tqdm(range(args.iter), desc="Running Inference"):
        result = inference_grammar_constrained(args, model, tokenizer)
        logging.info(f"result: {result}")
        res = result[0].split(".")[2]
        # print(f"res: {res}")
        if res in output:
            output[res] += 1
        else:
            output['other'] += 1

        faithful[res] = faithful.get(res, 0) + 1 # collect all the outputs instead of classifying to others
        if i % 10 == 0:
            logging.info(f"Iteration: {i+1}")
            logging.info(f"Output: {output}")
            logging.info(f"Faithful: {faithful}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time} seconds")
    logging.info(f"model_id: {args.model_id}")
    logging.info(f"repetition_penalty: {args.repetition_penalty}")
    # print(f"num_beams: {args.num_beams}")
    logging.info(f"temperature: {args.temperature}")
    logging.info(f"top_p: {args.top_p}")
    logging.info(f"max_new_tokens: {args.max_new_tokens}")
    logging.info(f"output: {output}")
    logging.info(f"faithful: {faithful}")
    logging.info(f"ideal: {ideal}")
    return output, faithful, ideal, elapsed_time

def run_inference_greedy(args):
    model, tokenizer = load_model_tokenizer_hf(args)
    tokenizer.pad_token = tokenizer.eos_token
    logging.basicConfig(filename=args.log_file, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    start_time = time.time()
    generations = []
    for i in range(args.iter):
        # print(f"iteration: {i}")
        generation = inference_greedy(args, model, tokenizer)
        generations.append(generation)
        logging.info(f"greedy generations: {generation}")
        print(f"greedy generations: {generation}")
    # generations = inference_greedy_vllm(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time} seconds")
    logging.info(f"Model: {args.model_id}")
    logging.info(f'Prompt: {args.prompt}')
    logging.info(f'Grammar: {args.grammar_file}')
    logging.info(f'Iterations: {args.iter}')
    logging.info(f'Do we sample when decoding? {args.do_sample}')
    logging.info(f'Repetition penalty: {args.repetition_penalty}')
    logging.info(f'max_new_tokens: {args.max_new_tokens}')
    if args.do_sample:
        logging.info(f'Top_p: {args.top_p}')
        # f.write(f'Top_k: {args.top_k}\n')
        logging.info(f'Temperature: {args.temperature}')
    print(f"end logging...")
    return generations

def run_inference_beam(args):
    model, tokenizer = load_model_tokenizer_hf(args)
    tokenizer.pad_token = tokenizer.eos_token

    generations = []
    for i in range(args.iter):
        generation = inference_beam(args, model, tokenizer)
        generations.append(generation)
        logging.info(f"beam generations: {generation}")
        print(f"beam generations: {generation}")
    return generations


def log_results(args, output, faithful, ideal, elapsed_time):
    logging.basicConfig(filename=args.log_file, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime('%y-%m-%d-%H:%M:%S')
    logging.info(f"Elapsed time: {elapsed_time} seconds")

    with open(args.log_file, 'a') as f:
        print(f"Start Logging...")
        f.write(f'Model: {args.model_id}\n')
        f.write(f'Prompt: {args.prompt}\n')
        f.write(f'Grammar: {args.grammar_file}\n')
        f.write(f'Iterations: {args.iter}\n')
        f.write(f'Do we sample when decoding? {args.do_sample}\n')
        # f.write(f'Number of beams: {args.num_beams}\n')
        f.write(f'Repetition penalty: {args.repetition_penalty}\n')
        f.write(f'max_new_tokens: {args.max_new_tokens}\n')
        if args.do_sample:
            # f.write(f'Top_p: {args.top_p}\n')
            # f.write(f'Top_k: {args.top_k}\n')
            f.write(f'Number of beams: {args.num_beams}\n')
            f.write(f'Temperature: {args.temperature}\n')
        f.write('All outputs:\n')
        f.write(json.dumps(faithful))
        f.write('\nClassified outputs:\n')
        f.write(json.dumps(output))
        f.write('\nIdea distribution:\n')
        f.write(json.dumps(ideal))
        f.write('\n')
        print(f"Logging complete...")
    return datetime_string

def plot_results(args, output, ideal, datetime_string):
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
    plt.title(f"Experiment on {model} w string length {args.string_length}")
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

    plt.savefig(f"/nobackup2/yf/mila/GD/plots/t-{datetime_string}_m-{model}_l-{args.string_length}_g-{grammar}_t-{args.temperature}_rp-{args.repetition_penalty}_tp-{args.top_p}_mnt-{args.max_new_tokens}_iter-{args.iter}.pdf")

if __name__ == "__main__":
    args = parse_args()

    print(f"model_id: {args.model_id}")
    print(f"repetition_penalty: {args.repetition_penalty}")
    print(f"grammar_file: {args.grammar_file}")
    # print(f"num_beams: {args.num_beams}")
    print(f"temperature: {args.temperature}")
    print(f"num_beam: {args.num_beams}")
    # print(f"top_p: {args.top_p}")
    print(f"max_new_tokens: {args.max_new_tokens}")

    output, faithful, ideal, elapsed_time = run_inference_grammar_constrained(args)


    # print(f"Output: {output}")
    # print(f"Faithful: {faithful}")
    # print(f"Ideal: {ideal}")
    # datetime_string = log_results(args, output, faithful, ideal, elapsed_time)
    # plot_results(args, output, ideal, datetime_string)

    # model, tokenizer = load_model_tokenizer_hf(args)
    # result = inference_beam(args, model, tokenizer)


    # generation = inference_greedy_vllm(args)
    # generation = inference_greedy(args)
    # generation = inference_grammar_constrained(args)
    # generations = run_inference_greedy(args)


