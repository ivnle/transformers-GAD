import torch
import json
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import random
from inference_utils import get_file, load_model_tokenizer_hf
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import json
import logging
from tqdm import tqdm
import time
from datetime import datetime


#models=("meta-llama/Llama-2-7b-hf"
#"meta-llama/Llama-2-13b-hf"
#"meta-llama/Llama-2-70b-hf"
#"mistralai/Mixtral-8x7B-Instruct-v0.1"
# "mistralai/Mistral-7B-Instruct-v0.1")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with grammar constraint decoding.")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.1",
                        help="pretrained model checkpoint.")
    parser.add_argument("--cache_dir", type=str, default='/nobackup2/yf/mila/GD_caches/',
                        help="Where to store cache tokenizers and models.")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Number of sequences to return.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                         help="Repetition penalty for greedy decoding.")
    parser.add_argument("--iter", type=int, default=1,
                        help="Number of iterations for inference.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top p for nucleus sampling.")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top k for sampling.")
    parser.add_argument("--log_file", type=str, default='/nobackup2/yf/mila/GD/log_GAD/track_scores_prob2.log',
                        help="Where to store log file.")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--sygus_prompt_file", type=str, default="/nobackup2/yf/mila/GD/prompts/pre_prompt.jsonl",
                        help="File path to prompts for sygus task.")
    parser.add_argument("--prompt_type", type=str, choices=["bare", "completion"], default="bare",
                        help="Prompt type for sygus task.")
    parser.add_argument("--output_folder", type=str, default="/nobackup2/yf/mila/GD/results/",
                        help="Output folder to store results.")

    args = parser.parse_args()
    return args

def get_sygus_prompt(filename, prompt_type):
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)

            if data.get('prompt_type') == prompt_type:
                return data['prompt']

        raise ValueError(f"Prompt type {prompt_type} not found in file {filename}")

def inference_bare(args, model, tokenizer, prompt):
    tokenizer.pad_token = tokenizer.eos_token
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
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=args.num_return_sequences,
        return_dict_in_generate=True,
    )

    input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    generated_tokens = output.sequences[:, input_length:]
    generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return generations

def run_inference_bare(args,output_file_path):
    model, tokenizer = load_model_tokenizer_hf(args)
    prompt = get_sygus_prompt(args.sygus_prompt_file, args.prompt_type)
    start_time = time.time()
    with open(output_file_path, 'a', encoding='utf-8') as outfile:
        for i in tqdm(range(args.iter), desc="Running Inference"):
            generations = inference_bare(args, model, tokenizer, prompt)
            result = {"answer": generations, "prompt": prompt, "prompt_type": args.prompt_type, "grammar": "PRE_100_10.sl"}
            print(f"result: {result}")

            json_record = json.dumps(result)
            outfile.write(json_record + '\n')
            outfile.flush()
            os.fsync(outfile.fileno())


    end_time = time.time()
    print(f"Results saved to {output_file_path}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

def construct_output_file_path(args):
    model_name = args.model_id.split("/")[-1]
    output_file_path = os.path.join(args.output_folder, f"bare_g-pre_100_10_{model_name}_p-{args.prompt_type}_iter-{args.iter}.jsonl")
    output_directory = os.path.dirname(output_file_path)
    # Ensure the directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_file_path

if __name__ == "__main__":
    args = parse_args()
    output_file_path = construct_output_file_path(args)

    print(f"model_id: {args.model_id}")
    print(f"repetition_penalty: {args.repetition_penalty}")
    print(f"temperature: {args.temperature}")
    print(f"top_p: {args.top_p}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"output_file_path: {output_file_path}")

    run_inference_bare(args, output_file_path)







