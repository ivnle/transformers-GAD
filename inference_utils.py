import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import pickle
import sys
import hashlib
import typing
import re
from arg_parser import ArgumentParser

GRAMMAR_PROMPT_TOKEN = "<|grammar_prompt|>"

def load_model_tokenizer_hf(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id,
                                              # use_fast=True,
                                              cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(args.device)
    model.to(device)
    if args.dtype:
        dtype = torch.float32
        if args.dtype == "float16":
            dtype = torch.float16
        elif args.dtype == "bfloat16":
            dtype = torch.bfloat16
        model.to(dtype=dtype)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def get_file(args):
    """Depreciated, get grammar file path, use only for test case."""
    base_dir = args.base_grammar_dir
    grammar_file = args.grammar_file
    return os.path.join(base_dir, grammar_file)

def get_grammar_file_path_by_prompt_type(args):
    grammar_prefix = get_grammar_prefix(args)
    grammar_file = f"{grammar_prefix}_{args.prompt_type}.ebnf"
    return os.path.join(args.base_grammar_dir, grammar_file)

def extract_prefix(filename):
    """
    Extracts the prefix of a filename, which is the part connected by the first underscore.
    Works for PRE_100 and find_inv, crci; deal with files in woosuk separately.
    """
    if "sygus" or "name" in filename:
        return filename
    else:
        pattern = r"^([^_]+)_([^_]+)"
        match = re.match(pattern, filename)
        if match:
            # Combine the first two captured groups with an underscore
            return "_".join(match.groups())
        else:
            raise ValueError(f"Filename {filename} does not match pattern {pattern}")

def get_grammar_prefix(args):
    grammar_prompt_name = args.grammar_prompt_file.split("/")[-1]
    grammar_prompt_name = grammar_prompt_name.split(".")[0]
    return extract_prefix(grammar_prompt_name)

def get_prompt(args, prompt_type):
    """depreciated, use construct_sygus_prompt instead for generalized version."""
    with open(args.instruct_prompt_file, 'r') as file:
        for line in file:
            data = json.loads(line)

            if data.get('prompt_type') == prompt_type:
                return data['prompt']

        raise ValueError(f"Prompt type {prompt_type} not found in file {args.instruct_prompt_file}")

def construct_sygus_prompt(args, prompt_type):
    with open(args.grammar_prompt_file, "r") as file:
        grammar_str = file.read()
    if "bare" in prompt_type:
        grammar_prefix = None
    else:
        grammar_prefix = get_grammar_prefix(args)
        # print(f"grammar_prefix: {grammar_prefix}")

    with open(args.instruct_prompt_file, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue

            # Check for prompt_type match
            if data.get('prompt_type') == prompt_type:
                # Handle the "bare" case or ensure grammar_prefix matches
                if data.get('prompt_type') == prompt_type and (
                        grammar_prefix is None or grammar_prefix == data.get('grammar_prefix')):
                    instruct_prompt = data['instruct_prompt']
                    # Debugging print
                    print(f"instruct_prompt: {instruct_prompt}")
                    prompt = instruct_prompt.replace(GRAMMAR_PROMPT_TOKEN, grammar_str)
                    return prompt

            # If no matching line was found, raise an error
    raise ValueError(f"Prompt type {prompt_type} not found in file {args.instruct_prompt_file}")


def save_trie_to_pkl(trie, file_path):
    sys.setrecursionlimit(10000000)
    with open(file_path, 'wb') as f:
        pickle.dump(trie, f)

def construct_trie_file(args, trie_status=None):
    model_name = args.model_id.split("/")[-1]
    grammar_prompt_file = args.grammar_prompt_file.split("/")[-1]
    grammar_prompt_name = grammar_prompt_file.split(".")[0]
    if trie_status is None:
        trie_file = f"trie_{grammar_prompt_name}_{args.prompt_type}_{model_name}_i{args.iter}_{args.device}.pkl"
    else:
        trie_file = f"trie_{grammar_prompt_name}_{args.prompt_type}_{model_name}_i{args.iter}_{args.device}_{trie_status}.pkl"
    trie_file_path = os.path.join(args.trie_folder, trie_file)
    return trie_file_path

def stable_hash(s: str) -> int:
    """Returns a stable hash of the given string, stable between runs."""
    hash_obj = hashlib.sha256()
    hash_obj.update(bytes(s, "UTF-8"))
    return int.from_bytes(hash_obj.digest(), "big")

if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    prompt = construct_sygus_prompt(args, "completion")
    print(prompt)
    print(extract_prefix("sygus_iter_26_0"))