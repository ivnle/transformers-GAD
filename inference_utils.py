import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import pickle
import sys
import hashlib
import typing

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

    return model, tokenizer

def get_file(args):
    base_dir = args.base_grammar_dir
    grammar_file = args.grammar_file
    return os.path.join(base_dir, grammar_file)

def get_grammar_file_path_by_prompt_type(args):
    base_dir = args.base_grammar_dir
    grammar_file = f"{args.grammar_name}_{args.prompt_type}.ebnf"
    return os.path.join(base_dir, grammar_file)

def get_sygus_prompt(filename, prompt_type):
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)

            if data.get('prompt_type') == prompt_type:
                return data['prompt']

        raise ValueError(f"Prompt type {prompt_type} not found in file {filename}")

def save_trie_to_pkl(trie, file_path):
    sys.setrecursionlimit(10000000)
    with open(file_path, 'wb') as f:
        pickle.dump(trie, f)

def construct_trie_file(args, trie_status=None):
    model_name = args.model_id.split("/")[-1]
    if trie_status is None:
        trie_file = f"trie_{args.grammar_name}_{args.prompt_type}_{model_name}_iter-{args.iter}_{args.device}.pkl"
    else:
        trie_file = f"trie_{args.grammar_name}_{args.prompt_type}_{model_name}_iter-{args.iter}_{args.device}_{trie_status}.pkl"
    trie_file_path = os.path.join(args.trie_folder, trie_file)
    return trie_file_path

def stable_hash(s: str) -> int:
    """Returns a stable hash of the given string, stable between runs."""
    hash_obj = hashlib.sha256()
    hash_obj.update(bytes(s, "UTF-8"))
    return int.from_bytes(hash_obj.digest(), "big")