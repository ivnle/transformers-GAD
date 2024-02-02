import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_tokenizer_hf(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id,
                                              # use_fast=True,
                                              cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    return model, tokenizer


def get_file(args):
    base_dir = args.base_grammar_dir
    grammar_file = args.grammar_file
    return os.path.join(base_dir, grammar_file)