import torch
import json
import pickle
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarConstrainedLogitsProcessor, GrammarAlignedLogitsProcessor
from transformers_gad.generation.gad_logits_processor import GrammarAlignedGroundTruthLogitsProcessor
from transformers_gad.build_oracle.build_oracle_trie import run_demo_trie_string_01_len_3
from transformers_gad.generation.gad_logits_processor_oracle import GrammarAlignedOracleLogitsProcessor
from transformers_gad.build_oracle.build_oracle_trie import Trie, TrieNode
import argparse
import os
import random
from inference_utils import get_file, load_model_tokenizer_hf
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import get_desired_string_dict
from get_desired_string_dict import stringsofLenk_max, stringsofLenk, convert_grammar
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
#"mistralai/Mixtral-8x7B-Instruct-v0.1"
# "mistralai/Mistral-7B-Instruct-v0.1")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with grammar constraint decoding.")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.1",
                        help="pretrained model checkpoint.")
    parser.add_argument("--cache_dir", type=str, default='/nobackup2/yf/mila/GD_caches/',
                        help="Where to store cache tokenizers and models.")
    parser.add_argument("--base_grammar_dir", type=str, default="/nobackup2/yf/mila/GD/examples/grammars/",
                        help="Base directory for test grammars.")
    parser.add_argument("--grammar_file", type=str, default="string_start_w_1_all_0_len_3.ebnf",
                        help="Grammar file to test.")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Number of sequences to return.")
    # parser.add_argument("--max_length", type=int, default=50,
    #                     help="Maximum length of generated sequences when do not sample.")
    # parser.add_argument("--seed", type=int, default=42,
    #                     help="Random seed for reproducibility.")
    # parser.add_argument("--num_beams", type=int, default=5,
    #                     help="Number of beams for beam search.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                         help="Repetition penalty for greedy decoding.")
    parser.add_argument("--string_length", type=int, default=3,
                        help="Length of string to generate.")
    parser.add_argument("--prompt", type=str, default=f"Be a helpful assistant. Generate a random binary string of length 3? Directly show the generated string without explanation.",
                        help="Prompt for model inference.")
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
    parser.add_argument("--max_new_tokens", type=int, default=4,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--sygus_prompt_file", type=str, default="/nobackup2/yf/mila/GD/prompts/pre_prompt.jsonl",
                        help="File path to prompts for sygus task.")
    parser.add_argument("--prompt_type", type=str, choices=["bare", "completion"], default="bare",
                        help="Prompt type for sygus task.")

    args = parser.parse_args()
    return args

def load_oracle_trie(trie_file):
    with open(trie_file, 'rb') as f:
        trie = pickle.load(f)
    return trie

def construct_gad_output_file_path(args):
    model_name = args.model_id.split("/")[-1]
    output_file_path = os.path.join(args.output_folder, f"gad_g-pre_100_10_{model_name}_p-{args.prompt_type}_iter-{args.iter}.jsonl")
    output_directory = os.path.dirname(output_file_path)
    # Ensure the directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_file_path

def get_sygus_prompt(filename, prompt_type):
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)

            if data.get('prompt_type') == prompt_type:
                return data['prompt']

        raise ValueError(f"Prompt type {prompt_type} not found in file {filename}")

def inference_gad(args, model, tokenizer, prompt, grammar, trie):
    """
    latest version of gad test function prepared for run inference for iterations
    """
    gad_oracle_processor = GrammarAlignedOracleLogitsProcessor(grammar, trie)
    inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
    logits_processors = LogitsProcessorList([
        inf_nan_remove_processor,
        gad_oracle_processor,
    ])

    # Generate
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
        logits_processor=logits_processors,
        repetition_penalty=args.repetition_penalty,
        # early_stopping=True,
        num_return_sequences=args.num_return_sequences,
        return_dict_in_generate=True,
        output_scores=True,
    )

    input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    generated_tokens = output.sequences[:, input_length:]
    acceptance_details_history = gad_oracle_processor.acceptance_details_history
    adjusted_acceptance_details_history = gad_oracle_processor.adjusted_acceptance_details_history
    generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    # print(f"grammar constrained generations: {generations}")
    return generated_tokens, acceptance_details_history,adjusted_acceptance_details_history, generations




if __name__ == "__main__":
    args = parse_args()

    print(f"model_id: {args.model_id}")
    print(f"repetition_penalty: {args.repetition_penalty}")
    print(f"grammar_file: {args.grammar_file}")
    # print(f"num_beams: {args.num_beams}")
    print(f"temperature: {args.temperature}")
    print(f"top_p: {args.top_p}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"log_file: {args.log_file}")

    # output, faithful, ideal, elapsed_time = run_inference_grammar_constrained_track_scores(args)
    # result, non_inf_scores_with_index, output_sequences, elapsed_time = run_inference_grammar_constrained_track_scores(args)
    # result, non_inf_scores_with_index, output_sequences, elapsed_time = run_inference_track_scores(args)
    model, tokenizer = load_model_tokenizer_hf(args)
    # sequences, scores, generations = inference_track_scores(args, model, tokenizer)
    # print(f"sequences: {sequences}")
    # print(f"scores: {scores}")
    # print(f"generations: {generations}")

    ### run inference_gcd_get_logits_for_oracle ###
    # (sequences,
    #         scores,
    #         generations,
    #         generated_tokens,
    #         acceptance_details_history) = inference_gcd_get_logits_for_oracle(args, model, tokenizer)

    # print(f"sequences: {sequences}")
    # print(f"scores: {scores}")
    # print(f"generations: {generations}")
    # print(f"generated_tokens: {generated_tokens}")
    # print(f"acceptance_details_history: {acceptance_details_history}")


    ### run inference_grammar_aligned_track_full_history ###
    (sequences,
     scores,
     generations, accepted_tokens_history, accepted_indices_history, acceptance_raw_scores_history,
     acceptance_logits_history,
     acceptance_details_history, adjusted_acceptance_detailed_history) = inference_grammar_aligned_track_full_history(args, model, tokenizer, trie)


    print(f"sequences: {sequences}")
    print(f"scores: {scores}")
    print(f"generations: {generations}")
    print(f"accepted_tokens_history: {accepted_tokens_history}")
    print(f"accepted_indices_history: {accepted_indices_history}")
    print(f"acceptance_raw_scores_history: {acceptance_raw_scores_history}")
    print(f"acceptance_logits_history: {acceptance_logits_history}")
    print(f"acceptance_details_history: {acceptance_details_history}")
    print(f"adjusted_acceptance_detailed_history: {adjusted_acceptance_detailed_history}")

    # ### run gad ###
    # output, faithful, ideal, elapsed_time = run_inference_grammar_aligned(args, trie)





