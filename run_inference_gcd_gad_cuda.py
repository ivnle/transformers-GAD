import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_gad.build_oracle.build_oracle_trie import Trie, update_oracle_trie
from run_inference_gad import inference_gad, load_oracle_trie, construct_gad_output_file_path
import argparse
import os
import random
from inference_utils import (get_file,
                             load_model_tokenizer_hf, load_model_tokenizer_hf_with_device,
                             get_sygus_prompt,
                             get_grammar_file_path_by_prompt_type,
                             save_trie_to_pkl,
                             construct_trie_file, construct_trie_file_cuda)
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
#"mistralai/Mixtral-8x7B-Instruct-v0.1")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with grammar constraint decoding.")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.1",
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
    # parser.add_argument("--num_beams", type=int, default=5,
    #                     help="Number of beams for beam search.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                         help="Repetition penalty for greedy decoding.")
    parser.add_argument("--prompt", type=str, default=f"Generate a program.",
                        help="Depreciated, warning: only test prompt for the model.")
    parser.add_argument("--iter", type=int, default=1,
                        help="Number of iterations for inference.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling.")
    parser.add_argument("--do_sample", action='store_true',
                        help="Whether to sample from the model.")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top p for nucleus sampling.")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top k for sampling.")
    # parser.add_argument("--log_file", type=str, default='/nobackup2/yf/mila/GD/log/test_log.txt',
    #                     help="Where to store log file.")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--sygus_prompt_file", type=str, default="/nobackup2/yf/mila/GD/prompts/pre_prompt.jsonl",
                        help="File path to prompts for sygus task.")
    parser.add_argument("--prompt_type", type=str, choices=["bare", "completion"], default="bare",
                        help="Prompt type for sygus task.")
    parser.add_argument("--output_folder", type=str, default="/nobackup2/yf/mila/GD/results/",
                        help="Output folder to store results.")
    parser.add_argument("--grammar_name", type=str, default="PRE_100",
                        help="Name of the grammar, mainly used for call grammar file.")
    parser.add_argument("--trie_folder", type=str, default="/nobackup2/yf/mila/GD/results_trie/",
                        help="Folder to store trie files.")


    args = parser.parse_args()
    return args


def inference_grammar_constrained(args, model, tokenizer):
    """
    depreciated, only apply for generating binary strings.
    """
    test_file = get_file(args)

    # Load grammar
    with open(test_file, "r") as file:
        grammar_str = file.read()
    print(f"grammar_str: {grammar_str}")
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prompt = args.prompt
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]
    # tensor([[16968,   368,  5706,   263,  7581,  1347,   310,  3309,   472,  1556,
    #          29871, 29946, 29973]])

    # if args.do_sample == False:
    #     output = model.generate(
    #         input_ids,
    #         do_sample=args.do_sample,
    #         max_length=args.max_length,
    #         num_beams=args.nums_beams,
    #         logits_processor=[grammar_processor],
    #         repetition_penalty=args.repetition_penalty,
    #         num_return_sequences=args.num_return_sequences,
    #     )
    #
    # else:
    output = model.generate(
        input_ids,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
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

def run_inference_grammar_constrained(args):
    """
    depreciated, only apply for generating binary strings.
    """
    model, tokenizer = load_model_tokenizer_hf(args)
    tokenizer.pad_token = tokenizer.eos_token

    def get_current_time_as_string():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    log_file_path = args.log_file
    start_time = time.time()

    with open(get_file(args), 'r') as f:
        input_grammar = f.read()
    # output = stringsofLenk_max(input_grammar, args.string_length)
    output = stringsofLenk(input_grammar, args.string_length)
    ideal = {key: round(args.iter / len(output.keys())) for key in output.keys()}
    faithful = output.copy()
    output['other'] = 0
    ideal['other'] = 0
    with open(log_file_path, 'a') as log:
        log.write(f"{get_current_time_as_string()} - input_grammar: {input_grammar}\n")
        for i in tqdm(range(args.iter), desc="Running Inference"):
            result = inference_grammar_constrained(args, model, tokenizer)
            log.write(f"{get_current_time_as_string()} - result: {result}\n")
            log.flush()
            # print(f'start logging...')
            res = result[0].split(".")[2]
            # print(f"res: {res}")
            if res in output:
                output[res] += 1
            else:
                output['other'] += 1

            faithful[res] = faithful.get(res, 0) + 1 # collect all the outputs instead of classifying to others
            if i % 10 == 0:
                log.write(f"{get_current_time_as_string()} - Iteration: {i+1}\n")
                log.flush()
                log.write(f"{get_current_time_as_string()} - Output: {output}\n")
                log.flush()
                log.write(f"{get_current_time_as_string()} - Faithful: {faithful}\n")
                log.flush()
        end_time = time.time()
        elapsed_time = end_time - start_time
        log.write(f"Elapsed time: {elapsed_time} seconds\n")
        log.flush()
        log.write(f"model_id: {args.model_id}\n")
        log.flush()
        log.write(f"repetition_penalty: {args.repetition_penalty}\n")
        log.flush()
        # print(f"num_beams: {args.num_beams}")
        log.write(f"temperature: {args.temperature}\n")
        log.flush()
        log.write(f"top_p: {args.top_p}\n")
        log.flush()
        log.write(f"max_new_tokens: {args.max_new_tokens}\n")
        log.flush()
        log.write(f"{get_current_time_as_string()} - output: {output}\n")
        log.flush()
        log.write(f"{get_current_time_as_string()} - faithful: {faithful}\n")
        log.flush()
        log.write(f"{get_current_time_as_string()} - ideal: {ideal}\n")
        log.flush()
    return output, faithful, ideal, elapsed_time

def inference_gcd(args, model, tokenizer):
    """
    latest version of gcd test function
    """
    test_file = get_grammar_file_path_by_prompt_type(args)
    prompt = args.prompt

    # Load grammar
    with open(test_file, "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
    inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
    logits_processors = LogitsProcessorList([
        inf_nan_remove_processor,
        grammar_processor,
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
    acceptance_details_history = grammar_processor.acceptance_details_history
    generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(f"prompt: {prompt}")
    print(f"grammar constrained generations: {generations}")
    return generated_tokens, acceptance_details_history, generations

def inference_gcd_build_oracle_trie(args, model, tokenizer, prompt, grammar_str):
    """
    latest version of gcd test function
    """
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
    inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
    logits_processors = LogitsProcessorList([
        inf_nan_remove_processor,
        grammar_processor,
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
    acceptance_details_history = grammar_processor.acceptance_details_history
    generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    # print(f"grammar constrained generations: {generations}")
    return generated_tokens, acceptance_details_history, generations

def run_inference_gcd_construct_oracle_trie(args):
    model, tokenizer = load_model_tokenizer_hf(args)
    output_file_path = construct_gcd_output_file_path(args)
    trie_file = construct_trie_file_cuda(args)
    trie = Trie()
    prompt = get_sygus_prompt(args.sygus_prompt_file, args.prompt_type)
    test_file = get_grammar_file_path_by_prompt_type(args)

    # #### only for test purpose ####
    # prompt = args.prompt
    # test_file = get_file(args)

    # Load grammar
    with open(test_file, "r") as file:
        grammar_str = file.read()

    start_time = time.time()


    with open(output_file_path, 'a', encoding='utf-8') as outfile:
        for i in tqdm(range(args.iter), desc="Running Inference"):
            generated_tokens, acceptance_details_history, generations = inference_gcd_build_oracle_trie(args, model, tokenizer, prompt, grammar_str)
            result = {"answer": generations, "prompt": prompt, "prompt_type": args.prompt_type,
                      "grammar": "PRE_100_10.sl"}
            print(f"result: {result}")
            # print(f"generated_tokens: {generated_tokens}, acceptance_details_history: {acceptance_details_history}")
            update_oracle_trie(trie, generated_tokens, acceptance_details_history)
            json_record = json.dumps(result)
            outfile.write(json_record + '\n')
            outfile.flush()
            os.fsync(outfile.fileno())
    save_trie_to_pkl(trie, trie_file)
    print(f"Trie saved to {trie_file}")
    end_time = time.time()
    print(f"GCD results saved to {output_file_path}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

    ##### also run gad after gcd #####

    gad_output_file_path = construct_gad_output_file_path(args)
    start_time = time.time()

    with open(gad_output_file_path, 'a', encoding='utf-8') as outfile:
        trie = load_oracle_trie(trie_file)
        before_trie_status = "gad_before"
        after_trie_status = "gad_after"
        adjusted_trie_before = Trie()
        adjustd_trie_after = Trie()
        for i in tqdm(range(args.iter), desc="Running Inference"):
            generated_tokens, acceptance_details_history,adjusted_acceptance_details_history, generations = inference_gad(args, model, tokenizer, prompt, grammar, trie)
            result = {"answer": generations, "prompt": prompt, "prompt_type": args.prompt_type,
                      "grammar": "PRE_100_10.sl"}
            print(f"result: {result}")
            # print(f"generated_tokens: {generated_tokens}, acceptance_details_history: {acceptance_details_history}")
            update_oracle_trie(adjusted_trie_before, generated_tokens, acceptance_details_history)
            update_oracle_trie(adjustd_trie_after, generated_tokens, adjusted_acceptance_details_history)
            json_record = json.dumps(result)
            outfile.write(json_record + '\n')
            outfile.flush()
            os.fsync(outfile.fileno())

    trie_file_before = construct_trie_file_cuda(args, before_trie_status)
    trie_file_after = construct_trie_file_cuda(args, after_trie_status)
    save_trie_to_pkl(adjusted_trie_before, trie_file_before)
    print(f"GAD before trie saved to {trie_file_before}")
    save_trie_to_pkl(adjustd_trie_after, trie_file_after)
    print(f"GAD after trie saved to {trie_file_after}")
    end_time = time.time()
    print(f"GAD results saved to {gad_output_file_path}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")


def construct_gcd_output_file_path(args):
    model_name = args.model_id.split("/")[-1]
    output_file_path = os.path.join(args.output_folder, f"gcd_g-pre_100_10_{model_name}_p-{args.prompt_type}_iter-{args.iter}_cuda.jsonl")
    output_directory = os.path.dirname(output_file_path)
    # Ensure the directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_file_path


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = load_model_tokenizer_hf_with_device(args)

    print(f"model_id: {args.model_id}")
    print(f"repetition_penalty: {args.repetition_penalty}")
    print(f"temperature: {args.temperature}")
    print(f"top_p: {args.top_p}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"output_folder: {args.output_folder}")

    run_inference_gcd_construct_oracle_trie(args)
    # inference_gcd(args, model, tokenizer)



