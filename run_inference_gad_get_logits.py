import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarConstrainedLogitsProcessor, GrammarAlignedLogitsProcessor
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
    parser.add_argument("--string_length", type=int, default=5,
                        help="Length of string to generate.")
    parser.add_argument("--prompt", type=str, default=f"Be a helpful assistant. Generate a random binary string of length 3? Directly show the generated string without explanation.",
                        help="Prompt for model inference.")
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
    parser.add_argument("--log_file", type=str, default='/nobackup2/yf/mila/GD/log_GAD/track_scores_prob2.log',
                        help="Where to store log file.")
    parser.add_argument("--max_new_tokens", type=int, default=4,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--output_scores", action='store_true',
                        help="Whether to output scores.")
    parser.add_argument("--return_dict_in_generate", action='store_true',
                        help="Whether to return dict in generate.")

    args = parser.parse_args()
    return args


def inference_grammar_constrained_track_scores(args, model, tokenizer):
    test_file = get_file(args)
    tokenizer.pad_token = tokenizer.eos_token

    # Load grammar
    with open(test_file, "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
    grammar_aligned_processor = GrammarAlignedLogitsProcessor(grammar)

    # Generate
    prompt = args.prompt
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]
    # tensor([[16968,   368,  5706,   263,  7581,  1347,   310,  3309,   472,  1556,
    #          29871, 29946, 29973]])

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
        logits_processor=[grammar_aligned_processor],
        repetition_penalty=args.repetition_penalty,
        # early_stopping=True,
        num_return_sequences=args.num_return_sequences,
        return_dict_in_generate=True,
        output_scores=True
    )

    # (accepted_tokens_history,
    #  accepted_indices_history,
    #  acceptance_raw_scores_history,
    #  acceptance_logits_history,
    #  acceptance_details_history)\
    #     = grammar_processor.get_history()

    (accepted_tokens_history,
     accepted_indices_history,
     acceptance_raw_scores_history,
     acceptance_logits_history,
     acceptance_details_history,
     adjusted_acceptance_detailed_history) \
        = grammar_aligned_processor.get_history()

    # input_ids_history = grammar_processor.get_input_ids_history()

    print(f"raw_scores_history: {acceptance_raw_scores_history}")
    print(f"softmax_logits_history: {acceptance_logits_history}, length: {len(acceptance_logits_history)}")
    print(f"accepted_tokens_history: {accepted_tokens_history}")
    print(f"accepted_indices_history: {accepted_indices_history}")
    print(f"acceptance_details_history: {acceptance_details_history}")
    print(f"adjusted_acceptance_detailed_history: {adjusted_acceptance_detailed_history}")
    # print(f"input_ids_history: {input_ids_history}")

    transition_scores = model.compute_transition_scores(
        output.sequences, output.scores, normalize_logits=True
    )

    input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    generated_tokens = output.sequences[:, input_length:]
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        # | token | token string | logits | probability
        print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")

    # decode output
    print(f"output sequences: {output.sequences}")
    print(f"scores: {output.scores}")
    generations = tokenizer.batch_decode(output[0], skip_special_tokens=True)
    print(f"grammar constrained generations: {generations}")
    return (output.sequences,
            output.scores,
            generations, accepted_tokens_history, accepted_indices_history, acceptance_raw_scores_history,
            acceptance_logits_history,
            acceptance_details_history)

def softmax(selected_scores, scores):
    exp_selected_scores = torch.exp(selected_scores)
    exp_scores = torch.exp(scores)
    sum_exp_scores = torch.sum(exp_scores)
    return exp_selected_scores / sum_exp_scores

def inference_track_scores(args, model, tokenizer):
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
    force_words = ["000"]
    force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids

    output = model.generate(
        input_ids,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        num_beams=2,
        top_k=50,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        # early_stopping=True,
        num_return_sequences=args.num_return_sequences,
        return_dict_in_generate=True,
        output_scores=True,
        force_words_ids=force_words_ids
    )

    transition_scores = model.compute_transition_scores(
        output.sequences, output.scores, normalize_logits=True
    )

    input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    generated_tokens = output.sequences[:, input_length:]
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        # | token | token string | logits | probability
        print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")

    # decode output
    selected_scores = np.array([14.586828231811523, 14.703150749206543])
    scores = output.scores[1].numpy()
    for selected_score in selected_scores:
        print(f"selected_score: {selected_score}")
        print(f"softmax: {np.exp(selected_score) / np.sum(np.exp(scores))}")

    print(f"output sequences: {output.sequences}")
    # print(f"scores: {output.scores}")
    generations = tokenizer.batch_decode(output[0], skip_special_tokens=True)
    print(f"grammar constrained generations: {generations}")
    return output.sequences, output.scores, generations


def run_inference_grammar_constrained_track_scores(args):
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
            output_sequences, scores, result = inference_grammar_constrained_track_scores(args, model, tokenizer)

            # deal with scores:
            # List to collect non -inf scores
            non_inf_scores_with_index = []

            for index, score_tensor in enumerate(scores):
                mask = score_tensor != -float('inf')
                filtered_scores = score_tensor[mask]
                for score in filtered_scores:
                    non_inf_scores_with_index.append((index, score.item()))

            # Print non -inf scores with their respective tensor index
            for index, score in non_inf_scores_with_index:
                print(f"Tensor {index}: Score {score}")

            log.write(f"{get_current_time_as_string()} - result: {result}\n")
            log.flush()
            log.write(f"{get_current_time_as_string()} - scores: {scores}\n")
            log.write(f"{get_current_time_as_string()} - non_inf_scores: {non_inf_scores_with_index}\n")
            log.flush()
            log.write(f"{get_current_time_as_string()} - output_sequences: {output_sequences}\n")
            log.flush()
            # # print(f'start logging...')
            # res = result[0].split(".")[2]
            # # print(f"res: {res}")
            # if res in output:
            #     output[res] += 1
            # else:
            #     output['other'] += 1
            #
            # faithful[res] = faithful.get(res, 0) + 1 # collect all the outputs instead of classifying to others
            # if i % 10 == 0:
            #     log.write(f"{get_current_time_as_string()} - Iteration: {i+1}\n")
            #     log.flush()
            #     log.write(f"{get_current_time_as_string()} - Output: {output}\n")
            #     log.flush()
            #     log.write(f"{get_current_time_as_string()} - Faithful: {faithful}\n")
            #     log.flush()
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
        # log.write(f"{get_current_time_as_string()} - output: {output}\n")
        # log.flush()
        # log.write(f"{get_current_time_as_string()} - faithful: {faithful}\n")
        # log.flush()
        # log.write(f"{get_current_time_as_string()} - ideal: {ideal}\n")
        # log.flush()
    # return output, faithful, ideal, elapsed_time
    return result, non_inf_scores_with_index, output_sequences, elapsed_time

def run_inference_track_scores(args):
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
            output_sequences, scores, result = inference_track_scores(args, model, tokenizer)

            # deal with scores:
            # List to collect non -inf scores
            non_inf_scores_with_index = []

            for index, score_tensor in enumerate(scores):
                mask = score_tensor != -float('inf')
                filtered_scores = score_tensor[mask]
                for score in filtered_scores:
                    non_inf_scores_with_index.append((index, score.item()))

            # Print non -inf scores with their respective tensor index
            # for index, score in non_inf_scores_with_index:
            #     print(f"Tensor {index}: Score {score}")

            log.write(f"{get_current_time_as_string()} - result: {result}\n")
            log.flush()
            log.write(f"{get_current_time_as_string()} - scores: {scores}\n")
            log.write(f"{get_current_time_as_string()} - non_inf_scores: {non_inf_scores_with_index}\n")
            log.flush()
            log.write(f"{get_current_time_as_string()} - output_sequences: {output_sequences}\n")
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
    # return output, faithful, ideal, elapsed_time
    return result, non_inf_scores_with_index, output_sequences, elapsed_time


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
    (sequences, scores, generations,
     accepted_tokens_history, accepted_indices_history, acceptance_raw_scores_history,
            acceptance_logits_history,
            acceptance_details_history) = inference_grammar_constrained_track_scores(args, model, tokenizer)

