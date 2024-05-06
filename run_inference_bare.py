import torch
import json
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import random
from inference_utils import get_file, load_model_tokenizer_hf
import numpy as np
import json
import logging
from tqdm import tqdm
import time
from inference_utils import get_prompt_in_test_folder, fix_seed


#models=("meta-llama/Llama-2-7b-hf"
#"meta-llama/Llama-2-13b-hf"
#"meta-llama/Llama-2-70b-hf"
#"mistralai/Mixtral-8x7B-Instruct-v0.1"
# "mistralai/Mistral-7B-Instruct-v0.1")
from arg_parser import ArgumentParser

def get_sygus_prompt(filename, prompt_type):
    """depreciated, only works for prompt store in jsonl file."""
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)

            if data.get('prompt_type') == prompt_type:
                return data['prompt']

        raise ValueError(f"Prompt type {prompt_type} not found in file {filename}")


@torch.inference_mode()
def inference_bare(args, model, tokenizer, prompt):
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]
    input_ids = input_ids.to(model.device)

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

    transition_scores = model.compute_transition_scores(
        output.sequences, output.scores, normalize_logits=True
    )

    metas = []
    sum_log_prob = 0
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        meta = {
            "token_id": int(tok),
            "token_str": tokenizer.decode(tok),
            "norm_score": float(score.cpu().numpy()),
            "prob": float(np.exp(score.cpu().numpy()))
        }
        metas.append(meta)
        sum_log_prob += float(score.cpu().numpy())
    return generations, metas, sum_log_prob

@torch.inference_mode()
def run_inference_bare(args,output_file_path, test_filename):
    model, tokenizer = load_model_tokenizer_hf(args)
    # if "binary" in args.prompt_type:
    #     prompt = get_prompt(args, args.prompt_type)
    #     # test_file = get_file(args)
    #     # grammar_constr_name = test_file.split("/")[-1]
    #     grammar_prompt_file = None
    # else:
    #     prompt = construct_sygus_prompt(args, args.prompt_type)
    #     # test_file = get_grammar_file_path_by_prompt_type(args)
    #     # grammar_constr_name = test_file.split("/")[-1]
    #     grammar_prompt_file = args.grammar_prompt_file.split("/")[-1]
    prompt = get_prompt_in_test_folder(args, test_filename)
    grammar_prompt_file = f"{test_filename}.sl"
    grammar_constr_name = f"{test_filename}.ebnf"
    start_time = time.time()
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for i in tqdm(range(args.iter), desc="Running Inference"):
            generations, metas, sum_log_prob = inference_bare(args, model, tokenizer, prompt)
            result = {"answer": generations,
                      "metas": metas,
                      "sum_log_prob": sum_log_prob,
                      "prompt": prompt,
                      "grammar_prompt": grammar_prompt_file,
                      "grammar_constr": grammar_constr_name
                      }
            print(f"result: {result}")

            json_record = json.dumps(result)
            outfile.write(json_record + '\n')
            outfile.flush()
            os.fsync(outfile.fileno())


    end_time = time.time()
    print(f"Results saved to {output_file_path}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

def construct_bare_output_file_path(args):
    model_name = args.model_id.split("/")[-1]
    grammar_prompt_file = args.grammar_prompt_file.split("/")[-1]
    grammar_prompt_name = grammar_prompt_file.split(".")[0]
    output_file_path = os.path.join(args.output_folder, f"bare_g-{grammar_prompt_name}_{model_name}_p-{args.prompt_type}_i{args.iter}_{args.device}_sd{args.seed}_{args.dtype}.jsonl")
    output_directory = os.path.dirname(output_file_path)
    # Ensure the directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_file_path

def construct_bare_output_file_path_from_folder(args, test_filename):
    model_name = args.model_id.split("/")[-1]
    output_file_path = os.path.join(args.output_folder, f"{test_filename}")
    output_file_path = os.path.join(output_file_path, f"bare_{model_name}_i{args.iter}_{args.device}_sd{args.seed}_{args.dtype}.jsonl")
    output_directory = os.path.dirname(output_file_path)
    # Ensure the directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_file_path

if __name__ == "__main__":
    arg_parser = ArgumentParser(version="bare")
    args = arg_parser.parse_args()
    print(f"model_id: {args.model_id}")
    print(f"repetition_penalty: {args.repetition_penalty}")
    print(f"temperature: {args.temperature}")
    print(f"top_p: {args.top_p}")
    print(f"max_new_tokens: {args.max_new_tokens}")

    directory = args.test_folder
    for filename in os.listdir(directory):
        if filename.endswith(".sl"):
            test_filename = filename[:-3]
            print(f"test_filename: {test_filename}")
            output_file_path = construct_bare_output_file_path_from_folder(args, test_filename)
            fix_seed(args.seed)
            run_inference_bare(args, output_file_path, test_filename)
    print("Bare Inference Done!")

    # # Test for pre_prompt
    # output_file_path = construct_bare_output_file_path(args)
    # run_inference_bare(args, output_file_path)

    # # debug
    # model, tokenizer = load_model_tokenizer_hf(args)
    # prompt = args.prompt
    # generations = inference_bare(args, model, tokenizer, prompt)
    # print(f"bare_generations: {generations}")

