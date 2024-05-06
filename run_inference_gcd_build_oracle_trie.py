from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_gad.build_oracle.build_oracle_trie import Trie, update_oracle_trie
from run_inference_gad import inference_gad, load_oracle_trie, construct_gad_output_file_path, construct_gad_output_file_path_from_folder
import torch
import os
from inference_utils import (get_file,
                             load_model_tokenizer_hf,
                             get_prompt,
                             get_grammar_file_path_by_prompt_type,
                             save_trie_to_pkl,
                             construct_trie_file,
                             construct_sygus_prompt,
                             construct_trie_file_from_folder,
                             get_prompt_in_test_folder,
                             fix_seed)
import json
from tqdm import tqdm
import time
from datetime import datetime
from arg_parser import ArgumentParser
import numpy as np


#models=("meta-llama/Llama-2-7b-hf"
#"meta-llama/Llama-2-13b-hf"
#"meta-llama/Llama-2-70b-hf"
#"mistralai/Mixtral-8x7B-Instruct-v0.1")

@torch.inference_mode()
def inference_gcd(args, model, tokenizer):
    """
    latest version of gcd test function
    """
    test_file = get_grammar_file_path_by_prompt_type(args)
    print(f"test_file: {test_file}")
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

    # pipe = pipeline(
    #     "text-generation",
    #     model="HuggingFaceH4/starchat2-15b-v0.1",
    #     device_map="auto",
    #     torch_dtype=torch.bfloat16,
    # )
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are StarChat2, an expert programming assistant",
    #     },
    #     {"role": "user",
    #      "content": "Write a simple website in HTML. When a user clicks the button, it shows a random Chuck Norris joke."},
    # ]
    # outputs = pipe(
    #     messages,
    #     max_new_tokens=512,
    #     do_sample=True,
    #     temperature=0.7,
    #     top_k=50,
    #     top_p=0.95,
    #     logits_processor=logits_processors,
    #     stop_sequence="<|im_end|>",
    # )
    # print(outputs[0]["generated_text"][-1]["content"])

    # Generate
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]

    model.resize_token_embeddings(len(tokenizer))

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

    transition_scores = model.compute_transition_scores(
        output.sequences, output.scores, normalize_logits=True
    )
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        # | token | token string | logits | probability
        print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")

    # decode output
    print(f"output sequences: {output.sequences}")
    print(f"scores: {output.scores}")
    print(f"prompt: {prompt}")
    print(f"grammar constrained generations: {generations}")
    return generated_tokens, acceptance_details_history, generations

@torch.inference_mode()
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
    # print(f"grammar constrained generations: {generations}")
    return generated_tokens, acceptance_details_history, generations, metas, sum_log_prob

@torch.inference_mode()
def run_inference_gcd_construct_oracle_trie(args, test_filename, model, tokenizer):
    output_file_path = construct_gcd_output_file_path_from_folder(args, test_filename)
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if len(lines) >= args.iter:
                print(f"Skipping {output_file_path} as it already contains 100 or more lines.")
                return
    # model, tokenizer = load_model_tokenizer_hf(args)
    trie_file = construct_trie_file_from_folder(args, test_filename)
    trie = Trie()
    # if "binary" in args.prompt_type:
    #     prompt = get_prompt(args, args.prompt_type)
    #     test_file = get_file(args)
    #     grammar_constr_name = test_file.split("/")[-1]
    #     grammar_prompt_file = None
    # else:
    #     prompt = construct_sygus_prompt(args, args.prompt_type)
    #     test_file = get_grammar_file_path_by_prompt_type(args)
    #     grammar_constr_name = test_file.split("/")[-1]
    #     grammar_prompt_file = args.grammar_prompt_file.split("/")[-1]

    # #### only for test purpose ####
    # prompt = args.prompt
    # test_file = get_file(args)

    # Load grammar
    test_file = os.path.join(args.grammar_folder, f"{test_filename}.ebnf")
    with open(test_file, "r") as file:
        grammar_str = file.read()

    prompt = get_prompt_in_test_folder(args, test_filename)
    grammar_prompt_file = f"{test_filename}.sl"
    grammar_constr_name = f"{test_filename}.ebnf"

    start_time = time.time()

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for i in tqdm(range(args.iter), desc="Running Inference"):
            generated_tokens, acceptance_details_history, generations, metas, sum_log_prob = inference_gcd_build_oracle_trie(args, model, tokenizer, prompt, grammar_str)
            result = {"answer": generations,
                      "sum_log_prob": sum_log_prob,
                      "metas": metas,
                      "prompt": prompt,
                      # "prompt_type": args.prompt_type,
                      "grammar_prompt": grammar_prompt_file,
                      "grammar_constr": grammar_constr_name}
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


def construct_gcd_output_file_path(args):
    model_name = args.model_id.split("/")[-1]
    grammar_prompt_file = args.grammar_prompt_file.split("/")[-1]
    grammar_prompt_name = grammar_prompt_file.split(".")[0]
    output_file_path = os.path.join(args.output_folder, f"gcd_g-{grammar_prompt_name}_{model_name}_p-{args.prompt_type}_i{args.iter}_{args.device}_sd{args.seed}_{args.dtype}.jsonl")
    output_directory = os.path.dirname(output_file_path)
    # Ensure the directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_file_path

def construct_gcd_output_file_path_from_folder(args, test_filename):
    model_name = args.model_id.split("/")[-1]
    output_file_path = os.path.join(args.output_folder, f"{test_filename}")
    output_file_path = os.path.join(output_file_path, f"gcd_{model_name}_i{args.iter}_{args.device}_sd{args.seed}_{args.dtype}.jsonl")
    output_directory = os.path.dirname(output_file_path)
    # Ensure the directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_file_path


if __name__ == "__main__":
    arg_parser = ArgumentParser(version="gad")
    args = arg_parser.parse_args()

    print(f"model_id: {args.model_id}")
    print(f"repetition_penalty: {args.repetition_penalty}")
    print(f"temperature: {args.temperature}")
    print(f"top_p: {args.top_p}")
    print(f"max_new_tokens: {args.max_new_tokens}")

    model, tokenizer = load_model_tokenizer_hf(args)
    directory = args.test_folder
    for filename in os.listdir(directory):
        if filename.endswith(".sl"):
            test_filename = filename[:-3]
            print(f"test_filename: {test_filename}")
            fix_seed(args.seed)
            run_inference_gcd_construct_oracle_trie(args, test_filename, model, tokenizer)
    print("GCD Inference Done!")

    # print(f"output_folder: {args.output_folder}")

    # test to see whether grammar file works
    # model, tokenizer = load_model_tokenizer_hf(args)
    # inference_gcd(args, model, tokenizer)

    # run inference and build trie
    # run_inference_gcd_construct_oracle_trie(args)



