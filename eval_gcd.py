import argparse
import json
import os
import pprint

import torch
import wandb
from datasets import load_dataset
from torch.nn.functional import log_softmax
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
)

from transformers_gad.generation.logits_process import (
    GrammarAlignedOracleLogitsProcessor,
)
from transformers_gad.grammar_utils import IncrementalGrammarConstraint


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ebmoon/GAD-dataset", help="Dataset to use.")
    parser.add_argument("--split", type=str, default="BV4", help="Dataset split to use.")
    parser.add_argument("--id", type=str, default="find_inv_eq_bvlshr0_4bit", help="Dataset id (of split) to use.")
    parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations for inference.")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Model ID to use.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for model.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty for generation.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter.")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling parameter.")
    parser.add_argument("--do_sample", type=bool, default=True, help="Whether to use sampling for generation.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search. 1 means no beam search.")
    parser.add_argument("--mask", type=str, default='gcd', choices=['gcd', 'asap'], help="Strategy for masking tokens.")
    parser.add_argument("--log_every", type=int, default=5, help="Log every n iterations.")
    parser.add_argument("--do_compile", action="store_true", help="Whether to compile the model before running.")
    return parser.parse_args()

def init_logits_proc(grammar):
    gad_oracle_processor = GrammarAlignedOracleLogitsProcessor(grammar)
    inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
    logits_processors = LogitsProcessorList(
        [
            inf_nan_remove_processor,
            gad_oracle_processor,
        ]
    )
    return gad_oracle_processor, inf_nan_remove_processor, logits_processors


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def eval_prob(model, tokenizer, id, prompt, grammar_str, args):
    # Load EBNF grammar
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)

    # Tokenize prompt into ids
    inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Inference Loop
    outputs = []
    history = []

    logprob_best = float("-inf")
    raw_likelihood_best = float("-inf")

    for iter in tqdm(range(args.num_iter), desc="Running Inference"):
        if (args.mask == "gcd") or ((args.mask == "asap") and (iter == 0)):
            gad_oracle_processor, _, logits_processors = init_logits_proc(grammar)

        # Generate sequences
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=args.do_sample,
            # pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            logits_processor=logits_processors,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
            num_beams=args.num_beams,
        )
        # print(f"{input_ids.shape}=")
        # foo
        input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
        generated_tokens = output.sequences[0, input_length:].tolist()
        raw_likelihood = gad_oracle_processor.oracle_trie.raw_likelihood(
            generated_tokens
        )
        h = {"tokens": generated_tokens, "raw_likelihood": raw_likelihood}

        # generations = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # print(f"{generations=}")
        # print(f"{raw_likelihood=}")
        # print(f"{input_ids.shape=}")
        # print(f"{output.sequences.shape=}")
        # print(f"{len(generated_tokens)=}")

        # Compute conditional log probabilities
        with torch.no_grad():
            logits = model(output.sequences).logits
            logits_comp = logits[:, input_length - 1 : -1, :]
            log_probs = log_softmax(logits_comp, dim=-1)  # [1, 36, 32768]
            log_probs = (
                log_probs.gather(2, output.sequences[:, input_length:].unsqueeze(-1))
                .squeeze(-1)
                .sum(dim=1)
            )
            log_probs = log_probs.item()

        if log_probs > logprob_best:
            logprob_best = log_probs
        if raw_likelihood > raw_likelihood_best:
            raw_likelihood_best = raw_likelihood

        if iter % args.log_every == 0:
            to_log = {
                "logprob": log_probs,
                "raw_likelihood": raw_likelihood,
                "best_logprob": logprob_best,
                "best_raw_likelihood": raw_likelihood_best,
            }
            wandb.log(to_log)
        # Save history
        history.append(h)

        # if args.mask == "asap":
        # Incremental parser state must be reset after each generation
        # TODO don't understand why we need to reset if oracle processor gets reinit every time this for loop runs
        gad_oracle_processor.reset()

    # Save the history as JSON
    result_path = f"results/{args.split}"
    make_dir(f"{result_path}/{id}")
    with open(f"{result_path}/{id}/{id}_gcd.jsonl", "w") as f:
        for h in history:
            f.write(json.dumps(h))
            f.write("\n")


def main():
    # torch.set_float32_matmul_precision('high')
    args = parse_arguments()
    pprint.pprint(vars(args), indent=4)
    wandb.init(project="gad", config=vars(args))

    wandb.define_metric("logprob", summary="max")
    wandb.define_metric("raw_likelihood", summary="max")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, cache_dir="/trunk/model-hub"
    )
    # tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, cache_dir="/trunk/model-hub"
    )
    model.to(device)
    model.to(dtype=dtype)
    model.resize_token_embeddings(len(tokenizer))
    if args.do_compile:
        model = torch.compile(model)

    dataset = load_dataset(args.dataset, split=args.split)
    dataset = dataset.filter(lambda example: example["id"] == args.id)

    for data in dataset:
        id = data["id"]
        prompt = data["prompt"]
        grammar = data["grammar"]

        chat = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        eval_prob(model, tokenizer, id, prompt, grammar, args)

        print(f"Evaluation finished: {id}")


if __name__ == "__main__":
    main()