import argparse
import json
import os
import pprint
from typing import Optional, Tuple, Union
import time
import copy

import torch
import wandb
from datasets import load_dataset
from torch.nn.functional import log_softmax
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
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
    parser.add_argument(
        "--dataset", type=str, default="ebmoon/GAD-dataset", help="Dataset to use."
    )
    parser.add_argument(
        "--split", type=str, default="BV4", help="Dataset split to use."
    )
    parser.add_argument(
        "--id",
        type=str,
        default="find_inv_eq_bvlshr0_4bit",
        help="Dataset id (of split) to use.",
    )
    parser.add_argument(
        "--num_iter", type=int, default=50, help="Number of iterations for inference."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Huggingface model id.",
    )
    parser.add_argument(
        "--model_id_assist",
        type=str,
        default=None,
        help="Huggingface model id for speculative decoding.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on."
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="Data type for model."
    )
    # parser.add_argument(
    #     "--max_new_tokens",
    #     type=int,
    #     default=512,
    #     help="Maximum number of new tokens to generate.",
    # )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling."
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for generation.",
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Top-p sampling parameter."
    )
    parser.add_argument(
        "--top_k", type=int, default=0, help="Top-k sampling parameter."
    )
    # parser.add_argument(
    #     "--do_sample",
    #     action="store_true",
    #     help="Whether to use sampling for generation.",
    # )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search. 1 means no beam search.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default="none",
        choices=["gcd", "asap", "none"],
        help="Strategy for masking tokens.",
    )
    parser.add_argument(
        "--log_every", type=int, default=5, help="Log every n iterations."
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether to compile the model before running.",
    )
    parser.add_argument(
        "--tc",
        default=None,
        choices=["none", "medium", "high", "highest"],
        help="Tensor core precision.",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default="/trunk/model-hub",
    )
    parser.add_argument(
        "--use_prefix_cache",
        action="store_true",
        help="Whether to share KV cache for prompt across generations.",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--min_new_tokens",
        default=1,
        type=int,
    )
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


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    """
    Implements a variant of the Gumbel-Max trick, which is used to sample from a categorical distribution. It divides probs_sort by q, effectively adding Gumbel noise to the log-probabilities. The torch.argmax(..., dim=-1, keepdim=True) finds the index of the maximum value along the last dimension, which corresponds to the sampled category. The result is then converted to an integer type with .to(dtype=torch.int).
    """
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token(
    model, x: torch.Tensor, temperature: float, top_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    # assert input_pos.shape[-1] == 1
    # logits = model(x, input_pos)
    logits = model(x).logits
    return sample(logits, temperature=temperature, top_k=top_k)


@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    # batch_size: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    print(f"{prompt.shape=}")
    n_tokens = 0
    for _ in tqdm(range(max_new_tokens), desc="Generating tokens"):
        idx_next, probs = decode_one_token(
            model, x=prompt, temperature=temperature, top_k=top_k
        )
        # print(idx_next, probs.shape)
        n_tokens += 1
    print(probs)
    return n_tokens


def eval_prob(model, tokenizer, id, prompt, grammar_str, args):
    # Load EBNF grammar
    # grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)

    # Tokenize prompt into ids
    inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"]  # .to(model.device)
    wandb.run.summary["prompt_len"] = input_ids.shape[-1]
    attention_mask = inputs["attention_mask"]  # .to(model.device)

    # Inference Loop
    outputs = []
    history = []

    logprob_best = float("-inf")
    raw_likelihood_best = float("-inf")

    generate(model, prompt=input_ids, max_new_tokens=256, temperature=1.0, top_k=None)

    # for iter in tqdm(range(args.num_iter), desc="Running Inference"):
    #     if (args.mask == "gcd") or ((args.mask == "asap") and (iter == 0)):
    #         gad_oracle_processor, _, logits_processors = init_logits_proc(grammar)

    #     # Generate sequences
    #     output = model.generate(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         do_sample=args.do_sample,
    #         # pad_token_id=tokenizer.eos_token_id,
    #         eos_token_id=tokenizer.eos_token_id,
    #         max_new_tokens=args.max_new_tokens,
    #         top_p=args.top_p,
    #         top_k=args.top_k,
    #         temperature=args.temperature,
    #         logits_processor=logits_processors,
    #         repetition_penalty=args.repetition_penalty,
    #         num_return_sequences=1,
    #         return_dict_in_generate=True,
    #         output_scores=True,
    #         num_beams=args.num_beams,
    #     )
    #     # print(f"{input_ids.shape}=")
    #     # foo
    #     input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    #     generated_tokens = output.sequences[0, input_length:].tolist()
    #     raw_likelihood = gad_oracle_processor.oracle_trie.raw_likelihood(
    #         generated_tokens
    #     )
    #     h = {"tokens": generated_tokens, "raw_likelihood": raw_likelihood}

    # generations = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # print(f"{generations=}")
    # print(f"{raw_likelihood=}")
    # print(f"{input_ids.shape=}")
    # print(f"{output.sequences.shape=}")
    # print(f"{len(generated_tokens)=}")

    # Compute conditional log probabilities
    # with torch.no_grad():
    #     logits = model(output.sequences).logits
    #     logits_comp = logits[:, input_length - 1 : -1, :]
    #     log_probs = log_softmax(logits_comp, dim=-1)  # [1, 36, 32768]
    #     log_probs = (
    #         log_probs.gather(2, output.sequences[:, input_length:].unsqueeze(-1))
    #         .squeeze(-1)
    #         .sum(dim=1)
    #     )
    #     log_probs = log_probs.item()

    # if log_probs > logprob_best:
    #     logprob_best = log_probs
    # if raw_likelihood > raw_likelihood_best:
    #     raw_likelihood_best = raw_likelihood

    # if iter % args.log_every == 0:
    #     to_log = {
    #         "logprob": log_probs,
    #         "raw_likelihood": raw_likelihood,
    #         "best_logprob": logprob_best,
    #         "best_raw_likelihood": raw_likelihood_best,
    #     }
    #     wandb.log(to_log)
    # # Save history
    # history.append(h)

    # # if args.mask == "asap":
    # # Incremental parser state must be reset after each generation
    # # TODO don't understand why we need to reset if oracle processor gets reinit every time this for loop runs
    # gad_oracle_processor.reset()

    # Save the history as JSON
    # result_path = f"results/{args.split}"
    # make_dir(f"{result_path}/{id}")
    # with open(f"{result_path}/{id}/{id}_gcd.jsonl", "w") as f:
    #     for h in history:
    #         f.write(json.dumps(h))
    #         f.write("\n")


def compute_tokens_per_second(t0, n_tokens):
    """
    Compute the number of tokens processed per second.

    Args:
        t0 (float): The start time in seconds.
        tokens (int): The number of tokens processed.

    Returns:
        float: The number of tokens processed per second.
    """
    elapsed_time = time.time() - t0
    if elapsed_time > 0:
        return n_tokens / elapsed_time
    else:
        return float(
            "inf"
        )  # Return infinity if no time has elapsed to avoid division by zero


def main():
    args = parse_arguments()
    pprint.pprint(vars(args), indent=4)
    wandb.init(project="gad", config=vars(args))
    wandb.define_metric("logprob", summary="max")
    wandb.define_metric("raw_likelihood", summary="max")

    # Enable tensor cores.
    # TODO Does this do anything if model weights are already in bf16?
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-tensor-cores
    if args.tc is not None:
        torch.set_float32_matmul_precision(args.tc)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, cache_dir=args.model_cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        cache_dir=args.model_cache_dir,
        torch_dtype="auto",
    )

    print("Sanity checking dtype of model...")
    layer_count = 0
    for name, param in model.named_parameters():
        if layer_count >= 3:  # Check the first 3 layers
            break
        if "weight" in name or "bias" in name:
            print(f"Layer '{name}' has dtype: {param.dtype}")
            layer_count += 1
    print()

    # Speculative decoding
    # https://huggingface.co/docs/transformers/v4.46.0/llm_optims?spec-decoding=sampling&static-kv=advanced+usage%3A+control+Static+Cache#speculative-decoding
    # https://huggingface.co/blog/assisted-generation
    if args.model_id_assist is not None:
        assistant_model = AutoModelForCausalLM.from_pretrained(
            args.model_id_assist,
            device_map="auto",
            cache_dir=args.model_cache_dir,
            torch_dtype="auto",
        )
        # TODO support caching
        # TODO suport compile

    # TODO figure out what mode and fullgraph do
    # fullgraph checks for breaks in the computation graph
    if args.compile:
        if not args.use_prefix_cache:
            # see "basic usage" in https://huggingface.co/docs/transformers/v4.46.0/llm_optims?spec-decoding=sampling&static-kv=basic+usage%3A+generation_config#static-kv-cache-and-torchcompile
            model.generation_config.cache_implementation = "static"
        model.forward = torch.compile(
            model.forward, mode="reduce-overhead", fullgraph=True
        )
        # model.forward = torch.compile(
        #     model.forward, mode="max-autotune"
        # )
        # compiling just fwd instead of whole model is faster
        # model = torch.compile(
        #     model, mode="reduce-overhead", fullgraph=True
        # )

        # TODO spec decode + compile might not be supported. currently breaks
        if args.model_id_assist is not None:
            if not args.use_prefix_cache:
                assistant_model.generation_config.cache_implementation = "static"
            assistant_model.forward = torch.compile(
                assistant_model.forward, mode="reduce-overhead", fullgraph=True
            )

    # Prepare prompt
    dataset = load_dataset(args.dataset, split=args.split)
    dataset = dataset.filter(lambda example: example["id"] == args.id)
    id = dataset[0]["id"]
    prompt = dataset[0]["prompt"]
    grammar = dataset[0]["grammar"]
    messages = [
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if args.mask in ["gcd", "asap"]:
        grammar = IncrementalGrammarConstraint(grammar, "root", tokenizer)

    # Padding prompts to multiples of 8 might improve tensor core usage
    # https://huggingface.co/docs/transformers/v4.46.0/llm_optims?spec-decoding=sampling&static-kv=advanced+usage%3A+control+Static+Cache#static-kv-cache-and-torchcompile
    # https://github.com/huggingface/tokenizers/issues/991
    # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    # input_ids = tokenizer(
    #     input_text,
    #     return_tensors="pt",
    #     pad_to_multiple_of=8,
    #     padding=True,
    #     padding_side="left",
    # ).to(model.device)

    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

    ids, mask = input_ids.input_ids, input_ids.attention_mask
    ids_copy, mask_copy = ids.clone(), mask.clone()
    # HACK drop the final token add it back later to prevent HF from complaining
    # when using static cache
    ids, mask = ids[:, :-1], mask[:, :-1]
    prompt_length = ids.shape[1]
    print(f"{prompt_length=}")

    if args.use_prefix_cache:
        # https://huggingface.co/docs/transformers/v4.46.0/kv_cache#re-use-cache-to-continue-generation
        # https://huggingface.co/docs/transformers/v4.46.0/llm_optims?spec-decoding=sampling&static-kv=advanced+usage%3A+control+Static+Cache#static-kv-cache-and-torchcompile
        prompt_cache = StaticCache(
            config=model.config,
            batch_size=1,
            max_cache_len=prompt_length * 2,
            device=model.device,
            dtype=model.dtype,
        )
        with torch.no_grad():
            prompt_cache = model(
                ids, attention_mask=mask, past_key_values=prompt_cache
            ).past_key_values
            print(f"{type(prompt_cache)=}")

    n_tokens = 0
    warmup_iters = 5
    logits_processors = None
    # past_key_values = prompt_cache
    for iter in tqdm(range(args.num_iter)):
        # warm up GPUs before profiling
        if iter == warmup_iters:
            t0 = time.time()
        if args.use_prefix_cache:
            past_key_values = copy.deepcopy(prompt_cache)

        if (args.mask == "gcd") or (iter == 0 and args.mask == "asap"):
            gad_oracle_processor = GrammarAlignedOracleLogitsProcessor(grammar)
            inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
            logits_processors = LogitsProcessorList(
                [
                    inf_nan_remove_processor,
                    gad_oracle_processor,
                ]
            )

        outputs = model.generate(
            ids_copy,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens if (args.min_new_tokens > 1) else 1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=mask_copy,
            past_key_values=past_key_values if args.use_prefix_cache else None,
            assistant_model=assistant_model
            if (args.model_id_assist is not None)
            else None,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            logits_processor=logits_processors,
        )
        # print(f"{past_key_values.get_seq_length()=}")
        # for layer_idx in range(len(past_key_values.key_cache)):
        #     # In-place ops prevent breaking the static address
        #     past_key_values.key_cache[layer_idx][0, 0, prompt_length:].zero_()
        #     past_key_values.value_cache[layer_idx][0, 0, prompt_length:].zero_()
        # print(f"{past_key_values.get_seq_length()=}")

        # foo
        if iter >= warmup_iters:
            n_tokens += outputs.shape[1] - ids_copy.shape[-1]
        if args.mask in ["gcd", "asap"]:
            gad_oracle_processor.reset()

    tok_per_sec = compute_tokens_per_second(t0, n_tokens)
    print(f"{n_tokens=}")
    print(f"{tok_per_sec=}")
    wandb.run.summary["tokens_per_second"] = tok_per_sec
    foo

    past_key_values = StaticCache(
        config=model.config,
        batch_size=2,
        max_cache_len=4096,
        device=torch_device,
        dtype=model.dtype,
    )

    wandb.run.summary["prompt_len"] = prompt_length
    # Inference Loop
    outputs = []
    history = []

    logprob_best = float("-inf")
    raw_likelihood_best = float("-inf")

    t0 = time.time()
    n_tokens = generate(
        model, prompt=input_ids, max_new_tokens=512, temperature=1.0, top_k=None
    )

    tok_per_sec = compute_tokens_per_second(t0, n_tokens)
    wandb.run.summary["tokens_per_second"] = tok_per_sec

    # for data in dataset:
    #     id = data["id"]
    #     prompt = data["prompt"]
    #     grammar = data["grammar"]

    #     chat = [
    #         {"role": "user", "content": prompt},
    #     ]
    #     prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    #     eval_prob(model, tokenizer, id, prompt, grammar, args)

    #     print(f"Evaluation finished: {id}")


if __name__ == "__main__":
    main()
