import copy
import math
import pprint

import torch
import logging
from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)
from transformers.utils import add_start_docstrings

# logger = logging.getLogger(__name__)


class GrammarConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, grammar_constraint, parse_start_index=None):
        self.last_size = None
        self.grammar_constraint = grammar_constraint
        self.batch_stacks = None
        self.parse_start_index = None


    def mask_logits(self, logits, device):
        # resolve each stack to a tensor of True/False for each token
        # indicating acceptance
        # acceptance = self.grammar_acceptor.filter_vocab(self.stacks, device)
        acceptance = self.grammar_constraint.batch_filter_vocab(
            self.batch_stacks, device
        )
        # acceptance is a tensor of shape (batch_size, vocab_size)
        # get the indices of the accepted tokens
        # do the following operation only in debug mode

        ##### debug mode #####
        # convert acceptance to numpy array
        batch_size, vocab_size = acceptance.shape
        acceptance_np = acceptance.cpu().numpy()
        accepted_x, accepted_y = acceptance_np.nonzero()
        # dict of {batch_index: [accepted_token_indices]}
        # initialize the dict with empty list
        accepted_token_indices = {i: [] for i in range(batch_size)}
        for x, y in zip(accepted_x, accepted_y):
            accepted_token_indices[x].append(y)
        logger.debug("Accepted token indices for the current batch:")
        logger.debug("\n" + pprint.pformat(accepted_token_indices))
        # convert token_ids to tokens
        accepted_tokens = {
            i: [
                self.grammar_constraint.tokenizer.decode([token_id])
                for token_id in token_ids
            ]
            for i, token_ids in accepted_token_indices.items()
        }
        logger.debug("Accepted tokens for the current batch:")
        logger.debug("\n" + pprint.pformat(accepted_tokens))

        # Logits to -inf where False
        logits[~acceptance] = -math.inf

    # TODO: batching
    def process_logits(self, input_ids, scores):
        """
        :param input_ids:
        :param scores:
        :return:
        """
        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        if self.batch_stacks is None:
            self.batch_stacks = [
                # self.grammar_constraint.init_stacks()
                copy.deepcopy(self.grammar_constraint.grammar.stacks)
                for _ in range(len(input_ids))
            ]

        if os.getenv("DEBUG_MODE") == "True":
            print("-" * 80)

        logger.debug("input_ids: \n" + pprint.pformat(input_ids))
        # logger.debug("scores: \n" + pprint.pformat(scores))
        logger.debug("last_size: \n" + pprint.pformat(self.last_size))
        logger.debug(
            "num of stacks: \n"
            + pprint.pformat([len(stack) for stack in self.batch_stacks])
        )
        logger.debug("stacks: \n" + pprint.pformat(self.batch_stacks))

        self.batch_stacks = self.grammar_constraint.advance_token_ids(
            input_ids, self.batch_stacks, self.parse_start_index
        )

        self.mask_logits(scores, scores.device)
        return scores

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores)

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import sys
    import os

    sys.path.append('/nobackup2/yf/mila/GD/transformers_gad')
    os.environ["DEBUG_MODE"] = "True"
    from transformers_gad.grammar_utils import IncrementalGrammarConstraint

    # set logging level
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler('/nobackup2/yf/mila/GD/log/debug.log')
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    test_file = "/nobackup2/yf/mila/GD/examples/grammars/string_start_w_1_all_0.ebnf"

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", cache_dir="/nobackup2/yf/mila/GD_caches")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", cache_dir="/nobackup2/yf/mila/GD_caches")

    # Load grammar
    with open(test_file, "r") as file:
        grammar_str = file.read()

    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
    grammar_processor_mask_logits = grammar_processor.mask_logits
    print("masked_logits: ", grammar_processor_mask_logits)
    grammar_processor_process_logits = grammar_processor.process_logits
    print("process_logits: ", grammar_processor_process_logits)

    print(f"grammar: {grammar}")
    print(f"grammar_processor: {grammar_processor}")

    prompt = "Generate a binary string of length 2."
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]


    output = model.generate(
        input_ids,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # num_beams=args.num_beams,
        max_new_tokens=10,
        top_p=1.0,
        # top_k=args.top_k,
        temperature=0.7,
        logits_processor=[grammar_processor],
        repetition_penalty=1,
        # early_stopping=True,
        num_return_sequences=1
    )

    print("output: ", output)
    print("tokenizer.decode(output[0]): ", tokenizer.decode(output[0]))