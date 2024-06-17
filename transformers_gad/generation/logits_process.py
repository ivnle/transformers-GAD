import copy
import math
import pprint
import torch.nn.functional as F
import os

import torch
import logging
from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)

from transformers.utils import add_start_docstrings

class GrammarConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, grammar_constraint, parse_start_index=None, logger=None):
        self.grammar_constraint = grammar_constraint
        self.last_size = None
        self.batch_accept_states = None
        self.parse_start_index = parse_start_index
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.accepted_indices_history = []  # To store indices of accepted tokens
        self.accepted_tokens_history = []
        self.acceptance_raw_scores_history = []
        self.acceptance_logits_history = []
        self.acceptance_details_history = []

    def mask_scores(self, scores, device):
        # resolve each stack to a tensor of True/False for each token
        # indicating acceptance
        # acceptance = self.grammar_acceptor.filter_vocab(self.stacks, device)
        acceptance = self.grammar_constraint.batch_filter_vocab(
            self.batch_accept_states, device
        )

        self.get_accepted_tokens(acceptance)
        self.get_detailed_history(acceptance, scores)
        # print(f"accpetance_details_history: {self.acceptance_details_history}")

        # store raw scores and logits for acceptance tokens before applying the mask
        # First, calculate the logits for the entire scores tensor
        logits = F.softmax(scores, dim=-1)

        # For raw scores of accepted tokens
        accepted_raw_scores = scores[acceptance].clone().detach()
        self.acceptance_raw_scores_history.append(accepted_raw_scores.cpu())

        # For logits of accepted tokens
        accepted_logits = logits[acceptance].clone().detach()
        self.acceptance_logits_history.append(accepted_logits.cpu())

        # Scores to -inf where False
        scores[~acceptance] = float('-inf')

    # TODO: batching
    def process_gcd_scores(self, input_ids, scores):
        """
        :param input_ids:
        :param scores:
        :return:
        """

        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        if self.batch_accept_states is None:
            self.batch_accept_states = [
                # self.grammar_constraint.init_stacks()
                copy.deepcopy(
                    self.grammar_constraint.string_recognizer.get_initial_accept_state()
                )
                for _ in range(len(input_ids))
            ]

        self.batch_accept_states = self.grammar_constraint.advance_token_ids(
            input_ids, self.batch_accept_states, self.parse_start_index
        )

        self.mask_scores(scores, scores.device)
        return scores

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.process_gcd_scores(input_ids, scores)

    def get_accepted_tokens(self, acceptance):
        """
        Stores the indices of accepted tokens and their corresponding string values for each item in the batch.

        Parameters:
        - acceptance (torch.Tensor): A boolean tensor indicating accepted tokens for each item in the batch.
        """
        batch_size, vocab_size = acceptance.shape
        acceptance_np = acceptance.cpu().numpy()
        accepted_x, accepted_y = acceptance_np.nonzero()

        # Initialize the dictionary with empty lists for indices
        accepted_token_indices = {i: [] for i in range(batch_size)}
        for x, y in zip(accepted_x, accepted_y):
            accepted_token_indices[x].append(y)

        # Store accepted indices for history
        self.accepted_indices_history.append(accepted_token_indices)

        # Convert token IDs to tokens
        accepted_tokens = {
            i: [self.grammar_constraint.tokenizer.decode([token_id]) for token_id in token_ids]
            for i, token_ids in accepted_token_indices.items()
        }

        # Store accepted tokens for history
        self.accepted_tokens_history.append(accepted_tokens)

    def get_detailed_history(self, acceptance, scores):
        """
        Processes and stores information for accepted tokens including their IDs, tokens,
        raw scores, and logits.

        Parameters:
        - acceptance (torch.Tensor): A boolean tensor indicating accepted tokens for each item in the batch.
        - scores (torch.Tensor): The raw scores from the model output.
        """
        logits = F.softmax(scores, dim=-1)

        # Initializing the list to store detailed information for each step
        detailed_accepted_info = []

        for batch_index in range(acceptance.size(0)):  # Iterate over batch items
            accepted_info = []
            accepted_indices = acceptance[batch_index].nonzero().squeeze(-1)

            for idx in accepted_indices:
                token_id = idx.item()
                raw_score = scores[batch_index, idx].item()
                logit = logits[batch_index, idx].item()
                token = self.grammar_constraint.tokenizer.decode([token_id])

                # Store detailed information as a dictionary
                accepted_info.append({
                    "token_id": token_id,
                    "token": str(token),
                    "raw_score": raw_score,
                    "raw_logit": logit
                })

            detailed_accepted_info.append(accepted_info)

        # Store this detailed information in the history
        self.acceptance_details_history.append(detailed_accepted_info)

    def get_history(self):
        return (self.accepted_tokens_history, self.accepted_indices_history,
                self.acceptance_raw_scores_history, self.acceptance_logits_history, self.acceptance_details_history)

    def get_acceptance_details_history(self):
        return self.acceptance_details_history



if __name__ == "__main__":
    from transformers import AutoTokenizer
    import sys
    import os

    from transformers_gad.grammar_utils import IncrementalGrammarConstraint

    # set logging level
    # logging.basicConfig(level=logging.DEBUG)

    test_file = "/nobackup2/yf/mila/GD/examples/grammars/string_start_w_1_all_0.ebnf"

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", cache_dir="/nobackup2/yf/mila/GD_caches")

    # Load grammar
    with open(test_file, "r") as file:
        grammar_str = file.read()

    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    print(f"grammar_processor: {grammar_processor}")

