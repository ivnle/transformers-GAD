import copy
import math
import torch.nn.functional as F

import torch
import logging
from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)
from transformers.utils import add_start_docstrings
from transformers_gad.oracle.oracle_trie import Trie, update_oracle_trie
from transformers_gad.token_grammar_recognizer import IncrementalGrammarConstraint

class GrammarAlignedOracleLogitsProcessor(LogitsProcessor):
    def __init__(self, grammar_constraint, oracle_trie=Trie(), parse_start_index=None, save_log=False):
        # Parser variables
        self.grammar_constraint = grammar_constraint
        self.batch_accept_states = None
        self.parse_start_index = parse_start_index

        # ASAp oracle trie
        self.oracle_trie = oracle_trie

        # To start with a longer prefix in enumerative search
        self.generate_start_index = None
        self.generated_tokens = None

        # Generation Log
        self.save_log = save_log
        self.history = []

    def mask_scores(self, scores, device):
        """
        resolve each stack to a tensor of True/False for each token
        indicating acceptance
        """
        masked_scores = scores.clone()
        acceptance = self.grammar_constraint.batch_filter_vocab(
            self.batch_accept_states, device
        )
        
        if self.save_log:
            self.store_detailed_history(acceptance, scores)
        
        # Scores to -inf where False
        masked_scores[~acceptance] = -math.inf

        return masked_scores

    def process_scores(self, input_ids, scores):
        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        if self.batch_accept_states is None:
            self.batch_accept_states = [
                copy.deepcopy(
                    self.grammar_constraint.string_recognizer.get_initial_accept_state()
                )
                for _ in range(len(input_ids))
            ]

        # assume the generation starts from the same index
        if self.generate_start_index is None:
            # the default is the end of input sequence of tokens
            self.generate_start_index = self.parse_start_index \
                if self.parse_start_index else input_ids.size(1)
        self.generated_tokens = input_ids[:, self.generate_start_index:]

        # Advance parser states
        self.batch_accept_states = self.grammar_constraint.advance_token_ids(
            input_ids, self.batch_accept_states, self.parse_start_index
        )

        masked_scores = self.mask_scores(scores, scores.device)
        return masked_scores

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.process_scores(input_ids, scores)

    def reset_parser(self):
        self.batch_parsing_states = None
        if isinstance(self.grammar_constraint, IncrementalGrammarConstraint):
            self.grammar_constraint.reset()

    def get_accepted_tokens(self, acceptance):
        """
        Get the indices of accepted tokens and their corresponding string values for each item in the batch.

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

        # Convert token IDs to tokens
        accepted_tokens = {
            i: [self.grammar_constraint.tokenizer.decode([token_id]) for token_id in token_ids]
            for i, token_ids in accepted_token_indices.items()
        }

        return accepted_tokens

    def store_detailed_history(self, acceptance, scores):
        """
        Processes and stores information for accepted tokens including their IDs, tokens,
        raw scores, and logits.

        Parameters:
        - acceptance (torch.Tensor): A boolean tensor indicating accepted tokens for each item in the batch.
        - scores (torch.Tensor): The raw scores from the model output.
        - adjusted_scores (torch.Tensor): The adjusted scores after applying expected future grammaticality.
        """
        likelihoods = F.softmax(scores, dim=-1)

        # Initializing the list to store detailed information for each step
        batch_accepted_info = []

        for batch_index in range(acceptance.size(0)):  # Iterate over batch items
            accepted_info = []
            accepted_indices = acceptance[batch_index].nonzero().squeeze(-1)

            for idx in accepted_indices:
                token_id = idx.item()
                raw_score = scores[batch_index, idx].item()
                likelihood = likelihoods[batch_index, idx].item()
                token = self.grammar_constraint.tokenizer.decode([token_id])

                # Store detailed information as a dictionary
                accepted_info.append({
                    "token_id": token_id,
                    "token": str(token),
                    "raw_score": raw_score,
                    "raw_likelihood": likelihood
                })

            batch_accepted_info.append(accepted_info)

        # Store this detailed information in the history
        self.history.append(batch_accepted_info)

    def acceptance_detailed_history(self):
        return self.history
