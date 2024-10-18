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

class GrammarAlignedOracleLogitsProcessor(LogitsProcessor):
    def __init__(self, grammar_constraint, oracle_trie, parse_start_index=None):
        self.grammar_constraint = grammar_constraint
        self.oracle_trie = oracle_trie
        self.last_size = None
        self.batch_accept_states = None
        self.parse_start_index = parse_start_index
        self.generate_start_index = None
        self.accepted_indices_history = []  # To store indices of accepted tokens
        self.accepted_tokens_history = []
        self.acceptance_raw_scores_history = []
        self.acceptance_likelihoods_history = []
        self.acceptance_details_history = [] # history for building oracle tree
        self.adjusted_acceptance_details_history = [] # record after applying score adjustment to unbiased distribution
        self.generated_tokens = None

    def mask_scores(self, input_ids, scores, device):
        """
        resolve each stack to a tensor of True/False for each token
        indicating acceptance
        """
        acceptance = self.grammar_constraint.batch_filter_vocab(
            self.batch_accept_states, device
        )

        self.get_accepted_tokens(acceptance)
        self.get_detailed_history(acceptance, scores, self.acceptance_details_history)

        # store raw scores and logits for acceptance tokens before applying the mask
        # First, calculate the logits for the entire scores tensor
        likelihoods = F.softmax(scores, dim=-1)

        # For raw scores of accepted tokens
        accepted_raw_scores = scores[acceptance].clone().detach()
        self.acceptance_raw_scores_history.append(accepted_raw_scores.cpu())

        # For logits of accepted tokens
        accepted_likelihoods = likelihoods[acceptance].clone().detach()
        self.acceptance_likelihoods_history.append(accepted_likelihoods.cpu())

        current_parent = self.oracle_trie.search_last_parent(self.generated_tokens)
        self.apply_oracle_adjustments(acceptance, scores, current_parent)
        self.get_detailed_history(acceptance, scores, self.adjusted_acceptance_details_history)
        
        # Scores to -inf where False
        scores[~acceptance] = float('-inf')

    def apply_oracle_adjustments(self, acceptance, scores, current_parent):
        likelihoods = F.softmax(scores, dim=-1)
        log_likelihoods = torch.log(likelihoods)

        for batch_index in range(acceptance.size(0)):
            accepted_indices = acceptance[batch_index].nonzero().squeeze(-1)

            for idx in accepted_indices:
                token_id = idx.item()
                likelihood = likelihoods[batch_index, idx].item()
                log_likelihood = log_likelihoods[batch_index, idx].item()
                
                # Get theta (log of expected future grammaticality) for this specific token
                success_rate = self.oracle_trie.get_success_rate_for_candidate_token(current_parent, token_id)

                if not isinstance(success_rate, torch.Tensor):
                    success_rate = torch.tensor(success_rate, dtype=torch.float)
                log_theta = torch.log(success_rate)
                
                # Calculate adjusted score
                adjusted_score = log_likelihood + log_theta

                # Here you could either adjust the score in-place or store this information for later use
                scores[batch_index, idx] = adjusted_score

    # TODO: batching
    def process_gad_scores(self, input_ids, scores):
        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        if self.batch_accept_states is None:
            self.batch_accept_states = [
                copy.deepcopy(
                    self.grammar_constraint.string_recognizer.get_initial_accept_state()
                )
                for _ in range(len(input_ids))
            ]

        if self.generate_start_index is None:
            self.generate_start_index = self.parse_start_index \
                if self.parse_start_index else input_ids.size(1)
        self.generated_tokens = input_ids[:, self.generate_start_index:]

        self.batch_accept_states = self.grammar_constraint.advance_token_ids(
            input_ids, self.batch_accept_states, self.parse_start_index
        )
        self.mask_scores(input_ids, scores, scores.device)
        return scores

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.process_gad_scores(input_ids, scores)

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

    def get_detailed_history(self, acceptance, scores, history):
        """
        Processes and stores information for accepted tokens including their IDs, tokens,
        raw scores, and logits.

        Parameters:
        - acceptance (torch.Tensor): A boolean tensor indicating accepted tokens for each item in the batch.
        - scores (torch.Tensor): The raw scores from the model output.
        """
        likelihoods = F.softmax(scores, dim=-1)

        # Initializing the list to store detailed information for each step
        detailed_accepted_info = []

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

            detailed_accepted_info.append(accepted_info)

        # Store this detailed information in the history
        history.append(detailed_accepted_info)

    def get_history(self):
        return (self.accepted_tokens_history, self.accepted_indices_history,
                self.acceptance_raw_scores_history, self.acceptance_likelihoods_history, self.acceptance_details_history, self.adjusted_acceptance_details_history)

    def acceptance_details_history(self):
        return self.acceptance_details_history

    def adjusted_acceptance_details_history(self):
        return self.adjusted_acceptance_details_history
