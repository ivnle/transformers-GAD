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
        self.last_size = None
        self.grammar_constraint = grammar_constraint
        self.batch_stacks = None
        self.parse_start_index = None
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.accepted_indices_history = []  # To store indices of accepted tokens
        self.accepted_tokens_history = []
        self.acceptance_raw_scores_history = []
        self.acceptance_logits_history = []
        self.acceptance_details_history = []
        self.input_ids_history = []

    def mask_scores(self, scores, device):
        # resolve each stack to a tensor of True/False for each token
        # indicating acceptance
        # acceptance = self.grammar_acceptor.filter_vocab(self.stacks, device)
        acceptance = self.grammar_constraint.batch_filter_vocab(
            self.batch_stacks, device
        )

        self.get_accepted_tokens(acceptance)
        self.get_detailed_history(acceptance, scores)

        # store raw scores and logits for acceptance tokens before applying the mask
        # First, calculate the logits for the entire scores tensor
        logits = F.softmax(scores, dim=-1)

        # We'll use torch.nonzero to find the indices of accepted tokens
        # Note: This operation flattens the acceptance mask, so we need to adjust indices accordingly
        accepted_indices = acceptance.nonzero()

        # For raw scores of accepted tokens
        accepted_raw_scores = scores[acceptance].clone().detach()
        self.acceptance_raw_scores_history.append(accepted_raw_scores.cpu())

        # For logits of accepted tokens
        accepted_logits = logits[acceptance].clone().detach()
        self.acceptance_logits_history.append(accepted_logits.cpu())


        # Scores to -inf where False
        scores[~acceptance] = -math.inf

    # TODO: batching
    def process_gcd_scores(self, input_ids, scores):
        """
        :param input_ids:
        :param scores:
        :return:
        """
        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        self.input_ids_history.append(input_ids)

        if self.batch_stacks is None:
            self.batch_stacks = [
                # self.grammar_constraint.init_stacks()
                copy.deepcopy(self.grammar_constraint.grammar.stacks)
                for _ in range(len(input_ids))
            ]

        if os.getenv("DEBUG_MODE") == "True":
            print("-" * 80)

            self.logger.debug("input_ids: \n" + pprint.pformat(input_ids))
            # logger.debug("scores: \n" + pprint.pformat(scores))
            self.logger.debug("last_size: \n" + pprint.pformat(self.last_size))
            self.logger.debug(
                "num of stacks: \n"
                + pprint.pformat([len(stack) for stack in self.batch_stacks])
            )
            self.logger.debug("stacks: \n" + pprint.pformat(self.batch_stacks))

        self.batch_stacks = self.grammar_constraint.advance_token_ids(
            input_ids, self.batch_stacks, self.parse_start_index
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
                    "token": token,
                    "raw_score": raw_score,
                    "raw_logit": logit
                })

            detailed_accepted_info.append(accepted_info)

        # Store this detailed information in the history
        self.acceptance_details_history.append(detailed_accepted_info)

    def get_history(self):
        return (self.accepted_tokens_history, self.accepted_indices_history,
                self.acceptance_raw_scores_history, self.acceptance_logits_history, self.acceptance_details_history)

    def get_input_ids_history(self):
        return self.input_ids_history

class GrammarAlignedLogitsProcessor(LogitsProcessor):
    def __init__(self, grammar_constraint, parse_start_index=None, logger=None):
        self.last_size = None
        self.grammar_constraint = grammar_constraint
        self.batch_stacks = None
        self.parse_start_index = None
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.accepted_indices_history = []  # To store indices of accepted tokens
        self.accepted_tokens_history = []
        self.acceptance_raw_scores_history = []
        self.acceptance_logits_history = []
        self.acceptance_details_history = []
        self.input_ids_history = []
        self.parse_start_index = None
        self.adjusted_acceptance_details_history = []

    def mask_scores(self, input_ids, scores, device):
        # resolve each stack to a tensor of True/False for each token
        # indicating acceptance
        # acceptance = self.grammar_acceptor.filter_vocab(self.stacks, device)
        acceptance = self.grammar_constraint.batch_filter_vocab(
            self.batch_stacks, device
        )

        self.get_accepted_tokens(acceptance)
        self.get_detailed_history(acceptance, scores)

        # store raw scores and logits for acceptance tokens before applying the mask
        # First, calculate the logits for the entire scores tensor
        logits = F.softmax(scores, dim=-1)

        # For raw scores of accepted tokens
        accepted_raw_scores = scores[acceptance].clone().detach()
        self.acceptance_raw_scores_history.append(accepted_raw_scores.cpu())

        # For logits of accepted tokens
        accepted_logits = logits[acceptance].clone().detach()
        self.acceptance_logits_history.append(accepted_logits.cpu())

        # help me fill in the code here to assign score to acceptance tokens with their corresponding log(logits) * theta
        sequence_to_theta, avg_sequence_to_theta, w_s_R, w_s_T, w_s_Z = self.get_reweigh_factor()


        self.apply_theta_adjustments(input_ids, acceptance, scores, w_s_R, w_s_T, w_s_Z)
        self.get_adjusted_detailed_history(acceptance, scores)
        # Scores to -inf where False
        scores[~acceptance] = -math.inf

    def apply_theta_adjustments(self, input_ids, acceptance, scores, w_s_R, w_s_T, w_s_Z):
        logits = F.softmax(scores, dim=-1)
        log_logits = torch.log(logits)

        # Assume theta_values is a tensor or function providing theta for each token
        # This is simplified; the actual implementation will depend on how theta is determined

        for batch_index in range(acceptance.size(0)):
            accepted_indices = acceptance[batch_index].nonzero().squeeze(-1)

            for idx in accepted_indices:
                token_id = idx.item()
                # print(f"token_id: {token_id}")
                logit = logits[batch_index, idx].item()
                # print(f"logit: {logit}")
                log_logit = log_logits[batch_index, idx].item()
                # print(f"log_logit: {log_logit}")
                # Assume a method to get theta for this specific token
                theta = self.get_theta_for_token(input_ids, w_s_R, w_s_T, w_s_Z)
                log_theta = torch.log(theta)
                # print(f"log_theta: {log_theta}")
                # Calculate adjusted score
                adjusted_score = log_logit + log_theta
                # print(f"adjusted_score: {adjusted_score}")

                # Here you could either adjust the score in-place or store this information for later use
                scores[batch_index, idx] = adjusted_score  # Direct adjustment


    def get_theta_for_token(self, input_ids, w_s_R, w_s_T, w_s_Z):
        # TODO: Implement a method to get theta for a specific token, generalized to a specified tree, now only apply for 01 strings
        generated_start_idx = self.parse_start_index
        # Ensure there's at least one generated token
        if input_ids.size(1) > generated_start_idx:
            # Extract the first generated token ID for each sequence in the batch
            first_generated_token_ids = input_ids[:, generated_start_idx]

            # Initialize an empty tensor for theta values
            theta = torch.empty(first_generated_token_ids.size(0), dtype=torch.float)

            # Loop over each token to determine the correct theta value
            for i, token_id in enumerate(first_generated_token_ids):
                if token_id == 28740:
                    theta[i] = w_s_T
                elif token_id == 28734:
                    theta[i] = w_s_Z
                else:
                    # If the token ID does not match any expected value, raise an error
                    raise ValueError(f"Unexpected input ID: {token_id.item()}")
        else:
            # If there are no generated tokens, use the default w_s_R for all sequences in the batch
            theta = torch.full((input_ids.size(0),), w_s_R, dtype=torch.float)

        return theta


    # TODO: batching
    def process_gad_scores(self, input_ids, scores):
        self.input_ids_history.append(input_ids)
        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        if self.batch_stacks is None:
            self.batch_stacks = [
                # self.grammar_constraint.init_stacks()
                copy.deepcopy(self.grammar_constraint.grammar.stacks)
                for _ in range(len(input_ids))
            ]

        if self.parse_start_index is None:
            self.parse_start_index = input_ids.size(1)  # Assuming the initial size is the prompt length

        self.batch_stacks = self.grammar_constraint.advance_token_ids(
            input_ids, self.batch_stacks, self.parse_start_index
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
                    "token": token,
                    "raw_score": raw_score,
                    "raw_logit": logit
                })

            detailed_accepted_info.append(accepted_info)

        # Store this detailed information in the history
        self.acceptance_details_history.append(detailed_accepted_info)

    def get_adjusted_detailed_history(self, acceptance, scores):
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
                    "token": token,
                    "raw_score": raw_score,
                    "raw_logit": logit
                })

            detailed_accepted_info.append(accepted_info)

        # Store this detailed information in the history
        self.adjusted_acceptance_details_history.append(detailed_accepted_info)

    def get_history(self):
        return (self.accepted_tokens_history, self.accepted_indices_history,
                self.acceptance_raw_scores_history, self.acceptance_logits_history, self.acceptance_details_history, self.adjusted_acceptance_details_history)

    def get_acceptance_details_history(self):
        return self.acceptance_details_history

    def get_input_ids_history(self):
        return self.input_ids_history

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

