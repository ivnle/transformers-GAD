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

class GrammarAlignedGroundTruthLogitsProcessor(LogitsProcessor):
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
        self.acceptance_details_history = [] # history for building oracle tree
        self.input_ids_history = []
        self.adjusted_acceptance_details_history = [] # record after applying score adjustment to unbiased distribution
        # TODO: fix generated tokens
        self.generated_tokens = []

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


        self.apply_theta_adjustments(input_ids, acceptance, scores, sequence_to_theta)
        self.get_adjusted_detailed_history(acceptance, scores)
        # Scores to -inf where False
        scores[~acceptance] = float('-inf')

    def apply_theta_adjustments(self, input_ids, acceptance, scores, sequence_to_theta):
        logits = F.softmax(scores, dim=-1)
        log_logits = torch.log(logits)

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
                theta = self.get_theta_for_token(input_ids, token_id, sequence_to_theta)

                if not isinstance(theta, torch.Tensor):
                    theta = torch.tensor(theta, dtype=torch.float)

                log_theta = torch.log(theta)
                # print(f"log_theta: {log_theta}")
                # Calculate adjusted score
                adjusted_score = log_logit + log_theta
                # print(f"adjusted_score: {adjusted_score}")

                # Here you could either adjust the score in-place or store this information for later use
                scores[batch_index, idx] = adjusted_score  # Direct adjustment

    def get_theta_for_token(self, input_ids, token_id, sequence_to_theta):
        # TODO: generalize to a specified tree, now only apply for 01 strings
        generated_start_idx = self.parse_start_index

        # Extract the generated tokens up to the current token
        generated_tokens = input_ids[:, generated_start_idx:]

        # Append the current token_id to form the sequence to check
        sequence_to_check = torch.cat((generated_tokens, torch.tensor([[token_id]])), dim=1)
        sequence_list = sequence_to_check.squeeze().tolist()
        # Ensure the list is iterable (important for single-element cases)
        if isinstance(sequence_list, int):
            sequence_list = [sequence_list]

        # Convert the list to a tuple for dictionary lookup
        sequence_tuple = tuple(sequence_list)
        # print(f"sequence_tuple: {sequence_tuple}")

        # Lookup the sequence in the dictionary
        if sequence_tuple in sequence_to_theta:
            # print(f"theta: {sequence_to_theta[sequence_tuple]}")
            return sequence_to_theta[sequence_tuple]
        # TODO: note that this should be EOS token
        elif sequence_tuple[-1] == 2:
            return 1
        else:
            raise ValueError(f"Unexpected sequence: {sequence_tuple}")


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

    def get_input_ids_history(self):
        return self.input_ids_history

    def get_reweigh_factor(self):
        p_1_1 = 3.175825986545533e-05
        p_1_0 = 2.0020976080559194e-05
        p_2_1_given_1 = 0.17334003746509552
        p_2_0_given_1 = 0.22081588208675385
        p_2_0_given_0 = 0.25583988428115845
        p_3_1_given_11 = 0.2095051109790802
        p_3_0_given_11 = 0.06769692897796631
        p_3_1_given_10 = 0.1298251450061798
        p_3_0_given_10 = 0.11201044172048569
        p_3_0_given_00 = 0.21392203867435455

        # Calculating P values using the corrected interpretation for time steps
        P_111 = p_1_1 * p_2_1_given_1 * p_3_1_given_11
        P_110 = p_1_1 * p_2_1_given_1 * p_3_0_given_11
        P_101 = p_1_1 * p_2_0_given_1 * p_3_1_given_10
        P_100 = p_1_1 * p_2_0_given_1 * p_3_0_given_10
        P_000 = p_1_0 * p_2_0_given_0 * p_3_0_given_00

        # Calculating w^s values
        w_s_r = P_111 + P_110 + P_101 + P_100 + P_000
        w_s_a = (p_2_1_given_1) * ((p_3_1_given_11) + (p_3_0_given_11)) + (p_2_0_given_1) * (
                    (p_3_1_given_10) + (p_3_0_given_10))
        w_s_b = (p_3_1_given_11) + (p_3_0_given_11)
        w_s_c = (p_3_1_given_10) + (p_3_0_given_10)
        w_s_d = p_2_0_given_0 * p_3_0_given_00
        w_s_e = p_3_0_given_00
        w_s_R = w_s_r
        w_s_T = (w_s_a + w_s_b + w_s_c)/3
        w_s_Z = (w_s_d + w_s_e)/2
        # Building the dictionary
        success_rates = {
            "root": w_s_r,
            "a": w_s_a,
            "b": w_s_b,
            "c": w_s_c,
            "d": w_s_d,
            "e": w_s_e,
            "R": w_s_R,
            "T": w_s_T,
            "Z": w_s_Z
        }

        # Display the success rates
        # for non_terminal, success_rate in success_rates.items():
        #     print(f"{non_terminal}: {success_rate}")

        theta_1_1 = w_s_a / w_s_r
        theta_1_0 = w_s_d / w_s_r
        theta_2_1_given_1 = w_s_b / w_s_a
        theta_2_0_given_1 = w_s_c / w_s_a
        theta_2_0_given_0 = w_s_e / w_s_d
        theta_3_1_given_11 = 1 / w_s_b
        theta_3_0_given_11 = 1 / w_s_b
        theta_3_1_given_10 = 1 / w_s_c
        theta_3_0_given_10 = 1 / w_s_c
        theta_3_0_given_00 = 1 / w_s_e

        # Input IDs
        id_1 = 28740
        id_0 = 28734

        # Mapping input IDs to theta values
        sequence_to_theta = {
            (id_1,): theta_1_1,
            (id_0,): theta_1_0,
            (id_1, id_1): theta_2_1_given_1,
            (id_1, id_0): theta_2_0_given_1,
            (id_0, id_0): theta_2_0_given_0,
            (id_1, id_1, id_1): theta_3_1_given_11,
            (id_1, id_1, id_0): theta_3_0_given_11,
            (id_1, id_0, id_1): theta_3_1_given_10,
            (id_1, id_0, id_0): theta_3_0_given_10,
            (id_0, id_0, id_0): theta_3_0_given_00,
        }

        avg_sequence_to_theta = {
            (id_1,): w_s_R,
            (id_0,): w_s_R,
            (id_1, id_1): w_s_T,
            (id_0, id_0): w_s_Z,
            (id_1, id_1, id_1): w_s_T,
            (id_1, id_1, id_0): w_s_T,
            (id_1, id_0, id_1): w_s_T,
            (id_1, id_0, id_0): w_s_T,
            (id_0, id_0, id_0): w_s_Z,
        }
        return sequence_to_theta, avg_sequence_to_theta, w_s_R, w_s_T, w_s_Z