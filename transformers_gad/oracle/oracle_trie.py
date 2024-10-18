from graphviz import Digraph
import torch
import pickle

class TrieNode:
    def __init__(self, token_id=None, token=None, raw_likelihood=None, raw_score=None, tokenizer=None):
        self.children = {}
        self.parent = None
        self.token_id = token_id
        self.token = token
        self.raw_likelihood = raw_likelihood
        self.raw_score = raw_score
        self.success_rate = 1

        if tokenizer is None:
            self.eos_token_id = 2
        else:
            self.eos_token_id = tokenizer.eos_token_id

        self.is_end_of_sequence = False
        self.is_start_of_sequence = False

    def __repr__(self):
        parent_token_id = 'None (Root Node)' if self.parent is None else self.parent.token_id
        return (f"TrieNode(token_id={self.token_id}, token='{self.token}', "
                f"raw_likelihood={self.raw_likelihood}, raw_score={self.raw_score}, children={list(self.children.keys())}, "
                f"parent={parent_token_id}, success rate={self.success_rate})") # TODO: add prefix

class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.root.is_start_of_sequence = True

    def insert(self, parent_node: TrieNode, child_node: TrieNode):
        """
        Insert child_node into parent_node's children dictionary        
        """
        if child_node.token_id not in parent_node.children:
            parent_node.children[child_node.token_id] = child_node
            child_node.parent = parent_node 
            
            if child_node.token_id == self.eos_token_id:
                child_node.is_end_of_sequence = True
            
            # update the success rate of the parent node
            return self.update_success_rate(parent_node)
        else:
            return 0

    def update_success_rate(self, node: TrieNode):
        """
        Re-compute the success rate from the updated success rate of children
        """
        if node and node.children:
            total_success_rate = sum(child.raw_likelihood * child.success_rate for child in node.children.values())
            node.success_rate = total_success_rate
            
            # Get how much of unexplored nodes are covered with this update
            updated_rate = node.success_rate - total_success_rate

            # Back propagate the success rate
            if node.parent:
                return self.update_success_rate(node.parent)
            
            return updated_rate

    def search_last_parent(self, prefix: torch.LongTensor):
        """
        Search the longest prefix in the trie that matches to the input sequence of tokens 'prefix'
        """
        found_parent = []
        current_parent = self.root

        # Assume one batch of prefix
        for time_step, token_id in enumerate(prefix[0]):
            token_id = token_id.item()
            if token_id in current_parent.children.keys():
                current_parent = current_parent.children[token_id]
                found_parent.append(current_parent.token_id)
            else:
                print(
                    f"last parent found is {found_parent}; current {token_id} not found in the trie at time step {time_step}")
                return None
        return current_parent

    def search_token_from_parent(self, parent_node, candidate_token_id):
        """
        Check if the parent_node has a children with candidate_token_id
        Return the children node if it exists, return None otherwise
        """
        if parent_node is None:
            return None
        if candidate_token_id in parent_node.children.keys():
            return parent_node.children[candidate_token_id]
        else:
            return None

    def get_success_rate_for_candidate_token(self, parent_node, candidate_token_id):
        """
        Return Approximated Expected Future Grammaticality 
        of the candidate_token_id from the parent_node
        """
        if parent_node is None:
            return 1
        if candidate_token_id in parent_node.children.keys():
            return parent_node.children[candidate_token_id].success_rate
        else:
            return 1

    def search(self, sequence):
        """
        Return the sequence of nodes that exactly matches with the input
        """
        node = self.root
        for token_id in sequence:
            if token_id not in node.children:
                return False
            node = node.children[token_id]
        return node.is_end_of_sequence

    def print_trie(self, node=None, prefix=None):
        """
        Print all the leaves in the trie
        """
        if node is None:
            node = self.root
        if prefix is None:
            prefix = []

        # If current node marks the end of a sequence, print the prefix as a list
        if node.is_end_of_sequence or len(node.children) == 0:
            print(prefix)

        # Recursively call print_trie for all children, appending the current character/token to the prefix
        for char, child_node in node.children.items():
            self.print_trie(child_node, prefix + [char])

    def has_full_information(self):
        """
        Checks if all paths in the trie end with an is_end_of_sequence node set to True.
        Returns True if the trie has full information, False otherwise.
        """
        return self._check_full_information(self.root)

    def _check_full_information(self, node):
        # If the node has no children, check if it is marked as the end of a sequence
        if not node.children:
            return node.is_end_of_sequence

        # Recursively check all children
        return all(self._check_full_information(child) for child in node.children.values())

    def print_all_nodes(self, node=None, depth=0):
        """
        Print all the nodes in the trie (including non-leaves)
        """

        if node is None:
            node = self.root

        # Print current node's details
        indent = "  " * depth  # Create indentation based on the depth in the trie
        node_details = (f"{indent}TrieNode(token_id={node.token_id}, token='{node.token}', "
                        f"raw_likelihood={node.raw_likelihood}, raw_score={node.raw_score}, success rate={node.success_rate}, "
                        f"children={list(node.children.keys())}, "
                        f"parent={node.parent.token_id if node.parent else None}, "
                        f"is_end_of_sequence={node.is_end_of_sequence})")
        print(node_details)

        # Recursively call print_all_nodes for all children
        for child_node in node.children.values():
            self.print_all_nodes(child_node, depth + 1)

def create_nodes_from_history(detailed_history):
    nested_nodes = []
    for time_step in detailed_history:
        step_nodes = []
        for batch in time_step:
            batch_nodes = []
            for node_info in batch:
                node = TrieNode(token_id=node_info['token_id'], token=node_info['token'], raw_likelihood=node_info['raw_likelihood'], raw_score=node_info['raw_score'])
                batch_nodes.append(node)
            step_nodes.append(batch_nodes)
        nested_nodes.append(step_nodes)
    return nested_nodes

def get_selected_token_id_at_time_step(tokens, time_step):
    # Assuming there is only one batch TODO: handle multiple batches
    if time_step < len(tokens[0]):
        return tokens[0][time_step].item()
    else:
        raise ValueError(f"Time step {time_step} is out of range for the given tokens")

def insert_nodes_by_generated_tokens(trie, generated_tokens, nodes):
    current_parent = trie.root

    updated_total = 0

    for time_step, candidate_list in enumerate(nodes):
        selected_token_id = get_selected_token_id_at_time_step(generated_tokens, time_step)
        found_parent_for_next_step = False

        for batch in candidate_list:
            for node in batch:
                # Insert node only if it doesn't already exist as a child of the current parent.
                if node.token_id not in current_parent.children.keys():
                    # print(f"current_parent={current_parent} at time step {time_step}")
                    # print(f"Inserting node {node.token_id} at time step {time_step}")
                    updated_total += trie.insert(current_parent, node)

                else:
                    # if args.verbose:
                        # print(f"Node {node.token_id} already exists as a child of the current parent at time step {time_step}")
                    pass

                # Check if this node matches the next token ID and should be the next parent.
                if node.token_id == selected_token_id:
                    next_parent_candidate = current_parent.children[node.token_id]
                    found_parent_for_next_step = True

        if found_parent_for_next_step:
            current_parent = next_parent_candidate
            # print(f"current_parent_token_id={current_parent.token_id} at time step {time_step}")
        else:
            print(f"No matching child found for next parent at time step {time_step}")

    return updated_total

def update_oracle_trie(trie, generated_tokens, detailed_history):
    nodes = create_nodes_from_history(detailed_history)
    updated_rate = insert_nodes_by_generated_tokens(trie, generated_tokens, nodes)

    return trie, updated_rate