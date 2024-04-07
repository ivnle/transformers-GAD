import pickle
import torch

class TrieNode:
    def __init__(self, token_id=None, token=None, raw_logit=None, raw_score=None):
        self.children = {}
        self.parent = None
        self.token_id = token_id
        self.token = token
        self.raw_logit = raw_logit
        # self.raw_score = raw_score
        self.successful_rate = 1
        # isEndOfWord is True if node represent EOS
        self.is_end_of_sequence = False
        self.is_start_of_sequence = False

    def __repr__(self):
        parent_token_id = 'None (Root Node)' if self.parent is None else self.parent.token_id
        # raw_score = self.raw_score if self.raw_score is not None else 'None' # only for old tries
        return (f"TrieNode(token_id={self.token_id}, token='{self.token}', "
                f"raw_logit={self.raw_logit}, "
                # f"raw_score={self.raw_score}, "
                f"children={list(self.children.keys())}, "
                f"parent={parent_token_id}, successful rate={self.successful_rate})") # TODO: add prefix



class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.root.is_start_of_sequence = True

    def insert(self, parent_node: TrieNode, child_node: TrieNode):
        # Insert child_node into parent_node's children dictionary
        if child_node.token_id not in parent_node.children:
            parent_node.children[child_node.token_id] = child_node
            child_node.parent = parent_node  # Set the parent of the child_node
            if child_node.token_id == 2: # TODO: replace to real EOS token
                child_node.is_end_of_sequence = True
            # update the successful rate of the parent node
            self.update_successful_rate(parent_node)

    def update_successful_rate(self, node: TrieNode):
        if node and node.children:
            total_success_rate = sum(child.raw_logit * child.successful_rate for child in node.children.values())
            node.successful_rate = total_success_rate
            if node.parent:
                self.update_successful_rate(node.parent)

    def search_last_parent(self, prefix: torch.LongTensor):
        found_parent = []
        current_parent = self.root
        for time_step, token_id in enumerate(prefix[0]): # still assume one batch
            token_id = token_id.item()
            if token_id in current_parent.children.keys():
                current_parent = current_parent.children[token_id]
                found_parent.append(current_parent.token_id)
            else:
                print(
                    f"last parent found is {found_parent}; current {token_id} not found in the trie at time step {time_step}")
                return None
        return current_parent

    def get_successful_rate_for_candidate_token(self, parent_node, candidate_token_id):
        if parent_node is None:
            return 1
        if candidate_token_id in parent_node.children.keys():
            return parent_node.children[candidate_token_id].successful_rate
        else:
            return 1

    def search(self, sequence):
        node = self.root
        for token_id in sequence:
            if token_id not in node.children:
                return False
            node = node.children[token_id]
        return node.is_end_of_sequence # TODO: also return if not end of sequence

    def print_trie(self, node=None, prefix=None):
        if node is None:
            node = self.root
        if prefix is None:
            prefix = []

        # If current node marks the end of a sequence, print the prefix as a list
        if node.is_end_of_sequence: # TODO: also print if not end of sequence
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
        if node is None:
            node = self.root

        # Print current node's details
        indent = "  " * depth  # Create indentation based on the depth in the trie
        # raw_score = node.raw_score if node.raw_score is not None else 'None'
        node_details = (f"{indent}TrieNode(token_id={node.token_id}, token='{node.token}', "
                        f"raw_logit={node.raw_logit}, "
                        # f"raw_score={node.raw_score}, "
                        f"successful rate={node.successful_rate}, "
                        f"children={list(node.children.keys())}, "
                        f"parent={node.parent.token_id if node.parent else None}, "
                        f"is_end_of_sequence={node.is_end_of_sequence})")
        print(node_details)

        # Recursively call print_all_nodes for all children
        for child_node in node.children.values():
            self.print_all_nodes(child_node, depth + 1)

if __name__ == '__main__':
    # tries= ["trie_PRE_100_bare_starcoder2-15b_iter-100.pkl", "trie_test_len_3.pkl", "trie_PRE_100_bare_Mistral-7B-Instruct-v0.1_iter-1.pkl"]

    import sys

    sys.path.append('/nobackup2/yf/mila/GD/transformers_gad')

    with open('/nobackup2/yf/mila/GD/results_trie/trie_PRE_100_bare_starcoder2-15b_iter-100.pkl', 'rb') as f:
        trie = pickle.load(f)
        trie.print_all_nodes()
