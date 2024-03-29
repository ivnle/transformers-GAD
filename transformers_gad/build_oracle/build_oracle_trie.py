class TrieNode:
    def __init__(self):
        # Instead of storing single token details, store history for all considered tokens
        self.history = []
        self.children = {}  # children nodes

    def add_history(self, token_details):
        self.history.append(token_details)

    def add_or_get_child(self, token_id):
        if token_id not in self.children:
            self.children[token_id] = TrieNode()
        return self.children[token_id]


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, tokens, detailed_history):
        # Ensure detailed_history aligns with tokens for direct indexing
        for token_sequence, history_sequence in zip(tokens, detailed_history):
            current_node = self.root
            for token_id, history in zip(token_sequence, history_sequence):
                current_node.add_history(history)  # Add full history at current step
                # Move to the child node for the next step
                current_node = current_node.add_or_get_child(token_id)

    def print_trie(self, node=None, indent="", token_id=None):
        if node is None:
            node = self.root

        if token_id is not None:  # Skip root
            print(f"{indent}Token ID: {token_id}, History: {node.history}")

        for child_token_id, child_node in node.children.items():
            self.print_trie(child_node, indent + "  ", child_token_id)


if __name__ == "__main__":
    import torch

    # Your input
    generated_tokens = torch.tensor([[28740, 28734, 28740, 2]])
    detailed_history = [
        [[{'token_id': 28734, 'token': '0', 'raw_score': 2.6377220153808594, 'raw_logit': 2.0020976080559194e-05},
          {'token_id': 28740, 'token': '1', 'raw_score': 3.0990943908691406, 'raw_logit': 3.175825986545533e-05}]],
        [[{'token_id': 28734, 'token': '0', 'raw_score': 10.416403770446777, 'raw_logit': 0.22081588208675385},
          {'token_id': 28740, 'token': '1', 'raw_score': 10.17432975769043, 'raw_logit': 0.17334003746509552}]],
        [[{'token_id': 28734, 'token': '0', 'raw_score': 9.51977825164795, 'raw_logit': 0.11201044172048569},
          {'token_id': 28740, 'token': '1', 'raw_score': 9.667374610900879, 'raw_logit': 0.1298251450061798}]],
        [[{'token_id': 2, 'token': '</s>', 'raw_score': 13.21010684967041, 'raw_logit': 0.7505450248718262}]]
    ]

    # Assuming generated_tokens and detailed_history are defined as in the previous example

    trie = Trie()
    trie.insert(generated_tokens, detailed_history)

    trie.print_trie()

