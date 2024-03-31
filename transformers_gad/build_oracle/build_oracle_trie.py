from graphviz import Digraph
import torch

class TrieNode:
    def __init__(self, token_id=None, token=None, raw_logit=None):
        self.children = {}
        self.parent = None
        self.token_id = token_id
        self.token = token
        self.raw_logit = raw_logit
        self.successful_rate = 1
        # isEndOfWord is True if node represent EOS
        self.is_end_of_sequence = False
        self.is_start_of_sequence = False

    def __repr__(self):
        parent_token_id = 'None (Root Node)' if self.parent is None else self.parent.token_id
        return (f"TrieNode(token_id={self.token_id}, token='{self.token}', "
                f"raw_logit={self.raw_logit}, children={list(self.children.keys())}, "
                f"parent={parent_token_id}, successful rate={self.successful_rate})") # TODO: add prefix



class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.root.is_start_of_sequence = True

    def insert(self, parent_node: TrieNode, child_node: TrieNode): # TODO: update successful rate
        # Insert child_node into parent_node's children dictionary
        if child_node.token_id not in parent_node.children:
            parent_node.children[child_node.token_id] = child_node
            child_node.parent = parent_node  # Set the parent of the child_node
            if child_node.token_id == 2: # TODO: replace to real EOS token
                child_node.is_end_of_sequence = True
            # update the successful rate of the parent node
            self.update_successful_rate(parent_node)

    def update_successful_rate(self, node: TrieNode):
        # # If the node has children, calculate its successful rate based on its children
        # if node.children:
        #     total_success_rate = 0
        #     for child in node.children.values():
        #         # Add child's raw_logit multiplied by its successful rate to the total
        #         total_success_rate += child.raw_logit * child.successful_rate
        #     # Update the node's successful rate with the calculated total
        #     node.successful_rate = total_success_rate
        # # If the node is a leaf (no children), it keeps its own successful rate
        #
        # # Update the parent node recursively, if the node is not the root
        # if node.parent is not None:
        #     self.update_successful_rate(node.parent)

        if node and node.children:
            total_success_rate = sum(child.raw_logit * child.successful_rate for child in node.children.values())
            node.successful_rate = total_success_rate
            if node.parent:
                self.update_successful_rate(node.parent)

    def search(self, sequence):
        node = self.root
        for token_id in sequence:
            if token_id not in node.children:
                return False
            node = node.children[token_id]
        return node.is_end_of_sequence

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

def visualize_trie(trie_root): # TODO: problem with visualization
    def add_nodes_edges(node, graph, parent=None):
        # Create a unique identifier for the current node
        node_id = id(node)

        # Add the current node to the graph
        label = f"{node.token} (ID: {node.token_id}, Logit: {node.raw_logit})"
        graph.node(str(node_id), label=label)

        # If this node has a parent, add an edge from the parent to this node
        if parent:
            graph.edge(str(id(parent)), str(node_id))

        # Recursively add nodes/edges for the children
        for child in node.children.values():
            add_nodes_edges(child, graph, parent=node)

    graph = Digraph()
    add_nodes_edges(trie_root, graph)
    return graph

def create_nodes_from_history(detailed_history):
    # This list will store all created nodes
    all_nodes = []

    for step in detailed_history:
        for item_list in step:
            for item in item_list:
                node = TrieNode(
                    token_id=item['token_id'],
                    token=item['token'],
                    raw_logit=item['raw_logit']
                )
                all_nodes.append(node)

    return all_nodes

if __name__ == "__main__":

    # Your input
    generated_tokens = torch.tensor([[28740, 28734, 28740, 2]])
    detailed_history = [
        [[{'token_id': 28734, 'token': '0', 'raw_logit': 2.0020976080559194e-05},
          {'token_id': 28740, 'token': '1', 'raw_logit': 3.175825986545533e-05}]],
        [[{'token_id': 28734, 'token': '0', 'raw_logit': 0.22081588208675385},
          {'token_id': 28740, 'token': '1', 'raw_logit': 0.17334003746509552}]],
        [[{'token_id': 28734, 'token': '0', 'raw_logit': 0.11201044172048569},
          {'token_id': 28740, 'token': '1', 'raw_logit': 0.1298251450061798}]],
        [[{'token_id': 2, 'token': '</s>', 'raw_logit': 0.7505450248718262}]]
    ]

    trie = Trie()

    nodes = create_nodes_from_history(detailed_history)
    trie.insert(trie.root, nodes[0])
    trie.insert(trie.root, nodes[1])
    trie.insert(nodes[1], nodes[2])
    trie.insert(nodes[1], nodes[3])
    trie.insert(nodes[2], nodes[4])
    trie.insert(nodes[2], nodes[5])
    trie.insert(nodes[5], nodes[6])
    for node in nodes:
        print(node)
    print(trie.root)
    # list_of_nodes = [] # List of Node objects to be added
    # for i, token_id in enumerate(list_of_nodes):
    #     trie.insert([token_id], list_of_nodes[i])
    # # trie.insert(father_node, node_to_add)
    #
    # trie.insert(generated_tokens, detailed_history)

    # visualize_trie(trie.root)
    trie.print_trie(trie.root)
    sequence = [28740, 28734, 28740, 2]
    search_result = trie.search(sequence)
    print(f"Is the sequence in the trie? {search_result}")