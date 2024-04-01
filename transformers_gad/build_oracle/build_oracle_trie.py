from graphviz import Digraph
import torch
import pickle

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
        node_details = (f"{indent}TrieNode(token_id={node.token_id}, token='{node.token}', "
                        f"raw_logit={node.raw_logit}, successful rate={node.successful_rate}, "
                        f"children={list(node.children.keys())}, "
                        f"parent={node.parent.token_id if node.parent else None}, "
                        f"is_end_of_sequence={node.is_end_of_sequence})")
        print(node_details)

        # Recursively call print_all_nodes for all children
        for child_node in node.children.values():
            self.print_all_nodes(child_node, depth + 1)

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
    nested_nodes = []
    for time_step in detailed_history:
        step_nodes = []
        for batch in time_step:
            batch_nodes = []
            for node_info in batch:
                node = TrieNode(token_id=node_info['token_id'], token=node_info['token'], raw_logit=node_info['raw_logit'])
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

    for time_step, candidate_list in enumerate(nodes):
        selected_token_id = get_selected_token_id_at_time_step(generated_tokens, time_step)
        found_parent_for_next_step = False

        for batch in candidate_list:
            for node in batch:
                # Insert node only if it doesn't already exist as a child of the current parent.
                if node.token_id not in current_parent.children.keys():
                    print(f"current_parent={current_parent} at time step {time_step}")
                    print(f"Inserting node {node.token_id} at time step {time_step}")
                    trie.insert(current_parent, node)

                else:
                    print(f"Node {node.token_id} already exists as a child of the current parent at time step {time_step}")

                # Check if this node matches the next token ID and should be the next parent.
                if node.token_id == selected_token_id:
                    next_parent_candidate = current_parent.children[node.token_id]
                    found_parent_for_next_step = True

        if found_parent_for_next_step:
            current_parent = next_parent_candidate
            print(f"current_parent_token_id={current_parent.token_id} at time step {time_step}")
        else:
            print(f"No matching child found for next parent at time step {time_step}")

def run_demo_trie_string_01_len_3():
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
    insert_nodes_by_generated_tokens(trie, generated_tokens, nodes)
    #
    # for node in nodes:
    #     print(node)
    # print(trie.root)
    print(f"====updated trie====")
    updated_generated_tokens = torch.tensor([[28740, 28734, 28734, 2]])
    acceptance_details_history = [[[{'token_id': 28734, 'token': '0', 'raw_score': 2.6377220153808594,
                                    'raw_logit': 2.0020976080559194e-05},
                                   {'token_id': 28740, 'token': '1', 'raw_score': 3.0990943908691406,
                                    'raw_logit': 3.175825986545533e-05}]],
                                  [[{'token_id': 28734, 'token': '0', 'raw_score': 10.416403770446777,
                                    'raw_logit': 0.22081588208675385},
                                    {'token_id': 28740, 'token': '1', 'raw_score': 10.17432975769043,
                                    'raw_logit': 0.17334003746509552}]],
                                  [[{'token_id': 28734, 'token': '0', 'raw_score': 9.51977825164795,
                                    'raw_logit': 0.11201044172048569},
                                    {'token_id': 28740, 'token': '1', 'raw_score': 9.667374610900879,
                                    'raw_logit': 0.1298251450061798}]],
                                 [[{'token_id': 2, 'token': '</s>', 'raw_score': 11.959583282470703,
                                    'raw_logit': 0.35645970702171326}]]]

    updated_nodes = create_nodes_from_history(acceptance_details_history)
    insert_nodes_by_generated_tokens(trie, updated_generated_tokens, updated_nodes)

    # for node in nodes:
    #     print(node)
    # print(trie.root)

    trie.print_all_nodes()  # Print all nodes in the trie

    # visualize_trie(trie.root)
    trie.print_trie(trie.root)
    sequence = [28740, 28734, 28734, 2]
    prefix = torch.tensor([[28740, 28734, 28734]])
    last_parent = trie.search_last_parent(prefix)
    # print(f"last parent found is {last_parent}")
    search_result = trie.search(sequence)
    # print(f"Is the sequence in the trie? {search_result}")
    #
    # print(f"Does the trie have full information? {trie.has_full_information()}")

    return trie

if __name__ == "__main__":
    trie = run_demo_trie_string_01_len_3()

    with open('/nobackup2/yf/mila/GD/results_trie/trie_test_len_3.pkl', 'wb') as f:
        pickle.dump(trie, f)