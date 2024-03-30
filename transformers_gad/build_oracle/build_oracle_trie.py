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

    def __repr__(self):
        return f"TrieNode(token_id={self.token_id}, token='{self.token}', raw_logit={self.raw_logit})" # TODO: update successful rate, add parent, child



class Trie:
    def __init__(self):
        self.root = self.TrieNode()

    def insert(self, parent_node: TrieNode, child_node: TrieNode): # TODO: update successful rate
        # Insert child_node into parent_node's children dictionary
        if child_node.token_id not in parent_node.children:
            parent_node.children[child_node.token_id] = child_node
            child_node.parent = parent_node  # Set the parent of the child_node



def visualize_trie(node, graph=None, parent=None, edge_label=""):
    if graph is None:
        graph = Digraph(comment='Trie')
        graph.attr(size='10,5')
        graph.node('root', 'Root', shape='box')

    for child in node.children.values():
        node_name = f'{child.token_id}_{child.token}'
        graph.node(node_name, f'Token: {child.token}\nToken ID: {child.token_id}\nRaw Logit: {child.raw_logit}',
                   shape='ellipse')
        if parent is None:
            graph.edge('root', node_name, label=edge_label)
        else:
            parent_name = f'{parent.token_id}_{parent.token}'
            graph.edge(parent_name, node_name, label=edge_label)
        visualize_trie(child, graph, child)

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

    # Assuming generated_tokens and detailed_history are defined as in the previous example

    trie = Trie()

    # list_of_nodes = [] # List of Node objects to be added
    # for i, token_id in enumerate(list_of_nodes):
    #     trie.insert([token_id], list_of_nodes[i])
    # # trie.insert(father_node, node_to_add)
    #
    # trie.insert(generated_tokens, detailed_history)

    # Create nodes
    node_0 = TrieNode(token_id=28734, token='0', raw_logit=2.0020976080559194e-05)
    node_1 = TrieNode(token_id=28740, token='1', raw_logit=3.175825986545533e-05)

    # Insert nodes
    trie.insert(trie.root, node_0)  # Insert node_0 into the root
    trie.insert(trie.root, node_1)  # Insert node_1 into node_0
    #
    # # Insert each token directly under the root
    # for sequence in detailed_history:
    #     for token_info_list in sequence:
    #         for token_info in token_info_list:
    #             trie.insert(token_info['token_id'], token_info['token'], token_info['raw_logit'])

    new_children_data = [
        {'token_id': 28734, 'token': '0', 'raw_logit': 0.22081588208675385},
        {'token_id': 28740, 'token': '1', 'raw_logit': 0.17334003746509552}
    ]

    # Find the parent node with token_id 28740
    parent_node = trie.find_node(trie.root, 28740)

    # Insert new children under the found parent node
    for child_data in new_children_data:
        new_child_node = TrieNode(token_id=child_data['token_id'], token=child_data['token'],
                                  raw_logit=child_data['raw_logit'])
        trie.insert(parent_node, new_child_node)

    trie_graph = visualize_trie(trie.root)
    trie_graph.render('trie_visualization', format='png', cleanup=True)
