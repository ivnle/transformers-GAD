from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = {}
        self.value = 0  # Sum of values for strings with the current prefix

def insert(root, string, value):
    node = root
    for char in string:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
        node.value += value

def build_trie(outputs):
    root = TrieNode()
    for string, value in outputs.items():
        insert(root, string, value)
    return root

def traverse(node, prefix='', prefix_count={}):
    if node.value > 0:  # If there's a value, it means this prefix exists in the dataset
        prefix_count[prefix] = node.value
    for char, next_node in node.children.items():
        traverse(next_node, prefix + char, prefix_count)
    return prefix_count

if __name__ == '__main__':

    outputs = {'10010': 43, '10100': 29, '11010': 26, '11011': 24, '10110': 100,
               '11000': 1, '11100': 7, '10000': 11, '11110': 1, '10011': 66, '11101': 8,
               '11111': 1, '10111': 69, '10001': 31, '11001': 16, '10101': 67}

    prefix_count = defaultdict(int)

    for key, value in outputs.items():
        for i in range(1, len(key) + 1):
            prefix = key[:i]
            prefix_count[prefix] += value

    # If you need the result in a regular dict or need it sorted
    sorted_prefix_count = dict(sorted(prefix_count.items()))

    print(sorted_prefix_count)

    # Build the trie
    trie_root = build_trie(outputs)

    # Traverse the trie to calculate prefix counts
    prefix_count = traverse(trie_root)

    # Since the question might expect the prefixes to be sorted
    sorted_prefix_count_trie = dict(sorted(prefix_count.items()))
    print(sorted_prefix_count_trie)