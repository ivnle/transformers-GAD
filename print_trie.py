import sys
import pickle
# import graphviz

from transformers_gad.build_oracle.build_oracle_trie import visualize_trie

def load_oracle_trie(trie_file):
    with open(trie_file, 'rb') as f:
        trie = pickle.load(f)
    return trie

def main(trie_file):
    trie = load_oracle_trie(trie_file)

    trie.print_all_nodes()

    # graph = visualize_trie(trie.root)
    # graph.format = 'png'

    # graph.render(directory='trie-graph').replace('\\', '/')


if __name__ == "__main__":
    trie_file = sys.argv[1]

    main(trie_file)