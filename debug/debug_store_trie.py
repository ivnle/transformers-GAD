import pickle
from transformers_gad.build_oracle.build_oracle_trie import Trie, TrieNode

if __name__ == '__main__':
    with open('/nobackup2/yf/mila/GD/results_trie/trie_test_len_3.pkl', 'rb') as f:
        trie = pickle.load(f)
        trie.print_all_nodes()
