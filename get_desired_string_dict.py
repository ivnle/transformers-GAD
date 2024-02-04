def generate_strings(cfg, k):
    """
    Generate all strings of length at most k from the given context-free grammar.
    :param cfg: converted grammar dict
    :param k: length of string
    :return: list of all possible strings
    """
    def expand(symbol, current_length):
        # If the symbol is a terminal, return it as is
        if symbol not in cfg:
            return [symbol] if current_length < k else []

        # Explore all production rules for the current non-terminal symbol
        result = []
        for production in cfg[symbol]:
            if current_length + len(production) - 1 < k:
                # Generate all combinations for the current production
                combinations = ['']
                for prod_symbol in production:
                    new_combinations = []
                    for string in combinations:
                        for expansion in expand(prod_symbol, current_length + len(string)):
                            new_combinations.append(string + expansion)
                    combinations = new_combinations
                result.extend(combinations)
        return result

    # Start the expansion from the start symbol
    start_symbol = list(cfg.keys())[0]  # Assuming the first key is the start symbol
    return expand(start_symbol, 0) # ['0', '1', '00', '01', '000', '001', '010', '011', '10', '11', '100', '101', '110', '111']

def generate_strings_len_k(cfg, k):
    def expand(symbol, current_length):
        if symbol not in cfg:
            return [symbol] if current_length == k else []

        result = []
        for production in cfg[symbol]:
            # This initial check might not be necessary with the new logic
            if len(production) + current_length <= k:  # Adjust to ensure we're not exceeding length prematurely
                combinations = ['']
                for prod_symbol in production:
                    new_combinations = []
                    for string in combinations:
                        expanded = expand(prod_symbol, current_length + len(string))
                        new_combinations.extend([string + exp for exp in expanded])
                    combinations = new_combinations
                result.extend(combinations)
        return result

    start_symbol = list(cfg.keys())[0]
    return [s for s in expand(start_symbol, 0) if len(s) == k]

def convert_grammar(input_grammar):
    grammar = {}
    lines = input_grammar.strip().split('\n')

    for line in lines:
        # Split the line into the non-terminal and its productions
        non_terminal, productions = line.split('::=')
        non_terminal = non_terminal.strip()

        # Process each production, removing extra spaces and quotes
        productions = productions.strip().split('|')
        productions = [prod.strip().replace('"', '') for prod in productions]

        grammar[non_terminal] = productions

    return grammar # {'root': ['s'], 's': ['x', 'xs'], 'x': ['0', '1']}

def stringsofLenk_max(input_grammar, k):
    converted_grammar_dict = convert_grammar(input_grammar)
    lstStrings = generate_strings(converted_grammar_dict, k)
    Stringdict = {}
    for i in lstStrings:
        Stringdict[i] = 0
    return Stringdict # {'0': 0, '1': 0, '00': 0, '01': 0, '000': 0, '001': 0, '010': 0, '011': 0, '10': 0, '11': 0, '100': 0, '101': 0, '110': 0, '111': 0}

def stringsofLenk(input_grammar, k):
    # TODO: fix to return the specific length k of strings
    converted_grammar_dict = convert_grammar(input_grammar)
    lstStrings = generate_strings(converted_grammar_dict, k)
    return lstStrings

if __name__ == "__main__":
    f = open('./examples/grammars/string_start_w_1_all_0.ebnf')
    input_grammar = f.read()
    f.close()
    converted_grammar_dict = convert_grammar(input_grammar)
    # print(f"converted_grammar_dict: {converted_grammar_dict}")
    print(f"generate_strings: {generate_strings_len_k(converted_grammar_dict, 5)}")
    # print(f"string of len k: {stringsofLenk_max(input_grammar, 5)}")
