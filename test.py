from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarConstrainedLogitsProcessor


def generate_strings_len_k(cfg, k):
    """
    Generate all unique strings of exact length k from the given context-free grammar.
    This function correctly handles the expansion of non-terminal symbols and
    ensures generated strings are of length k.

    :param cfg: Context-free grammar as a dictionary with rules.
    :param k: Target length of strings to generate.
    :return: A set of all unique possible strings of length k.
    """

    def expand(symbol, sequence="", depth=0):
        # If depth reached k and current symbol is terminal or empty, return sequence
        if depth == k:
            return {sequence} if symbol == "" else set()

        # If symbol is not in CFG (it's terminal or an error), stop expanding
        if symbol not in cfg:
            return set()

        expansions = set()
        for production in cfg[symbol]:
            if 'x' in production:  # Expand 'x' according to its rules
                for x_expansion in cfg['x']:
                    new_sequence = sequence + x_expansion
                    if depth + 1 == k:  # Ensure new sequence is of length k
                        expansions.add(new_sequence)
            else:  # Handle '0s' and '1s'
                for char in production:
                    if char == 's':  # Continue expanding for 's'
                        expansions.update(expand('s', sequence, depth))
                    else:  # Directly add '0' or '1' to sequence
                        new_sequence = sequence + char
                        # Only continue if we haven't reached the final length
                        if len(new_sequence) < k:
                            expansions.update(expand('s', new_sequence, depth + 1))
                        elif len(new_sequence) == k:  # Add if it's the right length
                            expansions.add(new_sequence)

        return expansions

    return expand('root')

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

def stringsofLenk(input_grammar, k):
    # TODO: fix to return the specific length k of strings
    converted_grammar_dict = convert_grammar(input_grammar)
    lstStrings = generate_strings_len_k(converted_grammar_dict, k)
    Stringdict = {}
    for i in lstStrings:
        Stringdict[i] = 0
    return Stringdict



if __name__ == "__main__":
    # # Load model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir="/nobackup2/yf/mila/GD_caches")
    # tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir="/nobackup2/yf/mila/GD_caches")
    #
    # # Load json grammar
    # with open("./examples/grammars/string_0.ebnf", "r") as file:
    #     grammar_str = file.read()
    # grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    # grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
    #
    # # Generate
    # prefix1 = "This is a valid json string for http request:"
    # prefix2 = "This is a valid json string for shopping cart:"
    # input_ids = tokenizer([prefix1, prefix2], add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"]
    #
    # output = model.generate(
    #     input_ids,
    #     do_sample=False,
    #     max_length=50,
    #     num_beams=2,
    #     logits_processor=[grammar_processor],
    #     repetition_penalty=1.0,
    #     num_return_sequences=1,
    # )
    # # decode output
    # generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    # print(generations)

    # f = open('./examples/grammars/string_start_w_1_all_0.ebnf')
    f = open('./examples/grammars/string_01.ebnf')
    input_grammar = f.read()
    f.close()
    converted_grammar_dict = convert_grammar(input_grammar)
    print(f"converted_grammar_dict: {converted_grammar_dict}")
    # print(f"generated_strings: {generate_strings(converted_grammar_dict, 5)}")
    result = generate_strings_len_k(converted_grammar_dict, 5)
    print(f"generated_strings_of_length_k: {result}")
    # print(f"string of len k: {stringsofLenk_max(input_grammar, 5)}")
    print(f"string of len k: {stringsofLenk(input_grammar, 5)}")