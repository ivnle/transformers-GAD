import sys
import string
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType

from sygus.src.ast import *
from sygus.src.symbol_table_builder import SymbolTableBuilder
# from sygus.src.v1.parser import SygusV1Parser

def tokenize(s):
    """
    >>> tokenize("(quote (+ 1 2) + (1 (1)))")
    ['(', 'quote', '(', '+', 1, 2, ')', '+', '(', 1, '(', 1, ')', ')', ')']
    """
    return list(map(lambda s: int(s) if s.isnumeric() else s, filter(
        bool, s.replace('(', '( ').replace(')', ' )').split(" "))))

def parse(tokens):
    """
    >>> parse(['(', '(', '+', 1, 2, ')', '+', '(', 1, '(', 1, ')', ')', ')'])
    (('+', 1, 2), '+', (1, (1,)))
    """
    def _list():
        tokens.pop(0)
        while tokens[0] != ')':
            yield parse(tokens)
        tokens.pop(0)
    return tuple(_list()) if tokens[0] == '(' else tokens.pop(0)

def convert_term_to_ebnf(term, non_list):
    if isinstance(term, tuple):
        converted_terms = [convert_term_to_ebnf(t, non_list) for t in term]
        converted_join = ' " " '.join(converted_terms)
        return f'"(" {converted_join} ")"'
    elif (term in non_list):
        return term
    else:
        return f'"{term}"'

def convert_non_to_ebnf(non, non_list):
    non_id = non[0]
    rules = non[2]

    converted_rules = [convert_term_to_ebnf(rule, non_list) for rule in rules]
    rules_str = " | ".join(converted_rules)

    return f"{non_id} ::= {rules_str}"

def convert_to_ebnf(grammar_string, args):
    out = ""
    for c in string.whitespace:
        grammar_string = grammar_string.replace(c, " ")

    grammar_string = grammar_string.replace('","', 'comma')
    grammar_string = grammar_string.replace('"."', 'dot')
    grammar_string = grammar_string.replace('""', 'empty')
    grammar_string = grammar_string.replace('"Dr."', 'doctor_Dr')
    grammar_string = grammar_string.replace('"("', 'left_paren')
    grammar_string = grammar_string.replace('")"', 'right_paren')
    grammar_string = grammar_string.replace('"+"', 'plus')
    grammar_string = grammar_string.replace('"-"', 'minus')
    grammar_string = grammar_string.replace('"USA"', 'United_States')
    grammar_string = grammar_string.replace('"New York"', 'New_York')
    grammar_string = grammar_string.replace('"A"', 'capital_A')
    grammar_string = grammar_string.replace('"B"', 'capital_B')
    grammar_string = grammar_string.replace('"C"', 'capital_C')
    grammar_string = grammar_string.replace('"D"', 'capital_D')
    grammar_string = grammar_string.replace('"E"', 'capital_E')
    grammar_string = grammar_string.replace('"F"', 'capital_F')
    grammar_string = grammar_string.replace('"G"', 'capital_G')
    grammar_string = grammar_string.replace('"H"', 'capital_H')
    grammar_string = grammar_string.replace('"I"', 'capital_I')
    grammar_string = grammar_string.replace('"J"', 'capital_J')
    grammar_string = grammar_string.replace('"K"', 'capital_K')
    grammar_string = grammar_string.replace('"L"', 'capital_L')
    grammar_string = grammar_string.replace('"M"', 'capital_M')
    grammar_string = grammar_string.replace('"N"', 'capital_N')
    grammar_string = grammar_string.replace('"O"', 'capital_O')
    grammar_string = grammar_string.replace('"P"', 'capital_P')
    grammar_string = grammar_string.replace('"Q"', 'capital_Q')
    grammar_string = grammar_string.replace('"R"', 'capital_R')
    grammar_string = grammar_string.replace('"S"', 'capital_S')
    grammar_string = grammar_string.replace('"T"', 'capital_T')
    grammar_string = grammar_string.replace('"U"', 'capital_U')
    grammar_string = grammar_string.replace('"V"', 'capital_V')
    grammar_string = grammar_string.replace('"W"', 'capital_W')
    grammar_string = grammar_string.replace('"X"', 'capital_X')
    grammar_string = grammar_string.replace('"Y"', 'capital_Y')
    grammar_string = grammar_string.replace('"Z"', 'capital_Z')

    state_abbreviations = [
        "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL",
        "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
        "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR",
        "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI"
    ]

    for state in state_abbreviations:
        replacement_string = f'state_{state}'
        grammar_string = grammar_string.replace(f'"{state}"', replacement_string)


    grammar_string = grammar_string.replace('" "', '"whitespace"')
    # print(grammar_string)
    # print(tokenize(grammar_string))

    sexp = parse(tokenize(grammar_string))
    
    if args.source_sygus_standard == "2":
        sexp = sexp[1]
    else:
        sexp = sexp[0]

    non_list = [non[0] for non in sexp]

    for non in sexp:
        out += convert_non_to_ebnf(non, non_list) + "\n"

    out = out.replace('"whitespace"', " ")
    out = out.replace('comma', ",")
    out = out.replace('empty', "")
    out = out.replace('dot', ".")
    out = out.replace('doctor_Dr', '"Dr."')
    out = out.replace('capital_D', '"D"')
    out = out.replace('left_paren', '"("')
    out = out.replace('right_paren', '")"')
    out = out.replace('plus', '"+"')
    out = out.replace('minus', '"-"')
    out = out.replace('United_States', '"USA"')
    out = out.replace('Pennsylvania', '"PA"')
    out = out.replace('California', '"CA"')
    out = out.replace('New_York_abb', '"NY"')
    out = out.replace('Maryland', '"MD"')
    out = out.replace('Connecticut', '"CT"')
    out = out.replace('New_York', '"New York"')
    out = out.replace('capital_A', '"A"')
    out = out.replace('capital_B', '"B"')
    out = out.replace('capital_C', '"C"')
    out = out.replace('capital_D', '"D"')
    out = out.replace('capital_E', '"E"')
    out = out.replace('capital_F', '"F"')
    out = out.replace('capital_G', '"G"')
    out = out.replace('capital_H', '"H"')
    out = out.replace('capital_I', '"I"')
    out = out.replace('capital_J', '"J"')
    out = out.replace('capital_K', '"K"')
    out = out.replace('capital_L', '"L"')
    out = out.replace('capital_M', '"M"')
    out = out.replace('capital_N', '"N"')
    out = out.replace('capital_O', '"O"')
    out = out.replace('capital_P', '"P"')
    out = out.replace('capital_Q', '"Q"')
    out = out.replace('capital_R', '"R"')
    out = out.replace('capital_S', '"S"')
    out = out.replace('capital_T', '"T"')
    out = out.replace('capital_U', '"U"')
    out = out.replace('capital_V', '"V"')
    out = out.replace('capital_W', '"W"')
    out = out.replace('capital_X', '"X"')
    out = out.replace('capital_Y', '"Y"')
    out = out.replace('capital_Z', '"Z"')

    for state in state_abbreviations:
        replacement_string = f'state_{state}'
        out = out.replace(replacement_string, f'"{state}"')

    return out

def main(args):
    input_string = args.input_file.read()

    if args.source_sygus_standard == "2":
    # parser = SygusV1Parser()
        from sygus.src.v2.parser import SygusV2Parser
        parser = SygusV2Parser()
    else:
        from sygus.src.v1.parser import SygusV1Parser
        parser = SygusV1Parser()
    
    ast = parser.parse(input_string)
    sym_table = SymbolTableBuilder.run(ast)

    grammar = None
    for command in ast.commands:
        if isinstance(command, SynthFunCommand):
            func = command
            grammar = command.synthesis_grammar
    
    # print(grammar.nonterminals)
    # print(grammar.grouped_rule_lists)

    if args.source_sygus_standard == "2":
        from sygus.src.v2.printer import SygusV2ASTPrinter as printer
    else:
        from sygus.src.v1.printer import SygusV1ASTPrinter as printer

    grammar_str = printer.run(grammar, sym_table, vars(args))
    param_converted = [f"({t[0]} {printer.run(t[1], sym_table, vars(args))})" for t in func.parameters_and_sorts]
    param_str = " ".join(param_converted)
    range_str = printer.run(func.range_sort_expression, sym_table, vars(args))
    other_rules = convert_to_ebnf(f'({grammar_str})', args)
    top_non = grammar.nonterminals[0][0]

    top_rule = f'root ::= "(define-fun {func.function_symbol} ({param_str}) {range_str} " {top_non} ")"'

    print(top_rule)
    print(other_rules)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-s', '--source-sygus-standard', default='1', choices=['1','2'],
        help='The SyGuS language standard used in the input file. Use 1 to include the function definition in the ebnf grammar. Use 2 to exclude them.')

    parser.add_argument(
        'input_file', type=FileType('r'),
        help='Path to an input file (or stdin if "-")')

    main(parser.parse_args())

