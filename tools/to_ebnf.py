import sys
import string
import shlex
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
        bool, shlex.split(s.replace('(', '( ').replace(')', ' )'), posix=False))))

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

    grammar_string.replace('""', "empty_string")

    tokens = tokenize(grammar_string)
    tokens = [str(token).replace("\"", "\\\"") for token in tokens]
    sexp = parse(tokens)
    
    if args.source_sygus_standard == "2":
        sexp = sexp[1]
    else:
        sexp = sexp[0]

    non_list = [non[0] for non in sexp]

    for non in sexp:
        out += convert_non_to_ebnf(non, non_list) + "\n"

    out.replace("empty_string", '\\\"\\\"')
    out = out.replace("str.to.int", "str.to_int")
    out = out.replace("int.to.str", "str.from_int")

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
    
    input_string = input_string.replace("str.to_int", "str.to.int")
    input_string = input_string.replace("str.from_int", "int.to.str")

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

