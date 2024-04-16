import subprocess
import os
import json
from sygus.src.utilities import Location
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType

from sygus.src.ast import *
from sygus.src.symbol_table_builder import SymbolTableBuilder
from sygus.src.v2.printer import SygusV2ASTPrinter as printer

TMP_DIR = "tmp"
TMP_PATH = TMP_DIR + "/tmp.sl"
CVC5_PATH = "./cvc5-macOS"

def make_tempdir():
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

def write_tempfile(input_string):
    with open(TMP_PATH, "w") as f:
        f.write(input_string)

def run_cvc5(input_string):
    write_tempfile(input_string)

    output = subprocess.check_output(
        [CVC5_PATH, TMP_PATH],
        stderr=subprocess.PIPE)

    return output

def split_ast(ast):
    synth_fun = None
    constraints = []
    others = []

    for command in ast.commands:
        if isinstance(command, SynthFunCommand):
            synth_fun = command
        elif isinstance(command, ConstraintCommand):
            constraints.append(command)
        elif isinstance(command, CheckSynthCommand):
            pass
        else:
            others.append(command)

    return synth_fun, constraints, others

def test(args, sym_table, constraints, others, answer):  
    score = 0
    
    if args.source_sygus_standard == "1":
        answer = answer.replace("BitVec", "_ BitVec")

    for constraint in constraints:
        prog = ""
        for cmd in others:
            prog += printer.run(cmd, sym_table, vars(args)) + "\n"

        prog += answer + "\n"
        prog += "(synth-fun dummy () Bool)\n"
        prog += printer.run(constraint, sym_table, vars(args)) + "\n"
        prog += "(check-synth)"
 
        out = run_cvc5(prog)

        if b"fail" not in out:
            score += 1
    
    return score

def main(args):
    input_string = args.input_file.read()

    make_tempdir()

    if args.source_sygus_standard == "2":
    # parser = SygusV1Parser()
        from sygus.src.v2.parser import SygusV2Parser
        parser = SygusV2Parser()
    else:
        from sygus.src.v1.parser import SygusV1Parser
        parser = SygusV1Parser()  

    ast = parser.parse(input_string)
    sym_table = SymbolTableBuilder.run(ast)

    synth_fun, constraints, others = split_ast(ast)
    lines = args.answer_file.readlines()

    for line in lines:
        line_json = json.loads(line)
        answer = line_json["answer"][0]

        score = test(args, sym_table, constraints, others, answer)

        print(answer, "| score:", score)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-s', '--source-sygus-standard', default='1', choices=['1','2'],
        help='The SyGuS language standard used in the input file')

    parser.add_argument(
        '--answer-file', type=FileType('r'),
        help='JSONL file that contains candidate solutions')

    parser.add_argument(
        'input_file', type=FileType('r'),
        help='Path to an input file (or stdin if "-")')

    main(parser.parse_args())