import subprocess
import os
import json
from sygus.src.utilities import Location
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType

from sygus.src.ast import *
from sygus.src.symbol_table_builder import SymbolTableBuilder
from sygus.src.v2.printer import SygusV2ASTPrinter as printer

TMP_DIR = "temp"
TMP_PATH = TMP_DIR + "/temp.sl"
CVC5_PATH = "tools/cvc5-macOS"

MODEL = "Mistral-7B-Instruct-v0.2"
# MODEL = "Mistral-7B-Instruct-v0.2-gad-bv4nogram3-merged"
# MODEL = "Mistral-7B-Instruct-v0.2-gad-slianogram3-merged"
NUM_ITER = 2000

def get_input_path(args, prob_name):
    return f"{args.input_path}/{prob_name}.sl"

def get_unconstrained_answer_path(args, prob_name):
    return f"{args.answer_path}/{prob_name}/bare_{MODEL}_i{NUM_ITER}_cuda_sd42_float16.jsonl"

def get_gad_answer_path(args, prob_name):
    return f"{args.answer_path}/{prob_name}/gad_{MODEL}_i{NUM_ITER}_cuda_sd42_bfloat16.jsonl"

def get_gcd_answer_path(args, prob_name):
    return f"{args.answer_path}/{prob_name}/gcd_{MODEL}_i{NUM_ITER}_cuda_sd42_float16.jsonl"

def make_tempdir():
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

def write_tempfile(input_string):
    with open(TMP_PATH, "w") as f:
        f.write(input_string)

def run_cvc5(input_string):
    input_string = input_string.replace("str.to.int", "str.to_int")
    input_string = input_string.replace("int.to.str", "str.from_int")

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

def test_answer(args, sym_table, constraints, others, answer):  
    score = 0
    
    answer = answer.replace("BitVec", "_ BitVec")

    prog = ""
    for cmd in others:
        s = printer.run(cmd, sym_table, vars(args))
        if "inv" not in s:
            prog += s + "\n"

    prog += answer + "\n"
    prog += "(synth-fun dummy () Bool)\n"

    for cmd in others:
        s = printer.run(cmd, sym_table, vars(args))
        if "inv" in s:
            prog += s + "\n"

    for constraint in constraints:
        prog += printer.run(constraint, sym_table, vars(args)) + "\n"
    prog += "(check-synth)"

    try:
        out = run_cvc5(prog)
    except:
        out = b"fail"

    if b"fail" not in out:
        score += len(constraints)
    
    return score

def test_answer_file(args, sym_table, constraints, others, answer_path):
    if not os.path.isfile(answer_path):
        return None

    with open(answer_path, 'r') as f:
        lines = f.readlines()
    
    num_correct = 0
    for line in lines:
        line_json = json.loads(line)
        # print(line_json)
        answer = line_json["answer"][0]
        answer = answer.replace("</s>", "")

        score = test_answer(args, sym_table, constraints, others, answer)
        
        if score == len(constraints):
            num_correct += 1
    
    return num_correct

def test_prob(parser, args, prob_name):
    input_path = get_input_path(args, prob_name)
    unconstrained_answer_path = get_unconstrained_answer_path(args, prob_name)
    gcd_answer_path = get_gcd_answer_path(args, prob_name)
    gad_answer_path = get_gad_answer_path(args, prob_name)

    with open(input_path, "r") as f:
        input_string = f.read()

        input_string = input_string.replace("BitVec", "_ BitVec")

        input_string = input_string.replace("str.to_int", "str.to.int")
        input_string = input_string.replace("str.from_int", "int.to.str")

    # print(input_string)
    ast = parser.parse(input_string)
    sym_table = SymbolTableBuilder.run(ast)
    synth_fun, constraints, others = split_ast(ast)

    unconstrained = test_answer_file(args, sym_table, constraints, others, unconstrained_answer_path)
    gcd = test_answer_file(args, sym_table, constraints, others, gcd_answer_path)
    gad = test_answer_file(args, sym_table, constraints, others, gad_answer_path)

    return unconstrained, gcd, gad

def main(args):
    make_tempdir()

    if args.source_sygus_standard == "2":
    # parser = SygusV1Parser()
        from sygus.src.v2.parser import SygusV2Parser
        parser = SygusV2Parser()
    else:
        from sygus.src.v1.parser import SygusV1Parser
        parser = SygusV1Parser()  

    prob_names = [x[0] for x in os.walk(args.answer_path) if x[0] != args.answer_path]
    prob_names = [x.split("/")[-1] for x in prob_names]

    scores = []

    for prob_name in prob_names:
        unconstrained, gcd, gad = test_prob(parser, args, prob_name)

        scores.append((prob_name, unconstrained, gcd, gad))
        print(prob_name, unconstrained, gcd, gad)

    print(scores)

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-s', '--source-sygus-standard', default='1', choices=['1','2'],
        help='The SyGuS language standard used in the input file')

    parser.add_argument(
        '--answer-path',
        help='Path to JSONL files that contains candidate solutions')

    parser.add_argument(
        '--input-path',
        help='Path to input files')

    # parser.add_argument(
    #     '--output-path',
    #     help='Path to store output files'
    # )

    main(parser.parse_args())