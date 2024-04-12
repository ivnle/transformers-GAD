import sys

from sygus.src.v2.parser import SygusV2Parser
from sygus.src.v2.printer import SygusV2ASTPrinter, SymbolTable
# from sygus.src.v1.parser import SygusV1Parser

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        input_string = f.read()

    # parser = SygusV1Parser()
    parser = SygusV2Parser()
    ast = parser.parse(input_string)

    print(ast.commands)

    # sym_table = SymbolTable()
    # printer = SygusV2ASTPrinter(sym_table)
    # print(printer.run(ast, sym_table))

