from argparse import ArgumentParser
from typing import NamedTuple
from parse import Parser
from transform import normalize_rules, index_rules, IndexResult
from lower import lower
from report import report_error
from error import Error
from solve import solve, SolveResult, ExitStatus
import sys


def to_xdot(index: IndexResult, solve: SolveResult, out) -> str:
    all_materials = set(index.uses.keys()).union(set(index.constructions.keys()))

    out.write("digraph {\n")

    for material in all_materials:
        amount = solve[material]
        if amount == 0:
            continue
        out.write(f"{solve.index(material)}[shape=\"ellipse\", label=\"{amount:.2f} {material}\"]\n")

    for idx, recipe in enumerate(index.rules.recipes):
        if not solve.recipe_used(recipe):
            continue

        label = [f"{solve[out]:.2f} {out.name}" for out in recipe.outputs.variables()]
        label = ", ".join(label)
        out.write(f"r{idx}[shape=\"box\", label=\"{label}\"]")

        for inp in recipe.inputs.variables():
            # Dont draw Power usage lines to avoid clutter
            if inp.name == "MW":
                continue

            src = solve.index(inp.pool())
            amount = solve[inp]

            out.write(f"{src} -> r{idx}[label=\"{amount:.2f}\"]")

        for outp in recipe.outputs.variables():

            dest = solve.index(outp.pool())
            amount = solve[outp]

            out.write(f"r{idx} -> {dest}[label=\"{amount:.2f}\"]")

    out.write("}\n")


def main():
    cli = ArgumentParser(prog='Satisfactory Production Chain Optimizer',
                        description='Generates an optimal production chain given user suppied constraints.',
                        epilog='Have Fun!')
    cli.add_argument('filename')
    cli.add_argument('dot_out_file')

    args = cli.parse_args()

    with open(args.filename) as f:
        text = "\n".join(f.readlines())

        try:
            parser = Parser(text)
            rules = parser.parse()

            normalized_rules = normalize_rules(rules)
        except Error as error:
            report_error(error, text)
            return

        index_result = index_rules(normalized_rules)
        lowering_result = lower(index_result.rules, index_result.uses, index_result.constructions)
        solve_result = solve(lowering_result)
    
    if solve_result.status != ExitStatus.SUCCESS:
        print(f"Problem with Ruleset: {solve_result.status}")
    elif args.dot_out_file == '-':
        to_xdot(index_result, solve_result, sys.stdout)
    else:
        with open(args.dot_out_file) as out:
            to_xdot(index_result, solve_result, out)



if __name__ == '__main__':
    main()