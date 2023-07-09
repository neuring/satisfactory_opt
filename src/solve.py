from scipy.optimize import linprog
import numpy as np

from lower import LoweringResult, LinProgBound, LinProgConstraint

def solve(lower: LoweringResult):

    total_vars = len(lower.imap)

    eq_matrix = []
    eq_vector = []
    ineq_matrix = []
    ineq_vector = []

    for constraint in lower.constraints:
        match constraint.ty:
            case LinProgConstraint.Type.EQUALITY:
                matrix = eq_matrix
                vector = eq_vector
            case LinProgConstraint.Type.LESS_EQ:
                matrix = ineq_matrix
                vector = ineq_vector
        
        row = np.zeros(total_vars)
        for var_idx, factor in constraint.terms:
            row[var_idx] = factor
        matrix.append(row)
        vector.append(constraint.constant)
    
    bounds = [(None, None)] * total_vars
    for bound in lower.bounds:
        bounds[bound.variable_idx] = (bound.lower_bound, None)
    
    goal = np.zeros(total_vars)
    for var_idx, factor in lower.goal:
        goal[var_idx] = factor
    
    result = linprog(goal, ineq_matrix, ineq_vector, eq_matrix, eq_vector, bounds)
    print(result)

if __name__ == '__main__':
    import sys
    from parse import Parser
    from transform import normalize_rules, index_rules
    from lower import lower
    from report import report_error
    from error import Error

    with open(sys.argv[2]) as f:
        text = "\n".join(f.readlines())

        try:
            parser = Parser(text)
            rules = parser.parse()

            normalized_rules = normalize_rules(rules)
        except Error as error:
            report_error(error, text)
        else:
            index_result = index_rules(normalized_rules)

            lowering_result = lower(index_result.rules, index_result.uses, index_result.constructions)

            solve(lowering_result)