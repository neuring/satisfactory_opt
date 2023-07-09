from typing import NamedTuple
from enum import Enum, auto
from scipy.optimize import linprog, OptimizeResult
import numpy as np

from lower import LoweringResult, LinProgBound, LinProgConstraint, IndexMap
from transform import IndexedVar, Recipe

class ExitStatus(Enum):
    SUCCESS=auto()
    ITERATION_LIMIT=auto()
    PROBLEM_INFEASIBLE=auto()
    PROBLEM_UNBOUNDED=auto()
    NUMERICAL_ERROR=auto()

class SolveResult(NamedTuple):
    inner: OptimizeResult
    imap: IndexMap

    @property
    def status(self) -> ExitStatus:
        match self.inner.status:
            case 0:
                return ExitStatus.SUCCESS
            case 1:
                return ExitStatus.ITERATION_LIMIT
            case 2:
                return ExitStatus.PROBLEM_INFEASIBLE
            case 3:
                return ExitStatus.PROBLEM_UNBOUNDED
            case 4:
                return ExitStatus.NUMERICAL_ERROR
    
    def __getitem__(self, index) -> float:
        match index:
            case str():
                return self.inner.x[self.imap[IndexedVar(index, IndexedVar.Direction.POOL, 0)]]
            case IndexedVar():
                return self.inner.x[self.imap[index]]

    def index(self, index) -> float:
        match index:
            case str():
                return self.imap[IndexedVar(index, IndexedVar.Direction.POOL, 0)]
            case IndexedVar():
                return self.imap[index]
    
    def recipe_used(self, recipe: Recipe) -> bool:
        var = list(recipe.outputs.variables())[0]
        return self[var] > 0
            

def solve(lower: LoweringResult) -> SolveResult:

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
    return SolveResult(result, lower.imap)

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