from enum import Enum, auto
from typing import List, Tuple, NamedTuple, Dict
from transform import NormalizedRules, LinearExpression, IndexedVar, Recipe, Constraint, Goal
from parse import Goal as ASTGoal

# This module lowers normalized rules to linear equalities and inequalities suitable for linear programming

class LinProgConstraint(NamedTuple):
    class Type(Enum):
        EQUALITY=auto()
        LESS_EQ=auto()
    
    class VariableFactorPair(NamedTuple):
        variable_idx: int
        factor: float

        def __str__(self):
            return f"{self.factor}*{self.variable_idx}"
    
    terms: List[VariableFactorPair]
    ty: "LinProgConstraint.Type"
    constant: float

    def __str__(self):
        return f"{self.terms} {self.ty} {self.constant}"

class LinProgBound(NamedTuple):
    lower_bound: float
    variable_idx: int

    def __str__(self):
        return f"{self.lower_bound} <= {self.variable_idx}"

class IndexMap:
    """Map IndexedVariable to a unique integer"""

    def __init__(self):
        self.next_id = 0
        self.data = {}
        self.rev_map = {}

    def __getitem__(self, index):
        if index in self.data:
            return self.data[index]
        else:
            self.data[index] = self.next_id
            self.rev_map[self.next_id] = index
            result = self.next_id
            self.next_id += 1
            return result

    def reverse(self, idx):
        return self.rev_map[idx]

    def __str__(self):
        return self.data.__str__()
    
    def __len__(self) -> int:
        return self.next_id



def _lower_recipe(recipe: Recipe, imap: IndexMap) -> List[LinProgConstraint]:
    result = []

    outputs = list(recipe.outputs.variable_coefficients.items())

    anchor_var, anchor_factor = outputs[0]
    for in_var, in_factor in recipe.inputs.variable_coefficients.items():
        result.append(LinProgConstraint([
            LinProgConstraint.VariableFactorPair(imap[anchor_var], in_factor), 
            LinProgConstraint.VariableFactorPair(imap[in_var], -1 * anchor_factor)
        ], LinProgConstraint.Type.EQUALITY, 0.0))

    for in_var, in_factor in outputs[1:]:
        result.append(LinProgConstraint([
            LinProgConstraint.VariableFactorPair(imap[anchor_var], in_factor), 
            LinProgConstraint.VariableFactorPair(imap[in_var], -1 * anchor_factor)
        ], LinProgConstraint.Type.EQUALITY, 0.0))
    
    return result


def _lower_constraint(constraint: Constraint, imap: IndexMap) -> List[LinProgConstraint]:
    assert constraint.term.constant_part == 0

    term = [LinProgConstraint.VariableFactorPair(imap[var], val) for var, val in constraint.term.variable_coefficients.items()]

    match constraint.ty:
        case Constraint.Type.EQUALITY:
            ty = LinProgConstraint.Type.EQUALITY
        case Constraint.Type.LESS_EQ:
            ty = LinProgConstraint.Type.LESS_EQ

    return [LinProgConstraint(term, ty, constraint.bound)]

def _lower_goal(goal: Goal, imap: IndexMap) -> List[LinProgConstraint.VariableFactorPair]:
    assert goal.term.constant_part == 0

    # Express MAX as a MIN goal by negating all coefficients
    match goal.ty:
        case ASTGoal.Type.MAX:
            factor = -1.0
        case ASTGoal.Type.MIN:
            factor = 1.0

    return [LinProgConstraint.VariableFactorPair(imap[var], factor * val) for var, val in goal.term.variable_coefficients.items()]

def _vars_positive(imap: IndexMap) -> List[LinProgBound]:
    return [LinProgBound(0.0, idx) for idx in imap.data.values()]

def _join_in_out_pool_vars(uses: Dict[str, int], constructions: Dict[str, int], imap: IndexMap) -> List[LinProgConstraint]:
    result = []

    for var_name, use_amount in uses.items():
        if use_amount <= 1:
            continue

        in_pairs = [LinProgConstraint.VariableFactorPair(imap[IndexedVar(var_name, IndexedVar.Direction.IN, i)], 1.0) for i in range(use_amount)] 
        pool_pair = [LinProgConstraint.VariableFactorPair(imap[IndexedVar(var_name, IndexedVar.Direction.POOL, 0)], -1.0)]

        result.append(LinProgConstraint(in_pairs + pool_pair, LinProgConstraint.Type.EQUALITY, 0.0))

    for var_name, construction_amount in constructions.items():
        if construction_amount <= 1:
            continue

        out_pairs = [LinProgConstraint.VariableFactorPair(imap[IndexedVar(var_name, IndexedVar.Direction.OUT, i)], 1.0) for i in range(construction_amount)] 
        pool_pair = [LinProgConstraint.VariableFactorPair(imap[IndexedVar(var_name, IndexedVar.Direction.POOL, 0)], -1.0)]

        result.append(LinProgConstraint(out_pairs + pool_pair, LinProgConstraint.Type.EQUALITY, 0.0))
    
    return result

class LoweringResult(NamedTuple):
    constraints: List[LinProgConstraint]
    bounds: List[LinProgBound]
    goal: List[LinProgConstraint.VariableFactorPair]
    imap: IndexMap

def lower(rules: NormalizedRules, uses: Dict[str, int], constructions: Dict[str, int]) -> LoweringResult:

    imap = IndexMap()

    recipe_constraints = [c for recipe in rules.recipes for c in _lower_recipe(recipe, imap)]
    constraint_constraints = [c for constraint in rules.constraints for c in _lower_constraint(constraint, imap)]
    join_constraints = _join_in_out_pool_vars(uses, constructions, imap)

    goal_pairs = _lower_goal(rules.goal, imap)

    bounds = _vars_positive(imap)

    return LoweringResult(recipe_constraints + constraint_constraints + join_constraints, bounds, goal_pairs, imap)

if __name__ == '__main__':
    import sys
    from parse import Parser
    from transform import normalize_rules, index_rules
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

            print(lowering_result.constraints)
            print(lowering_result.bounds)
            print(lowering_result.goal)
            print(lowering_result.imap)