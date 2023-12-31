# This module transform the syntax representation in the parsing module to an equivalent representation where all expressions are linear.

import copy
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict
from typing import DefaultDict, Optional, Union, List, NamedTuple, Dict, Any
from error import Error
from parse import Expr as ASTExpr, Constant as ASTConstant, Term as ASTTerm, Variable as ASTVariable, Rules as ASTRules, Goal as ASTGoal, Recipe as ASTRecipe, Constraint as ASTConstraint
from report import report_error
from source import Span

@dataclass
class LinearExpression:
    # Maps variables to their coefficent.
    # The key `None` stores the constant part of the expression

    variable_coefficients: DefaultDict[Optional[Any], float]

    def add(self, other) -> "LinearExpression":
        result = defaultdict(float)

        for var, val in self.variable_coefficients.items():
            result[var] += val
        for var, val in other.variable_coefficients.items():
            result[var] += val
        
        return LinearExpression(result)

    def multiply(self, factor: Union[float, "_AggregatedFactor"]) -> "LinearExpression":
        result = defaultdict(float)

        if isinstance(factor, float):
            for var, val in self.variable_coefficients.items():
                result[var] = factor * val
        elif isinstance(factor, _AggregatedFactor):
            if factor.var is None:
                for var, val in self.variable_coefficients.items():
                    result[var] = factor.factor * val
            elif self.is_constant():
                result[factor.var] = self.constant_part * factor.val

        return LinearExpression(result)
    
    def is_constant(self) -> bool:
        for var, val in self.variable_coefficients.items():
            if var is not None and val != 0:
                return False
        return True

    @property
    def constant_part(self) -> float:
        return self.variable_coefficients[None] if None in self.variable_coefficients else 0.0
    
    def __str__(self) -> str:
        #terms = sorted(self.variable_coefficients.items(), key=lambda x: x[0])
        terms = self.variable_coefficients.items()
        return " + ".join([f"{val} {var if var is not None else ''}" for var, val in terms])
    
    def variables(self) -> List[str]:
        return [var for var in self.variable_coefficients.keys() if var is not None]

@dataclass
class NonLinearExpressionException(Error):
    rule: Union[ASTRecipe, ASTConstraint, ASTGoal, None]

    @property
    def span(self) -> List[Span]:
        if self.rule is None:
            assert False, "No rule for this NonLinearExpression defined"
        else:
            return [self.rule.span]
    
    def create_message(self) -> str:
        return f"Non-linear expression in {type(self.rule).__name__.lower()}"

@dataclass
class _AggregatedFactor:
    var: Optional[str]
    factor: float

    def multiply(self, value: Union[str, float, int]) -> "_AggregatedFactor" :
        new = copy.deepcopy(self)
        if isinstance(value, str):
            if self.var is not None:
                raise NonLinearExpressionException(None)
            new.var = value
        elif isinstance(value, float | int):
            new.factor *= value
        return new
    
    @staticmethod
    def default() -> "_AggregatedFactor":
        return _AggregatedFactor(None, 1)


def _linearize(expr: ASTExpr, context: LinearExpression, factor: _AggregatedFactor) -> LinearExpression :
    match expr:
        case ASTConstant(constant=c, span=_):
            result = factor.multiply(c)
            result_context = copy.deepcopy(context)
            result_context.variable_coefficients[result.var] += result.factor
            return result_context
        case ASTVariable(name=v, span=_):
            result = factor.multiply(v)
            result_context = copy.deepcopy(context)
            result_context.variable_coefficients[result.var] += result.factor
            return result_context
        case ASTTerm(lhs=lhs, op=op, rhs=rhs, span=_):
            match op:
                case ASTTerm.Op.ADD:
                    lhs_context = _linearize(lhs, context, factor)
                    rhs_context = _linearize(rhs, context, factor)
                    return lhs_context.add(rhs_context)
                case ASTTerm.Op.SUB:
                    if lhs is not None:
                        lhs_context = _linearize(lhs, context, factor)
                    else:
                        lhs_context = None

                    rhs_context = _linearize(rhs, context, factor.multiply(-1))

                    if lhs_context is None:
                        return rhs_context
                    else:
                        return lhs_context.add(rhs_context)

                case ASTTerm.Op.MUL:
                    lhs_context = _linearize(lhs, LinearExpression(defaultdict(float)), _AggregatedFactor.default())
                    rhs_context = _linearize(rhs, LinearExpression(defaultdict(float)), _AggregatedFactor.default())

                    # At least one side has to be constant, otherwise the result of multiplication would be nonlinear.
                    
                    if lhs_context.is_constant():
                        constant = lhs_context.constant_part
                        non_constant_context = rhs_context
                    elif rhs_context.is_constant():
                        constant = rhs_context.constant_part
                        non_constant_context = rhs_context
                    else:
                        constant = None

                    if constant is None:
                        print(expr.pretty_print())
                        raise NonLinearExpressionException(None)
                    
                    return context.add(non_constant_context.multiply(factor.multiply(constant)))
                case ASTTerm.Op.DIV:
                    rhs_context = _linearize(rhs, LinearExpression(defaultdict(float)), _AggregatedFactor.default())

                    if not rhs_context.is_constant():
                        raise NonLinearExpressionException()
                    
                    return _linearize(lhs, context, factor.multiply(1 / rhs_context.constant_part))
        case _:
            assert False, "Unreacheable"

def linearize(expr: ASTExpr) -> LinearExpression:
    return _linearize(expr, LinearExpression(defaultdict(float)), _AggregatedFactor.default())


class Constraint(NamedTuple):
    class Type(Enum):
        EQUALITY="=="
        LESS_EQ="<="

    term: LinearExpression
    ty: "Constraint.Type"
    bound: float

    def __str__(self) -> str:
        return f"{self.term} {self.ty.value} {self.bound}"


class Recipe(NamedTuple):
    inputs: LinearExpression
    outputs: LinearExpression

    def __str__(self) -> str:
        return f"{self.inputs} -> {self.outputs}"

class Goal(NamedTuple):
    ty: ASTGoal.Type
    term: LinearExpression 

    def __str__(self) -> str:
        return f"{self.ty} {self.term}"

class NormalizedRules(NamedTuple):
    recipes: List[Recipe]
    constraints: List[Constraint]
    goal: Goal

    def __str__(self) -> str:
        r = "\n".join([str(r) for r in self.recipes])
        c = "\n".join([str(c) for c in self.constraints])
        g = str(self.goal)

        return "\n\n".join([r, c, g])

# transforms the AST rules to the following normal form:
# - Linearized expressions
# - Constraints are constant on the left-hand side
def normalize_rules(rules: ASTRules) -> NormalizedRules:
    normalized_recipes = []
    for recipe in rules.recipes:
        try:
            linear_inputs = linearize(recipe.inputs)
            linear_outputs = linearize(recipe.outputs)
        except NonLinearExpressionException:
            raise NonLinearExpressionException(recipe)
        normalized_recipes.append(Recipe(linear_inputs, linear_outputs))

    try:
        normalized_goal = Goal(rules.goal.ty, linearize(rules.goal.term))
    except NonLinearExpressionException:
        raise NonLinearExpressionException(rules.goal)

    normalized_constraints = []
    for constraint in rules.constraints:
        try:
            lhs = linearize(constraint.lhs)
            rhs = linearize(constraint.rhs)
        except NonLinearExpressionException:
            raise NonLinearExpressionException(constraint)
        new_lhs = lhs.add(rhs.multiply(-1.0))

        constant_part = new_lhs.variable_coefficients[None]
        del new_lhs.variable_coefficients[None]

        match constraint.ty:
            case ASTConstraint.Type.EQUALITY:
                ty = Constraint.Type.EQUALITY
            case ASTConstraint.Type.LESS_EQ:
                ty = Constraint.Type.LESS_EQ
            case ASTConstraint.Type.GREATER_EQ:
                new_lhs = new_lhs.multiply(-1.0)
                constant_part *= -1
                ty = Constraint.Type.LESS_EQ

        normalized_constraints.append(Constraint(new_lhs, ty, -constant_part))
    
    return NormalizedRules(normalized_recipes, normalized_constraints, normalized_goal)


def _count_uses(recipes: List[Recipe]) -> Dict[str, int]:
    result = defaultdict(int)

    for recipe in recipes:
        for var in recipe.inputs.variables():
            result[var] += 1
    
    return result


def _count_constructions(recipes: List[Recipe]) -> Dict[str, int]:
    result = defaultdict(int)

    for recipe in recipes:
        for var in recipe.outputs.variables():
            result[var] += 1
    
    return result

_SUBSCRIPT_TRANSLATION = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
class IndexedVar(NamedTuple):
    class Direction(Enum):
        IN=auto()
        OUT=auto()
        POOL=auto()

    name: str
    dir: "IndexedVar.Direction"
    # Index is always zero if dir is POOL
    index: int

    def __eq__(self, other) -> bool:
        return self.name == other.name and self.dir == other.dir and self.index == other.index
    
    def __hash__(self):
        return hash((self.name, self.dir, self.index))
    
    def __str__(self):
        match self.dir:
            case IndexedVar.Direction.POOL:
                d = "ᴾᴼᴼᴸ"
            case IndexedVar.Direction.IN:
                d = "ᴵᴺ"
            case IndexedVar.Direction.OUT:
                d = "ᴼᵁᵀ"
        return f"{self.name}{d}{str(self.index).translate(_SUBSCRIPT_TRANSLATION)}"
    
    def pool(self) -> "IndexedVar":
        return IndexedVar(self.name, IndexedVar.Direction.POOL, 0)

def _index_expr(
    expr: LinearExpression, 
    dir: IndexedVar.Direction, 
    appearances: Dict[str, int], 
    next_index: Dict[str, int]
) -> LinearExpression:

    result = {}
    for var, val in expr.variable_coefficients.items():

        if var is None:
            result[None] = val
        
        match dir:
            case IndexedVar.Direction.POOL:
                result[IndexedVar(var, IndexedVar.Direction.POOL, 0)] = val
            case IndexedVar.Direction.IN | IndexedVar.Direction.OUT:
                if appearances[var] > 1:
                    idx = next_index[var]
                    next_index[var] += 1
                    result[IndexedVar(var, dir, idx)] = val
                else:
                    result[IndexedVar(var, IndexedVar.Direction.POOL, 0)] = val
    return LinearExpression(result)

class IndexResult(NamedTuple):
    uses: Dict[str, int]
    constructions: Dict[str, int]
    rules: NormalizedRules

def index_rules(rules: NormalizedRules) -> IndexResult:
    """
    In order to formulate recipes as a set of linear equations, variables that appear in multiple recipes
    need to be distinct.
    Then we can constrain the system so that all the sum of all out variables is equal to the sum of all input variables.
    This forces the the system to balance input and output.

    We therefore introduce three types of variables:
    - INPUT: Variable that appear in input of recipes (also called uses)
    - OUTPUT: Variable that appear in output of recipes (also called constructions)
    - POOL: This variable is equivalent to the sum of INPUT variables (likewise equivalent to the sum of OUTPUT variables).
            POOL Variables are used in constraints or the goal
    
    When a material is only used (created) in one recipe. We use the corresponding POOL variable instead to avoid unecessary variable bloat.
    """

    uses = _count_uses(rules.recipes)
    constructions = _count_constructions(rules.recipes)

    next_index_use = defaultdict(int)
    next_index_construct = defaultdict(int)

    indexed_recipes = []
    for recipe in rules.recipes:
        indexed_input = _index_expr(recipe.inputs, IndexedVar.Direction.IN, uses, next_index_use)
        indexed_output = _index_expr(recipe.outputs, IndexedVar.Direction.OUT, constructions, next_index_construct)
        indexed_recipes.append(Recipe(indexed_input, indexed_output))

    indexed_constraints = []
    for constraint in rules.constraints:
        indexed_expr = _index_expr(constraint.term, IndexedVar.Direction.POOL, None, None)
        indexed_constraints.append(Constraint(indexed_expr, constraint.ty, constraint.bound))

    indexed_goal_expr = _index_expr(rules.goal.term, IndexedVar.Direction.POOL, None, None)
    indexed_goal = Goal(rules.goal.ty, indexed_goal_expr)

    normalized_rules = NormalizedRules(indexed_recipes, indexed_constraints, indexed_goal)

    return IndexResult(uses, constructions, normalized_rules)


if __name__ == '__main__':
    import sys
    from parse import Parser

    if sys.argv[1] == "linearize":
        text = " ".join(sys.argv[2:])
        expr = Parser(text).parse_expr()
        linear_expr = linearize(expr)
        print(linear_expr)

    if sys.argv[1] in ["normalize", "index"]:
        with open(sys.argv[2]) as f:
            text = "\n".join(f.readlines())

            try:
                parser = Parser(text)
                rules = parser.parse()

                normalized_rules = normalize_rules(rules)
            except Error as error:
                report_error(error, text)
            else:
                if sys.argv[1] == "normalize":
                    print(normalized_rules)
                elif sys.argv[1] == "index":
                    indexed_rules = index_rules(normalized_rules)
                    print(indexed_rules.rules)

