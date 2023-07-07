# This module transform the syntax representation in the parsing module to an equivalent representation where all expressions are linear.

import copy
from dataclasses import dataclass
from collections import defaultdict
from typing import DefaultDict, Optional, Union
from parse import Expr, Constant, Term, Variable, Result, Goal, Recipe, Constraint

@dataclass
class LinearExpression:
    # Maps variables to their coefficent.
    # The key `None` stores the constant part of the expression
    variable_coefficients: DefaultDict[Optional[str], float]

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
                result[factor.var] = self.variable_coefficients[None] * factor.val

        return LinearExpression(result)
    
    def is_constant(self) -> bool:
        for var, val in self.variable_coefficients.items():
            if var is not None and val != 0:
                return False
        return True
    
    def __str__(self) -> str:
        #terms = sorted(self.variable_coefficients.items(), key=lambda x: x[0])
        terms = self.variable_coefficients.items()
        return " + ".join([f"{val} {var if var is not None else ''}" for var, val in terms])

@dataclass
class NonLinearExpression:
    pass

@dataclass
class _AggregatedFactor:
    var: Optional[str]
    factor: float

    def multiply(self, value: Union[str, float, int]) -> "_AggregatedFactor" :
        new = copy.deepcopy(self)
        if isinstance(value, str):
            if self.var is not None:
                raise NonLinearExpression()
            new.var = value
        elif isinstance(value, float | int):
            new.factor *= value
        return new
    
    @staticmethod
    def default() -> "_AggregatedFactor":
        return _AggregatedFactor(None, 1)


def _linearize(expr: Expr, context: LinearExpression, factor: _AggregatedFactor) -> LinearExpression :
    match expr:
        case Constant(c):
            result = factor.multiply(c)
            result_context = copy.deepcopy(context)
            result_context.variable_coefficients[result.var] += result.factor
            return result_context
        case Variable(v):
            result = factor.multiply(v)
            result_context = copy.deepcopy(context)
            result_context.variable_coefficients[result.var] += result.factor
            return result_context
        case Term(lhs, op, rhs):
            match op:
                case Term.Op.ADD:
                    lhs_context = _linearize(lhs, context, factor)
                    rhs_context = _linearize(rhs, context, factor)
                    return lhs_context.add(rhs_context)
                case Term.Op.SUB:
                    if lhs is not None:
                        lhs_context = _linearize(lhs, context, factor)
                    else:
                        lhs_context = None

                    rhs_context = _linearize(rhs, context, factor.multiply(-1))

                    if lhs_context is None:
                        return rhs_context
                    else:
                        return lhs_context.add(rhs_context)

                case Term.Op.MUL:
                    lhs_context = _linearize(lhs, LinearExpression(defaultdict(float)), _AggregatedFactor.default())
                    rhs_context = _linearize(rhs, LinearExpression(defaultdict(float)), _AggregatedFactor.default())

                    # At least one side has to be constant, otherwise the result of multiplication would be nonlinear.
                    
                    if lhs_context.is_constant():
                        constant = lhs_context.variable_coefficients[None]
                        non_constant_context = rhs_context
                    elif rhs_context.is_constant():
                        constant = rhs_context.variable_coefficients[None]
                        non_constant_context = rhs_context
                    else:
                        constant = None

                    if constant is None:
                        raise NonLinearExpression()
                    
                    return context.add(non_constant_context.multiply(factor.multiply(constant)))
                case Term.Op.DIV:
                    rhs_context = _linearize(rhs, LinearExpression(defaultdict(float)), _AggregatedFactor.default())

                    if not rhs_context.is_constant():
                        raise NonLinearExpression()
                    
                    return _linearize(lhs, context, factor.multiply(1 / rhs_context.variable_coefficients[None]))

def linearize(expr: Expr) -> LinearExpression:
    return _linearize(expr, LinearExpression(defaultdict(float)), _AggregatedFactor.default())

if __name__ == '__main__':
    import sys
    from parse import Parser
    text = " ".join(sys.argv[1:])
    expr = Parser(text).parse_expr()
    linear_expr = linearize(expr)
    print(linear_expr)