from enum import Enum, auto
from typing import List, Optional, Tuple, cast
from dataclasses import dataclass
import re

class Expr:

    def pretty_print(self):
        match self:
            case Term(lhs, op, rhs):
                if isinstance(lhs, Term) and lhs.op.binding_power() < op.binding_power():
                    lhs_str = f"({lhs.pretty_print()})"
                elif lhs is None:
                    lhs_str = ""
                else:
                    lhs_str = lhs.pretty_print()

                if isinstance(rhs, Term) and rhs.op.binding_power() < op.binding_power():
                    rhs_str = f"({rhs.pretty_print()})"
                else:
                    rhs_str = rhs.pretty_print()

                match op:
                    case Term.Op.ADD:
                        op = "+"
                    case Term.Op.SUB:
                        op = "-"
                    case Term.Op.MUL:
                        op = "*"
                    case Term.Op.DIV:
                        op = "/"

                return f"{lhs_str} {op} {rhs_str}"
            case Constant(c):
                return str(c)
            case Variable(v):
                return v

@dataclass
class Term(Expr):
    class Op(Enum):
        ADD = 0
        SUB = 1
        MUL = 2
        DIV = 3

        def binding_power(self) -> int:
            match self:
                case Term.Op.ADD | Term.Op.SUB:
                    return 1
                case Term.Op.MUL | Term.Op.DIV:
                    return 2

    lhs: Optional[Expr]
    op: "Term.Op"
    rhs: Expr

@dataclass
class Constant(Expr):
    constant: float

@dataclass
class Variable(Expr):
    name: str

@dataclass
class Recipe:
    inputs: Expr
    outputs: Expr

    def pretty_print(self) -> str:
        return f"{self.inputs.pretty_print()} -> {self.outputs.pretty_print()}"

@dataclass
class Constraint:
    class Type(Enum):
        EQUALITY=auto()
        LESS_EQ=auto()
        GREATER_EQ=auto()

    lhs: Expr
    ty: "Constraint.Type"
    rhs: Expr

    def pretty_print(self) -> str:
        match self.ty:
            case Constraint.Type.EQUALITY:
                op = "=="
            case Constraint.Type.LESS_EQ:
                op = "<="
            case Constraint.Type.GREATER_EQ:
                op = ">="

        return f"{self.lhs.pretty_print()} {op} {self.rhs.pretty_print()}"

@dataclass
class Goal:
    class Type(Enum):
        MAX=auto()
        MIN=auto()

    term: Expr
    ty: "Goal.Type"

    def pretty_print(self) -> str:
        match self.ty:
            case Goal.Type.MIN:
                op = "min"
            case Goal.Type.MAX:
                op = "max"

        return f"{op} {self.term.pretty_print()}"

@dataclass
class Result:
    recipes: List[Recipe]
    constraints: List[Constraint]
    goal: Goal

    def pretty_print(self) -> str:
        r = "\n".join([recipe.pretty_print() for recipe in self.recipes])
        c = "\n".join([constraint.pretty_print() for constraint in self.constraints])
        g = self.goal.pretty_print()

        return "\n\n".join([r, c, g])


@dataclass
class Span:
    start: int
    end: int

    @staticmethod
    def start_len(start: int, len: int) -> "Span":
        return Span(start=start, end=start+len)

    @staticmethod
    def empty() -> "Span":
        return Span(start=0, end=0)

    @property
    def len(self) -> int:
        assert(self.end >= self.start)
        return self.end - self.start

# GRAMMAR:
class Token:
    class Type(Enum):
        EQUAL=auto()
        LESS_EQ=auto()
        GREATER_EQ=auto()
        RIGHT_ARROW=auto()

        MIN=auto()
        MAX=auto()

        PLUS=auto()
        MINUS=auto()
        SLASH=auto()
        STAR=auto()

        LPAREN=auto()
        RPAREN=auto()
        
        NEWLINE=auto()

        STRING=auto()
        NUMBER=auto()

        EOF=auto()

        def as_str(self):
            return Token(self, Span.empty()).as_str()

    def __init__(self, ty: "Token.Type", span: Span, data: Optional[str]=None):
        assert(data is not None if ty in [Token.Type.STRING, Token.Type.NUMBER] else True)
        self.ty = ty
        self.data = data
        self.span = span

    def as_str(self) -> str:
        match self.ty:
            case Token.Type.PLUS:
                return "+"
            case Token.Type.MINUS:
                return "-"
            case Token.Type.SLASH:
                return "/"
            case Token.Type.STAR:
                return "*"
            case Token.Type.STRING | Token.Type.NUMBER:
                return str(self.data)
            case Token.Type.EQUAL:
                return "=="
            case Token.Type.LESS_EQ:
                return "<="
            case Token.Type.GREATER_EQ:
                return ">="
            case Token.Type.RIGHT_ARROW:
                return "->"
            case Token.Type.MIN:
                return "min"
            case Token.Type.MAX:
                return "max"
            case Token.Type.NEWLINE:
                return "\n"
            case Token.Type.LPAREN:
                return "("
            case Token.Type.RPAREN:
                return ")"
            case Token.Type.EOF:
                return ""
    def __repr__(self):
        data = f" '{self.data}'" if self.ty in [Token.Type.STRING, Token.Type.NUMBER] else ""
        return f"<Token {self.ty}{data}>"

@dataclass
class InvalidToken(Exception):
    position: int

@dataclass
class DuplicateGoal(Exception):
    first_goal: Goal 
    second_goal: Goal

@dataclass
class MissingGoal(Exception):
    pass

@dataclass
class UnexpectedToken(Exception):
    token: Token
    context: str  # The thing we were parsing at the moment


ParseException = InvalidToken | DuplicateGoal | MissingGoal | UnexpectedToken

_NUMBER_PATTERN = re.compile(r"^\d+(\.\d*)?")
_STRING_PATTERN = re.compile(r"^(([A-z][A-z ]*[A-z])|[A-z])")

class Lexer:
    def __init__(self, input: str):
        self.input: str = input
        self.current_pos: int = 0

    def next_token(self) -> Token:
        self._skip_whitespace_and_comment()
        remainder = self.input[self.current_pos:]

        for token_type in Token.Type:
            if token_type in [Token.Type.STRING, Token.Type.NUMBER, Token.Type.EOF]:
                continue
            
            if remainder.startswith(token_type.as_str()):
                token_len = len(token_type.as_str())
                span = Span.start_len(self.current_pos, token_len)
                self.current_pos += token_len
                return Token(token_type, span)

        if len(remainder) == 0:
            return Token(Token.Type.EOF, Span.start_len(self.current_pos, 0))
        elif (number := _NUMBER_PATTERN.search(remainder)) is not None:
            token_len = len(number.group(0))
            span = Span.start_len(self.current_pos, token_len)
            self.current_pos += token_len
            return Token(Token.Type.NUMBER, span, number.group(0))
        elif (string := _STRING_PATTERN.search(remainder)) is not None:
            token_len = len(string.group(0))
            span = Span.start_len(self.current_pos, token_len)
            self.current_pos += token_len
            return Token(Token.Type.STRING, span, string.group(0))
        else:
            raise InvalidToken(self.current_pos)


    def _skip_whitespace_and_comment(self):
        idx = self.current_pos

        nested_depth = 0
        in_line_comment = False

        while True:
            if len(self.input[idx:]) == 0:
                break
            if self.input[idx:].startswith("/*"):
                nested_depth += 1
                idx += 2
            elif nested_depth > 0 and self.input[idx:].startswith("*/"):
                nested_depth -= 1
                idx += 2
            elif self.input[idx:].startswith("//"):
                in_line_comment = True
                idx += 2
            elif self.input[idx] == '\n':
                in_line_comment = False
                break
            elif self.input[idx] in [' ', '\t'] or nested_depth > 0 or in_line_comment:
                idx += 1
            else:
                break

        self.current_pos = idx

# GRAMMAR Rules:
# S = (Recipe | Constraint | Goal)+
# Recipe = Expr -> Expr \n
# Constraint = Expr cOp Expr \n
# cOp = '<=' | '==' | '>='
# Goal = gOp Expr\n
# gOp = 'min' | 'max'
# Expr = Atom (tOp Expr)*
# tOp = '+' | '-' | '/' | '*'
# Atom = Constant | Variable | '(' Expr ')' | -Atom
# Constant = \d+(\.\d*)?
# Variable = \w[A-z ]*\w | \w


class Parser:
    def __init__(self, s: str):
        self.lexer = Lexer(s)
        self._current_token = None

    @property
    def current_token(self) -> Token:
        if self._current_token is None:
            return self.next_token()
        else:
            return self._current_token

    def next_token(self) -> Token:
        self._current_token = self.lexer.next_token()
        return self._current_token

    GOAL_START_TOKENS = [Token.Type.MAX, Token.Type.MIN]
    EXPR_START_TOKENS = [Token.Type.NUMBER, Token.Type.STRING, Token.Type.LPAREN]
    CONSTRAINT_OP_TOKENS = [Token.Type.EQUAL, Token.Type.LESS_EQ, Token.Type.GREATER_EQ]
    RULE_TERMINATOR_TOKENS = [Token.Type.NEWLINE, Token.Type.EOF]
    OPERATOR_TOKENS = [Token.Type.PLUS, Token.Type.MINUS, Token.Type.SLASH, Token.Type.STAR]

    def parse_goal(self) -> Goal:
        match self.current_token.ty:
            case Token.Type.MAX:
                ty = Goal.Type.MAX
            case Token.Type.MIN:
                ty = Goal.Type.MIN
            case _:
                assert False, "unreacheable"
        self.next_token()

        expr = self.parse_expr()
        return Goal(expr, ty)

    def parse_atom(self) -> Expr:

        negate = self.current_token.ty == Token.Type.MINUS

        if negate:
            self.next_token()

        match self.current_token.ty:
            case Token.Type.NUMBER:
                result = Constant(float(cast(str, self.current_token.data)))
                self.next_token()
            case Token.Type.STRING:
                result = Variable(cast(str, self.current_token.data))
                self.next_token()
            case Token.Type.LPAREN:
                self.next_token()
                result = self.parse_expr()
                if (unexpected_token := self.current_token).ty != Token.Type.RPAREN:
                    raise UnexpectedToken(unexpected_token, "Expression in parentheses")
                self.next_token()
            case _:
                raise UnexpectedToken(self.current_token, "Atom")

        if negate:
            result = Term(None, Term.Op.SUB, result)

        return result


    def parse_expr(self, min_precedence=0) -> Expr:
        lhs = self.parse_atom()

        while True:
            op = self.current_token

            implicit_multiplication =  op.ty in [Token.Type.STRING, Token.Type.NUMBER, Token.Type.LPAREN]

            if op.ty not in Parser.OPERATOR_TOKENS and not implicit_multiplication :
                break

            l_bp, r_bp = self.binding_power(op.ty, implicit_multiplication)

            if l_bp < min_precedence:
                break

            if not implicit_multiplication:
                self.next_token()

            rhs = self.parse_expr(r_bp)

            lhs = Term(lhs, self.token_op_to_term_op(op.ty, implicit_multiplication), rhs)

        return lhs


    def binding_power(self, token: Token.Type, implicit_multiplication: bool) -> Tuple[int, int]:
        match (token, implicit_multiplication):
            case (Token.Type.PLUS, False) | (Token.Type.MINUS, False):
                return (1, 2)
            case (Token.Type.SLASH, False) | (Token.Type.STAR, False) | (_, True):
                return (3, 4)
            case _:
                assert False, f"token {token} is not an operator"

    def token_op_to_term_op(self, token: Token.Type, implicit_multiplication: bool) -> Term.Op:
        match (token, implicit_multiplication):
            case (Token.Type.PLUS, False):
                return Term.Op.ADD
            case (Token.Type.MINUS, False):
                return Term.Op.SUB
            case (Token.Type.SLASH, False):
                return Term.Op.DIV
            case (Token.Type.STAR, False) | (_, True):
                return Term.Op.MUL
            case _:
                assert False, f"token {token} is not an operator"

    def parse(self) -> Result:
        constraints = []
        recipes = []
        goal = None

        while self.current_token.ty != Token.Type.EOF:
            if self.current_token.ty in Parser.GOAL_START_TOKENS:
                new_goal = self.parse_goal();
                if goal is None:
                    goal = new_goal
                else:
                    raise DuplicateGoal(goal, new_goal)
            elif self.current_token.ty in Parser.EXPR_START_TOKENS:
                lhs = self.parse_expr()

                if self.current_token.ty == Token.Type.RIGHT_ARROW:
                    # Parsing Recipe
                    self.next_token()
                    rhs = self.parse_expr()
                    recipes.append(Recipe(lhs, rhs))
                elif self.current_token.ty in Parser.CONSTRAINT_OP_TOKENS:
                    # Parsing Constraint
                    match self.current_token.ty:
                        case Token.Type.EQUAL:
                            ty = Constraint.Type.EQUALITY
                        case Token.Type.LESS_EQ:
                            ty = Constraint.Type.LESS_EQ
                        case Token.Type.GREATER_EQ:
                            ty = Constraint.Type.GREATER_EQ
                        case _:
                            assert False, "Unreacheable"
                    self.next_token()
                    rhs = self.parse_expr()
                    constraints.append(Constraint(lhs, ty, rhs))
                else:
                    raise UnexpectedToken(self.current_token, "recipe or constraint")
            elif self.current_token.ty == Token.Type.NEWLINE:
                self.next_token()
            else:
                raise UnexpectedToken(self.current_token, "Rule list")


        if goal is None:
            raise MissingGoal()

        return Result(recipes, constraints, goal)
        
if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f:
        text = "\n".join(f.readlines())

        parser = Parser(text)
        result = parser.parse_expr()

        print(result.pretty_print())

