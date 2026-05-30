# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Arith Eval
#
# > Single unified evaluator for ArithTerm (compile-time and runtime).

# %%
from typing import Any, Callable
from relann.pydantic_classes import ArithTerm, Var


def evaluate_arith_term(term: ArithTerm, resolver: Callable[[Var], Any]) -> Any:
    """
    Recursively evaluate ArithTerm using a resolver callback for Var resolution.

    Used for both compile-time (Engine symbol table) and runtime (Selection DataFrame)
    evaluation. The resolver takes a Var and returns its resolved value.

    Args:
        term: ArithTerm to evaluate
        resolver: Function that takes a Var and returns its resolved value

    Returns:
        Evaluated result (scalar, Series, tensor, etc. depending on resolver)
    """
    if term is None:
        return None
    if term.op is None:
        value = term.value
        if isinstance(value, Var):
            return resolver(value)
        return value

    if not term.sons or len(term.sons) == 0:
        raise ValueError(f"Arithmetic term with op '{term.op}' requires operands")
    operands = [evaluate_arith_term(s, resolver) for s in term.sons]
    match term.op:
        case "+":
            return operands[0] + operands[1]
        case "-":
            return operands[0] - operands[1]
        case "*":
            return operands[0] * operands[1]
        case "/":
            return operands[0] / operands[1]
        case "//":
            return operands[0] // operands[1]
        case "**":
            return operands[0] ** operands[1]
        case _:
            raise NotImplementedError(f"Unsupported arithmetic operator '{term.op}'")
