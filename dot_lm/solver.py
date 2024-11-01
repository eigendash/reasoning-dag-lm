"""A tiny decision procedure for the typed proposition blocks.

A proposition is a JSON object with one comparison at the top, over integer terms
that may themselves be small arithmetic expressions:

    {"gt": [4, 2]}                          4 > 2
    {"eq": [{"add": [3, 4]}, 7]}            3 + 4 = 7

This plays the role of the solver check the paper allows a validator to call for
typed blocks. It is deliberately a fragment small enough to decide by evaluation.
"""

import json
import operator

COMPARE = {
    "gt": operator.gt,
    "lt": operator.lt,
    "ge": operator.ge,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
}
ARITH = {"add": operator.add, "sub": operator.sub, "mul": operator.mul}
MAX_DEPTH = 4


class PropError(ValueError):
    pass


def _term(node, depth):
    if isinstance(node, bool):
        raise PropError("booleans are not terms")
    if isinstance(node, int):
        return node
    if isinstance(node, dict) and len(node) == 1 and depth < MAX_DEPTH:
        (op, args), = node.items()
        if op in ARITH and isinstance(args, list) and len(args) == 2:
            return ARITH[op](_term(args[0], depth + 1), _term(args[1], depth + 1))
    raise PropError(f"bad term: {node!r}")


def evaluate(prop):
    """Truth value of a parsed proposition. Raises PropError if it is malformed."""
    if not (isinstance(prop, dict) and len(prop) == 1):
        raise PropError("a proposition has exactly one top-level comparison")
    (op, args), = prop.items()
    if op not in COMPARE or not (isinstance(args, list) and len(args) == 2):
        raise PropError(f"bad comparison: {prop!r}")
    return bool(COMPARE[op](_term(args[0], 1), _term(args[1], 1)))


def parse_prop(text):
    """Parse and type-check a proposition string; return the object."""
    try:
        prop = json.loads(text)
    except json.JSONDecodeError as err:
        raise PropError(str(err)) from None
    evaluate(prop)
    return prop
