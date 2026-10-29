"""Shared bounded dimension declarations; no DSL or application evaluation."""

import operator


def validate_extent(expression):
    remaining = 1024

    def visit(node, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 32 or not isinstance(node, (tuple, list)) or not node:
            raise ValueError("A dimension expression must be a bounded declaration")
        op, *args = node
        if op == "dim":
            valid = len(args) == 1 and isinstance(args[0], str) and args[0].isidentifier()
        elif op == "const":
            valid = len(args) == 1 and type(args[0]) is int and 0 <= args[0] <= 2**31 - 1
        else:
            valid = op in ("add", "mul", "max", "min", "floordiv") and len(args) == 2
            if valid:
                for argument in args:
                    visit(argument, depth + 1)
        if not valid:
            raise ValueError(f"Invalid dimension expression {op!r}")

    visit(expression, 0)


def freeze_extent(expression):
    validate_extent(expression)

    def freeze(node):
        op, *args = node
        return (op, *args) if op in ("dim", "const") else (op, *(freeze(arg) for arg in args))

    return freeze(expression)


def evaluate_extent(expression, resolve):
    validate_extent(expression)

    def evaluate(node):
        op, *args = node
        if op == "dim":
            return operator.index(resolve(args[0]))
        if op == "const":
            return args[0]
        left, right = (evaluate(arg) for arg in args)
        if op == "add":
            result = left + right
        elif op == "mul":
            result = left * right
        elif op == "max":
            result = max(left, right)
        elif op == "min":
            result = min(left, right)
        elif op == "floordiv" and right > 0:
            result = left // right
        else:
            raise ValueError("A dimension divisor must be positive")
        if not 0 <= result <= 2**63 - 1:
            raise ValueError("Dimension expression exceeds its nonnegative i64 intermediate range")
        return result

    return evaluate(expression)


def extent_dependencies(expression):
    validate_extent(expression)
    dependencies = set()
    pending = [expression]
    while pending:
        op, *args = pending.pop()
        if op == "dim":
            dependencies.add(args[0])
        elif op != "const":
            pending.extend(args)
    return dependencies


def validate_constraints(constraints, dimensions, triggers):
    """Validate a bounded conjunction of declared dimension inequalities."""
    if not isinstance(constraints, dict) or len(constraints) > 1024:
        raise ValueError("Invalid component dimension constraints")
    for name, spec in constraints.items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or not isinstance(spec, dict)
            or set(spec) != {"left", "right", "when"}
            or not isinstance(spec["when"], str)
            or spec["when"] not in triggers
        ):
            raise ValueError(f"Invalid dimension constraint {name!r} or unused trigger")
        dependencies = extent_dependencies(spec["left"]) | extent_dependencies(spec["right"])
        if not dependencies or not dependencies <= dimensions:
            raise ValueError(f"Dimension constraint {name!r} requires declared dimensions")


def check_constraints(constraints, resolve, sources=None, *, entry=None):
    for name, spec in constraints.items():
        left, right = (evaluate_extent(spec[side], resolve) for side in ("left", "right"))
        if left > right:
            source = (sources or {}).get(name)
            location = f" (declared at {source[0]}:{source[1]})" if source else ""
            context = f"Component {entry!r}: " if entry else ""
            raise ValueError(
                context + f"Dimension constraint {name!r} for {spec['when']!r} failed: {left} > {right}{location}"
            )
