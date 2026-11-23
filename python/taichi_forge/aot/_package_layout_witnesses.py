"""Choose a symbolic capture layout for dimensions with disabled zero spans."""


def positive_layout_flags(schema):
    """Prefer the enabled span without enumerating models or static combinations.

    A capture includes all native allocation branches. The ordinary Host can
    represent an inactive domain by zero, which is never a physical Field
    extent. When a boolean gates a symbolic contribution against literal zero,
    its nonzero arm supplies a witness for the shared native recipe.
    """
    def zero(expr):
        if expr[0] == "const":
            return expr[1] == 0
        if expr[0] == "mul":
            return zero(expr[1]) or zero(expr[2])
        if expr[0] in ("add", "max", "min"):
            return zero(expr[1]) and zero(expr[2])
        return False

    choices = {}
    for item in schema.values():
        pending = list(item.get("dimension_expressions", {}).values())
        if item.get("expression"):
            pending.append(item["expression"])
        while pending:
            expr = pending.pop()
            if expr[0] == "if_layout" and expr[1][0] == "dim":
                yes, no = zero(expr[2]), zero(expr[3])
                if yes != no:
                    choices.setdefault(expr[1][1], set()).add(no)
            pending.extend(child for child in expr[1:] if isinstance(child, (tuple, list)))
    return {flag: next(iter(values)) for flag, values in choices.items() if len(values) == 1}
