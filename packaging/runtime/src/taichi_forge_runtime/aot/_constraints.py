"""Cold, symbolic requirements on the already selected execution bindings."""


def validate_constraints(constraints, used):
    from ._runtime import _name

    if not isinstance(constraints, list):
        raise ValueError("Execution constraints must be a list")

    def binding(value):
        _name(value)
        if value not in used:
            raise ValueError(f"Constraint references unused binding: {value}")

    def integer(value, *, positive=False):
        if type(value) is not int or not int(positive) <= value <= 0x7FFFFFFFFFFFFFFF:
            raise ValueError("Constraint dimensions and sizes must fit nonnegative int64")

    for constraint in constraints:
        if not isinstance(constraint, dict):
            raise ValueError("Execution constraints must be objects")
        if constraint.get("kind") == "array_layout":
            required = {"kind", "array", "record_bytes", "alignment"}
            if not required <= set(constraint) or set(constraint) - required - {
                "element_shape",
                "minimum_shape",
                "exact_shape",
            }:
                raise ValueError("Invalid array layout constraint fields")
            if ("minimum_shape" in constraint) == ("exact_shape" in constraint):
                raise ValueError("Array layout requires either minimum_shape or exact_shape")
            binding(constraint["array"])
            for key in ("minimum_shape", "exact_shape", "element_shape"):
                shape = constraint.get(key, [])
                if not isinstance(shape, list):
                    raise ValueError("Constraint shape must be a list")
                for axis in shape:
                    integer(axis, positive=key == "element_shape")
            integer(constraint["record_bytes"], positive=True)
            integer(constraint["alignment"], positive=True)
        elif constraint.get("kind") == "disjoint":
            if set(constraint) - {"allow_same_view"} != {"kind", "left", "right"}:
                raise ValueError("Invalid disjoint constraint fields")
            if type(constraint.get("allow_same_view", False)) is not bool:
                raise ValueError("allow_same_view must be a boolean")
            binding(constraint["left"])
            binding(constraint["right"])
        else:
            raise ValueError("Unsupported execution constraint")
