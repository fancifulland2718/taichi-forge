"""Cold, symbolic requirements on the already selected execution bindings."""


def validate_constraints(constraints, used):
    from ._runtime import _DTYPES, _name

    if not isinstance(constraints, list):
        raise ValueError("Execution constraints must be a list")

    references = set()

    def binding(value):
        _name(value)
        if value not in used:
            raise ValueError(f"Constraint references unused binding: {value}")
        references.add(value)

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
                "dtype",
            }:
                raise ValueError("Invalid array layout constraint fields")
            if ("minimum_shape" in constraint) == ("exact_shape" in constraint):
                raise ValueError("Array layout requires either minimum_shape or exact_shape")
            binding(constraint["array"])
            if "dtype" in constraint and (
                not isinstance(constraint["dtype"], str) or constraint["dtype"] not in _DTYPES
            ):
                raise ValueError("Unsupported array layout dtype")
            for key in ("minimum_shape", "exact_shape", "element_shape"):
                shape = constraint.get(key, [])
                if not isinstance(shape, list):
                    raise ValueError("Constraint shape must be a list")
                for axis in shape:
                    integer(axis, positive=key == "element_shape")
            integer(constraint["record_bytes"], positive=True)
            integer(constraint["alignment"], positive=True)
        elif constraint.get("kind") == "struct_array_layout":
            if set(constraint) != {"kind", "array", "layout", "exact_shape"}:
                raise ValueError("Invalid structure layout constraint fields")
            binding(constraint["array"])
            _name(constraint["layout"])
            if not isinstance(constraint["exact_shape"], list):
                raise ValueError("Constraint shape must be a list")
            for axis in constraint["exact_shape"]:
                integer(axis)
        elif constraint.get("kind") == "disjoint":
            if set(constraint) - {"allow_same_view"} != {"kind", "left", "right"}:
                raise ValueError("Invalid disjoint constraint fields")
            if type(constraint.get("allow_same_view", False)) is not bool:
                raise ValueError("allow_same_view must be a boolean")
            binding(constraint["left"])
            binding(constraint["right"])
        elif constraint.get("kind") == "disjoint_writes":
            if set(constraint) != {"kind", "reads", "writes"}:
                raise ValueError("Invalid access group fields")
            reads, writes = constraint["reads"], constraint["writes"]
            if not isinstance(reads, list) or not isinstance(writes, list) or not writes:
                raise ValueError("Access group needs reads and nonempty writes lists")
            for read in reads:
                binding(read)
            for write in writes:
                if not isinstance(write, dict) or set(write) - {"inplace_read"} != {"array"}:
                    raise ValueError("Invalid access group write fields")
                binding(write["array"])
                if "inplace_read" in write:
                    index = write["inplace_read"]
                    integer(index)
                    if index >= len(reads):
                        raise ValueError("In-place read index is outside the access group")
        elif constraint.get("kind") == "resource_alias":
            if set(constraint) != {"kind", "left", "right", "left_path", "right_path"}:
                raise ValueError("Invalid resource alias constraint fields")
            binding(constraint["left"])
            binding(constraint["right"])
            for key in ("left_path", "right_path"):
                if not isinstance(constraint[key], list):
                    raise ValueError("Resource alias path must be a list")
                for port in constraint[key]:
                    _name(port)
        else:
            raise ValueError("Unsupported execution constraint")
    return references
