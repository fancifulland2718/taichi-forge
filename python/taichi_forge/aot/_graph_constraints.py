"""Persist selected Graph binding requirements without producer storage owners."""

import json

from taichi_forge.lang.exception import TaichiRuntimeError


def export_constraints(spec, binding, *, additional=()):
    constraints = {}

    def add(value):
        # Private aliases can canonicalize multiple symbolic requirements to one
        # binding. Keep distinct requirements, but store exact duplicates once.
        constraints.setdefault(json.dumps(value, sort_keys=True), value)

    for requirement in spec.memory_layout_requirements:
        if len(requirement) == 4:
            name, minimum, record_bytes, alignment = requirement
            element_shape, layout = (), "scalar"
        elif len(requirement) == 6:
            name, minimum, record_bytes, alignment, element_shape, layout = requirement
        else:
            raise TaichiRuntimeError("AOT Graph has an invalid memory layout certificate")
        if isinstance(minimum, (tuple, list)):
            if layout != "row_major_2d" or len(minimum) != 2:
                raise TaichiRuntimeError("AOT Graph needs a supported packed array layout certificate")
            shape = list(minimum)
        elif layout in ("scalar", "aos"):
            shape = [minimum]
        else:
            raise TaichiRuntimeError("AOT Graph needs a supported packed array layout certificate")
        # The AOT ndarray ABI supplies packed storage. It is a qualified subset
        # of Graph's row-pitched view contract, not an affine view encoding.
        add(
            dict(
                kind="array_layout",
                array=binding(name),
                minimum_shape=shape,
                element_shape=list(element_shape),
                record_bytes=record_bytes,
                alignment=alignment,
            )
        )
    for left, right in spec.memory_disjoint_pairs:
        left, right = sorted((binding(left), binding(right)))
        add(dict(kind="disjoint", left=left, right=right))
    for requirement in additional:
        add(requirement)
    return list(constraints.values())
