"""Package projection of existing frontend types, never physical layout rules."""

import ast
import hashlib
import inspect
import json
from types import ModuleType

from taichi_forge.types._argument_descriptor import describe_element_type


def qualified_kind(node, kind):
    if isinstance(node, ast.Call):
        node = node.func
    result = kind(node)
    if result is None and isinstance(node, ast.Attribute):
        parent = qualified_kind(node.value, kind)
        if parent is not None:
            result = parent + "." + node.attr
    return result


def resolve_type(node, namespace):
    """Resolve only fixed type declarations; do not execute application factories."""
    import taichi_forge as ti

    def value(expression):
        if isinstance(expression, ast.Constant):
            return expression.value
        if isinstance(expression, ast.Name):
            defaults = dict(int=ti.i32, float=ti.f32, bool=ti.i32)
            if expression.id in defaults and expression.id not in namespace:
                return defaults[expression.id]
            return namespace[expression.id]
        if isinstance(expression, ast.Attribute):
            parent = value(expression.value)
            if not isinstance(parent, ModuleType):
                raise ValueError("Type paths must use module attributes")
            return inspect.getattr_static(parent, expression.attr)
        if isinstance(expression, ast.Call):
            factory = value(expression.func)
            if factory not in (ti.types.vector, ti.types.matrix, ti.types.struct):
                raise ValueError("Type construction requires a builtin vector/matrix/struct factory")
            if any(k.arg is None for k in expression.keywords):
                raise ValueError("Expanded type arguments are unsupported")
            return factory(*[value(a) for a in expression.args], **{k.arg: value(k.value) for k in expression.keywords})
        raise ValueError("Type declarations require fixed names, constants or builtin factories")

    result = value(node)
    describe_element_type(result)  # Reject arbitrary Host objects.
    return result


def type_spec(dtype, module):
    """Publish struct ABI through Module; tensor/scalar metadata is logical only."""
    from taichi_forge.lang.struct import StructType

    description = describe_element_type(dtype)
    if not description.is_complete:
        raise ValueError("Package types require fixed element types and dimensions")
    if description.category == "struct":
        if not isinstance(dtype, StructType):
            raise ValueError("Structure exports require their original frontend type")
        name = "type_" + hashlib.sha256(json.dumps(description.structural_key(), sort_keys=True).encode()).hexdigest()
        module.add_ndarray_layout(name, dtype)
        return dict(
            kind="struct",
            layout=name,
            members={member: type_spec(element, module) for member, element in dtype.members.items()},
        )
    return dict(kind=description.category, dtype=str(description.scalar_type), element_shape=list(description.shape))


def type_expression(spec):
    return ast.Call(
        ast.Attribute(ast.Name("_forge_package_runtime", ast.Load()), "type", ast.Load()),
        [ast.parse(repr(spec), mode="eval").body],
        [],
    )
