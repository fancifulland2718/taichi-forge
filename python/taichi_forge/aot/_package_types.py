"""Package projection of existing frontend types, never physical layout rules."""

import ast
import hashlib
import inspect
import json
from types import ModuleType

from taichi_forge.types._argument_descriptor import describe_element_type


def qualified_kind(node, kind):
    if getattr(node, "_package_role", None) is not None:
        return node._package_role
    if isinstance(node, ast.Call):
        node = node.func
    result = kind(node)
    if result is None and isinstance(node, ast.Attribute):
        parent = qualified_kind(node.value, kind)
        if parent is not None:
            result = parent + "." + node.attr
    return result


def resolve_type(node, namespace, *, allow_ndarray=False, allow_members=False):
    """Resolve only fixed type declarations; do not execute application factories."""
    import taichi_forge as ti

    def value(expression):
        if hasattr(expression, "_package_forwarded_argument"):
            return value(expression._package_forwarded_argument)
        if isinstance(expression, ast.Constant):
            return expression.value
        if isinstance(expression, (ast.Tuple, ast.List)):
            return tuple(value(item) for item in expression.elts)
        if allow_members and isinstance(expression, ast.Dict):
            if any(not isinstance(key, ast.Constant) or type(key.value) is not str for key in expression.keys):
                raise ValueError("Struct Field members require literal names")
            return {key.value: value(item) for key, item in zip(expression.keys, expression.values)}
        if isinstance(expression, ast.Name):
            defaults = dict(int=ti.i32, float=ti.f32, bool=ti.i32)
            if expression.id in defaults and expression.id not in namespace:
                return defaults[expression.id]
            return namespace[expression.id]
        if isinstance(expression, ast.Attribute):
            parent = value(expression.value)
            if allow_members and isinstance(parent, ti.lang.struct.StructType) and expression.attr == "members":
                return parent.members
            if not isinstance(parent, ModuleType):
                raise ValueError("Type paths must use module attributes")
            return inspect.getattr_static(parent, expression.attr)
        if isinstance(expression, ast.Call):
            factory = value(expression.func)
            factories = (ti.types.vector, ti.types.matrix, ti.types.struct)
            if allow_ndarray:
                factories += (ti.types.ndarray,)
            if factory not in factories:
                raise ValueError("Type construction requires a builtin vector/matrix/struct factory")
            if any(k.arg is None for k in expression.keywords):
                raise ValueError("Expanded type arguments are unsupported")
            return factory(*[value(a) for a in expression.args], **{k.arg: value(k.value) for k in expression.keywords})
        raise ValueError("Type declarations require fixed names, constants or builtin factories")

    result = value(node)
    if allow_members and type(result) is dict:
        result = ti.types.struct(**result)
    if allow_ndarray and isinstance(result, ti.types.ndarray):
        return result
    describe_element_type(result)  # Reject arbitrary Host objects.
    return result


def type_spec(dtype, module):
    """Publish struct ABI through Module; tensor/scalar metadata is logical only."""
    from taichi_forge.lang.struct import StructType
    from taichi_forge.types.ndarray_type import NdarrayType

    if isinstance(dtype, NdarrayType):
        if type(dtype.ndim) is not int or dtype.ndim < 0 or dtype.needs_grad:
            raise ValueError("Package ndarray annotations require a fixed rank and no gradients")
        return dict(
            kind="ndarray_annotation",
            element=type_spec(dtype.dtype, module),
            ndim=dtype.ndim,
            needs_grad=dtype.needs_grad,
            boundary=dtype.boundary.name,
        )

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
