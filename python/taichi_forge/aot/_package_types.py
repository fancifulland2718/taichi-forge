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


def project_tensor_axes(node, kind, dimension):
    """Mark definition-proven model axes on builtin type calls only."""
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        role = qualified_kind(call, kind)
        if isinstance(call.func, ast.Name) and call.func.id in ("_forge_package_vector", "_forge_package_matrix"):
            role = "types." + call.func.id.removeprefix("_forge_package_")
        axes = {"types.vector": ("n",), "types.matrix": ("n", "m")}.get(role)
        if axes is None:
            continue
        arguments = dict(zip((*axes, "dtype"), call.args))
        arguments.update((k.arg, k.value) for k in call.keywords)
        for axis in axes:
            if axis in arguments:
                projected = dimension(arguments[axis])
                if isinstance(projected, str):
                    arguments[axis]._package_model_extent = projected
                elif type(projected) is int:
                    # A declared definition constant is a fixed mathematical
                    # type axis, not a model tensor dimension or a Host lookup.
                    replacement = ast.copy_location(ast.Constant(projected), arguments[axis])
                    position = axes.index(axis)
                    if position < len(call.args):
                        call.args[position] = replacement
                    else:
                        for keyword in call.keywords:
                            if keyword.arg == axis:
                                keyword.value = replacement


def expand_local_type(node, bindings, kind):
    """Substitute single-assignment builtin type declarations, never call Host code."""
    import copy

    class Expand(ast.NodeTransformer):
        def visit_Name(self, node):
            if hasattr(node, "_package_forwarded_argument"):
                return self.visit(copy.deepcopy(node._package_forwarded_argument))
            return self.visit(copy.deepcopy(bindings[node.id])) if node.id in bindings else node

        def visit_Attribute(self, node):
            node = self.generic_visit(node)
            if node.attr == "members" and qualified_kind(node.value, kind) == "types.struct":
                call = node.value
                if call.args or any(k.arg is None for k in call.keywords):
                    raise ValueError("Local Struct type requires explicit named members")
                return ast.Dict([ast.Constant(k.arg) for k in call.keywords], [k.value for k in call.keywords])
            return node

    return Expand().visit(copy.deepcopy(node))


def member_initialization(node, members, source_members):
    """Resolve literal policies over fixed member names, not model-dependent maps."""
    if node is None:
        return {}

    def literal(value):
        try:
            return ast.literal_eval(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Member initialization requires literal names, policies and filters") from error

    if isinstance(node, ast.Dict) and all(k is not None for k in node.keys):
        pairs = [(literal(k), literal(v)) for k, v in zip(node.keys, node.values)]
    elif isinstance(node, ast.DictComp) and len(node.generators) == 1:
        loop = node.generators[0]
        if (
            loop.is_async
            or not isinstance(loop.target, ast.Name)
            or not isinstance(node.key, ast.Name)
            or node.key.id != loop.target.id
        ):
            raise ValueError("Member initialization requires one fixed member-name iteration")
        sequence = list(members) if ast.dump(loop.iter) == ast.dump(source_members) else literal(loop.iter)
        if not isinstance(sequence, (tuple, list)) or any(type(name) is not str for name in sequence):
            raise ValueError("Member initialization requires fixed member names")
        selected = list(sequence)
        for condition in loop.ifs:
            if not (
                isinstance(condition, ast.Compare)
                and isinstance(condition.left, ast.Name)
                and condition.left.id == loop.target.id
                and len(condition.ops) == 1
                and isinstance(condition.ops[0], (ast.In, ast.NotIn, ast.Eq, ast.NotEq))
            ):
                raise ValueError("Member initialization filters require literal member comparison")
            choices = literal(condition.comparators[0])
            if isinstance(condition.ops[0], (ast.Eq, ast.NotEq)):
                if type(choices) is not str:
                    raise ValueError("Member initialization equality requires a literal member name")
                choices = (choices,)
            if not isinstance(choices, (tuple, list)) or any(type(name) is not str for name in choices):
                raise ValueError("Member initialization filters require literal member names")
            selected = [
                name for name in selected if (name in choices) == isinstance(condition.ops[0], (ast.In, ast.Eq))
            ]
        policy = literal(node.value)
        pairs = [(name, policy) for name in selected]
    else:
        raise ValueError("Member initialization requires a literal map or fixed member-name comprehension")
    result = {}
    for name, policy in pairs:
        if (
            type(name) is not str
            or name not in members
            or name in result
            or policy not in ("zero", "consumer_explicit")
        ):
            raise ValueError("Member initialization requires unique declared members and existing policies")
        result[name] = policy
    return result


def resolve_type(node, namespace, *, allow_ndarray=False, allow_members=False, model_tensors=None, witness=None):
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
            if allow_members:
                defaults["dict"] = dict
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
            if allow_members:
                factories += (dict,)
            if allow_ndarray:
                factories += (ti.types.ndarray,)
            if factory not in factories:
                raise ValueError("Type construction requires a builtin vector/matrix/struct factory")
            if any(k.arg is None for k in expression.keywords):
                raise ValueError("Expanded type arguments are unsupported")
            if factory is dict and expression.args:
                raise ValueError("Struct member dict requires explicit named type arguments")
            axes = ("n",) if factory is ti.types.vector else ("n", "m") if factory is ti.types.matrix else ()
            arguments = dict(zip((*axes, "dtype"), expression.args)) if axes else {}
            arguments.update((k.arg, k.value) for k in expression.keywords)
            model_axes = {axis: getattr(arguments.get(axis), "_package_model_extent", None) for axis in axes}

            def argument(item):
                extent = getattr(item, "_package_model_extent", None)
                if extent is not None:
                    if model_tensors is None or witness is None:
                        raise ValueError("Model tensor type requires its Field dimension projection")
                    return witness(extent)
                return value(item)

            result = factory(
                *[argument(a) for a in expression.args], **{k.arg: argument(k.value) for k in expression.keywords}
            )
            if any(model_axes.values()):
                model_tensors[id(result)] = tuple(
                    model_axes[axis] or size for axis, size in zip(axes, result.get_shape())
                )
            return result
        raise ValueError("Type declarations require fixed names, constants or builtin factories")

    result = value(node)
    if allow_members and type(result) is dict:
        result = ti.types.struct(**result)
    if allow_ndarray and isinstance(result, ti.types.ndarray):
        return result
    describe_element_type(result)  # Reject arbitrary Host objects.
    return result


def type_spec(dtype, module, *, model_tensors=None):
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
        members = {
            member: type_spec(element, module, model_tensors=model_tensors) for member, element in dtype.members.items()
        }
        # A model-sized Field is a native SNode recipe, not a witness-sized
        # StructNdarray ABI. Fixed Struct types keep the existing type export.
        dynamic = any(part.get("model_tensor") for part in members.values())
        if not dynamic:
            module.add_ndarray_layout(name, dtype)
        return dict(
            kind="struct",
            **({} if dynamic else {"layout": name}),
            members=members,
        )
    if model_tensors and id(dtype) in model_tensors:
        return dict(
            kind="tensor",
            dtype=str(description.scalar_type),
            element_shape=list(model_tensors[id(dtype)]),
            model_tensor=True,
        )
    return dict(kind=description.category, dtype=str(description.scalar_type), element_shape=list(description.shape))


def type_expression(spec):
    return ast.Call(
        ast.Attribute(ast.Name("_forge_package_runtime", ast.Load()), "type", ast.Load()),
        [ast.parse(repr(spec), mode="eval").body],
        [],
    )
