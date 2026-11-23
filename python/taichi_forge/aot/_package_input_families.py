"""Typed constructor contexts from declared calls, never from executed models."""

import ast
import copy
import hashlib
import inspect
import json
import types
import typing


def owner_annotation_types(annotation, namespace):
    """Read finite owner type annotations without evaluating Python expressions."""
    from ._package_factories import _source_value

    if isinstance(annotation, str):
        annotation = ast.parse(annotation, mode="eval").body
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return owner_annotation_types(annotation.value, namespace)
    def union(arms):
        choices = []
        for arm in arms:
            if arm is type(None) or isinstance(arm, ast.Constant) and arm.value is None:
                continue
            choice = owner_annotation_types(arm, namespace)
            if not choice:
                return None
            choices.extend(choice)
        return tuple(dict.fromkeys(choices))

    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return union((annotation.left, annotation.right))
    if isinstance(annotation, ast.Subscript):
        origin = _source_value(annotation.value, namespace)
        if origin is typing.Optional:
            return union((annotation.slice, type(None)))
        if origin is typing.Union:
            return union(annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else (annotation.slice,))
    if isinstance(annotation, ast.AST):
        annotation = _source_value(annotation, namespace)
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return union(typing.get_args(annotation))
    return (annotation,) if inspect.isclass(annotation) and annotation.__module__ != "builtins" else ()


def local_owner_annotations(constructor, namespace, stores, deleted):
    """Read single-write Host selections whose declared type is checked by owner_reference."""
    result = {}
    for statement in constructor.body:
        if not (isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
                and statement.value is not None and stores[statement.target.id] == 1
                and statement.target.id not in deleted and isinstance(statement.annotation, (ast.Name, ast.Attribute))):
            continue
        root = statement.annotation
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id not in stores:
            choices = owner_annotation_types(statement.annotation, namespace)
            if choices and len(choices) == 1:
                owner = choices[0]
                result[statement.target.id] = (statement.lineno, owner.__module__ + "." + owner.__qualname__)
    return result


def source_class(tree, name):
    return getattr(tree, "_package_owner_contexts", {}).get(name, name)


def split_constructor_inputs(sources, owner, inputs, sites, input_types):
    """Split resource contracts only for actual, fully proved argument tuples.

    Numerical definitions and the projected Host class remain single definitions.
    Model sizes and the ordering/count of constructed instances are not keys.
    """
    parameters = {name: item for (identity, name), item in inputs.items() if identity == owner}
    selections = []
    for caller, arguments, declaration in sites:
        selection = {}
        for name, item in parameters.items():
            values = input_types(caller, arguments.get(item["input_parameter"]))
            if values is None or len(values) != 1 or next(iter(values))[0] is None:
                raise ValueError("Typed resource inputs require one proven source at each constructor call: " + owner)
            identity, path = next(iter(values))
            if (item["kind"] == "owner_input") == bool(path):
                raise ValueError("Typed constructor input differs from its resource kind: " + owner)
            selection[name] = dict(source_type=identity, source_member=list(path))
        selections.append((selection, declaration))
    if not selections:
        raise ValueError("Typed resource inputs require declared constructor calls: " + owner)
    _publish_input_variants(sources, owner, parameters, selections)


def split_declared_owner_inputs(sources, owner, inputs, domains, sites, input_types):
    """Reuse typed contexts for a public finite union, including uncalled alternatives."""
    from itertools import product

    parameters = {name: inputs[owner, name] for name in domains}
    by_parameter = {item['input_parameter']: domains[name] for name, item in parameters.items()}
    selections = []
    for choices in product(*by_parameter.values()):
        chosen = dict(zip(by_parameter, choices))
        selections.append(({
            name: dict(source_type=chosen[item['input_parameter']], source_member=[])
            for name, item in parameters.items()
        }, None))
    for caller, arguments, declaration in sites:
        selection = {}
        for name, item in parameters.items():
            actual = input_types(caller, arguments.get(item['input_parameter']))
            if actual is None or len(actual) != 1:
                return False  # A forwarded source may resolve in a later preparation pass.
            identity, path = next(iter(actual))
            if path or identity not in domains[name]:
                raise ValueError('Owner input contradicts its declared source types: ' + owner + '.' + name)
            selection[name] = dict(source_type=identity, source_member=[])
        selections.append((selection, declaration))
    _publish_input_variants(sources, owner, parameters, selections)
    return True


def _publish_input_variants(sources, owner, parameters, selections):
    module, _, base = owner.rpartition(".")
    source = next(source for source in sources if source[0] == module)
    tree, schemas = source[4], source[6]
    contexts = getattr(tree, "_package_owner_contexts", {})
    families = getattr(tree, "_package_input_families", {})
    if base in contexts or base in families:
        raise ValueError("Recursive typed resource input contexts require resolved source definitions: " + owner)
    original = schemas.pop(base)
    variants = {}
    for selection, declaration in selections:
        digest = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
        name = base + "__input_" + digest
        if name not in variants:
            schema = copy.deepcopy(original)
            for attribute, origin in selection.items():
                if schema[attribute]["kind"] == "owner_input":
                    schema[attribute]["owner_type"] = origin["source_type"]
                else:
                    schema[attribute].update(origin)
            schemas[name] = schema
            contexts[name] = base
            variants[name] = {
                "owner": module + "." + name,
                "inputs": {
                    parameters[attribute]["input_parameter"]: {"kind": parameters[attribute]["kind"], **origin}
                    for attribute, origin in selection.items()
                },
            }
        if declaration is not None:
            declaration["owner_type"] = module + "." + name
    families[base] = list(variants.values())
    tree._package_owner_contexts = contexts
    tree._package_input_families = families
