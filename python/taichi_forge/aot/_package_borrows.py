"""Project proved source-scope aliases onto the existing Field layout contract."""

import ast
from types import MappingProxyType


def needs_source_proof(expression, schema):
    path, node = [], expression
    while isinstance(node, ast.Attribute):
        path.insert(0, node.attr)
        node = node.value
    return isinstance(node, ast.Name) and (
        node.id != "self"
        or bool(path) and schema.get(path[0], {}).get("kind") not in ("owner", "owner_input", "dense_field")
    )


def declaration(call, scope_path, schema, dimension, fail, *, preparation=False, namespace_check):
    """Admit known allocation paths, never infer a borrowed ABI from a sample."""
    if len(call.args) != 2 or not isinstance(call.args[0], ast.Constant):
        fail(call, "Package borrow requires a literal attribute and its declared source Field")
    name = call.args[0].value
    keywords = {k.arg: k.value for k in call.keywords}
    if (
        type(name) is not str
        or not name.isidentifier()
        or len(keywords) != len(call.keywords)
        or set(keywords) - {"dimensions", "source_scope", "external"}
        or not isinstance(keywords.get("source_scope"), ast.Constant)
        or keywords["source_scope"].value is not True
        or "external" in keywords
        and (not isinstance(keywords["external"], ast.Constant) or keywords["external"].value is not False)
    ):
        fail(call, "Package borrow currently requires source_scope=True and its visible allocation")
    path, node = [], call.args[1]
    while isinstance(node, ast.Attribute):
        path.insert(0, node.attr)
        node = node.value
    parameter = node.id if isinstance(node, ast.Name) and node.id != "self" and not path else None
    deferred = preparation and needs_source_proof(call.args[1], schema)
    if scope_path and parameter is not None and not deferred:
        fail(call, "Record borrowing requires a declared allocation path, not an untyped record input")
    if not deferred and parameter is None and (not isinstance(node, ast.Name) or node.id != "self" or not path):
        fail(call, "Borrowed Field requires a definition-proven self allocation path; untyped inputs are not sampled")
    source = schema.get(path[0], {}) if path else {}
    if (
        not deferred
        and parameter is None
        and source.get("kind") not in ("owner", "owner_input")
        and (
            source.get("kind") != "dense_field"
            or source.get("conditions")
            or len(path) not in ((1, 2) if source.get("struct_field") else (1,))
        )
    ):
        fail(call, "Borrowed Field requires an earlier unconditional declared allocation or Struct Field member")
    mapping = keywords.get("dimensions", ast.Dict([], []))
    if isinstance(mapping, ast.Constant) and mapping.value is None:
        mapping = ast.Dict([], [])
    # A normal declaration helper may return a fixed relation dictionary. Its
    # single-assignment local alias carries the same symbolic extents; retained
    # Host code still passes the actual dictionary to Runtime.borrow, which
    # checks every key/expression before binding (including later mutations).
    resolved = {}
    if isinstance(mapping, ast.Dict):
        for key, value in zip(mapping.keys, mapping.values):
            if not isinstance(key, ast.Constant) or type(key.value) is not str or key.value in resolved:
                fail(call, "Borrowed dimensions require unique literal source names")
            resolved[key.value] = dimension(value)
    else:
        resolved = dimension(mapping)
    if not isinstance(resolved, dict):
        fail(call, "Borrowed dimensions require a declared mapping of source names to target extents")
    relations = {}
    from taichi_forge_runtime._dimensions import Extent, extent_dependencies

    for key, target in resolved.items():
        if (
            not deferred
            and parameter is None
            and source.get("kind") not in ("owner", "owner_input")
            and (schema.get(key, {}).get("kind") != "dimension" or schema[key].get("path"))
        ):
            fail(call, "Borrowed dimension is absent from the original allocation namespace")
        if not isinstance(target, Extent) or any(
            schema.get(n, {}).get("kind") != "dimension" for n in extent_dependencies(target.expression)
        ):
            fail(call, "Borrowed target relation requires dimensions in its declaration namespace")
        namespace_check(target)
        relations[key] = target.expression
    from ._component import _capture_id

    return _capture_id((*scope_path, name)), dict(
        kind="borrowed_field",
        **({"path": [*scope_path, name]} if scope_path else {}),
        source_path=path,
        **(
            {"input_expression": call.args[1]}
            if deferred
            else {"input_parameter": parameter} if parameter is not None else {}
        ),
        dimensions=relations,
        borrow_declaration=[call.lineno, call.col_offset],
    )


def project(owner, environment, schema):
    """Use the same source declaration and Component alias paths, without allocation."""
    from ._component import (
        Dimension,
        DenseField,
        Record,
        _ConditionalDenseField,
        _ConditionalModelTensorField,
        _ModelTensorField,
        _capture_id,
        _source_scope_alias,
    )
    from ._package_records import write_member
    from taichi_forge_runtime._dimensions import extent_dependencies

    declarations, paths = dict(environment.declarations), dict(environment._paths)
    source_paths = dict(environment._source_scope_paths)
    constraints = dict(environment._constraints)
    for name, item in schema.items():
        if item["kind"] != "borrowed_field" or name in source_paths:
            continue
        path = tuple(item["source_path"])
        source_name = _capture_id(path)
        if source_name not in declarations:
            raise ValueError("Borrowed source member is absent from its declared Field type")
        source = next(
            (
                part
                for key, part in schema.items()
                if part["kind"] == "dense_field"
                and (
                    tuple(part.get("path", (key,))) == path
                    or part.get("struct_field")
                    and tuple(part.get("path", (key,))) == path[:-1]
                )
            ),
            None,
        )
        if source is None:
            raise ValueError("Borrowed Field requires a declared original allocation")
        if ("input_parameter" in item or "input_method" in item) and source.get("conditions"):
            # An actual Field argument proves existence, not a numerical option.
            # Retain the source predicate as admission on its allocation owner.
            item["source_presence"] = [
                dict(
                    path=list(paths[key]),
                    expected=expected,
                    **({"record": True} if isinstance(declarations[key], Record) else {}),
                )
                for key, expected in sorted(source["conditions"].items())
                if item.get("conditions", {}).get(key) is not expected
            ]
        elif any(
            item.get("conditions", {}).get(key) is not expected
            for key, expected in source.get("conditions", {}).items()
        ):
            raise ValueError("Borrowed Field requires its original allocation presence guards")
        whole_struct = source.get("struct_field") and isinstance(declarations[source_name], Record)
        if whole_struct and item.get("nonstruct_input"):
            raise ValueError("Source declaration helper excludes whole StructField inputs")
        if whole_struct and item["dimensions"]:
            raise ValueError("Whole StructField borrowing retains its original source namespace")
        namespace = path[:-2] if source.get("struct_field") and not whole_struct else path[:-1]
        checks = dict(item.get("constraints", {}))
        for dimension, expression in item["dimensions"].items():
            original = _capture_id((*namespace, dimension))
            if not isinstance(declarations.get(original), Dimension):
                raise ValueError("Borrowed dimension is absent from the original allocation namespace")
            original = ("dim", original)
            if original != expression:
                for side, left, right in (("source", original, expression), ("target", expression, original)):
                    identity = f"__package_borrow_{len(name)}_{name}_{len(dimension)}_{dimension}_{side}"
                    checks[identity] = dict(
                        spec=dict(left=left, right=right, when=name), source=["package", item["borrow_declaration"][0]]
                    )
        if checks:
            item["constraints"] = checks
            item["dimension_paths"] = {
                key: list(paths[key])
                for check in checks.values()
                for expression in (check["spec"]["left"], check["spec"]["right"])
                for key in extent_dependencies(expression)
            }
            constraints.update(checks)
        target_path = tuple(item.get("path", (name,)))
        members = source["element"]["members"] if whole_struct else (None,)
        if whole_struct:
            item["struct_borrow"] = True
            declarations[name], paths[name] = Record(optional=bool(item.get("conditions"))), target_path
        for member in members:
            alias_name = name if member is None else _capture_id((*target_path, member))
            alias_path = target_path if member is None else (*target_path, member)
            original_name = source_name if member is None else _capture_id((*path, member))
            _source_scope_alias(declarations, paths, source_paths, alias_name, alias_path, original_name)
            if item.get("source_presence"):
                original = declarations[alias_name]
                declarations[alias_name] = (
                    _ModelTensorField(original.dtype, original.shape, original.group, original.element_shape)
                    if isinstance(original, _ModelTensorField)
                    else DenseField(original.dtype, original.shape, original.group)
                )
            if item.get("conditions"):
                original = declarations[alias_name]
                conditions = tuple(sorted(item["conditions"].items()))
                declarations[alias_name] = (
                    _ConditionalModelTensorField(
                        original.dtype, original.shape, original.group, original.element_shape, conditions
                    )
                    if isinstance(original, _ModelTensorField)
                    else _ConditionalDenseField(original.dtype, original.shape, original.group, conditions)
                )
        value = owner
        for part in path:
            value = getattr(value, part)
        write_member(owner, name, value, schema)
    environment._set_declarations(declarations)
    environment._paths = MappingProxyType(paths)
    environment._source_scope_paths = MappingProxyType(source_paths)
    environment._constraints = MappingProxyType(constraints)
    return environment
