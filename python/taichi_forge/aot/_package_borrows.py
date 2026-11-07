"""Project proved source-scope aliases onto the existing Field layout contract."""

import ast
from types import MappingProxyType


def declaration(call, scope_path, schema, dimension, fail):
    """Admit known allocation paths, never infer a borrowed ABI from a sample."""
    if scope_path or len(call.args) != 2 or not isinstance(call.args[0], ast.Constant):
        fail(call, "Package borrow requires a literal root attribute and its declared source Field")
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
    if parameter is None and (not isinstance(node, ast.Name) or node.id != "self" or not path):
        fail(call, "Borrowed Field requires a definition-proven self allocation path; untyped inputs are not sampled")
    source = schema.get(path[0], {}) if path else {}
    if (
        parameter is None
        and source.get("kind") not in ("owner", "owner_input")
        and (
            source.get("kind") != "dense_field"
            or source.get("conditions")
            or bool(source.get("struct_field")) != (len(path) == 2)
        )
    ):
        fail(call, "Borrowed Field requires an earlier unconditional declared allocation or Struct Field member")
    mapping = keywords.get("dimensions", ast.Dict([], []))
    if isinstance(mapping, ast.Constant) and mapping.value is None:
        mapping = ast.Dict([], [])
    if not isinstance(mapping, ast.Dict):
        fail(call, "Borrowed dimensions require a literal mapping of source names to declared target extents")
    relations = {}
    from taichi_forge_runtime._dimensions import Extent, extent_dependencies

    for key, value in zip(mapping.keys, mapping.values):
        if not isinstance(key, ast.Constant) or type(key.value) is not str or key.value in relations:
            fail(call, "Borrowed dimensions require unique literal source names")
        if (
            parameter is None
            and source.get("kind") not in ("owner", "owner_input")
            and (schema.get(key.value, {}).get("kind") != "dimension" or schema[key.value].get("path"))
        ):
            fail(call, "Borrowed dimension is absent from the original allocation namespace")
        target = dimension(value)
        if not isinstance(target, Extent) or any(
            schema.get(n, {}).get("kind") != "dimension" or schema[n].get("path")
            for n in extent_dependencies(target.expression)
        ):
            fail(call, "Borrowed target relation requires dimensions in its declaration namespace")
        relations[key.value] = target.expression
    return name, dict(
        kind="borrowed_field",
        source_path=path,
        **({"input_parameter": parameter} if parameter is not None else {}),
        dimensions=relations,
        borrow_declaration=[call.lineno, call.col_offset],
    )


def project(owner, environment, schema):
    """Use the same source declaration and Component alias paths, without allocation."""
    from ._component import Dimension, _capture_id, _source_scope_alias
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
        if source is None or source.get("conditions"):
            raise ValueError("Borrowed Field requires an unconditional original allocation")
        namespace = path[:-2] if source.get("struct_field") else path[:-1]
        checks = {}
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
        _source_scope_alias(declarations, paths, source_paths, name, tuple(item.get("path", (name,))), source_name)
        value = owner
        for part in path:
            value = getattr(value, part)
        write_member(owner, name, value, schema)
    environment._set_declarations(declarations)
    environment._paths = MappingProxyType(paths)
    environment._source_scope_paths = MappingProxyType(source_paths)
    environment._constraints = MappingProxyType(constraints)
    return environment
