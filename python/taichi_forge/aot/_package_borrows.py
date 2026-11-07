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
    if not isinstance(node, ast.Name) or node.id != "self" or len(path) not in (1, 2):
        fail(call, "Borrowed Field requires a definition-proven self allocation path; untyped inputs are not sampled")
    source = schema.get(path[0], {})
    if (
        source.get("kind") != "dense_field"
        or source.get("conditions")
        or bool(source.get("struct_field")) != (len(path) == 2)
    ):
        fail(call, "Borrowed Field requires an earlier unconditional declared allocation or Struct Field member")
    mapping = keywords.get("dimensions", ast.Dict([], []))
    if not isinstance(mapping, ast.Dict):
        fail(call, "Borrowed dimensions require a literal mapping of source names to declared target extents")
    constraints, relations = {}, {}
    from taichi_forge_runtime._dimensions import Extent, extent_dependencies

    for key, value in zip(mapping.keys, mapping.values):
        if not isinstance(key, ast.Constant) or type(key.value) is not str or key.value in relations:
            fail(call, "Borrowed dimensions require unique literal source names")
        if schema.get(key.value, {}).get("kind") != "dimension" or schema[key.value].get("path"):
            fail(call, "Borrowed dimension is absent from the original allocation namespace")
        target = dimension(value)
        if not isinstance(target, Extent) or any(
            schema.get(n, {}).get("kind") != "dimension" or schema[n].get("path")
            for n in extent_dependencies(target.expression)
        ):
            fail(call, "Borrowed target relation requires dimensions in its declaration namespace")
        relations[key.value] = target.expression
        original = ("dim", key.value)
        if original != target.expression:
            for side, left, right in (("source", original, target.expression), ("target", target.expression, original)):
                identity = f"__package_borrow_{len(name)}_{name}_{len(key.value)}_{key.value}_{side}"
                constraints[identity] = dict(
                    spec=dict(left=left, right=right, when=name), source=["package", call.lineno]
                )
    return name, dict(
        kind="borrowed_field",
        source_path=path,
        dimensions=relations,
        borrow_declaration=[call.lineno, call.col_offset],
        **({"constraints": constraints} if constraints else {}),
    )


def project(owner, environment, schema):
    """Use the same source declaration and Component alias paths, without allocation."""
    from ._component import _capture_id, _source_scope_alias
    from ._package_records import write_member

    declarations, paths = dict(environment.declarations), dict(environment._paths)
    source_paths = dict(environment._source_scope_paths)
    for name, item in schema.items():
        if item["kind"] != "borrowed_field":
            continue
        path = tuple(item["source_path"])
        source_name = _capture_id(path)
        if source_name not in declarations:
            raise ValueError("Borrowed source member is absent from its declared Field type")
        _source_scope_alias(declarations, paths, source_paths, name, (name,), source_name)
        value = owner
        for part in path:
            value = getattr(value, part)
        write_member(owner, name, value, schema)
    environment._set_declarations(declarations)
    environment._paths = MappingProxyType(paths)
    environment._source_scope_paths = MappingProxyType(source_paths)
    return environment
