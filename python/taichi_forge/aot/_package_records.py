"""Paths for existing ResourceScope records; no application object construction."""

from types import MappingProxyType, SimpleNamespace


def read_member(owner, name, schema):
    for part in schema.get(name, {}).get("path", (name,)):
        owner = getattr(owner, part)
    return owner


def write_member(owner, name, value, schema):
    path = schema.get(name, {}).get("path", (name,))
    for part in path[:-1]:
        if not hasattr(owner, part):
            setattr(owner, part, SimpleNamespace())
        owner = getattr(owner, part)
    setattr(owner, path[-1], value)


def record_environment(declarations, schema):
    """Use Component's record paths while preserving already qualified dimensions."""
    from ._component import Component, Dimension, Record, _OptionalArray

    tree = {}
    for name, declaration in declarations.items():
        path = schema.get(name, {}).get("path", (name,))
        target = tree
        for part in path[:-1]:
            target = target.setdefault(part, {})
        if isinstance(declaration, Record) and schema.get(name, {}).get("kind") == "record":
            target.setdefault(path[-1], {})
        else:
            target[path[-1]] = (
                Dimension(declaration.kind) if isinstance(declaration, Dimension) and len(path) > 1 else declaration
            )

    from ._component import _capture_id

    def records(members, prefix=()):
        return {
            key: (
                Record(
                    optional=schema.get(_capture_id((*prefix, key)), {}).get("optional", False),
                    **records(value, (*prefix, key)),
                )
                if isinstance(value, dict)
                else value
            )
            for key, value in members.items()
        }

    environment = Component(**records(tree))
    environment._symbolic_dimensions = True
    flattened = dict(environment.declarations)
    for name, declaration in declarations.items():
        if isinstance(declaration, (Dimension, _OptionalArray)):
            flattened[name] = declaration
    environment._set_declarations(flattened)
    environment._constraints = MappingProxyType(
        {name: constraint for item in schema.values() for name, constraint in item.get("constraints", {}).items()}
    )
    return environment


def dimension_relations(environment, names):
    from taichi_forge_runtime._dimensions import extent_dependencies

    relations = {}

    def visit(name):
        if name in relations:
            return
        declaration = environment.declarations[name]
        relations[name] = dict(
            path=list(environment._paths[name]), kind=declaration.kind, expression=declaration.expression
        )
        for child in extent_dependencies(declaration.expression) if declaration.expression else ():
            visit(child)

    for name in names:
        visit(name)
    return relations
