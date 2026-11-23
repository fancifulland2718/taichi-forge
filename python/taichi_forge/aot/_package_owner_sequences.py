"""Use existing declaration proofs for independent Host-owned algorithms."""

import ast
import typing

from ._package_factory_inputs import constructor_calls


def _declared_element_types(cls, name, classes, imported):
    from ._package_factories import _source_value
    from ._package_input_families import owner_annotation_types

    for base in cls.__mro__:
        annotations = vars(base).get("__annotations__", {})
        if name not in annotations or base.__module__ not in imported:
            continue
        namespace = vars(imported[base.__module__])
        annotation = annotations[name]
        if isinstance(annotation, str):
            annotation = ast.parse(annotation, mode="eval").body
            if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
                annotation = ast.parse(annotation.value, mode="eval").body
            if not isinstance(annotation, ast.Subscript) or not isinstance(annotation.slice, ast.Tuple):
                break
            origin = _source_value(annotation.value, namespace)
            if isinstance(annotation.value, ast.Name) and annotation.value.id == "tuple":
                origin = namespace.get("tuple", tuple)
            args = annotation.slice.elts
            variable_length = len(args) == 2 and isinstance(args[1], ast.Constant) and args[1].value is Ellipsis
        else:
            origin, args = typing.get_origin(annotation), typing.get_args(annotation)
            variable_length = len(args) == 2 and args[1] is Ellipsis
        if origin in (tuple, typing.Tuple) and variable_length:
            element = args[0].__forward_arg__ if isinstance(args[0], typing.ForwardRef) else args[0]
            kinds = owner_annotation_types(element, namespace)
            if kinds and all(kind in classes for kind in kinds):
                return tuple(classes[kind] for kind in kinds)
        break
    return ()


def constructor_items(schema):
    for item in schema.values():
        if item["kind"] == "host_owner_sequence":
            yield from item.get("constructors", ())
        elif item.get("owner_alternatives"):
            for choice in item["owner_alternatives"]:
                for key in ("declaration_module", "declaration_owner"):
                    if key in item:
                        choice.setdefault(key, item[key])
                yield choice
        else:
            yield item


def resolve_sequences(sources, imported):
    from ._package_source_sequences import SourceSequences

    inventories = SourceSequences(sources, imported)
    classes = {
        getattr(imported[module], name): module + "." + name
        for module, _, _, _, _, _, schemas, _ in sources
        for name in schemas
    }
    for module, _, _, _, tree, _, schemas, _ in sources:
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name not in schemas:
                continue
            schema = schemas[node.name]
            sequences = {key: item for key, item in schema.items() if item["kind"] == "host_owner_sequence"}
            if not sequences:
                continue
            constructor = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")

            def recipe(value):
                return classes.get(value) if isinstance(value, type) else None

            calls, _ = constructor_calls(
                constructor,
                schema,
                vars(imported[module]),
                recipe,
                attributes=sequences.keys(),
                host_sequence=True,
                source_aliases=inventories.constructor_aliases(constructor, vars(imported[module])),
            )
            for name, item in sequences.items():
                domain = _declared_element_types(getattr(imported[module], node.name), name, classes, imported)
                declarations = [
                    dict(kind="owner", owner_type=owner, constructor_call=call, declaration_module=module)
                    for attribute, call, owner in calls
                    if attribute.split(".")[0] == name
                ]
                if not declarations and not domain:
                    raise ValueError(
                        "Host owner sequence has no proven constructor input domain: " + module + "." + name
                    )
                if any(call["constructor_call"].lineno >= item["sequence_declaration"][0] for call in declarations):
                    raise ValueError("Host owner sequence must be declared after its construction")
                if domain:
                    if any(child["owner_type"] not in domain for child in declarations):
                        raise ValueError("Host owner sequence construction contradicts its element type: " + name)
                    item["owner_types"] = domain
                item["constructors"] = declarations
