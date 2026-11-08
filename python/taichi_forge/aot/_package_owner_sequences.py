"""Use existing declaration proofs for independent Host-owned algorithms."""

import ast

from ._package_factory_inputs import constructor_calls


def constructor_items(schema):
    for item in schema.values():
        if item["kind"] == "host_owner_sequence":
            yield from item.get("constructors", ())
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
                declarations = [
                    dict(kind="owner", owner_type=owner, constructor_call=call, declaration_module=module)
                    for attribute, call, owner in calls
                    if attribute.split(".")[0] == name
                ]
                if not declarations:
                    raise ValueError(
                        "Host owner sequence has no proven constructor input domain: " + module + "." + name
                    )
                if any(call["constructor_call"].lineno >= item["sequence_declaration"][0] for call in declarations):
                    raise ValueError("Host owner sequence must be declared after its construction")
                item["constructors"] = declarations
