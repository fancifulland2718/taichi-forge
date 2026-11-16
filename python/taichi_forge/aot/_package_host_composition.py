"""Keep pure Host composition out of instance resource specialization.

Declaration schemas still prove the types passed to independent numerical
entries. Only resource-free objects never consumed by numerical definitions
may discard those producer declarations in the retained Host projection.
"""

import ast

from ._package_types import qualified_kind


def mark_host_compositions(sources, function_inputs, factory_calls):
    schemas, families, nodes = {}, {}, {}
    required = {key for key, calls in factory_calls.items() if calls}
    for module, _, _, _, tree, kind, owners, definitions in sources:
        schemas.update((module + "." + name, schema) for name, schema in owners.items())
        families.update((module + "." + base, tuple(v["owner"] for v in variants))
                        for base, variants in getattr(tree, "_package_input_families", {}).items())
        for definition in definitions:
            if definition["owner"] is not None:
                required.add(module + "." + definition["owner"])
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                nodes[module + "." + node.name] = (node, kind)
    required.update(function_inputs)
    for calls in function_inputs.values():
        for call in calls:
            required.update(call["owner_inputs"].values())
            required.update(item["owner_type"] for item in call["field_inputs"].values()
                            if "owner_type" in item)

    def references(schema):
        for item in schema.values():
            for key in ("owner_type", "source_type"):
                if key in item:
                    yield item[key]
            for choice in (*item.get("owner_alternatives", ()), *item.get("constructors", ())):
                if "owner_type" in choice:
                    yield choice["owner_type"]

    # Numerical capture of any enclosing owner keeps its entire declared
    # dependency chain managed, including Host-looking intermediate records.
    pending = list(required)
    while pending:
        key = pending.pop()
        for child in (*families.get(key, ()), *references(schemas.get(key, {}))):
            if child not in required:
                required.add(child)
                pending.append(child)

    candidates = set()
    for key, (node, kind) in nodes.items():
        contexts = families.get(key, (key,))
        if node.bases or key in required or any(context in required for context in contexts):
            continue
        if not all(context in schemas for context in contexts):
            continue
        if any(item["kind"] not in ("owner_input", "owner", "static", "value", "record")
               for context in contexts for item in schemas[context].values()):
            continue
        scopes = {target.id for part in ast.walk(node) if isinstance(part, ast.Assign)
                  and isinstance(part.value, ast.Call)
                  and qualified_kind(part.value, kind) == "experimental.ResourceScope"
                  and len(part.value.args) == 1 and not part.value.keywords
                  and isinstance(part.value.args[0], ast.Name) and part.value.args[0].id == "self"
                  for target in part.targets if isinstance(target, ast.Name)}
        if not scopes:
            continue
        parents = {child: parent for parent in ast.walk(node) for child in ast.iter_child_nodes(parent)}
        valid = True
        for part in ast.walk(node):
            if isinstance(part, ast.Name) and part.id in scopes and isinstance(part.ctx, ast.Load):
                # Scopes must be used directly as attribute namespaces only;
                # no aliases, escaping callbacks or generic declaration helpers.
                parent = parents.get(part)
                if not isinstance(parent, ast.Attribute) or parent.value is not part:
                    valid = False
            if (isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute)
                    and isinstance(part.func.value, ast.Name) and part.func.value.id in scopes):
                if part.func.attr not in ("static", "record", "owner_reference"):
                    valid = False
                elif (not isinstance(parents.get(part), ast.Expr) or not part.args
                      or not isinstance(part.args[0], ast.Constant)
                      or type(part.args[0].value) is not str
                      or not part.args[0].value.isidentifier()
                      or part.func.attr != "static" and len(part.args) != 2):
                    valid = False
        if valid:
            candidates.add(key)

    # Constructed children must themselves be resource-free compositions.
    # Otherwise removing the managed parent would lose failure rollback.
    changed = True
    while changed:
        changed = False
        for key in tuple(candidates):
            if any(item["kind"] == "owner" and item.get("owner_type") not in candidates
                   for context in families.get(key, (key,)) for item in schemas[context].values()):
                candidates.remove(key)
                changed = True
    for module, _, _, _, tree, *_ in sources:
        tree._package_host_compositions = {key[len(module) + 1:] for key in candidates
                                          if key.rpartition(".")[0] == module}


class ProjectHostComposition(ast.NodeTransformer):
    def __init__(self, kind):
        self.kind, self.scopes = kind, set()

    def visit_Assign(self, node):
        if (isinstance(node.value, ast.Call)
                and qualified_kind(node.value, self.kind) == "experimental.ResourceScope"):
            self.scopes.update(target.id for target in node.targets if isinstance(target, ast.Name))
            node.value = ast.copy_location(ast.Name("self", ast.Load()), node.value)
        return self.generic_visit(node)

    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id in self.scopes):
            if call.func.attr == "static":
                return ast.copy_location(ast.Pass(), node)
            if call.func.attr in ("record", "owner_reference") and len(call.args) == 2:
                return ast.copy_location(ast.Expr(ast.Call(
                    ast.Attribute(ast.Name("_forge_package_runtime", ast.Load()), "host_reference", ast.Load()),
                    [ast.Name("self", ast.Load()), *call.args],
                    [*call.keywords, ast.keyword("record", ast.Constant(call.func.attr == "record"))])), node)
        return self.generic_visit(node)
