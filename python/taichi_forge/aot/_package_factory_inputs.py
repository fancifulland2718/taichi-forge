"""Project finite factory inputs from declarations, without running Host code.

Sequence positions are independent uses, not configurations of a combined
component. Only the static inputs reaching one factory call are expanded.
"""

import ast
import builtins
import copy
import itertools

from taichi_forge_runtime._static_values import read_static_key
from ._package_factories import _source_value


def constructor_calls(
    constructor, schema, namespace, recipe_for, *, attributes=None, host_sequence=False, source_aliases=None
):
    domains = {
        tuple(item.get("path", [name])): tuple(read_static_key(v) for v in item["choices"])
        for name, item in schema.items()
        if item["kind"] == "static" and not item["kernel_only"]
    }
    stores = {}
    for node in ast.walk(constructor):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            stores[node.id] = stores.get(node.id, 0) + 1
        elif isinstance(node, ast.arg):
            stores[node.arg] = stores.get(node.arg, 0) + 1
        elif isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Load):
            root = node.value
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name):
                stores[root.id] = stores.get(root.id, 0) + 2
    # Binding shadowing and value escape are different facts. Passing a tuple
    # result to Host code can invalidate the result's alias, but cannot rebind
    # the builtin used to construct it (or every other tuple in this owner).
    bound_names = frozenset(stores)
    for node in ast.walk(constructor):
        if not isinstance(node, ast.Call):
            continue
        if recipe_for(_source_value(node.func, namespace)) is not None:
            root = node.func
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in stores:
                raise ValueError("Cold kernel factory binding is shadowed in the constructor")
            continue
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in ("tuple", "zip", "len")
            and node.func.id not in bound_names
            and namespace.get(node.func.id, getattr(builtins, node.func.id)) is getattr(builtins, node.func.id)
        ):
            continue
        # A local sequence passed to an unknown call (including a method) may
        # be mutated. Do not forward its earlier syntax as an immutable fact.
        exposed = [*node.args, *(k.value for k in node.keywords)]
        if isinstance(node.func, ast.Attribute):
            exposed.append(node.func.value)
        for expression in exposed:
            for part in ast.walk(expression):
                if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load):
                    stores[part.id] = stores.get(part.id, 0) + 2
    # Mutation through an alias also invalidates its source. This conservative
    # dependency closure prevents copying the original syntax of an escaped list.
    dependencies = {}
    for node in ast.walk(constructor):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    dependencies.setdefault(target.id, set()).update(
                        part.id
                        for part in ast.walk(node.value)
                        if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load)
                    )
    pending = [name for name, count in stores.items() if count > 1]
    while pending:
        for name in dependencies.get(pending.pop(), ()):
            if stores.get(name, 0) <= 1:
                stores[name] = 2
                pending.append(name)
    aliases, covered, result = {}, set(), []

    def path(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.insert(0, node.attr)
            node = node.value
        return tuple(parts) if isinstance(node, ast.Name) and node.id == "self" else None

    class Substitute(ast.NodeTransformer):
        def __init__(self, names=None, selected=None):
            self.names, self.selected = names or {}, selected or {}

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in self.names:
                return copy.deepcopy(self.names[node.id])
            return node

        def visit_Attribute(self, node):
            key = path(node)
            if isinstance(node.ctx, ast.Load) and key in self.selected:
                return ast.copy_location(ast.Constant(self.selected[key]), node)
            return self.generic_visit(node)

        def visit_GeneratorExp(self, node):
            previous = self.names
            try:
                for generator in node.generators:
                    generator.iter = self.visit(generator.iter)
                    bound = {n.id for n in ast.walk(generator.target) if isinstance(n, ast.Name)}
                    self.names = {key: value for key, value in self.names.items() if key not in bound}
                    generator.ifs = [self.visit(n) for n in generator.ifs]
                node.elt = self.visit(node.elt)
            finally:
                self.names = previous
            return node

        visit_ListComp = visit_GeneratorExp

    def rewrite(node, names=None, selected=None):
        return Substitute(names, selected).visit(copy.deepcopy(node))

    def builtin(node, name):
        builtin_namespace = namespace.get("__builtins__", vars(builtins))
        if not isinstance(builtin_namespace, dict):
            builtin_namespace = vars(builtin_namespace)
        expected = getattr(builtins, name)
        return (
            isinstance(node, ast.Name)
            and node.id == name
            and node.id not in bound_names
            and namespace.get(name, expected) is expected
            and builtin_namespace.get(name) is expected
        )

    def sequence(node):
        if (host_sequence and isinstance(node, ast.Name) and stores.get(node.id) == 1
                and node.id in {arg.arg for arg in (*constructor.args.posonlyargs,
                    *constructor.args.args, *constructor.args.kwonlyargs)}):
            # A sequence parameter carries a source domain, not a fixed instance
            # count. Resolve its declared callers with the existing input pass.
            item = copy.deepcopy(node)
            item._package_source_sequence_parameter = node.id
            return [item]
        if isinstance(node, (ast.Tuple, ast.List)):
            return node.elts
        if isinstance(node, ast.Call) and builtin(node.func, "tuple") and len(node.args) == 1 and not node.keywords:
            return sequence(node.args[0])
        if isinstance(node, ast.Call) and builtin(node.func, "zip") and node.args and not node.keywords:
            return [ast.Tuple(list(items), ast.Load()) for items in zip(*(sequence(arg) for arg in node.args))]
        raise ValueError("Cold factory sequence requires fixed tuple/list/zip declaration inputs")

    def bind_target(target, value):
        if isinstance(target, ast.Name):
            return {target.id: value}
        if isinstance(target, (ast.Tuple, ast.List)):
            items = sequence(value)
            if len(items) == len(target.elts):
                result = {}
                for child, item in zip(target.elts, items):
                    bindings = bind_target(child, item)
                    if result.keys() & bindings.keys():
                        raise ValueError("Cold factory iteration target must name distinct inputs")
                    result.update(bindings)
                return result
        raise ValueError("Cold factory iteration target does not match its fixed sequence")

    def selections(node, selected):
        needed = sorted(
            {
                path(n)
                for n in ast.walk(node)
                if isinstance(n, ast.Attribute) and path(n) in domains and path(n) not in selected
            }
        )
        for values in itertools.product(*(domains[key] for key in needed)):
            yield {**selected, **dict(zip(needed, values))}

    def predicate(node):
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not predicate(node.operand)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            try:
                left, right = ast.literal_eval(node.left), ast.literal_eval(node.comparators[0])
            except (ValueError, TypeError):
                raise ValueError("Cold factory selection requires a declared finite condition") from None
            operation = node.ops[0]
            if isinstance(operation, (ast.Is, ast.IsNot)) and (left is None or right is None):
                return (left is right) if isinstance(operation, ast.Is) else (left is not right)
            if isinstance(operation, (ast.Eq, ast.NotEq)):
                return (left == right) if isinstance(operation, ast.Eq) else (left != right)
        raise ValueError("Cold factory selection requires a declared finite condition")

    def uses(node, attribute, selected=None):
        selected = {} if selected is None else selected
        if isinstance(node, ast.IfExp):
            if host_sequence:
                # Host decides whether to construct these independent objects.
                # Only their possible input contracts matter to numerical code.
                # Neither the condition nor the whole collection is specialized.
                yield from uses(node.body, attribute, selected)
                yield from uses(node.orelse, attribute, selected)
                return
            for selection in selections(node.test, selected):
                chosen = node.body if predicate(rewrite(node.test, selected=selection)) else node.orelse
                yield from uses(chosen, attribute, selection)
            return
        if isinstance(node, ast.Constant) and node.value is None:
            return
        if isinstance(node, ast.Call):
            recipe = recipe_for(_source_value(node.func, namespace))
            if recipe is not None:
                for selection in selections(node, selected):
                    yield attribute, rewrite(node, selected=selection), recipe
                return
            if builtin(node.func, "tuple") and len(node.args) == 1 and not node.keywords:
                yield from uses(node.args[0], attribute, selected)
                return
        if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
            # Only independently prepared owner instances may be filtered at
            # Host time. The input type pool stays conservative; fixed factory
            # sequences still need their exact original selection semantics.
            def identity_filter(test, target):
                if not (
                    isinstance(target, ast.Name)
                    and isinstance(test, ast.Compare)
                    and len(test.ops) == len(test.comparators) == 1
                ):
                    return False
                left, right = test.left, test.comparators[0]
                if isinstance(test.ops[0], (ast.In, ast.NotIn)):

                    def identity(node):
                        return (
                            isinstance(node, ast.Call)
                            and builtin(node.func, "id")
                            and len(node.args) == 1
                            and not node.keywords
                            and (isinstance(node.args[0], ast.Name) or path(node.args[0]) is not None)
                        )

                    collection = isinstance(right, ast.Name) or (
                        isinstance(right, (ast.Set, ast.Tuple, ast.List)) and all(identity(node) for node in right.elts)
                    )
                    if isinstance(right, ast.SetComp) and len(right.generators) == 1:
                        generator = right.generators[0]
                        names = {node.id for node in ast.walk(generator.target) if isinstance(node, ast.Name)}
                        collection = (
                            not generator.is_async
                            and not generator.ifs
                            and (isinstance(generator.iter, ast.Name) or path(generator.iter) is not None)
                            and identity(right.elt)
                            and isinstance(right.elt.args[0], ast.Name)
                            and right.elt.args[0].id in names
                        )
                    return (
                        isinstance(left, ast.Call)
                        and builtin(left.func, "id")
                        and len(left.args) == 1
                        and not left.keywords
                        and isinstance(left.args[0], ast.Name)
                        and left.args[0].id == target.id
                        and collection
                    )
                if not isinstance(test.ops[0], (ast.Is, ast.IsNot)):
                    return False
                if isinstance(right, ast.Name) and right.id == target.id:
                    left, right = right, left
                return (
                    isinstance(left, ast.Name)
                    and left.id == target.id
                    and (path(right) is not None or isinstance(right, ast.Constant) and right.value is None)
                )

            if (
                len(node.generators) != 1
                or node.generators[0].is_async
                or (node.generators[0].ifs and not host_sequence)
                or any(not identity_filter(test, node.generators[0].target) for test in node.generators[0].ifs)
            ):
                raise ValueError(
                    "Cold factory comprehension requires one fixed sequence with passive identity filters only"
                )
            generator = node.generators[0]
            for index, item in enumerate(sequence(generator.iter)):
                yield from uses(
                    rewrite(node.elt, names=bind_target(generator.target, item)), attribute + "." + str(index), selected
                )
            return
        if isinstance(node, (ast.Tuple, ast.List)):
            for index, item in enumerate(node.elts):
                yield from uses(item, attribute + "." + str(index), selected)
            return
        raise ValueError("Cold factory input domain has an unproven selection or call")

    for statement in constructor.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)) or statement.value is None:
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if len(targets) != 1:
            continue
        target = targets[0]
        expression = rewrite(statement.value, names=aliases)
        proven = {
            n.id: source_aliases[n.id]
            for n in ast.walk(target)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in (source_aliases or {})
        }
        if proven:
            aliases.update(proven)
            continue
        if isinstance(target, ast.Name) and stores[target.id] == 1:
            # Aliases forward syntax only. Mutated/rebound locals are not facts.
            aliases[target.id] = expression
            continue
        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"):
            continue
        if attributes is not None and target.attr not in attributes:
            continue
        call_nodes = [
            n
            for n in ast.walk(statement.value)
            if isinstance(n, ast.Call) and recipe_for(_source_value(n.func, namespace)) is not None
        ]
        if call_nodes:
            result.extend(uses(expression, target.attr))
            covered.update(id(n) for n in call_nodes)
    return result, covered
