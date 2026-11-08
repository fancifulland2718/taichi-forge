"""Prove free-kernel receiver domains from ordinary source calls.

Only owner type/path facts flow through Host functions. This does not execute
constructors, Host branches or numerical definitions to discover input types.
"""

import ast
from collections import deque
from dataclasses import dataclass
import inspect

from ._package_factories import _source_value
from ._package_input_families import source_class


@dataclass(frozen=True)
class _Owner:
    identity: str


def discover(sources, imported):
    from taichi_forge.types.annotations import Template

    owners, classes, functions, kernels = {}, {}, {}, {}
    for module, _, _, _, tree, _, schemas, definitions in sources:
        namespace = vars(imported[module])
        for name, schema in schemas.items():
            identity = module + "." + name
            owners[identity] = schema
            classes[identity] = namespace[source_class(tree, name)]
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                function = namespace.get(node.name)
                if inspect.isfunction(function) and not node.decorator_list:
                    functions[function] = (module, node)
            elif isinstance(node, ast.ClassDef):
                cls = namespace.get(node.name)
                if inspect.isclass(cls):
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef) and not method.decorator_list:
                            function = vars(cls).get(method.name)
                            if inspect.isfunction(function):
                                functions[function] = (module, method)
        for definition in definitions:
            if definition["kind"] != "kernel" or definition["owner"] is not None:
                continue
            entry = namespace.get(definition["node"].name)
            kernel = getattr(entry, "_primal", None)
            if kernel is None:
                continue
            templates = [a.name for a in kernel.arguments if isinstance(a.annotation, Template)]
            if len(templates) == 1:
                kernels[entry] = (module, definition, templates[0])
    if not kernels:
        return {}, []

    pending = deque()
    for identity, cls in classes.items():
        seen = set()
        for base in cls.__mro__:
            for name, function in vars(base).items():
                if name in seen:
                    continue
                seen.add(name)
                if inspect.isfunction(function) and function in functions:
                    parameters = list(inspect.signature(function).parameters)
                    if parameters and parameters[0] == "self":
                        pending.append((function, {"self": _Owner(identity)}))

    visited, domains, covered, failures = set(), {}, set(), []
    invalid = set()

    def member(owner, name):
        if not isinstance(owner, _Owner):
            return None
        item = owners[owner.identity].get(name, {})
        if item.get("kind") in ("owner", "owner_input") and item.get("owner_type") in owners:
            return _Owner(item["owner_type"])
        return None

    while pending:
        function, arguments = pending.popleft()
        identity = (function, tuple(sorted(arguments.items())))
        if identity in visited:
            continue
        visited.add(identity)
        module, node = functions[function]
        namespace = function.__globals__
        # A rebinding invalidates an earlier owner fact. Nested definitions have
        # their own lexical scope and cannot lend assignments to this call.
        nodes = []

        def walk(statement):
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                return
            nodes.append(statement)
            for child in ast.iter_child_nodes(statement):
                walk(child)

        for statement in node.body:
            walk(statement)
        stores, aliases = {}, {}
        for part in nodes:
            if isinstance(part, ast.Name) and not isinstance(part.ctx, ast.Load):
                stores[part.id] = stores.get(part.id, 0) + 1
            if isinstance(part, ast.Assign) and len(part.targets) == 1 and isinstance(part.targets[0], ast.Name):
                aliases[part.targets[0].id] = part.value
        local_names = set(stores) | set(inspect.signature(function).parameters)

        def value(expression, visiting=()):
            if isinstance(expression, ast.Name):
                name = expression.id
                if name in arguments and name not in stores:
                    return arguments[name]
                if name in aliases and stores.get(name) == 1 and name not in visiting:
                    return value(aliases[name], (*visiting, name))
                return None
            if isinstance(expression, ast.Attribute):
                return member(value(expression.value, visiting), expression.attr)
            return None

        def target(expression):
            root = expression
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in local_names:
                if isinstance(expression, ast.Attribute):
                    receiver = value(expression.value)
                    if isinstance(receiver, _Owner):
                        cls = classes[receiver.identity]
                        method = inspect.getattr_static(cls, expression.attr, None)
                        if inspect.isfunction(method) and method in functions:
                            return method, receiver
                return None, None
            result = _source_value(expression, namespace)
            return (result, None) if inspect.isfunction(result) else (None, None)

        for call in (part for part in nodes if isinstance(part, ast.Call)):
            callee, receiver = target(call.func)
            if callee not in kernels and callee not in functions:
                continue
            if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
                if callee in kernels:
                    failures.append(f"{module}:{call.lineno}: free Kernel inputs require explicit argument sources")
                    invalid.add(callee)
                continue
            values = ([receiver] if receiver is not None else []) + [value(arg) for arg in call.args]
            kwargs = {item.arg: value(item.value) for item in call.keywords}
            signature = inspect.signature(callee)
            try:
                bound = signature.bind(*values, **kwargs)
            except TypeError as error:
                if callee in kernels:
                    failures.append(f"{module}:{call.lineno}: free Kernel argument mismatch: {error}")
                    invalid.add(callee)
                continue
            if callee in kernels:
                origin, definition, parameter = kernels[callee]
                owner = bound.arguments.get(parameter)
                if not isinstance(owner, _Owner):
                    failures.append(
                        f"{module}:{call.lineno}: free Kernel {origin}.{definition['name']} "
                        f"receiver {parameter!r} has no definition-proven owner source"
                    )
                    invalid.add(callee)
                    continue
                domains.setdefault(callee, set()).add(owner.identity)
                covered.add(id(call))
            elif any(isinstance(item, _Owner) for item in bound.arguments.values()):
                pending.append(
                    (callee, {key: item for key, item in bound.arguments.items() if isinstance(item, _Owner)})
                )

    # An input domain must cover each ordinary direct use in the scanned source,
    # not just the first convenient model path. Unknown forwarding stays a gap.
    for function, (module, node) in functions.items():
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and id(call) not in covered:
                callee = _source_value(call.func, function.__globals__)
                if inspect.isfunction(callee) and callee in domains:
                    failures.append(f"{module}:{call.lineno}: free Kernel call has no proven receiver context")
                    invalid.add(callee)

    result = {}
    for kernel, identities in domains.items():
        if kernel in invalid:
            continue
        module, definition, receiver = kernels[kernel]
        definition["free_template"] = receiver
        for identity in sorted(identities):
            result.setdefault(identity, []).append(
                dict(
                    kernel=kernel,
                    definition=module + "." + definition["name"],
                    receiver=receiver,
                    node=definition["node"],
                )
            )
    return result, list(dict.fromkeys(failures))
