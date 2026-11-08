"""Prove free-kernel receiver domains from ordinary source calls.

Only owner type/path facts flow through Host functions. This does not execute
constructors, Host branches or numerical definitions to discover input types.
"""

import ast
import builtins
from collections import deque
from dataclasses import dataclass
import inspect

from taichi_forge_runtime._static_values import static_key
from ._package_factories import _source_value
from ._package_input_families import source_class


@dataclass(frozen=True)
class _Owner:
    identity: str


@dataclass(frozen=True)
class _Graph:
    pass


@dataclass(frozen=True)
class _Kernel:
    function: object
    receiver: object = None


@dataclass(frozen=True)
class _Mapping:
    items: tuple


@dataclass(frozen=True, eq=False)
class _Literal:
    value: object

    def __hash__(self):
        return hash(static_key(self.value))

    def __eq__(self, other):
        return isinstance(other, _Literal) and static_key(self.value) == static_key(other.value)


def _literal_key(value):
    if isinstance(value, _Literal):
        return static_key(value.value)
    if type(value) is tuple:
        return ("tuple", tuple(_literal_key(item) for item in value))
    raise ValueError("Template value has no immutable definition source")


def discover(sources, imported):
    from taichi_forge.types.annotations import Template
    from taichi_forge.graph import GraphBuilder

    owners, classes, functions, kernels, kernel_definitions = {}, {}, {}, {}, {}
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
            if definition["kind"] != "kernel":
                continue
            if definition["owner"] is None:
                entry = namespace.get(definition["node"].name)
            else:
                cls = namespace.get(source_class(tree, definition["owner"]))
                entry = inspect.getattr_static(cls, definition["node"].name, None) if inspect.isclass(cls) else None
            kernel = getattr(entry, "_primal", None)
            if kernel is None:
                continue
            templates = [a.name for a in kernel.arguments if isinstance(a.annotation, Template)]
            if len(templates) > (1 if definition["owner"] is not None else 0):
                kernels.setdefault(entry, (module, definition, tuple(templates)))
                kernel_definitions.setdefault(entry, []).append(definition)
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
                        pending.append((function, {"self": _Owner(identity)}, (function,)))

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
        function, arguments, chain = pending.popleft()
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

        def alias(target, expression):
            if isinstance(target, ast.Name):
                aliases[target.id] = expression
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(expression, (ast.Tuple, ast.List)):
                if len(target.elts) == len(expression.elts):
                    for child, item in zip(target.elts, expression.elts):
                        alias(child, item)

        for part in nodes:
            if isinstance(part, ast.Name) and not isinstance(part.ctx, ast.Load):
                stores[part.id] = stores.get(part.id, 0) + 1
            if isinstance(part, ast.Assign) and len(part.targets) == 1:
                alias(part.targets[0], part.value)
        local_names = set(stores) | set(inspect.signature(function).parameters)

        def graph_source(expression, visiting=()):
            if isinstance(expression, ast.Name):
                if expression.id in arguments and expression.id not in stores:
                    return isinstance(arguments[expression.id], _Graph)
                if expression.id in aliases and stores.get(expression.id) == 1 and expression.id not in visiting:
                    return graph_source(aliases[expression.id], (*visiting, expression.id))
            if isinstance(expression, ast.IfExp):
                return graph_source(expression.body, visiting) and graph_source(expression.orelse, visiting)
            if isinstance(expression, ast.Call):
                if isinstance(expression.func, ast.Attribute) and expression.func.attr == "create_sequential":
                    return graph_source(expression.func.value, visiting)
                root = expression.func
                while isinstance(root, ast.Attribute):
                    root = root.value
                return (
                    isinstance(root, ast.Name)
                    and root.id not in local_names
                    and _source_value(expression.func, namespace) is GraphBuilder
                    and not expression.args
                    and not expression.keywords
                )
            return False

        # A mapping passed through ordinary forwarding must remain unchanged.
        # Unknown calls and item writes invalidate its source, including aliases.
        tainted, dependencies = set(), {}
        for name, expression in aliases.items():
            if not isinstance(expression, (ast.Dict, ast.DictComp, ast.Tuple, ast.List)):
                dependencies[name] = {part.id for part in ast.walk(expression) if isinstance(part, ast.Name)}
        reads = {}
        for part in nodes:
            if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load):
                reads[part.id] = reads.get(part.id, 0) + 1

        def retained(name, visited=()):
            if name in visited:
                return False
            return reads.get(name, 0) > 1 or any(
                retained(item, (*visited, name)) for item in dependencies.get(name, ())
            )

        for part in nodes:
            if isinstance(part, ast.Subscript) and not isinstance(part.ctx, ast.Load):
                tainted.update(n.id for n in ast.walk(part.value) if isinstance(n, ast.Name))
            if isinstance(part, ast.Call):
                resolved = _source_value(part.func, namespace)
                known = inspect.isfunction(resolved) and (resolved in functions or resolved in kernels)
                # Known local forwarding parameters are resolved below. Their
                # mapping effects are checked in their own source context.
                if isinstance(part.func, ast.Name) and isinstance(arguments.get(part.func.id), _Kernel):
                    known = True
                pure_mapping_use = isinstance(part.func, ast.Attribute) and (
                    (part.func.attr == "dispatch" and graph_source(part.func.value))
                    or (part.func.attr == "items" and not part.args and not part.keywords)
                )
                if not pure_mapping_use:
                    exposed = [*part.args, *(k.value for k in part.keywords)]
                    if isinstance(part.func, ast.Attribute):
                        exposed.append(part.func.value)
                    tainted.update(
                        n.id
                        for item in exposed
                        for n in ast.walk(item)
                        if isinstance(n, ast.Name) and (not known or retained(n.id))
                    )
        pending_taints = list(tainted)
        while pending_taints:
            for dependency in dependencies.get(pending_taints.pop(), ()):
                if dependency not in tainted:
                    tainted.add(dependency)
                    pending_taints.append(dependency)

        def value(expression, visiting=(), bindings=None):
            bindings = bindings or {}

            def child(item):
                return value(item, visiting, bindings)

            def source(item):
                root = item
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and (root.id in local_names or root.id in bindings):
                    return None
                return _source_value(item, namespace)

            if isinstance(expression, ast.Name):
                name = expression.id
                if name in bindings:
                    return bindings[name]
                if name in arguments and name not in stores:
                    item = arguments[name]
                    return None if isinstance(item, _Mapping) and name in tainted else item
                if name in aliases and stores.get(name) == 1 and name not in visiting:
                    item = value(aliases[name], (*visiting, name), bindings)
                    return None if isinstance(item, _Mapping) and name in tainted else item
                if name not in local_names:
                    resolved = _source_value(expression, namespace)
                    if inspect.isfunction(resolved) and resolved in kernels:
                        return _Kernel(resolved)
                return None
            if isinstance(expression, ast.Attribute):
                receiver = child(expression.value)
                item = member(receiver, expression.attr)
                if item is not None:
                    return item
                if isinstance(receiver, _Owner):
                    method = inspect.getattr_static(classes[receiver.identity], expression.attr, None)
                    if inspect.isfunction(method) and method in kernels:
                        return _Kernel(method, receiver)
                resolved = source(expression)
                if inspect.isfunction(resolved) and resolved in kernels:
                    return _Kernel(resolved)
            if isinstance(expression, (ast.Tuple, ast.List)):
                return tuple(child(item) for item in expression.elts)
            if isinstance(expression, ast.Constant) and (
                expression.value is None or type(expression.value) in (bool, int, float, str)
            ):
                return _Literal(expression.value)
            if isinstance(expression, ast.Dict):
                if all(isinstance(key, ast.Constant) and isinstance(key.value, str) for key in expression.keys):
                    names = [key.value for key in expression.keys]
                    if len(names) == len(set(names)):
                        return _Mapping(
                            tuple((key.value, child(item)) for key, item in zip(expression.keys, expression.values))
                        )
            if isinstance(expression, ast.DictComp) and len(expression.generators) == 1:
                generator = expression.generators[0]
                sequence = child(generator.iter)
                target = generator.target
                if (
                    not generator.is_async
                    and isinstance(sequence, tuple)
                    and isinstance(target, (ast.Tuple, ast.List))
                    and all(isinstance(n, ast.Name) for n in target.elts)
                ):
                    items = []
                    for pair in sequence:
                        if not isinstance(pair, tuple) or len(pair) != len(target.elts):
                            return None
                        context = dict(bindings, **{name.id: item for name, item in zip(target.elts, pair)})
                        conditions = [value(test, visiting, context) for test in generator.ifs]
                        if any(not isinstance(test, _Literal) or type(test.value) is not bool for test in conditions):
                            return None
                        if not all(test.value for test in conditions):
                            continue
                        key = value(expression.key, visiting, context)
                        if not isinstance(key, _Literal) or not isinstance(key.value, str):
                            return None
                        items.append((key.value, value(expression.value, visiting, context)))
                    if len(set(key for key, _ in items)) == len(items):
                        return _Mapping(tuple(items))
            if isinstance(expression, ast.IfExp):
                condition = child(expression.test)
                if isinstance(condition, _Literal) and type(condition.value) is bool:
                    return child(expression.body if condition.value else expression.orelse)
                left, right = child(expression.body), child(expression.orelse)
                return left if left == right else None
            if isinstance(expression, ast.Call):
                if (
                    isinstance(expression.func, ast.Name)
                    and expression.func.id == "isinstance"
                    and "isinstance" not in local_names
                    and namespace.get("isinstance", builtins.isinstance) is builtins.isinstance
                    and len(expression.args) == 2
                    and not expression.keywords
                ):
                    item = child(expression.args[0])
                    cls = source(expression.args[1])
                    if isinstance(item, _Owner) and inspect.isclass(cls) and type(cls) is type:
                        return _Literal(issubclass(classes[item.identity], cls))
                if (
                    isinstance(expression.func, ast.Attribute)
                    and expression.func.attr == "items"
                    and not expression.args
                    and not expression.keywords
                ):
                    mapping = child(expression.func.value)
                    if isinstance(mapping, _Mapping):
                        return tuple((_Literal(key), item) for key, item in mapping.items)
                if source(expression.func) is GraphBuilder:
                    root = expression.func
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if (
                        isinstance(root, ast.Name)
                        and root.id not in local_names
                        and not expression.args
                        and not expression.keywords
                    ):
                        return _Graph()
                if (
                    isinstance(expression.func, ast.Attribute)
                    and expression.func.attr == "create_sequential"
                    and isinstance(child(expression.func.value), _Graph)
                ):
                    return _Graph()
            return None

        def target(expression):
            root = expression
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in local_names:
                if isinstance(expression, ast.Name):
                    item = value(expression)
                    if isinstance(item, _Kernel):
                        return item.function, item.receiver
                if isinstance(expression, ast.Attribute):
                    receiver = value(expression.value)
                    if isinstance(receiver, _Owner):
                        cls = classes[receiver.identity]
                        method = inspect.getattr_static(cls, expression.attr, None)
                        if inspect.isfunction(method) and (method in functions or method in kernels):
                            return method, receiver
                return None, None
            result = _source_value(expression, namespace)
            return (result, None) if inspect.isfunction(result) else (None, None)

        def record(callee, inputs, call):
            origin, definition, parameters = kernels[callee]
            receivers = [key for key in parameters if isinstance(inputs.get(key), _Owner)]
            if len(receivers) != 1:
                failures.append(
                    f"{module}:{call.lineno}: free Kernel {origin}.{definition['name']} "
                    "requires one definition-proven owner receiver"
                )
                invalid.add(callee)
                return
            receiver = receivers[0]
            try:
                values = tuple((key, _literal_key(inputs.get(key))) for key in parameters if key != receiver)
            except ValueError:
                failures.append(
                    f"{module}:{call.lineno}: free Kernel {origin}.{definition['name']} has an unproven template input"
                )
                invalid.add(callee)
                return
            domains.setdefault(callee, set()).add((inputs[receiver].identity, receiver, values))
            covered.add(id(call))

        for call in (part for part in nodes if isinstance(part, ast.Call)):
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "dispatch"
                and isinstance(value(call.func.value), _Graph)
                and call.args
            ):
                kernel = value(call.args[0])
                if isinstance(kernel, _Kernel):
                    inputs = next((value(k.value) for k in call.keywords if k.arg == "template_args"), None)
                    parameters = kernels[kernel.function][2]
                    mapping = dict(inputs.items) if isinstance(inputs, _Mapping) else {}
                    if kernel.receiver is not None:
                        receiver = parameters[0]
                        if receiver in mapping and mapping[receiver] != kernel.receiver:
                            mapping = {}
                        else:
                            mapping[receiver] = kernel.receiver
                    record(kernel.function, mapping if set(mapping) == set(parameters) else {}, call)
                continue
            callee, receiver = target(call.func)
            if callee not in kernels and callee not in functions:
                continue
            values = [receiver] if receiver is not None else []
            unknown_expansion = any(k.arg is None for k in call.keywords)
            for argument in call.args:
                if isinstance(argument, ast.Starred):
                    items = value(argument.value)
                    if isinstance(items, tuple):
                        values.extend(items)
                    else:
                        unknown_expansion = True
                else:
                    values.append(value(argument))
            if unknown_expansion:
                if callee in kernels:
                    failures.append(f"{module}:{call.lineno}: free Kernel inputs require explicit argument sources")
                    invalid.add(callee)
                continue
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
                record(callee, bound.arguments, call)
            elif any(item is not None for item in bound.arguments.values()):
                inputs = {
                    key: _Mapping(tuple(item.items())) if isinstance(item, dict) else item
                    for key, item in bound.arguments.items()
                    if item is not None
                }
                if callee in chain:
                    if (callee, tuple(sorted(inputs.items()))) not in visited:
                        failures.append(
                            f"{module}:{call.lineno}: recursive Kernel input forwarding changes its source context"
                        )
                else:
                    pending.append((callee, inputs, (*chain, callee)))

    # An input domain must cover each ordinary direct use in the scanned source,
    # not just the first convenient model path. Unknown forwarding stays a gap.
    for function, (module, node) in functions.items():
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and id(call) not in covered:
                callee = _source_value(call.func, function.__globals__)
                if isinstance(call.func, ast.Attribute) and call.func.attr == "dispatch" and call.args:
                    callee = _source_value(call.args[0], function.__globals__)
                if inspect.isfunction(callee) and callee in domains:
                    failures.append(f"{module}:{call.lineno}: free Kernel call has no proven receiver context")
                    invalid.add(callee)

    result = {}
    for kernel, identities in domains.items():
        if kernel in invalid:
            continue
        module, definition, parameters = kernels[kernel]
        receivers = {receiver for _, receiver, _ in identities}
        if len(receivers) != 1:
            failures.append(f"{module}.{definition['name']}: template receiver position changes between definitions")
            continue
        receiver = next(iter(receivers))
        public_identity = kernel.__module__ + "." + kernel.__qualname__
        for item in kernel_definitions[kernel]:
            item["free_template"] = receiver
            item["function_templates"] = parameters
            item["function_identity"] = public_identity
        for identity, _, values in sorted(identities, key=repr):
            result.setdefault(identity, []).append(
                dict(
                    kernel=kernel,
                    definition=public_identity,
                    receiver=receiver,
                    templates=dict(values),
                    node=definition["node"],
                )
            )
    return result, list(dict.fromkeys(failures))
