"""Prove free-kernel receiver domains from ordinary source calls.

Only owner type/path facts flow through Host functions. This does not execute
constructors, Host branches or numerical definitions to discover input types.
"""

import ast
import builtins
import copy
import functools
from collections import deque
from dataclasses import dataclass, fields, is_dataclass, MISSING
import inspect
import importlib.util

from taichi_forge_runtime._static_values import static_key
from ._package_factories import _source_value
from ._package_input_families import source_class


@dataclass(frozen=True)
class _Owner:
    identity: str
    root: str = ""
    path: tuple = ()


@dataclass(frozen=True)
class _Resource:
    owner: _Owner
    path: tuple
    optional: bool = False


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


@dataclass(frozen=True)
class _Record:
    cls: type
    items: tuple


@dataclass(frozen=True)
class _Choice:
    source: tuple
    alternatives: tuple


def _choice_contexts(arguments):
    """Keep repeated reads of one declared owner choice correlated."""
    def expand(value, selected):
        if isinstance(value, _Choice):
            indices = (selected[value.source],) if value.source in selected else range(len(value.alternatives))
            for index in indices:
                yield from expand(value.alternatives[index], {**selected, value.source: index})
        elif isinstance(value, (_Mapping, _Record)):
            for items, choices in entries(value.items, selected):
                yield (_Record(value.cls, items) if isinstance(value, _Record) else _Mapping(items)), choices
        elif isinstance(value, tuple):
            for items, choices in entries(tuple(enumerate(value)), selected):
                yield tuple(item for _, item in items), choices
        else:
            yield value, selected

    def entries(items, selected):
        if not items:
            yield (), selected
            return
        (name, value), *rest = items
        for item, choices in expand(value, selected):
            for tail, choices in entries(rest, choices):
                yield ((name, item), *tail), choices

    for items, _ in entries(tuple(arguments.items()), {}):
        yield dict(items)


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


def _plain_record(cls):
    """Only generated frozen record constructors preserve their input members."""
    return (
        inspect.isclass(cls)
        and type(cls) is type
        and is_dataclass(cls)
        and cls.__dataclass_params__.frozen
        and inspect.isfunction(cls.__init__)
        and cls.__init__.__code__.co_filename == "<string>"
        and not hasattr(cls, "__post_init__")
        and cls.__new__ is object.__new__
        and cls.__getattribute__ is object.__getattribute__
        and all(not hasattr(vars(base).get(field.name), "__get__") for base in cls.__mro__ for field in fields(cls))
    )


def _returns_on_all_paths(body):
    if not body:
        return False
    last = body[-1]
    return isinstance(last, ast.Return) or (
        isinstance(last, ast.If) and _returns_on_all_paths(last.body) and _returns_on_all_paths(last.orelse)
    )


def discover(sources, imported, *, preparation_receivers=None, resource_inventory=None):
    from taichi_forge.types.annotations import Template
    from taichi_forge.graph import GraphBuilder
    from taichi_forge.lang.field import ScalarField
    from taichi_forge.lang.matrix import MatrixField
    from taichi_forge.lang.struct import StructField

    owners, classes, functions, kernels, kernel_definitions = {}, {}, {}, {}, {}
    cached_sources = set()
    member_writes = {}
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
                    if resource_inventory is not None:
                        member_writes[cls] = {
                            part.attr for part in ast.walk(node)
                            if isinstance(part, ast.Attribute) and not isinstance(part.ctx, ast.Load)
                            and isinstance(part.value, ast.Name) and part.value.id == 'self'
                        }
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef) and not method.decorator_list:
                            function = vars(cls).get(method.name)
                            if inspect.isfunction(function):
                                functions[function] = (module, method)
                        elif (resource_inventory is not None and isinstance(method, ast.FunctionDef)
                              and len(method.decorator_list) == 1
                              and _source_value(method.decorator_list[0], namespace) is functools.cached_property):
                            descriptor = vars(cls).get(method.name)
                            if type(descriptor) is functools.cached_property and inspect.isfunction(descriptor.func):
                                functions[descriptor.func] = (module, method)
                                cached_sources.add(descriptor.func)
        for definition in definitions:
            if definition["kind"] != "kernel" and resource_inventory is None:
                continue
            if definition["owner"] is None:
                entry = namespace.get(definition["node"].name)
            else:
                cls = namespace.get(source_class(tree, definition["owner"]))
                entry = inspect.getattr_static(cls, definition["node"].name, None) if inspect.isclass(cls) else None
            if resource_inventory is not None and inspect.isfunction(entry):
                functions[entry] = (module, definition["node"])
            kernel = getattr(entry, "_primal", None)
            if kernel is None:
                continue
            templates = [a.name for a in kernel.arguments if isinstance(a.annotation, Template)]
            if resource_inventory is not None or len(templates) > (1 if definition["owner"] is not None else 0):
                kernels.setdefault(entry, (module, definition, tuple(templates)))
                kernel_definitions.setdefault(entry, []).append(definition)
    requested = {}
    for identity, schema in owners.items():
        if resource_inventory is not None:
            break
        for name, item in schema.items():
            if "input_expression" not in item:
                continue
            method = inspect.getattr_static(classes[identity], item["input_method"], None)
            if method not in functions:
                raise ValueError("Prepared input method is outside its ordinary source definition: " + identity)
            requested.setdefault(method, []).append((identity, name, item))
    if not kernels and not requested and preparation_receivers is None:
        return {}, []

    pending = deque()
    for identity, cls in classes.items():
        if resource_inventory is not None:
            break
        seen = set()
        for base in cls.__mro__:
            for name, function in vars(base).items():
                if name in seen:
                    continue
                seen.add(name)
                if inspect.isfunction(function) and function in functions:
                    parameters = list(inspect.signature(function).parameters)
                    if parameters and parameters[0] == "self":
                        pending.append((function, {"self": _Owner(identity)}, (function,), None))

    inventory = set()
    if resource_inventory is not None:
        identity, method = resource_inventory
        function = inspect.getattr_static(classes[identity], method, None)
        if function not in functions or function in kernels:
            raise ValueError("Write inventory requires an ordinary recording definition: " + identity + "." + method)
        try:
            inputs = inspect.signature(function).bind(_Owner(identity), _Graph()).arguments
        except TypeError as error:
            raise ValueError("Write inventory recording requires its owner and a Graph builder") from error
        pending.append((function, dict(inputs), (function,), None))

    visited, domains, covered, failures = set(), {}, set(), []
    invalid = set()
    return_cache = {}
    binding_domains, binding_calls, binding_covered = {}, set(), set()

    def source_callable(expression, namespace):
        result = _source_value(expression, namespace)
        if result is None and isinstance(expression, ast.Attribute):
            cls = _source_value(expression.value, namespace)
            if inspect.isclass(cls):
                result = inspect.getattr_static(cls, expression.attr, None)
        return result

    namespaces = {}
    sequence_projector = None
    iteration_failures = {}

    def function_namespace(function):
        if function in namespaces:
            return namespaces[function]
        _, node = functions[function]
        bindings, shadowed = {}, set(inspect.signature(function).parameters)

        def walk(part):
            if isinstance(part, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                shadowed.add(part.name)
                return
            if isinstance(part, ast.Lambda):
                return
            if isinstance(part, ast.Name) and not isinstance(part.ctx, ast.Load):
                shadowed.add(part.id)
            if isinstance(part, (ast.Import, ast.ImportFrom)):
                origin = None
                if isinstance(part, ast.ImportFrom):
                    name = "." * part.level + (part.module or "")
                    if part.level:
                        name = importlib.util.resolve_name(name, function.__globals__["__package__"])
                    origin = imported.get(name)
                for alias in part.names:
                    if origin is not None:
                        value = vars(origin).get(alias.name, imported.get(name + "." + alias.name))
                    elif isinstance(part, ast.Import):
                        value = imported.get(alias.name if alias.asname else alias.name.split(".")[0])
                    else:
                        value = None
                    key = alias.asname or (alias.name if isinstance(part, ast.ImportFrom) else alias.name.split(".")[0])
                    if key in bindings and bindings[key] is not value:
                        shadowed.add(key)
                    bindings[key] = value
            for child in ast.iter_child_nodes(part):
                walk(child)

        for statement in node.body:
            walk(statement)
        namespace = dict((inspect.unwrap(function) if resource_inventory is not None else function).__globals__)
        for name, value in bindings.items():
            # Read only modules already admitted to the source closure. Imports
            # remain ordinary Host statements; discovery does not execute them.
            namespace.pop(name, None)
            if name not in shadowed and value is not None:
                namespace[name] = value
        namespaces[function] = namespace
        return namespace

    def member(owner, name):
        if isinstance(owner, _Record):
            return dict(owner.items).get(name)
        if not isinstance(owner, _Owner):
            return None
        item = owners[owner.identity].get(name, {})
        alternatives = tuple(dict.fromkeys(choice.get('owner_type') for choice in item.get('owner_alternatives', ())))
        if resource_inventory is not None and len(alternatives) > 1 and all(kind in owners for kind in alternatives):
            root, path = owner.root or owner.identity, (*owner.path, name)
            return _Choice((root, path), tuple(_Owner(kind, root, path) for kind in alternatives))
        if resource_inventory is not None and isinstance(inspect.getattr_static(classes[owner.identity], name, None), property):
            inventory.add((owner.identity, ()))
        if resource_inventory is not None and (
            item.get("kind") == "host_owner_sequence"
            or item.get("kind") in ("owner", "owner_input") and item.get("owner_type") not in owners
            or len({choice["owner_type"] for choice in item.get("owner_alternatives", ())}) > 1
        ):
            inventory.add((owner.identity, ()))
        if item.get("kind") in ("owner", "owner_input") and item.get("owner_type") in owners:
            if len({choice["owner_type"] for choice in item.get("owner_alternatives", ())}) > 1:
                return None  # A choice is not evidence for its first receiver type.
            return _Owner(item["owner_type"], owner.root or owner.identity, (*owner.path, name))
        if item.get("kind") in ("dense_field", "model_tensor_field", "borrowed_field", "fixed_snode_field") or (
            resource_inventory is not None and item.get("kind") == "field_like"
        ):
            return _Resource(owner, tuple(item.get("path", (name,))))
        return None

    def attribute(receiver, name):
        if isinstance(receiver, _Choice):
            return _Choice(receiver.source, tuple(attribute(item, name) for item in receiver.alternatives))
        item = member(receiver, name)
        if item is not None:
            return item
        if isinstance(receiver, _Owner):
            method = inspect.getattr_static(classes[receiver.identity], name, None)
            if inspect.isfunction(method) and method in kernels:
                return _Kernel(method, receiver)
        return None

    def build_context(function, arguments, return_chain):
        nonlocal sequence_projector
        module, node = functions[function]
        namespace = function_namespace(function)
        # A rebinding invalidates an earlier owner fact. Nested definitions have
        # their own lexical scope and cannot lend assignments to this call.
        nodes = []
        condition_value = None

        def walk(statement):
            nonlocal sequence_projector
            if isinstance(statement, ast.Lambda):
                parameters = statement.args
                if not (
                    parameters.posonlyargs
                    or parameters.args
                    or parameters.kwonlyargs
                    or parameters.vararg
                    or parameters.kwarg
                    or any(
                        isinstance(part, (ast.NamedExpr, ast.Yield, ast.YieldFrom)) for part in ast.walk(statement.body)
                    )
                ):
                    # A zero-argument Host callback retains its enclosing
                    # stable input facts. Discover its body without invoking
                    # the callback or its scheduler/profiler.
                    walk(statement.body)
                return
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return
            if (resource_inventory is not None and condition_value is not None
                    and isinstance(statement, ast.For)):
                from ._package_source_sequences import SourceSequences, UnprovenSourceSequence

                if sequence_projector is None:
                    sequence_projector = SourceSequences(sources, imported, inventory=True)
                    for callee, (_, definition) in functions.items():
                        if not definition.decorator_list or callee in cached_sources:
                            sequence_projector.functions[callee] = (definition, callee.__globals__)
                try:
                    environment = {name: ast.Name(name, ast.Load()) for name in arguments}
                    environment.update({name: expression for name, expression in aliases.items()
                                        if stores.get(name) == 1})
                    # Keep the proven receiver resolver local to this lexical
                    # context; nested discovery must not replace another call's
                    # bindings. Choice IDs remain shared by the source analysis.
                    projection = copy.copy(sequence_projector)
                    projection.resolve_method = sequence_method
                    projection.resolve_cached = sequence_cached
                    projection.resolve_namespace = function_namespace
                    sequence = projection.sequence(projection.value(statement.iter, environment, namespace))
                    if not sequence_projector.immutable(sequence):
                        raise UnprovenSourceSequence('Iteration requires an immutable source sequence')
                    names = {part.id for part in ast.walk(statement.target) if isinstance(part, ast.Name)}
                    if any(isinstance(part, ast.Name) and part.id in names and not isinstance(part.ctx, ast.Load)
                           for body in statement.body for part in ast.walk(body)):
                        raise UnprovenSourceSequence('Iteration source is rebound in the loop body')
                except (UnprovenSourceSequence, TypeError) as error:
                    if (isinstance(statement.iter, ast.Call)
                            and source_callable(statement.iter.func, namespace) in sequence_projector.functions
                            or isinstance(statement.iter, ast.Attribute) and sequence_cached(statement.iter) is not None):
                        iteration_failures.setdefault(function, set()).add(str(error))
                else:
                    for item in sequence.values:
                        bindings = {}
                        sequence_projector.bind(statement.target, item, bindings)

                        class BindIteration(ast.NodeTransformer):
                            def visit_Name(self, part):
                                if isinstance(part.ctx, ast.Load) and part.id in bindings:
                                    return ast.copy_location(sequence_projector.syntax(bindings[part.id]), part)
                                return part

                            def visit_FunctionDef(self, part):
                                return part

                            visit_AsyncFunctionDef = visit_ClassDef = visit_FunctionDef

                        for body in statement.body:
                            walk(BindIteration().visit(copy.deepcopy(body)))
                    for body in statement.orelse:
                        walk(body)
                    return
            nodes.append(statement)
            if condition_value is not None and isinstance(statement, (ast.If, ast.IfExp)):
                choice = condition_value(statement.test)
                if choice is not None:
                    walk(statement.test)
                    selected = statement.body if choice else statement.orelse
                    for child in selected if isinstance(selected, list) else [selected]:
                        walk(child)
                    return
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
            if isinstance(part, (ast.Import, ast.ImportFrom)):
                for item in part.names:
                    name = item.asname or (item.name if isinstance(part, ast.ImportFrom) else item.name.split(".")[0])
                    if name in arguments:
                        stores[name] = stores.get(name, 0) + 1
            if isinstance(part, ast.Assign) and len(part.targets) == 1:
                alias(part.targets[0], part.value)
        local_names = set(stores) | set(inspect.signature(function).parameters)

        def source_value(expression):
            if (
                isinstance(expression, ast.Name)
                and expression.id in ("getattr", "dict", "isinstance")
                and expression.id not in local_names
            ):
                builtin = getattr(builtins, expression.id)
                supplied = namespace.get('__builtins__', vars(builtins))
                supplied = supplied if isinstance(supplied, dict) else vars(supplied)
                if namespace.get(expression.id, supplied.get(expression.id)) is builtin:
                    return builtin
            return source_callable(expression, namespace)

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
        tainted, record_tainted, dependencies = set(), set(), {}
        for name, expression in aliases.items():
            if not isinstance(expression, (ast.Dict, ast.DictComp, ast.Tuple, ast.List)):
                dependencies[name] = {part.id for part in ast.walk(expression) if isinstance(part, ast.Name)}
        record_dependencies = {
            name: dependencies[name]
            for name, expression in aliases.items()
            if isinstance(expression, (ast.Name, ast.Attribute))
        }
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

        def record_retained(name, visited=()):
            if name in visited:
                return False
            return reads.get(name, 0) > 1 or any(
                record_retained(item, (*visited, name)) for item in record_dependencies.get(name, ())
            )

        def declared_receiver(expression, visiting=()):
            if isinstance(expression, ast.Name):
                name = expression.id
                if name in arguments and name not in stores:
                    return arguments[name]
                if name in aliases and stores.get(name) == 1 and name not in visiting:
                    return declared_receiver(aliases[name], (*visiting, name))
            if isinstance(expression, ast.Attribute):
                return member(declared_receiver(expression.value, visiting), expression.attr)
            return None

        for part in nodes:
            if isinstance(part, (ast.Subscript, ast.Attribute)) and not isinstance(part.ctx, ast.Load):
                tainted.update(n.id for n in ast.walk(part.value) if isinstance(n, ast.Name))
                record_tainted.update(n.id for n in ast.walk(part.value) if isinstance(n, ast.Name))
            if isinstance(part, ast.Call):
                resolved = source_value(part.func)
                if isinstance(part.func, ast.Attribute):
                    receiver = declared_receiver(part.func.value)
                    if isinstance(receiver, _Owner):
                        resolved = inspect.getattr_static(classes[receiver.identity], part.func.attr, None)
                known = (
                    (inspect.isfunction(resolved) and (resolved in functions or resolved in kernels))
                    or _plain_record(resolved)
                    or resolved is builtins.getattr
                )
                # Known local forwarding parameters are resolved below. Their
                # mapping effects are checked in their own source context.
                if isinstance(part.func, ast.Name) and isinstance(arguments.get(part.func.id), _Kernel):
                    known = True
                pure_mapping_use = isinstance(part.func, ast.Attribute) and (
                    (part.func.attr == "dispatch" and graph_source(part.func.value))
                    or (part.func.attr == "items" and not part.args and not part.keywords)
                )
                if not pure_mapping_use and not getattr(part, "_package_resource_metadata", False):
                    exposed = [*part.args, *(k.value for k in part.keywords)]
                    if isinstance(part.func, ast.Attribute):
                        exposed.append(part.func.value)
                    tainted.update(
                        n.id
                        for item in exposed
                        for n in ast.walk(item)
                        if isinstance(n, ast.Name) and (not known or retained(n.id))
                    )
                    # Reading several members of a frozen record to supply a
                    # Kernel does not expose the record itself for mutation.
                    # Unknown Host calls and repeated whole-record forwarding
                    # retain the conservative escape check.
                    record_tainted.update(
                        n.id
                        for item in exposed
                        for n in ast.walk(item)
                        if isinstance(n, ast.Name)
                        and (
                            not known
                            or resolved not in kernels
                            and resolved is not builtins.getattr
                            and isinstance(item, ast.Name)
                            and record_retained(n.id)
                        )
                    )
        for taint_set, edges in ((tainted, dependencies), (record_tainted, record_dependencies)):
            pending_taints = list(taint_set)
            while pending_taints:
                for dependency in edges.get(pending_taints.pop(), ()):
                    if dependency not in taint_set:
                        taint_set.add(dependency)
                        pending_taints.append(dependency)

        def value(expression, visiting=(), bindings=None):
            if getattr(expression, '_package_unknown_iteration', False):
                return None
            bindings = bindings or {}

            def child(item):
                return value(item, visiting, bindings)

            def source(item):
                root = item
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and (root.id in local_names or root.id in bindings):
                    return None
                return source_value(item)

            if isinstance(expression, ast.Name):
                name = expression.id
                if name in bindings:
                    return bindings[name]
                if name in arguments and name not in stores:
                    item = arguments[name]
                    return (
                        None
                        if (
                            isinstance(item, _Mapping)
                            and name in tainted
                            or isinstance(item, _Record)
                            and name in record_tainted
                        )
                        else item
                    )
                if name in aliases and stores.get(name) == 1 and name not in visiting:
                    item = value(aliases[name], (*visiting, name), bindings)
                    return (
                        None
                        if (
                            isinstance(item, _Mapping)
                            and name in tainted
                            or isinstance(item, _Record)
                            and name in record_tainted
                        )
                        else item
                    )
                if name not in local_names:
                    resolved = _source_value(expression, namespace)
                    if inspect.isfunction(resolved) and resolved in kernels:
                        return _Kernel(resolved)
                return None
            if isinstance(expression, ast.Attribute):
                receiver = child(expression.value)
                item = attribute(receiver, expression.attr)
                if item is not None:
                    return item
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
                if resource_inventory is not None and hasattr(expression, '_package_source_choice'):
                    return _Choice(('source_sequence', expression._package_source_choice),
                                   (child(expression.body), child(expression.orelse)))
                condition = child(expression.test)
                if isinstance(condition, _Literal) and type(condition.value) is bool:
                    return child(expression.body if condition.value else expression.orelse)
                left, right = child(expression.body), child(expression.orelse)
                for resource, absent in ((left, right), (right, left)):
                    if isinstance(resource, _Resource) and isinstance(absent, _Literal) and absent.value is None:
                        return _Resource(resource.owner, resource.path, optional=True)
                return left if left == right else None
            if isinstance(expression, ast.Call):
                cls = source(expression.func)
                if cls is builtins.dict and not expression.args and all(k.arg is not None for k in expression.keywords):
                    return _Mapping(tuple((k.arg, child(k.value)) for k in expression.keywords))
                if cls is builtins.getattr and len(expression.args) in (2, 3) and not expression.keywords:
                    name = child(expression.args[1])
                    if isinstance(name, _Literal) and type(name.value) is str:
                        return child(ast.Attribute(expression.args[0], name.value, ast.Load()))
                if (
                    _plain_record(cls)
                    and not any(isinstance(arg, ast.Starred) for arg in expression.args)
                    and all(item.arg is not None for item in expression.keywords)
                ):
                    try:
                        bound = inspect.signature(cls).bind(
                            *(child(arg) for arg in expression.args),
                            **{k.arg: child(k.value) for k in expression.keywords},
                        )
                    except TypeError:
                        return None
                    members = dict(bound.arguments)
                    for field in fields(cls):
                        if field.name not in members:
                            try:
                                static_key(field.default)
                                members[field.name] = _Literal(field.default) if field.default is not MISSING else None
                            except ValueError:
                                members[field.name] = None
                    return _Record(cls, tuple(members.items()))
                if (
                    isinstance(expression.func, ast.Name)
                    and expression.func.id == "isinstance"
                    and "isinstance" not in local_names
                    and source(expression.func) is builtins.isinstance
                    and len(expression.args) == 2
                    and not expression.keywords
                ):
                    item = child(expression.args[0])
                    cls = source(expression.args[1])
                    if inspect.isclass(cls) and type(cls) is type:
                        if isinstance(item, _Literal) and type(item.value) in (bool, int, float, str, type(None)):
                            return _Literal(isinstance(item.value, cls))
                        if isinstance(item, _Owner):
                            return _Literal(issubclass(classes[item.identity], cls))
                        if isinstance(item, _Resource):
                            # Allocation-proven Fields cannot be arbitrary Host
                            # adapters. Keep the family broad until its element
                            # type is resolved; only unanimous answers are safe.
                            candidates = (ScalarField, MatrixField, StructField)
                            if item.optional:
                                candidates += (type(None),)
                            answers = {issubclass(candidate, cls) for candidate in candidates}
                            if len(answers) == 1:
                                return _Literal(answers.pop())
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
                callee, receiver = call_target(expression.func, visiting, bindings)
                if (
                    callee in functions
                    and callee not in return_chain
                    and _returns_on_all_paths(functions[callee][1].body)
                    and not any(isinstance(arg, ast.Starred) for arg in expression.args)
                    and all(k.arg is not None for k in expression.keywords)
                ):
                    args = ([receiver] if receiver is not None else []) + [child(arg) for arg in expression.args]
                    try:
                        bound = inspect.signature(callee).bind(
                            *args, **{k.arg: child(k.value) for k in expression.keywords}
                        )
                    except TypeError:
                        return None
                    for name, parameter in inspect.signature(callee).parameters.items():
                        if name not in bound.arguments and parameter.default is not inspect.Parameter.empty:
                            try:
                                static_key(parameter.default)
                                bound.arguments[name] = _Literal(parameter.default)
                            except ValueError:
                                pass
                    return_inputs = {
                        key: _Mapping(tuple(item.items())) if isinstance(item, dict) else item
                        for key, item in bound.arguments.items()
                    }
                    return_key = (callee, tuple(sorted(return_inputs.items())))
                    if return_key not in return_cache:
                        evaluate, _, returned_nodes, _ = build_context(callee, return_inputs, (*return_chain, callee))
                        results = [evaluate(part.value) for part in returned_nodes if isinstance(part, ast.Return)]
                        return_cache[return_key] = (
                            results[0] if results and all(item == results[0] for item in results) else None
                        )
                    return return_cache[return_key]
            return None

        def call_target(expression, visiting=(), bindings=None):
            root = expression
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in local_names:
                if isinstance(expression, ast.Name):
                    item = value(expression, visiting, bindings)
                    if isinstance(item, _Kernel):
                        return item.function, item.receiver
                if isinstance(expression, ast.Attribute):
                    receiver = value(expression.value, visiting, bindings)
                    if isinstance(receiver, _Owner):
                        cls = classes[receiver.identity]
                        method = inspect.getattr_static(cls, expression.attr, None)
                        if inspect.isfunction(method) and (method in functions or method in kernels):
                            return method, receiver
                return None, None
            result = source_callable(expression, namespace)
            return (result, None) if inspect.isfunction(result) else (None, None)

        def proven_condition(expression):
            fact = value(expression)
            if isinstance(fact, _Literal) and type(fact.value) in (bool, int, float, str, type(None)):
                return bool(fact.value)
            if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
                answer = proven_condition(expression.operand)
                return None if answer is None else not answer
            if isinstance(expression, ast.BoolOp):
                answers = [proven_condition(item) for item in expression.values]
                if isinstance(expression.op, ast.And):
                    return False if False in answers else True if all(answer is True for answer in answers) else None
                return True if True in answers else False if all(answer is False for answer in answers) else None
            if (isinstance(expression, ast.Compare) and len(expression.ops) == 1
                    and isinstance(expression.ops[0], (ast.Is, ast.IsNot))):
                left, right = value(expression.left), value(expression.comparators[0])
                if isinstance(left, _Literal) and left.value is None:
                    left, right = right, left
                if isinstance(right, _Literal) and right.value is None:
                    if isinstance(left, _Literal):
                        absent = left.value is None
                    elif isinstance(left, _Owner) or isinstance(left, _Resource) and not left.optional:
                        absent = False
                    else:
                        return None
                    return absent if isinstance(expression.ops[0], ast.Is) else not absent
            return None

        def sequence_method(expression):
            callee, receiver = call_target(expression)
            if (isinstance(receiver, _Owner) and callee in functions
                    and not functions[callee][1].decorator_list):
                return callee
            return None

        def sequence_cached(expression):
            receiver = value(expression.value)
            if isinstance(receiver, _Owner):
                cls = classes[receiver.identity]
                if (expression.attr in owners[receiver.identity]
                        or any(expression.attr in member_writes.get(base, ()) for base in cls.__mro__)):
                    return None
                descriptor = inspect.getattr_static(cls, expression.attr, None)
                if type(descriptor) is functools.cached_property and descriptor.func in cached_sources:
                    return descriptor.func
            return None

        # Keep the original conservative rebinding/escape facts above.
        # Inventory and signature discovery must agree on reachable calls;
        # unknown or rebound optional inputs still contribute both paths.
        condition_value = proven_condition
        nodes = []
        for statement in node.body:
            walk(statement)
        return value, call_target, nodes, module

    while pending:
        function, arguments, chain, call_site = pending.popleft()
        identity = (function, tuple(sorted(arguments.items())))
        value, target, nodes, module = build_context(function, arguments, (function,))
        for owner, name, item in requested.get(function, ()):
            receiver = arguments.get("self")
            if not isinstance(receiver, _Owner) or receiver.identity != owner:
                continue
            fact = value(item["input_expression"])
            if isinstance(fact, _Literal) and fact.value is None and item.get("optional_input"):
                if call_site is not None:
                    binding_covered.add((function, call_site, owner, name))
            if isinstance(fact, _Resource) and (not fact.optional or item.get("optional_input")):
                source = owners[fact.owner.identity].get(fact.path[0], {})
                if source.get("kind") == "dense_field" and (
                    len(fact.path) == 1 or source.get("struct_field") and len(fact.path) == 2
                ):
                    binding_domains.setdefault((owner, name), set()).add((fact.owner.identity, fact.path))
                    if call_site is not None:
                        binding_covered.add((function, call_site, owner, name))
        if preparation_receivers is not None and function in preparation_receivers:
            receivers = [(key, item.identity) for key, item in arguments.items() if isinstance(item, _Owner)]
            if not receivers:
                # An ordinary Host wrapper can forward an external owner to an
                # already declared allocation entry. Check that boundary below;
                # an unknown caller never establishes a new receiver contract.
                continue
            if len(receivers) != 1:
                raise ValueError(f"{module}:{functions[function][1].lineno}: preparation requires one proven owner")
            else:
                preparation_receivers[function].add(receivers[0])
                if call_site is not None:
                    covered.add(call_site)
        if identity in visited:
            continue
        visited.add(identity)

        def inventory_input(item):
            if isinstance(item, _Resource):
                inventory.add((item.owner.identity, item.path))
            elif isinstance(item, _Owner):
                # An opaque escape needs the whole declared domain. It cannot
                # establish that a cyclic or unresolved child is unused.
                inventory.add((item.identity, ()))
            elif isinstance(item, (_Mapping, _Record)):
                for _, child in item.items:
                    inventory_input(child)
            elif isinstance(item, _Choice):
                for child in item.alternatives:
                    inventory_input(child)
            elif isinstance(item, tuple):
                for child in item:
                    inventory_input(child)
            elif isinstance(item, _Graph):
                inventory.add((resource_inventory[0], ()))

        if resource_inventory is not None:
            # This is a conservative source/type domain, not a second effects
            # analysis. The existing compiler still selects actual writes.
            for part in nodes:
                if isinstance(part, (ast.Name, ast.Attribute)):
                    fact = value(part)
                    if isinstance(fact, _Resource):
                        inventory_input(fact)

        def record(callee, inputs, call):
            if resource_inventory is not None:
                if any(inputs.get(name) is None for name in kernels[callee][2]):
                    missing = ', '.join(name for name in kernels[callee][2] if inputs.get(name) is None)
                    failures.append(f"{module}:{call.lineno}: write inventory Kernel {callee.__qualname__} "
                                    f"has unproven template inputs: {missing}")
                    return
                pending.append((callee, inputs, (*chain, callee), id(call)))
                return
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
            owner = inputs[receiver]
            try:
                values, resources = [], []
                for key in parameters:
                    if key == receiver:
                        continue
                    item = inputs.get(key)
                    if isinstance(item, _Resource):
                        if item.optional:
                            raise ValueError("Optional Field forwarding requires its declared presence binding")
                        path = (*item.owner.path, *item.path)
                        if (item.owner.root or item.owner.identity) != (owner.root or owner.identity) or path[
                            : len(owner.path)
                        ] != owner.path:
                            raise ValueError("Resource is outside its template receiver")
                        resources.append((key, path[len(owner.path) :]))
                    else:
                        values.append((key, _literal_key(item)))
            except ValueError:
                failures.append(
                    f"{module}:{call.lineno}: free Kernel {origin}.{definition['name']} has an unproven template input"
                )
                invalid.add(callee)
                return
            domains.setdefault(callee, set()).add((owner.identity, receiver, tuple(values), tuple(resources)))
            covered.add(id(call))

        for call in (part for part in nodes if isinstance(part, ast.Call)):
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "dispatch"
                and isinstance(value(call.func.value), _Graph)
                and call.args
            ):
                kernel = value(call.args[0])
                inputs = next((value(k.value) for k in call.keywords if k.arg == "template_args"), None)
                for context in _choice_contexts(dict(kernel=kernel, inputs=inputs)):
                    kernel, inputs = context['kernel'], context['inputs']
                    if not isinstance(kernel, _Kernel):
                        if resource_inventory is not None:
                            route = ' -> '.join(item.__qualname__ for item in chain)
                            reasons = sorted({reason for item in chain for reason in iteration_failures.get(item, ())})
                            detail = '; iteration source: ' + '; '.join(reasons) if reasons else ''
                            failures.append(f"{module}:{call.lineno}: write inventory dispatch has no proven Kernel via {route}{detail}")
                        continue
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
                if resource_inventory is not None:
                    fact = value(call)
                    if isinstance(fact, (_Mapping, _Record, _Kernel, _Resource, _Literal, tuple)):
                        # Proven builtin/record projections expose no owner to
                        # an arbitrary callee. A Field lookup still contributes
                        # its conservative resource dependency.
                        if isinstance(fact, _Resource):
                            inventory_input(fact)
                        continue
                    if isinstance(call.func, ast.Name) and call.func.id == 'super':
                        inventory_input(arguments.get('self'))
                    for expression in [*call.args, *(k.value for k in call.keywords)]:
                        inventory_input(value(expression))
                    if isinstance(call.func, ast.Attribute):
                        inventory_input(value(call.func.value))
                continue
            for owner, name, _ in requested.get(callee, ()):
                if isinstance(receiver, _Owner) and receiver.identity == owner:
                    binding_calls.add((callee, id(call), owner, name))
            values = [receiver] if receiver is not None else []
            unknown_expansion = any(k.arg is None for k in call.keywords)
            unknown_position = False
            for argument in call.args:
                if isinstance(argument, ast.Starred):
                    items = value(argument.value)
                    if isinstance(items, tuple) and not unknown_position:
                        values.extend(items)
                    else:
                        unknown_expansion = True
                        unknown_position = True
                elif not unknown_position:
                    values.append(value(argument))
            if unknown_expansion and callee in kernels:
                failures.append(f"{module}:{call.lineno}: free Kernel inputs require explicit argument sources")
                invalid.add(callee)
                continue
            kwargs = {item.arg: value(item.value) for item in call.keywords if item.arg is not None}
            signature = inspect.signature(callee)
            try:
                # A successful ordinary Host call preserves its explicit
                # positional prefix and named arguments even when other options
                # are runtime data. Missing arguments provide no source facts;
                # duplicate bindings still fail in the normal Python call.
                bind = signature.bind_partial if unknown_expansion else signature.bind
                bound = bind(*values, **kwargs)
            except TypeError as error:
                if callee in kernels:
                    failures.append(f"{module}:{call.lineno}: free Kernel argument mismatch: {error}")
                    invalid.add(callee)
                elif resource_inventory is not None:
                    for item in [*values, *kwargs.values()]:
                        inventory_input(item)
                continue
            for owner, name, _ in requested.get(callee, ()):
                actual = bound.arguments.get("self")
                if isinstance(actual, _Owner) and actual.identity == owner:
                    binding_calls.add((callee, id(call), owner, name))
            if callee in kernels:
                record(callee, bound.arguments, call)
            elif any(item is not None for item in bound.arguments.values()):
                for name, parameter in signature.parameters.items():
                    if name not in bound.arguments and parameter.default is not inspect.Parameter.empty:
                        try:
                            static_key(parameter.default)
                            bound.arguments[name] = _Literal(parameter.default)
                        except ValueError:
                            pass
                inputs = {
                    key: _Mapping(tuple(item.items())) if isinstance(item, dict) else item
                    for key, item in bound.arguments.items()
                    if item is not None
                }
                for context in _choice_contexts(inputs):
                    if callee in chain:
                        if (callee, tuple(sorted(context.items()))) not in visited:
                            failures.append(
                                f"{module}:{call.lineno}: recursive Kernel input forwarding changes its source context"
                            )
                    else:
                        pending.append((callee, context, (*chain, callee), id(call)))

    if resource_inventory is not None:
        return inventory, list(dict.fromkeys(failures))

    if preparation_receivers is not None:
        from ._package_preparations import _declares_group

        def guarded_forward(function, node, callee, call):
            receivers = {name for name, _ in preparation_receivers[callee]}
            if len(receivers) != 1 or not _declares_group(functions[callee][1], callee.__globals__):
                return False
            # The projected allocation entry wraps its entire body in the
            # existing preparation_groups owner/lease transaction. Only admit
            # direct, unchanged Host parameter forwarding to that checked entry.
            if any(isinstance(arg, ast.Starred) for arg in call.args) or any(k.arg is None for k in call.keywords):
                return False
            try:
                arguments = inspect.signature(callee).bind(*call.args, **{k.arg: k.value for k in call.keywords})
            except TypeError:
                return False
            actual = arguments.arguments.get(next(iter(receivers)))
            if not isinstance(actual, ast.Name) or actual.id not in inspect.signature(function).parameters:
                return False
            return not any(
                isinstance(part, ast.Name)
                and part.id == actual.id
                and not isinstance(part.ctx, ast.Load)
                or isinstance(part, (ast.Global, ast.Nonlocal))
                and actual.id in part.names
                or isinstance(part, ast.alias)
                and (part.asname or part.name.split(".")[0]) == actual.id
                or isinstance(part, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler))
                and part.name == actual.id
                for part in ast.walk(node)
            )

        # Use the same source flow as free Kernels, before new preparation
        # declarations exist. Numerical/input validation runs after this pass.
        for function, (module, node) in functions.items():
            for call in ast.walk(node):
                if not isinstance(call, ast.Call) or id(call) in covered:
                    continue
                callee = source_callable(call.func, function_namespace(function))
                if inspect.isfunction(callee) and callee in preparation_receivers:
                    if guarded_forward(function, node, callee, call):
                        continue
                    raise ValueError(f"{module}:{call.lineno}: preparation call has no proven owner receiver")
        return {}, []

    # An input domain must cover each ordinary direct use in the scanned source,
    # not just the first convenient model path. Unknown forwarding stays a gap.
    for function, (module, node) in functions.items():
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and id(call) not in covered:
                callee = source_callable(call.func, function_namespace(function))
                for owner, name, _ in requested.get(callee, ()):
                    binding_calls.add((callee, id(call), owner, name))
                if isinstance(call.func, ast.Attribute) and call.func.attr == "dispatch" and call.args:
                    callee = _source_value(call.args[0], function.__globals__)
                if inspect.isfunction(callee) and callee in domains:
                    failures.append(f"{module}:{call.lineno}: free Kernel call has no proven receiver context")
                    invalid.add(callee)

    for entries in requested.values():
        for owner, name, item in entries:
            identities = binding_domains.get((owner, name), ())
            if len(identities) != 1 or any(
                key not in binding_covered for key in binding_calls if key[2:] == (owner, name)
            ):
                failures.append(
                    f"{owner}.{name}: prepared Field input requires one proven allocation source at every call"
                )
                continue
            source_type, source_member = next(iter(identities))
            private_source = "_forge_package_source_" + name
            item.update(
                source_type=source_type, source_member=list(source_member), source_path=[private_source, *source_member]
            )
            del item["input_expression"]
            if item.get("optional_input"):
                presence = "_forge_package_input_present_" + name
                owners[owner][presence] = dict(
                    kind="static",
                    choices=[("bool", False), ("bool", True)],
                    kernel_only=False,
                    presence_field=name,
                )
                item["input_presence_attribute"] = presence
                item["conditions"] = {**item.get("conditions", {}), presence: True}
            owners[owner][private_source] = dict(
                kind="owner_input",
                owner_type=source_type,
                field_input=name,
                binding_phase="preparation",
                **({"conditions": item["conditions"]} if item.get("conditions") else {}),
            )

    result = {}
    for kernel, identities in domains.items():
        if kernel in invalid:
            continue
        module, definition, parameters = kernels[kernel]
        receivers = {receiver for _, receiver, _, _ in identities}
        if len(receivers) != 1:
            failures.append(f"{module}.{definition['name']}: template receiver position changes between definitions")
            continue
        receiver = next(iter(receivers))
        public_identity = kernel.__module__ + "." + kernel.__qualname__
        for item in kernel_definitions[kernel]:
            item["free_template"] = receiver
            item["function_templates"] = parameters
            item["function_identity"] = public_identity
        for identity, _, values, resources in sorted(identities, key=repr):
            result.setdefault(identity, []).append(
                dict(
                    kernel=kernel,
                    definition=public_identity,
                    receiver=receiver,
                    templates=dict(values),
                    resources=dict(resources),
                    node=definition["node"],
                )
            )
    return result, list(dict.fromkeys(failures))
