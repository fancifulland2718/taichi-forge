"""Prove free-kernel receiver domains from ordinary source calls.

Only owner type/path facts flow through Host functions. This does not execute
constructors, Host branches or numerical definitions to discover input types.
"""

import ast
import builtins
import copy
import functools
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, MISSING
import inspect
import importlib.util
import types

from taichi_forge_runtime._static_values import static_key
from ._package_factories import _source_value
from ._package_input_families import source_class


def source_function_namespace(function, node, imported):
    """Resolve local imports only from the already admitted source closure."""
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
    namespace = dict(function.__globals__)
    for name, value in bindings.items():
        # Read only modules already admitted to the source closure. Imports
        # remain ordinary Host statements; discovery does not execute them.
        namespace.pop(name, None)
        if name not in shadowed and value is not None:
            namespace[name] = value
    return namespace


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
class _Host:
    function: object
    receiver: object = None


@dataclass(frozen=True)
class _CallableMetadata:
    source: object


@dataclass(frozen=True)
class _ScopeReference:
    owner: object


@dataclass(frozen=True)
class _SourceDomain:
    items: tuple


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


def _complete_fact(value):
    if isinstance(value, tuple):
        return all(_complete_fact(item) for item in value)
    if isinstance(value, (_Record, _Mapping)):
        return all(_complete_fact(item) for _, item in value.items)
    if isinstance(value, _SourceDomain):
        return all(_complete_fact(item) for item in value.items)
    if isinstance(value, _Choice):
        return all(_complete_fact(item) for item in value.alternatives)
    return value is not None


def _select_source_fact(value, source, index):
    """Refine one correlated source choice, leaving independent choices symbolic."""
    if isinstance(value, _Choice):
        if value.source == source:
            return _select_source_fact(value.alternatives[index], source, index)
        return _Choice(value.source, tuple(_select_source_fact(item, source, index) for item in value.alternatives))
    if isinstance(value, tuple):
        return tuple(_select_source_fact(item, source, index) for item in value)
    if isinstance(value, (_Mapping, _Record)):
        items = tuple((name, _select_source_fact(item, source, index)) for name, item in value.items)
        return _Record(value.cls, items) if isinstance(value, _Record) else _Mapping(items)
    if isinstance(value, _SourceDomain):
        return _SourceDomain(tuple(_select_source_fact(item, source, index) for item in value.items))
    if isinstance(value, (_Kernel, _Host)):
        return type(value)(value.function, _select_source_fact(value.receiver, source, index))
    if isinstance(value, _CallableMetadata):
        return _CallableMetadata(_select_source_fact(value.source, source, index))
    if isinstance(value, _ScopeReference):
        return _ScopeReference(_select_source_fact(value.owner, source, index))
    return value


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
        and all(not hasattr(vars(base).get(field.name), "__get__")
                or type(vars(base).get(field.name)) is types.MemberDescriptorType
                for base in cls.__mro__ for field in fields(cls))
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
                              and isinstance(method.decorator_list[0], ast.Name)
                              and method.decorator_list[0].id == 'staticmethod'
                              and namespace.get('staticmethod', builtins.staticmethod) is builtins.staticmethod):
                            descriptor = vars(cls).get(method.name)
                            if type(descriptor) is staticmethod and inspect.isfunction(descriptor.__func__):
                                functions[descriptor.__func__] = (module, method)
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
    # Free preparation already has a source-proven owner projection. Reuse its
    # normalized declarations (including helper-local aliases), with the actual
    # Python parameter name restored for ordinary call binding.
    preparations = {}
    if resource_inventory is None:
        for function, (module, node) in list(functions.items()):
            preparation = getattr(node, "_package_free_preparation", None)
            if preparation is None:
                continue
            owner, receiver, normalized = preparation

            class RestoreReceiver(ast.NodeTransformer):
                def __init__(self, parameter):
                    self.parameter = parameter

                def visit_Name(self, part):
                    if part.id == "self":
                        part.id = self.parameter
                    return part

                def visit_arg(self, part):
                    if part.arg == "self":
                        part.arg = self.parameter
                    return part

            restore = RestoreReceiver(receiver)
            functions[function] = (module, restore.visit(copy.deepcopy(normalized)))
            preparations[(module, owner, normalized.name)] = (function, receiver, restore)
    requested, request_sources = {}, {}
    for identity, schema in owners.items():
        if resource_inventory is not None:
            break
        for name, item in schema.items():
            if "input_expression" not in item:
                continue
            preparation = preparations.get((
                item.get("declaration_module"), item.get("declaration_owner"), item["input_method"]
            ))
            if preparation is not None:
                method, receiver, restore = preparation
                expression = restore.visit(copy.deepcopy(item["input_expression"]))
            else:
                method = inspect.getattr_static(classes[identity], item["input_method"], None)
                receiver, expression = "self", item["input_expression"]
            if method not in functions:
                raise ValueError("Prepared input method is outside its ordinary source definition: "
                                 + identity + "." + item["input_method"] + " (" + name + ")")
            request_sources[id(item)] = receiver, expression
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
                        inputs = {"self": _Owner(identity)}
                        # Public recording methods may receive their builder
                        # from the consumer instead of a call in this package.
                        # Use the explicit API type, never the parameter name
                        # or the spelling of a dispatch method. Read the AST
                        # so postponed annotations need no eval/import hooks.
                        node = functions[function][1]
                        for parameter in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                            if (parameter.arg != "self" and parameter.annotation is not None
                                    and _source_value(parameter.annotation, function.__globals__) is GraphBuilder):
                                inputs[parameter.arg] = _Graph()
                        pending.append((function, inputs, (function,), None))

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
    binding_domains = {}

    def source_callable(expression, namespace):
        if resource_inventory is not None and hasattr(expression, '_package_source_callable'):
            return expression._package_source_callable
        result = _source_value(expression, namespace)
        if result is None and isinstance(expression, ast.Attribute):
            cls = _source_value(expression.value, namespace)
            if inspect.isclass(cls):
                result = inspect.getattr_static(cls, expression.attr, None)
        return result

    namespaces = {}
    scope_factories = {}

    def scope_factory(callee, call):
        from ._package_factories import prove_factory
        key = (callee, ast.dump(call))
        if key not in scope_factories:
            try:
                normalized = prove_factory(callee, call)
            except (TypeError, ValueError, OSError):
                normalized = None
            scope_factories[key] = (normalized if getattr(normalized, '_package_role', None)
                                   == 'experimental.ResourceScope' else None)
        return scope_factories[key]
    sequence_projector = None
    iteration_failures = {}

    def function_namespace(function):
        if function in namespaces:
            return namespaces[function]
        _, node = functions[function]
        namespace = source_function_namespace(
            inspect.unwrap(function) if resource_inventory is not None else function, node, imported
        )
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
        if isinstance(receiver, _CallableMetadata):
            return receiver
        if isinstance(receiver, _ScopeReference):
            item = member(receiver.owner, name)
            if item is not None:
                return item
            return receiver  # Unknown scope use retains its owner at the call boundary.
        if isinstance(receiver, _Choice):
            return _Choice(receiver.source, tuple(attribute(item, name) for item in receiver.alternatives))
        item = member(receiver, name)
        if item is not None:
            return item
        if isinstance(receiver, _Owner):
            method = inspect.getattr_static(classes[receiver.identity], name, None)
            if resource_inventory is not None and type(method) is staticmethod and method.__func__ in functions:
                return _Host(method.__func__)
            if inspect.isfunction(method) and method in kernels:
                return _Kernel(method, receiver)
            if resource_inventory is not None and inspect.isfunction(method) and method in functions:
                return _Host(method, receiver)
        if resource_inventory is not None and isinstance(receiver, _Record):
            method = inspect.getattr_static(receiver.cls, name, None)
            if inspect.isfunction(method) and method in functions:
                return _Host(method, receiver)
        return None

    def build_context(function, arguments, return_chain):
        nonlocal sequence_projector
        module, node = functions[function]
        namespace = function_namespace(function)
        # A rebinding invalidates an earlier owner fact. Nested definitions have
        # their own lexical scope and cannot lend assignments to this call.
        nodes = []
        source_only_calls = set()
        source_expression_choices = {}
        condition_value = None

        def source_projection():
            nonlocal sequence_projector
            from ._package_source_sequences import SourceSequences
            if sequence_projector is None:
                sequence_projector = SourceSequences(sources, imported, inventory=True)
                for callee, (_, definition) in functions.items():
                    if not definition.decorator_list or callee in cached_sources:
                        sequence_projector.functions[callee] = (definition, callee.__globals__)
            projection = copy.copy(sequence_projector)
            projection.resolve_method = sequence_method
            projection.resolve_cached = sequence_cached
            projection.resolve_namespace = function_namespace
            return projection

        def sequence_environment():
            def syntax(fact):
                if (isinstance(fact, _Choice) and fact.alternatives
                        and all(isinstance(item, tuple) for item in fact.alternatives)
                        and len({len(item) for item in fact.alternatives}) == 1):
                    # Preserve positional unpacking and the common choice ID,
                    # without selecting every other operation in the recipe.
                    return ast.Tuple([
                        syntax(_Choice(fact.source, tuple(item[index] for item in fact.alternatives)))
                        for index in range(len(fact.alternatives[0]))
                    ], ast.Load())
                if isinstance(fact, (tuple, _SourceDomain)):
                    expression = ast.Tuple([syntax(item) for item in (fact.items if isinstance(fact, _SourceDomain) else fact)], ast.Load())
                    if isinstance(fact, _SourceDomain):
                        expression._package_source_domain = True
                    return expression
                expression = ast.Name('_package_input_fact', ast.Load())
                expression._package_input_fact = fact
                return expression
            return {name: syntax(fact) if isinstance(fact, (tuple, _SourceDomain)) else ast.Name(name, ast.Load())
                    for name, fact in arguments.items()}

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

                projection = source_projection()
                try:
                    iteration = statement.iter
                    environment = sequence_environment()
                    class LocalAliases(ast.NodeTransformer):
                        def __init__(self):
                            self.active = set()
                            self.used = []

                        def visit_Name(self, part):
                            if (isinstance(part.ctx, ast.Load) and part.id in aliases
                                    and stores.get(part.id) == 1):
                                if part.id in tainted:
                                    raise UnprovenSourceSequence('Inventory local sequence escaped or was mutated')
                                if part.id in self.active:
                                    raise UnprovenSourceSequence('Recursive local inventory alias')
                                self.active.add(part.id)
                                self.used.append(aliases[part.id])
                                replacement = self.visit(copy.deepcopy(aliases[part.id]))
                                self.active.remove(part.id)
                                return replacement
                            return part

                        def visit_Lambda(self, part):
                            return part

                    substitution = LocalAliases()
                    iteration = substitution.visit(copy.deepcopy(statement.iter))
                    from taichi_forge.lang.impl import static
                    static_iteration = (isinstance(iteration, ast.Call) and source_callable(iteration.func, namespace) is static
                                        and len(iteration.args) == 1 and not iteration.keywords)
                    if static_iteration:
                        iteration = iteration.args[0]
                    # Keep the proven receiver resolver local to this lexical
                    # context; nested discovery must not replace another call's
                    # bindings. Choice IDs remain shared by the source analysis.
                    sequence = (projection.iteration(iteration, environment, namespace, ()) if static_iteration else
                                projection.sequence(projection.value(iteration, environment, namespace)))
                    if not sequence_projector.immutable(sequence):
                        raise UnprovenSourceSequence('Iteration requires an immutable source sequence')
                    names = {part.id for part in ast.walk(statement.target) if isinstance(part, ast.Name)}
                    if any(isinstance(part, ast.Name) and part.id in names and not isinstance(part.ctx, ast.Load)
                           for body in statement.body for part in ast.walk(body)):
                        raise UnprovenSourceSequence('Iteration source is rebound in the loop body')
                except (UnprovenSourceSequence, TypeError) as error:
                    if (isinstance(iteration, ast.Call)
                            and source_callable(iteration.func, namespace) in sequence_projector.functions
                            or isinstance(statement.iter, ast.Attribute) and sequence_cached(statement.iter) is not None):
                        iteration_failures.setdefault(function, set()).add(str(error))
                else:
                    # The source-only projector proved these expressions without
                    # executing application code. Do not rediscover a selected
                    # local helper as an opaque owner escape in the generic pass.
                    source_only_calls.update(id(part) for expression in substitution.used
                                             for part in ast.walk(expression) if isinstance(part, ast.Call))
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
            if (resource_inventory is not None and condition_value is not None
                    and isinstance(statement, (ast.If, ast.IfExp))):
                fact = value(statement.test)
                if isinstance(fact, _Choice) and fact.source[0] == 'source_sequence':
                    class SelectSource(ast.NodeTransformer):
                        def __init__(self, index):
                            self.index = index

                        def visit_Name(self, part):
                            if isinstance(part.ctx, ast.Load):
                                original = value(part)
                                selected = _select_source_fact(original, fact.source, self.index)
                                if selected != original:
                                    part._package_input_fact = selected
                            return part

                        def visit_IfExp(self, part):
                            if getattr(part, '_package_source_choice', None) == fact.source[1]:
                                return self.visit(part.body if self.index == 0 else part.orelse)
                            original = value(part)
                            if isinstance(original, _Choice) and original.source == fact.source:
                                return self.visit(part.body if self.index == 0 else part.orelse)
                            return self.generic_visit(part)
                    for index in range(len(fact.alternatives)):
                        walk(SelectSource(index).visit(copy.deepcopy(statement)))
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
                and expression.id in ("getattr", "hasattr", "callable", "dict", "isinstance", "id")
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
                    (part.func.attr in ("dispatch", "private_scalar") and graph_source(part.func.value))
                    or (part.func.attr == "items" and not part.args and not part.keywords)
                )
                # A proven Graph private_scalar consumes a Python numeric value;
                # reading a frozen record member here cannot mutate the record.
                # An unrelated object's method with this name is still opaque.
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
            if resource_inventory is not None and hasattr(expression, '_package_input_fact'):
                return expression._package_input_fact
            bindings = bindings or {}

            def child(item):
                return value(item, visiting, bindings)

            def source(item):
                if resource_inventory is not None and hasattr(item, '_package_source_callable'):
                    return item._package_source_callable
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
                    if resource_inventory is not None and (name in namespace):
                        from ._package_factories import _literal
                        if _literal(resolved):
                            return _Literal(resolved)
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
                items = tuple(child(item) for item in expression.elts)
                return (_SourceDomain(items) if resource_inventory is not None
                        and getattr(expression, '_package_source_domain', False) else items)
            if isinstance(expression, ast.Constant) and (
                expression.value is None or type(expression.value) in (bool, int, float, str)
            ):
                return _Literal(expression.value)
            if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
                left, right = child(expression.left), child(expression.right)
                if isinstance(left, _Literal) and isinstance(right, _Literal) and type(left.value) is type(right.value) is str:
                    return _Literal(left.value + right.value)
            if resource_inventory is not None and isinstance(expression, ast.Subscript):
                parent, index = child(expression.value), child(expression.slice)
                if isinstance(parent, _Literal) and type(parent.value) is tuple and isinstance(index, _Literal) and type(index.value) is int:
                    try:
                        return _Literal(parent.value[index.value])
                    except IndexError:
                        return None
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
                condition = proven_condition(expression.test, visiting, bindings)
                if condition is not None:
                    return child(expression.body if condition else expression.orelse)
                left, right = child(expression.body), child(expression.orelse)
                for resource, absent in ((left, right), (right, left)):
                    if isinstance(resource, _Resource) and isinstance(absent, _Literal) and absent.value is None:
                        return _Resource(resource.owner, resource.path, optional=True)
                if resource_inventory is not None and left != right:
                    projection = source_projection()
                    key = (getattr(expression, 'lineno', None), getattr(expression, 'col_offset', None),
                           projection.source_key(expression))
                    if key not in source_expression_choices:
                        source_expression_choices[key] = projection.choice(expression.test)[1]
                    return _Choice(('source_sequence', source_expression_choices[key]), (left, right))
                return left if left == right else None
            if isinstance(expression, ast.Call):
                cls = source(expression.func)
                if resource_inventory is not None and cls is builtins.callable and len(expression.args) == 1 and not expression.keywords:
                    source_only_calls.add(id(expression))
                    candidate = child(expression.args[0])
                    def known_callable(item):
                        return (all(known_callable(arm) for arm in item.alternatives)
                                if isinstance(item, _Choice) else isinstance(item, (_Host, _Kernel)))
                    return _Literal(True) if known_callable(candidate) else None
                if (resource_inventory is not None and cls is builtins.hasattr and len(expression.args) == 2
                        and not expression.keywords and isinstance(child(expression.args[0]), _Graph)
                        and isinstance(child(expression.args[1]), _Literal)):
                    # Known Graph objects expose frontend metadata. Do not
                    # claim a particular Graph subtype or prune either path.
                    source_only_calls.add(id(expression))
                    return None
                if resource_inventory is not None and cls is builtins.id and len(expression.args) == 1 and not expression.keywords:
                    source_only_calls.add(id(expression))
                    return None  # Identity is Host data, never a code or resource identity.
                if cls is builtins.dict and not expression.args and all(k.arg is not None for k in expression.keywords):
                    return _Mapping(tuple((k.arg, child(k.value)) for k in expression.keywords))
                if cls is builtins.getattr and len(expression.args) in (2, 3) and not expression.keywords:
                    name = child(expression.args[1])
                    if isinstance(name, _Literal) and type(name.value) is str:
                        receiver = child(expression.args[0])
                        def callable_metadata(item):
                            return (all(callable_metadata(arm) for arm in item.alternatives)
                                    if isinstance(item, _Choice) else isinstance(item, (_Kernel, _Host)))
                        if resource_inventory is not None and name.value == '_primal' and callable_metadata(receiver):
                            # Reading a proven function's metadata does not
                            # invoke it or expose its bound owner to writes.
                            return _CallableMetadata(receiver)
                        result = child(ast.Attribute(expression.args[0], name.value, ast.Load()))
                        if resource_inventory is not None and result is None:
                            # Same provenance as an ordinary attribute read.
                            # Properties retain member()'s conservative effects;
                            # a Host scalar lookup does not escape its owner.
                            source_only_calls.add(id(expression))
                        return result
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
                    if inspect.isclass(cls) and (type(cls) is type or cls is Mapping):
                        def classify(item):
                            if isinstance(item, _Choice):
                                answers = tuple(classify(arm) for arm in item.alternatives)
                                return answers[0] if all(answer == answers[0] for answer in answers) else _Choice(item.source, answers)
                            if isinstance(item, _Literal) and type(item.value) in (bool, int, float, str, type(None)):
                                return _Literal(isinstance(item.value, cls))
                            if isinstance(item, _Owner):
                                return _Literal(issubclass(classes[item.identity], cls))
                            if isinstance(item, _Record):
                                return _Literal(issubclass(item.cls, cls))
                            if isinstance(item, (_Kernel, _Host)) and _plain_record(cls):
                                return _Literal(False)
                            if isinstance(item, _Resource):
                                # Allocation-proven Fields cannot be arbitrary Host
                                # adapters. Optional inputs also admit None.
                                candidates = (ScalarField, MatrixField, StructField)
                                if item.optional:
                                    candidates += (type(None),)
                                answers = {issubclass(candidate, cls) for candidate in candidates}
                                if len(answers) == 1:
                                    return _Literal(answers.pop())
                            return None
                        return classify(item)
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
                if resource_inventory is not None and isinstance(expression.func, ast.Attribute) and isinstance(child(expression.func.value), _Graph):
                    method = expression.func.attr
                    if method == 'compile':
                        return _Graph()
                    if method in ('private_scalar', 'private_ndarray'):
                        source_only_calls.add(id(expression))
                        return None
                    children = expression.args
                    if ((method == 'append' and len(children) == 1)
                            or (method == 'while_loop' and len(children) == 2)) and all(isinstance(child(item), _Graph) for item in children):
                        source_only_calls.add(id(expression))
                        return None
                callee, receiver = call_target(expression.func, visiting, bindings)
                if resource_inventory is not None and receiver is None and callee in functions and callee not in kernels:
                    factory = scope_factory(callee, expression)
                    if factory is not None:
                        return _ScopeReference(child(factory.args[0]))
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
                        evaluate, _, returned_nodes, _, _ = build_context(callee, return_inputs, (*return_chain, callee))
                        results = [evaluate(part.value) for part in returned_nodes if isinstance(part, ast.Return)]
                        if results and all(item == results[0] for item in results):
                            returned = results[0]
                        elif (any(isinstance(item, _Record) for item in results)
                              and all(isinstance(item, _Record)
                                      or isinstance(item, _Literal) and item.value is None for item in results)):
                            # An optional frozen binding retains each member's
                            # allocation source. Presence stays a Host choice;
                            # distinct source contracts are still checked below.
                            returned = _Choice(('record_return', return_key), tuple(dict.fromkeys(results)))
                        else:
                            returned = None
                        return_cache[return_key] = returned
                    if return_cache[return_key] is not None and (
                            resource_inventory is None or _complete_fact(return_cache[return_key])):
                        return return_cache[return_key]
                if resource_inventory is not None and callee in functions and callee not in return_chain:
                    from ._package_source_sequences import UnprovenSourceSequence
                    try:
                        projection = source_projection()
                        sequence = projection.sequence(projection.value(expression, sequence_environment(), namespace))
                        if projection.immutable(sequence):
                            result = child(projection.syntax(sequence, allow_domain=True))
                            source_only_calls.add(id(expression))
                            return result
                    except (UnprovenSourceSequence, TypeError):
                        pass
            return None

        def call_target(expression, visiting=(), bindings=None):
            if resource_inventory is not None:
                resolved = value(expression, visiting, bindings)
                if isinstance(resolved, (_Kernel, _Host)):
                    return resolved.function, resolved.receiver
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

        def proven_condition(expression, visiting=(), bindings=None):
            fact = value(expression, visiting, bindings)
            if isinstance(fact, _Literal) and type(fact.value) in (bool, int, float, str, type(None)):
                return bool(fact.value)
            if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
                answer = proven_condition(expression.operand, visiting, bindings)
                return None if answer is None else not answer
            if isinstance(expression, ast.BoolOp):
                answers = [proven_condition(item, visiting, bindings) for item in expression.values]
                if isinstance(expression.op, ast.And):
                    return False if False in answers else True if all(answer is True for answer in answers) else None
                return True if True in answers else False if all(answer is False for answer in answers) else None
            if (isinstance(expression, ast.Compare) and len(expression.ops) == 1
                    and isinstance(expression.ops[0], (ast.Is, ast.IsNot))):
                left, right = value(expression.left, visiting, bindings), value(expression.comparators[0], visiting, bindings)
                if isinstance(left, _Literal) and left.value is None:
                    left, right = right, left
                if isinstance(right, _Literal) and right.value is None:
                    if isinstance(left, _Literal):
                        absent = left.value is None
                    elif isinstance(left, (_Owner, _Graph, _Host, _Kernel, _Record)) or isinstance(left, _Resource) and not left.optional:
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
        return value, call_target, nodes, module, source_only_calls

    while pending:
        function, arguments, chain, call_site = pending.popleft()
        identity = (function, tuple(sorted(arguments.items())))
        module = functions[function][0]
        # Call-site coverage is per incoming edge, but source facts are per
        # function/input context. In particular, repeated Graph dispatches must
        # not rebuild the same lexical and resource analysis before this guard.
        if preparation_receivers is not None and function in preparation_receivers:
            receivers = [(key, item.identity) for key, item in arguments.items() if isinstance(item, _Owner)]
            if not receivers:
                # An ordinary Host wrapper can forward an external owner to an
                # already declared allocation entry. Check that boundary below;
                # an unknown caller never establishes a new receiver contract.
                continue
            if len(receivers) != 1:
                raise ValueError(f"{module}:{functions[function][1].lineno}: preparation requires one proven owner")
            preparation_receivers[function].add(receivers[0])
            if call_site is not None:
                covered.add(call_site)
        if identity in visited:
            continue
        visited.add(identity)
        value, target, nodes, module, source_only_calls = build_context(function, arguments, (function,))
        for owner, name, item in requested.get(function, ()):
            parameter, expression = request_sources[id(item)]
            receiver = arguments.get(parameter)
            if not isinstance(receiver, _Owner) or receiver.identity != owner:
                continue
            fact = value(expression)
            alternatives = [fact]
            while alternatives:
                candidate = alternatives.pop()
                if isinstance(candidate, _Choice):
                    alternatives.extend(candidate.alternatives)
                elif isinstance(candidate, _Resource) and (not candidate.optional or item.get("optional_input")):
                    source = owners[candidate.owner.identity].get(candidate.path[0], {})
                    if source.get("kind") == "dense_field" and (
                        len(candidate.path) == 1 or source.get("struct_field") and len(candidate.path) == 2
                    ):
                        binding_domains.setdefault((owner, name), set()).add((candidate.owner.identity, candidate.path))
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
            elif isinstance(item, _SourceDomain):
                for child in item.items:
                    inventory_input(child)
            elif isinstance(item, (_Kernel, _Host)):
                inventory_input(item.receiver)
            elif isinstance(item, _CallableMetadata):
                inventory_input(item.source)
            elif isinstance(item, _ScopeReference):
                inventory_input(item.owner)
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

        def forward_call(callee, receiver, call, selected_values=None):
            if resource_inventory is not None and receiver is None and callee not in kernels:
                if scope_factory(callee, call) is not None:
                    return  # Reuse the existing proof; construction performs no device work.
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
                return
            kwargs = {item.arg: value(item.value) for item in call.keywords if item.arg is not None}
            if selected_values is not None:
                positional, keywords = selected_values
                values = ([receiver] if receiver is not None else []) + list(positional)
                kwargs = dict(keywords)
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
                return
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
                # Host recipes retain choices until the selected operation is
                # used. Expanding every element of a forwarded sequence here
                # creates a Cartesian product of unrelated stages. Actual
                # dispatch still resolves the correlated Kernel/receiver pair.
                contexts = (inputs,) if resource_inventory is not None else _choice_contexts(inputs)
                for context in contexts:
                    if callee in chain:
                        if (callee, tuple(sorted(context.items()))) not in visited:
                            failures.append(
                                f"{module}:{call.lineno}: recursive Kernel input forwarding changes its source context"
                            )
                    else:
                        pending.append((callee, context, (*chain, callee), id(call)))


        for call in (part for part in nodes if isinstance(part, ast.Call)):
            if id(call) in source_only_calls:
                continue
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
                    callback = value(call.func)
                    if (isinstance(callback, _Choice)
                            and not any(isinstance(arg, ast.Starred) for arg in call.args)
                            and all(k.arg is not None for k in call.keywords)):
                        choices = list(_choice_contexts(dict(
                            callback=callback, args=tuple(value(arg) for arg in call.args),
                            kwargs=_Mapping(tuple((k.arg, value(k.value)) for k in call.keywords)))))
                        if all(isinstance(choice['callback'], (_Host, _Kernel)) for choice in choices):
                            for choice in choices:
                                method = choice['callback']
                                forward_call(method.function, method.receiver, call,
                                             (choice['args'], choice['kwargs'].items))
                            continue
                    fact = value(call)
                    if id(call) in source_only_calls:
                        continue
                    if isinstance(fact, (_Mapping, _Record, _Kernel, _Host, _Resource, _Literal, _Graph,
                                         _CallableMetadata, _ScopeReference, tuple)):
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
            forward_call(callee, receiver, call)

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

    # Free Kernel template domains must cover all direct uses. Prepared Field
    # borrows instead infer one allocation contract below and use the existing
    # binding boundary for caller-provided values; unknown calls are not proof
    # of that contract, nor do they require a second implementation.
    for function, (module, node) in functions.items():
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and id(call) not in covered:
                callee = source_callable(call.func, function_namespace(function))
                if isinstance(call.func, ast.Attribute) and call.func.attr == "dispatch" and call.args:
                    callee = _source_value(call.args[0], function.__globals__)
                if inspect.isfunction(callee) and callee in domains:
                    failures.append(f"{module}:{call.lineno}: free Kernel call has no proven receiver context")
                    invalid.add(callee)

    for entries in requested.values():
        for owner, name, item in entries:
            identities = binding_domains.get((owner, name), ())
            # Source-scope borrowing retains one known allocation role. All
            # proven contexts must agree; an unknown external caller adds no
            # source fact. Its actual Field still goes through the same borrow
            # binding contract before use. No evidence cannot infer an ABI.
            if len(identities) != 1:
                failures.append(
                    f"{owner}.{name}: prepared Field input requires one definition-proven allocation contract; "
                    "use a typed external borrow for independent allocation sources"
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
