"""Resolve cold numerical factories from normal definitions, without calling them."""

import ast
import copy
import functools
import inspect
from dataclasses import dataclass
from types import CodeType, FunctionType, MappingProxyType

from ._package_factories import _frontend_module, _source_value


@dataclass(frozen=True)
class FieldPath:
    path: tuple


def _factory_function(factory, namespace):
    """Unwrap only the standard cache, never call an application wrapper."""
    function = namespace.get(factory.name)
    if not factory.decorator_list:
        return function if inspect.isfunction(function) else None
    if len(factory.decorator_list) != 1 or type(function) is not functools._lru_cache_wrapper:
        return None
    decorator = factory.decorator_list[0]
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    target = _source_value(target, namespace)
    if target is not functools.lru_cache and target is not functools.cache:
        return None
    if isinstance(decorator, ast.Call):
        if (
            target is functools.cache
            or len(decorator.args) > 2
            or any(k.arg not in ("maxsize", "typed") for k in decorator.keywords)
        ):
            return None
        try:
            options = inspect.signature(functools.lru_cache).bind(
                *(ast.literal_eval(arg) for arg in decorator.args),
                **{k.arg: ast.literal_eval(k.value) for k in decorator.keywords},
            )
        except (ValueError, TypeError, SyntaxError):
            return None
        if any(
            (name == "maxsize" and value is not None and type(value) is not int)
            or (name == "typed" and type(value) is not bool)
            for name, value in options.arguments.items()
        ):
            return None
    original = vars(function).get("__wrapped__")
    return original if inspect.isfunction(original) and "__wrapped__" not in vars(original) else None


def _callable(value):
    return inspect.isfunction(value) or type(value) is functools._lru_cache_wrapper


def discover(module, tree, kind, namespace, fail):
    factories, definitions = {}, []
    for factory in tree.body:
        if not isinstance(factory, ast.FunctionDef) or any(kind(d) for d in factory.decorator_list):
            continue
        nested = [
            node
            for node in ast.walk(factory)
            if isinstance(node, ast.FunctionDef) and any(kind(d) == "kernel" for d in node.decorator_list)
        ]
        if not nested:
            continue
        function = _factory_function(factory, namespace)
        if function is None:
            fail(
                module, factory, "Cold kernel factory requires an undecorated function or the standard functools cache"
            )
        if len({node.name for node in nested}) != len(nested):
            fail(module, factory, "Cold factory numerical branches require distinct definition names")
        parents = {id(child): parent for parent in ast.walk(factory) for child in ast.iter_child_nodes(parent)}
        for node in nested:
            if len(node.decorator_list) != 1:
                fail(module, node, "Cold kernel factory decorator combinations require further support")
            parent = parents[id(node)]
            while isinstance(parent, ast.If):
                parent = parents[id(parent)]
            if parent is not factory:
                fail(module, node, "Cold numerical definitions require direct conditional factory branches")
            block_parent = parents[id(node)]
            block = block_parent.body if node in block_parent.body else block_parent.orelse
            tail = block[block.index(node) + 1 :]
            if not (
                len(tail) == 1
                and isinstance(tail[0], ast.Return)
                and isinstance(tail[0].value, ast.Name)
                and tail[0].value.id == node.name
            ):
                fail(module, node, "Cold kernel factory must return its numerical definition")
            decorator = node.decorator_list[0]
            if isinstance(decorator, ast.Call) and (
                decorator.args or any(k.arg != "opt_level" for k in decorator.keywords)
            ):
                fail(module, node, "Cold kernel decorator requires existing explicit compile options")
            options = dict((k.arg, k.value) for k in decorator.keywords) if isinstance(decorator, ast.Call) else {}
            definitions.append(
                dict(
                    owner=None,
                    node=node,
                    kind="kernel",
                    name=factory.name + "." + node.name,
                    requires_owner=True,
                    error_interface=False,
                    closure_definition=True,
                    closure_names=list(_code(function, node).co_freevars),
                    closure_options=options,
                    conditional_factory=len(nested) > 1 or bool(factory.decorator_list),
                )
            )
        factories[namespace[factory.name]] = dict(module=module, function=function, factory=factory, nodes=nested)
    return factories, definitions


def _code(function, node):
    matches = [item for item in function.__code__.co_consts if isinstance(item, CodeType) and item.co_name == node.name]
    if len(matches) != 1:
        raise ValueError("Cold kernel source does not uniquely identify its original code object")
    return matches[0]


def calls(sources, imported, fail):
    factories = {
        function: recipe
        for *_, tree, kind, schemas, definitions in sources
        for function, recipe in tree._package_closure_factories.items()
    }
    result, covered = {}, set()
    for module, _, _, _, tree, _, schemas, _ in sources:
        for owner in tree.body:
            if not isinstance(owner, ast.ClassDef) or owner.name not in schemas:
                continue
            constructor = next((n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
            if constructor is None:
                continue
            for statement in constructor.body:
                if not (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Attribute)
                    and isinstance(statement.targets[0].value, ast.Name)
                    and statement.targets[0].value.id == "self"
                    and isinstance(statement.value, ast.Call)
                ):
                    continue
                call = statement.value
                target = _source_value(call.func, vars(imported[module]))
                recipe = factories.get(target) if _callable(target) else None
                if recipe is not None:
                    key = module + "." + owner.name
                    result.setdefault(key, []).append((statement.targets[0].attr, call, recipe))
                    covered.add(id(call))
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and id(call) not in covered:
                target = _source_value(call.func, vars(imported[module]))
                if _callable(target) and target in factories:
                    fail(module, call, "Cold kernel factory requires declared constructor resource inputs")
    used = {id(recipe) for items in result.values() for _, _, recipe in items}
    for recipe in factories.values():
        if id(recipe) in used:
            continue
        fail(recipe["module"], recipe["factory"], "Cold kernel factory has no declared input domain")
    return result


def materialize(recipe, call, owner, environment, kernels):
    """Evaluate only type/declaration expressions; numerical AST goes to Forge."""
    import taichi_forge as ti
    from taichi_forge.lang._compiled_template import bind
    from taichi_forge.types._argument_descriptor import describe_element_type
    from ._component import DenseField, _capture_id
    from taichi_forge_runtime._static_values import static_key

    function, factory = recipe["function"], recipe["factory"]
    builtins = dict(tuple=tuple, len=len)
    for name, value in builtins.items():
        if function.__globals__.get(name, value) is not value or function.__builtins__.get(name) is not value:
            raise ValueError("Cold factory cannot use shadowed declaration builtins")

    def input_value(expr):
        if isinstance(expr, (ast.Tuple, ast.List)):
            return tuple(input_value(item) for item in expr.elts)
        path = []
        root = expr
        while isinstance(root, ast.Attribute):
            path.insert(0, root.attr)
            root = root.value
        if isinstance(root, ast.Name) and root.id == "self" and path:
            declaration = environment.declarations.get(_capture_id(path))
            if len(path) == 1 and isinstance(declaration, DenseField) and not getattr(declaration, "conditions", ()):
                return FieldPath(tuple(path))
        try:
            value = ast.literal_eval(expr)
            static_key(value)
            return value
        except (ValueError, TypeError, SyntaxError):
            raise ValueError("Cold factory input requires a declared Field or immutable literal") from None

    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(k.arg is None for k in call.keywords):
        raise ValueError("Cold factory input requires explicit arguments")
    arguments = inspect.signature(function).bind(
        *[input_value(arg) for arg in call.args], **{k.arg: input_value(k.value) for k in call.keywords}
    )
    arguments.apply_defaults()
    local = dict(arguments.arguments)

    def value(expr):
        if isinstance(expr, ast.Constant):
            static_key(expr.value)
            return expr.value
        if isinstance(expr, ast.Name):
            if expr.id in local:
                return local[expr.id]
            result = function.__globals__.get(expr.id, builtins.get(expr.id))
            if _frontend_module(result):
                return result
            if any(result is item for item in builtins.values()):
                return result
        if isinstance(expr, ast.Attribute):
            parent = value(expr.value)
            if _frontend_module(parent):
                return vars(parent)[expr.attr]
        if isinstance(expr, (ast.Tuple, ast.List)):
            return tuple(value(item) for item in expr.elts)
        if isinstance(expr, ast.Call):
            target = value(expr.func)
            args = [value(item) for item in expr.args]
            if target is tuple and len(args) == 1 and not expr.keywords and type(args[0]) is tuple:
                return args[0]
            if target is len and len(args) == 1 and not expr.keywords and type(args[0]) is tuple:
                return len(args[0])
            if any(target is item for item in (ti.types.vector, ti.types.matrix)):
                if any(k.arg is None for k in expr.keywords):
                    raise ValueError("Cold factory types require explicit keyword arguments")
                return target(*args, **{k.arg: value(k.value) for k in expr.keywords})
        if isinstance(expr, ast.BinOp):
            left, right = value(expr.left), value(expr.right)
            if type(left) is int and type(right) is int:
                if isinstance(expr.op, ast.Add):
                    return left + right
                if isinstance(expr.op, ast.Sub):
                    return left - right
                if isinstance(expr.op, ast.Mult):
                    return left * right
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            result = value(expr.operand)
            if type(result) is bool:
                return not result
        if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
            left, right = value(expr.left), value(expr.comparators[0])
            if type(left) is type(right) and type(left) in (int, str, bool):
                operations = {
                    ast.Lt: lambda: left < right,
                    ast.LtE: lambda: left <= right,
                    ast.Gt: lambda: left > right,
                    ast.GtE: lambda: left >= right,
                    ast.Eq: lambda: left == right,
                    ast.NotEq: lambda: left != right,
                }
                if type(expr.ops[0]) in operations:
                    return operations[type(expr.ops[0])]()
        raise ValueError("Unproven cold factory declaration expression: " + ast.unparse(expr))

    def select(statements):
        # Only declaration/control syntax is evaluated here. The selected
        # numerical body is always compiled by the normal Forge AST pipeline.
        for statement in statements:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and type(statement.value.value) is str
            ):
                continue
            if isinstance(statement, ast.FunctionDef) and statement in recipe["nodes"]:
                return statement  # discover proved its immediate return.
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                local[statement.targets[0].id] = value(statement.value)
            elif isinstance(statement, ast.If):
                guard = value(statement.test)
                if type(guard) is not bool:
                    raise ValueError("Cold factory condition requires a proven fixed boolean")
                selected = select(statement.body if guard else statement.orelse)
                if selected is not None:
                    return selected
            elif isinstance(statement, ast.Raise):
                raise ValueError("Cold factory declaration violates its input guard")
            else:
                raise ValueError("Cold factory has an unproven declaration or side effect")
        return None

    node = select(factory.body)
    if node is None:
        raise ValueError("Cold factory has no numerical return for its declared inputs")

    def syntax_binding(item):
        if isinstance(item, FieldPath):
            return ("path", item.path)
        if type(item) is tuple:
            return ("sequence", tuple(syntax_binding(child) for child in item))
        return ("static", static_key(item))

    def actual(item):
        if isinstance(item, FieldPath):
            return environment.read(owner, _capture_id(item.path))
        return tuple(actual(child) for child in item) if type(item) is tuple else item

    code = _code(function, node)
    bindings = {name: syntax_binding(local[name]) for name in code.co_freevars}
    if not any(isinstance(item, FieldPath) for item in _leaves(tuple(local[name] for name in code.co_freevars))):
        raise ValueError("Cold factory requires at least one declared allocation source")
    cell = lambda item: (lambda: item).__closure__[0]
    numerical = FunctionType(
        code, function.__globals__, node.name, closure=tuple(cell(actual(local[name])) for name in code.co_freevars)
    )
    numerical.__qualname__ = function.__qualname__ + ".<locals>." + node.name
    numerical.__module__ = function.__module__
    annotations = {arg.arg: value(arg.annotation) for arg in node.args.args if arg.annotation is not None}
    if node.returns is not None:
        annotations["return"] = value(node.returns)
    original = copy.deepcopy(node)
    original.decorator_list = []
    bind(
        lambda: ast.Module(body=[copy.deepcopy(original)], type_ignores=[]),
        globals_getter=lambda: function.__globals__,
        closure_getter=lambda: {
            name: cell.cell_contents for name, cell in zip(code.co_freevars, numerical.__closure__)
        },
        is_class=False,
        filename=function.__code__.co_filename,
        annotations=annotations,
    )(numerical)
    decorator = node.decorator_list[0]
    options = {}
    if isinstance(decorator, ast.Call):
        if decorator.args or any(k.arg != "opt_level" for k in decorator.keywords):
            raise ValueError("Cold kernel decorator requires existing explicit compile options")
        options = {k.arg: value(k.value) for k in decorator.keywords}
    # Host call locations and owner identities do not create numerical objects.
    # Compiler dependency guards decide whether a shared object can reuse code.
    key = (
        function,
        node.name,
        tuple(sorted((name, describe_element_type(dtype).structural_key()) for name, dtype in annotations.items())),
        tuple(sorted(options.items())),
    )
    kernel = kernels.get(key)
    if kernel is None:
        kernel = kernels[key] = ti.kernel(numerical, **options)
    numerical = kernel._primal.func
    environment._lexical_bindings = MappingProxyType(
        {**environment._lexical_bindings, **{(numerical, name): binding for name, binding in bindings.items()}}
    )
    selected = dict(node=node, identity=recipe["module"] + "." + factory.name + "." + node.name, options=options)
    return kernel, bindings, selected


def _leaves(value):
    if type(value) is tuple:
        for item in value:
            yield from _leaves(item)
    else:
        yield value
