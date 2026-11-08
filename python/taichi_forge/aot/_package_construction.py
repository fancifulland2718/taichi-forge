"""Find ordinary construction methods without executing or flattening Host code."""

import ast
import copy
import hashlib
import inspect

from ._package_factories import _source_value, prove_factory
from ._package_contexts import prove_context
from ._package_types import qualified_kind


def ordinary_owner_lookup(owner):
    from taichi_forge.lang.kernel_impl import data_oriented

    def ordinary(base):
        lookup = vars(base).get("__getattribute__", object.__getattribute__)
        return lookup is object.__getattribute__ or (
            inspect.isfunction(lookup) and lookup.__code__ in data_oriented.__code__.co_consts
        )

    return (
        inspect.isclass(owner)
        and all(ordinary(base) for base in owner.__mro__)
        and (inspect.getattr_static(owner, "__setattr__") is object.__setattr__)
    )


def construction_functions(node):
    functions = [
        child
        for child in node.body
        if isinstance(child, ast.FunctionDef)
        and (
            child.name == "__init__"
            or getattr(child, "_package_construction_method", False)
            or getattr(child, "_package_preparation_method", False)
        )
    ]
    return sorted(
        functions,
        key=lambda function: (function.name != "__init__", getattr(function, "_package_construction_order", 0)),
    )


def materialize_inherited_construction(tree, namespace, module, fail, kind):
    """Keep ordinary same-module construction hooks in their concrete context.

    A subclass may add allocation hooks or supply fixed type members. Existing
    allocation methods still cannot be replaced: that would silently erase the
    base resource contract. Only Host methods are copied, never numerical code.
    """
    nodes = {
        namespace[n.name]: n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and inspect.isclass(namespace.get(n.name))
    }
    methods = {(cls, n.name): n for cls, node in nodes.items() for n in node.body if isinstance(n, ast.FunctionDef)}
    facts = []
    for cls, node in nodes.items():
        if "__init__" in vars(cls) or not getattr(cls, "_data_oriented", False):
            continue
        base = next((b for b in cls.__mro__[1:] if "__init__" in vars(b)), object)
        if (base, "__init__") not in methods:
            continue
        selected, active = {}, set()
        overridden = False

        def collect(name):
            nonlocal overridden
            origin = next((b for b in cls.__mro__ if name in vars(b)), None)
            method = methods.get((origin, name))
            if method is None or method.decorator_list:
                return
            if name in active:
                fail(module, method, "Recursive inherited construction requires explicit declarations")
            if name in selected:
                return
            active.add(name)
            inherited = inspect.getattr_static(base, name, None)
            actual = inspect.getattr_static(cls, name, None)
            if name != "__init__" and actual is not inherited:
                original = next((methods.get((b, name)) for b in base.__mro__ if name in vars(b)), None)
                if original is not None and any(
                    isinstance(n, ast.Call)
                    and qualified_kind(n, kind)
                    in ("experimental.ResourceScope", "field", "Vector.field", "Matrix.field", "Struct.field")
                    for n in ast.walk(original)
                ):
                    fail(module, method, "Overridden construction allocation requires its own constructor contract")
                overridden = True
            selected[name] = method
            for call in ast.walk(method):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "self"
                ):
                    collect(call.func.attr)
            active.remove(name)

        collect("__init__")
        if not overridden:
            continue
        if not ordinary_owner_lookup(cls):
            fail(module, node, "Inherited construction requires ordinary owner lookup")
        for name, method in selected.items():
            if (cls, name) in methods:
                continue
            if any(
                isinstance(n, ast.Name)
                and n.id in ("super", "__class__")
                or isinstance(n, ast.Attribute)
                and n.attr.startswith("__")
                and not n.attr.endswith("__")
                for n in ast.walk(method)
            ):
                fail(module, method, "Inherited construction with lexical class state requires further support")
            node.body.append(copy.deepcopy(method))
            facts.append(
                dict(
                    kind="inherited_construction",
                    module=module,
                    definition=f"{module}.{node.name}.{name}",
                    sha256=hashlib.sha256(ast.dump(method).encode()).hexdigest(),
                )
            )
        node._package_inherited_constructor = True
    return facts


def discover_construction_methods(tree, namespace, module, fail, kind):
    facts = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        constructor = methods.get("__init__")
        if constructor is None:
            continue
        selected, active = set(), set()

        def calls(function):
            return [
                n
                for n in ast.walk(function)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "self"
                and n.func.attr in methods
                and not methods[n.func.attr].decorator_list
                and n.func.attr != "__init__"
            ]

        def allocates(function):
            for call in (n for n in ast.walk(function) if isinstance(n, ast.Call)):
                if qualified_kind(call, kind) == "ndarray" and not any(
                    isinstance(statement, (ast.Assign, ast.AnnAssign))
                    and statement.value is call
                    and any(
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        for target in (statement.targets if isinstance(statement, ast.Assign) else [statement.target])
                    )
                    for statement in ast.walk(function)
                ):
                    # Ordinary Host helpers may allocate temporary argument
                    # arrays. The existing Runtime projection handles those;
                    # they are not persistent owner declaration methods.
                    continue
                if qualified_kind(call, kind) in (
                    "experimental.ResourceScope",
                    "field",
                    "ndarray",
                    "Vector.field",
                    "Matrix.field",
                    "Struct.field",
                ):
                    return True
                target = _source_value(call.func, namespace)
                try:
                    normalized = prove_factory(target, call)
                    if normalized is not None and qualified_kind(normalized, kind) == "experimental.ResourceScope":
                        return True
                    if prove_context(target, call) is not None:
                        return True
                except (TypeError, ValueError, OSError):
                    # The normal factory/context analyzer supplies the precise
                    # rejection; an unproved allocation cannot disappear.
                    return True
            return False

        needed = {name for name, function in methods.items() if not function.decorator_list and allocates(function)}
        while True:
            expanded = needed | {
                name
                for name, function in methods.items()
                if not function.decorator_list and any(c.func.attr in needed for c in calls(function))
            }
            if expanded == needed:
                break
            needed = expanded

        def visit(function):
            if function.name in active:
                fail(module, function, "Recursive construction methods require explicit resource declarations")
            active.add(function.name)
            for call in calls(function):
                method = methods[call.func.attr]
                if method.name not in needed:
                    continue
                if not any(isinstance(s, ast.Expr) and s.value is call for s in function.body):
                    fail(module, call, "Construction method calls require an unconditional standalone statement")
                args = method.args
                if (
                    call.args
                    or call.keywords
                    or args.posonlyargs
                    or len(args.args) != 1
                    or args.args[0].arg != "self"
                    or args.kwonlyargs
                    or args.vararg
                    or args.kwarg
                ):
                    fail(module, call, "Construction methods currently require only the owner receiver")
                if method.name in selected:
                    fail(module, call, "Construction resource methods must have one call site")
                if any(
                    isinstance(n, (ast.Return, ast.Yield, ast.YieldFrom, ast.AsyncFunctionDef))
                    for n in ast.walk(method)
                ):
                    fail(module, method, "Construction resource methods cannot return or yield")
                selected.add(method.name)
                method._package_construction_method = True
                method._package_construction_order = len(selected)
                visit(method)
                facts.append(
                    dict(
                        kind="construction_method",
                        module=module,
                        definition=f"{module}.{cls.name}.{method.name}",
                        line=call.lineno,
                        sha256=hashlib.sha256(ast.dump(method).encode()).hexdigest(),
                    )
                )
            active.remove(function.name)

        visit(constructor)
        # A declared resource group may be prepared after construction. Reading
        # its schema must not execute the method or move that allocation into
        # __init__. Only explicit group contexts have an independent lifetime;
        # raw/default-root allocations keep their existing rejection boundary.
        for name, method in methods.items():
            if name == "__init__" or name in selected or method.decorator_list or not allocates(method):
                continue
            contexts = []
            for part in ast.walk(method):
                if not isinstance(part, ast.With) or len(part.items) != 1:
                    continue
                call = part.items[0].context_expr
                if not isinstance(call, ast.Call):
                    continue
                try:
                    recipe = prove_context(_source_value(call.func, namespace), call)
                except (TypeError, ValueError, OSError) as error:
                    fail(module, part, f"Preparation resource context is unproven: {error}")
                if recipe is not None:
                    contexts.append(part)
            typed_inputs = any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "borrow"
                and {keyword.arg for keyword in call.keywords} >= {"external", "dtype", "shape"}
                for call in ast.walk(method)
            )
            # Typed ports bind caller storage without allocating a group. The
            # normal declaration parser below still proves the scope and types.
            if not contexts and not typed_inputs:
                continue
            if (
                not method.args.args
                or method.args.args[0].arg != "self"
                or method.args.posonlyargs
                or any(isinstance(n, (ast.Yield, ast.YieldFrom, ast.AsyncFunctionDef)) for n in ast.walk(method))
            ):
                fail(module, method, "Preparation resource methods require an ordinary synchronous owner receiver")
            if any(
                isinstance(n, ast.Call)
                and qualified_kind(n, kind)
                in (
                    "FieldsBuilder",
                    "field",
                    "Vector.field",
                    "Matrix.field",
                    "Struct.field",
                )
                for n in ast.walk(method)
            ):
                fail(module, method, "Preparation resource allocations require explicit group contexts")
            method._package_preparation_method = True
            method._package_construction_order = len(selected) + method.lineno
            facts.append(
                dict(
                    kind="preparation_method",
                    module=module,
                    definition=f"{module}.{cls.name}.{name}",
                    sha256=hashlib.sha256(ast.dump(method).encode()).hexdigest(),
                )
            )
            selected.add(name)
        for name in selected:
            for node in ast.walk(cls):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                    and node.attr == name
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                ):
                    fail(module, node, "Construction method cannot be replaced on the owner")
            owner = namespace.get(cls.name)
            if inspect.isclass(owner) and (not ordinary_owner_lookup(owner) or "__getattribute__" in methods):
                fail(module, methods[name], "Construction methods require ordinary owner attribute lookup")
    return facts
