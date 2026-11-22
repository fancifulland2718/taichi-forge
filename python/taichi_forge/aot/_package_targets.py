"""Select explicit construction and static Kernel branches for one backend."""

import ast
import copy

from taichi_forge_runtime._dimensions import PORTABLE_TARGETS

from ._package_types import qualified_kind

BACKEND_NAMES = frozenset(("cpu", "x64", "arm64", "cuda", "vulkan"))


def backend_constant(node, kind):
    if not isinstance(node, (ast.Name, ast.Attribute)):
        return None
    role = qualified_kind(node, kind)
    if role in BACKEND_NAMES:
        import taichi_forge as ti

        return getattr(ti, role).name
    return None


def backend_query(node, kind):
    if isinstance(node, ast.Attribute) and qualified_kind(node, kind) == "cfg.arch":
        return True
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "arch"
        and isinstance(node.value, ast.Call)
        and not node.value.args
        and not node.value.keywords
        and qualified_kind(node.value, kind) == "lang.impl.current_cfg"
    )


def target_condition(node, value):
    """Evaluate only the explicit target predicates shared with AST lowering."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        result = target_condition(node.operand, value)
        return None if result is None else not result
    if isinstance(node, ast.BoolOp):
        items = [target_condition(item, value) for item in node.values]
        if any(item is None for item in items):
            return None
        return all(items) if isinstance(node.op, ast.And) else any(items)
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    left = value(node.left)
    right = node.comparators[0]
    op = node.ops[0]
    if left is None:
        return None
    if isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, (ast.Tuple, ast.List)):
        choices = [value(item) for item in right.elts]
        if any(item is None for item in choices):
            return None
        result = left in choices
        return result if isinstance(op, ast.In) else not result
    right = value(right)
    if right is None or not isinstance(op, (ast.Eq, ast.NotEq)):
        return None
    return (left == right) if isinstance(op, ast.Eq) else (left != right)


def prepare_condition(node, kind):
    """Describe pure device queries without consulting the producer's device."""
    outcomes = [target_condition(node, lambda part: target if backend_query(part, kind)
                                 else backend_constant(part, kind)) for target in PORTABLE_TARGETS]
    if None not in outcomes:
        return ("target_mask", sum(1 << i for i, result in enumerate(outcomes) if result))
    if isinstance(node, ast.BoolOp):
        items = [prepare_condition(part, kind) for part in node.values]
        if all(items):
            return ("and" if isinstance(node.op, ast.And) else "or", *items)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        item = prepare_condition(node.operand, kind)
        return ("not", item) if item else None
    if not isinstance(node, ast.Call) or node.keywords:
        return None
    if (len(node.args) == 2 and kind(node.func) == "is_extension_supported"
            and backend_query(node.args[0], kind)):
        arg = node.args[1]
        if isinstance(arg, ast.Attribute) and kind(arg.value) == "extension":
            return ("extension", arg.attr)
    if len(node.args) != 1 or not isinstance(node.func, ast.Attribute) or node.func.attr != "get":
        return None
    caps = node.func.value
    if not (isinstance(caps, ast.Call) and not caps.args and not caps.keywords
            and isinstance(caps.func, ast.Attribute) and caps.func.attr == "get_device_caps"):
        return None
    program = caps.func.value
    if not (isinstance(program, ast.Attribute) and program.attr == "prog"
            and isinstance(program.value, ast.Call) and not program.value.args and not program.value.keywords
            and kind(program.value.func) == "get_runtime"):
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Attribute) and kind(arg.value) == "DeviceCapability":
        return ("capability", arg.attr)
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return ("capability", arg.value)
    return None


def resource_views(function, module, kind, fail, namespace):
    """Yield only distinct declaration paths. Device kernels are never visited here."""
    guards = [n._package_target_mask for n in ast.walk(function) if hasattr(n, "_package_target_mask")]
    if not guards:
        return
    # Normalize the retained Host constructor before copying it. Its literal
    # declaration loops must have the same binding sites as each device view.
    from ._package_resources import normalize_resource_declarations

    normalize_resource_declarations(module, function, kind, fail, namespace)
    paths = {}
    for i, target in enumerate(PORTABLE_TARGETS):
        key = tuple(bool(mask & (1 << i)) for mask in guards)
        paths.setdefault(key, []).append(target)

    for targets in paths.values():
        bit = 1 << PORTABLE_TARGETS.index(targets[0])

        class Select(ast.NodeTransformer):
            def visit_If(self, node):
                mask = getattr(node, "_package_target_mask", None)
                if mask is None:
                    return self.generic_visit(node)
                branch = ast.Module(body=node.body if mask & bit else node.orelse, type_ignores=[])
                return self.generic_visit(branch).body

        yield targets, Select().visit(copy.deepcopy(function))


def _same_declaration(left, right):
    """Device views contain independent copies of the same constructor syntax."""
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, ast.AST):
        return ast.dump(left, include_attributes=True) == ast.dump(right, include_attributes=True)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_declaration(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same_declaration(a, b) for a, b in zip(left, right))
    return left == right


def common_resource_schema(schemas, fail):
    """Give the producer a common typed slot; preserve device-specific allocation facts for prepare."""
    result = {}
    names = dict.fromkeys(name for members in schemas.values() for name in members)
    for name in names:
        views = {target: members.get(name) for target, members in schemas.items()}
        present = [item for item in views.values() if item is not None]
        first = present[0]
        if all(_same_declaration(item, first) for item in views.values()):
            result[name] = first
            continue
        kinds = {item["kind"] for item in present}
        item = first.copy()
        if kinds == {"dimension"} and (len(present) == len(views) or not first.get("expression")) and all(
            s.get("expression") == first.get("expression") for s in present
        ):
            # A dim() call in an extra allocation is another reference to the
            # same model value. Its source location is not a new binding.
            result[name] = item
            continue
        if kinds == {"static"}:
            if len({s.get("kernel_only") for s in present}) != 1:
                fail(f"Inconsistent static resource role for {name!r}")
            item["choices"] = []
            for declaration in present:
                for choice in declaration["choices"]:
                    if choice not in item["choices"]:
                        item["choices"].append(choice)
        elif kinds == {"constant", "dimension"}:
            item = next(s for s in present if s["kind"] == "dimension").copy()
            if any(s.get("value", ["unknown"])[0] != "int" for s in present if s["kind"] == "constant"):
                fail(f"Non-integer device dimension for {name!r}")
            item.pop("expression", None)
        elif len(kinds) != 1:
            fail(f"Inconsistent device resource types for {name!r}: {sorted(kinds)}")
        elif kinds == {"dense_field"}:
            def field_abi(decl):
                syntax = decl.get("type_node")
                return (decl.get("element"), ast.dump(syntax) if isinstance(syntax, ast.AST) else syntax,
                        len(decl.get("shape", ())))
            if any(field_abi(s) != field_abi(first) for s in present):
                fail(f"Inconsistent device field ABI for {name!r}")
            # A varying extent must remain a dimension in the common capture.
            item = max(present, key=lambda decl: sum(isinstance(axis, str) for axis in decl["shape"])).copy()
        item["_device_declarations"] = views
        result[name] = item
    return result


def device_resource(item, target):
    """Resolve the original declaration while retaining subsequent owner-graph inference."""
    views = item.get("_device_declarations")
    if views is None:
        return item
    selected = views[target]
    if selected is None:
        return None
    result = item.copy()
    missing = object()
    for key in set().union(*(record for record in views.values() if record is not None)):
        value = selected.get(key, missing)
        if all(record is None or _same_declaration(record.get(key, missing), value) for record in views.values()):
            continue
        if value is missing:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def resource_schema_for_mask(schema, mask):
    targets = [target for i, target in enumerate(PORTABLE_TARGETS) if mask & (1 << i)]
    local = {}
    for name, item in schema.items():
        if "_device_declarations" not in item:
            local[name] = item
            continue
        first, *rest = (device_resource(item, target) for target in targets)
        if any(not _same_declaration(view, first) for view in rest):
            local[name] = item
        elif first is not None:
            local[name] = first
    return local


def select_target_branches(tree, kind, target, module, fail, *, include_host=False):
    selections = []

    def key(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            return "self." + node.attr
        return None

    for scope in tree.body:
        members = scope.body if isinstance(scope, ast.ClassDef) else (scope,)
        for function in members:
            if not isinstance(function, ast.FunctionDef):
                continue
            decorators = [kind(d) for d in function.decorator_list]
            numerical = decorators == ["kernel"]
            # Ordinary Host helpers can own backend-only resources too (e.g.
            # resize and cold native-pipeline construction). Device functions
            # still follow the numerical compiler's own specialization rules.
            if not numerical and any(role in ("func", "real_func", "pyfunc") for role in decorators):
                continue
            if not numerical and not include_host and not (isinstance(scope, ast.ClassDef) and (
                function.name == "__init__"
                or getattr(function, "_package_construction_method", False)
                or getattr(function, "_package_preparation_method", False)
            )):
                continue
            writes = {}
            locals_ = {n.arg for n in ast.walk(function.args) if isinstance(n, ast.arg)}
            # Both checks read the same syntax. Branch rewriting starts only
            # after them; never reuse this snapshot on the rewritten body.
            nodes = tuple(ast.walk(function))
            for node in nodes:
                name = key(node)
                if name is not None and isinstance(getattr(node, "ctx", None), ast.Store):
                    writes[name] = writes.get(name, 0) + 1
                    if isinstance(node, ast.Name):
                        locals_.add(name)
            for node in nodes:
                if backend_query(node, kind) or backend_constant(node, kind) is not None:
                    root = node
                    while isinstance(root, (ast.Call, ast.Attribute)):
                        root = root.func if isinstance(root, ast.Call) else root.value
                    if isinstance(root, ast.Name) and root.id in locals_:
                        fail(module, node, "Backend selection cannot use a shadowed frontend name")
            del nodes
            aliases = set()

            def value(node, arch=target):
                constant = backend_constant(node, kind)
                if constant is not None:
                    return constant
                if backend_query(node, kind) or (not numerical and key(node) in aliases):
                    return arch
                return None

            def condition(node):
                return target_condition(node, value)

            def statements(body):
                output = []
                for node in body:
                    if isinstance(node, (ast.Assign, ast.AnnAssign)):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        if len(targets) == 1 and backend_query(node.value, kind):
                            name = key(targets[0])
                            if name is not None and writes.get(name) == 1:
                                aliases.add(name)
                    test = node.test if isinstance(node, ast.If) else None
                    if numerical:
                        # Numerical admission follows an explicit compile-time
                        # target guard. Ordinary runtime conditions and owner
                        # aliases cannot establish backend availability.
                        test = (test.args[0] if isinstance(test, ast.Call)
                                and qualified_kind(test, kind) == "static"
                                and len(test.args) == 1 and not test.keywords else None)
                    selected = condition(test) if test is not None and target is not None else None
                    if target is None and test is not None:
                        matches = [target_condition(test, lambda expr: value(expr, arch)) for arch in PORTABLE_TARGETS]
                        if None not in matches:
                            node._package_target_mask = sum(1 << i for i, result in enumerate(matches) if result)
                    if selected is None:
                        # Resource contexts and model conditions do not hide
                        # backend-only decisions. Arm-local aliases must not
                        # escape into a sibling or continuation.
                        if isinstance(node, (ast.With, ast.If)):
                            outer_aliases = set(aliases)
                            node.body = statements(node.body)
                            aliases.clear()
                            aliases.update(outer_aliases)
                            if isinstance(node, ast.If):
                                node.orelse = statements(node.orelse)
                                aliases.clear()
                                aliases.update(outer_aliases)
                        elif isinstance(node, ast.Try):
                            outer_aliases = set(aliases)
                            for owner, attribute in [(node, "body"), (node, "orelse"),
                                                     (node, "finalbody"),
                                                     *((handler, "body") for handler in node.handlers)]:
                                aliases.clear()
                                aliases.update(outer_aliases)
                                setattr(owner, attribute, statements(getattr(owner, attribute)))
                            aliases.clear()
                            aliases.update(outer_aliases)
                        output.append(node)
                        continue
                    location = f"{module}:{node.lineno}"
                    selections.append(dict(owner=scope.name if isinstance(scope, ast.ClassDef) else None,
                                           line=node.lineno, taken=selected))
                    if numerical:
                        # The original Kernel keeps the same ti.static guard
                        # for JIT compilation. The selected syntax lets normal
                        # declaration analysis recognize a terminal NoReturn
                        # interface before trying to infer its numerical ABI.
                        output.extend(statements(node.body if selected else node.orelse))
                        continue
                    if function.name == "__init__" and isinstance(scope, ast.ClassDef):
                        # A private kernel factory can belong exclusively to the
                        # other target. Keep just its unshadowed name, not its AST.
                        discarded = node.orelse if selected else node.body
                        hidden = getattr(tree, "_package_inactive_constructor_calls", set())
                        for stmt in discarded:
                            hidden.update(call.func.id for call in ast.walk(stmt)
                                          if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                                          and call.func.id.startswith("_") and call.func.id not in locals_)
                        tree._package_inactive_constructor_calls = hidden
                    # Keep the original test as a consumption-time guard. A Host
                    # call/property may have changed an owner alias after the
                    # assignment; choosing that different layout is not legal.
                    guard = ast.Expr(
                        ast.Call(
                            ast.Attribute(
                                ast.Name("_forge_package_runtime", ast.Load()), "check_target_branch", ast.Load()
                            ),
                            [copy.deepcopy(node.test), ast.Constant(selected), ast.Constant(location)],
                            [],
                        )
                    )
                    output.append(ast.copy_location(guard, node))
                    output.extend(statements(node.body if selected else node.orelse))
                return output

            function.body = statements(function.body)
    return selections
