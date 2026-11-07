"""Project existing resource declarations, never execute application constructors."""

import ast
import copy
import hashlib

from taichi_forge_runtime._dimensions import Extent, extent_dependencies, evaluate_extent
from ._package_types import qualified_kind, project_tensor_axes, expand_local_type
from ._package_fields import field_type_shape


def _normalize_field_loops(module, constructor, kind, fail, namespace):
    """Expand literal member declarations, never iterate model data or numerical code."""
    scope_names = set()
    for node in ast.walk(constructor):
        if getattr(node, "_package_resource_context", None) is not None:
            scope_names.add(node.items[0].optional_vars.id)
        if (
            isinstance(node, ast.Assign)
            and qualified_kind(node.value, kind) == "experimental.ResourceScope"
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            scope_names.add(node.targets[0].id)
    shadowed = namespace.get("setattr", setattr) is not setattr or any(
        isinstance(n, ast.Name)
        and n.id == "setattr"
        and not isinstance(n.ctx, ast.Load)
        or isinstance(n, ast.arg)
        and n.arg == "setattr"
        or isinstance(n, ast.alias)
        and (n.asname or n.name.split(".")[0]) == "setattr"
        for n in ast.walk(constructor)
    )

    class Declarations(ast.NodeTransformer):
        def visit_For(self, node):
            if (
                len(node.body) != 1
                or not isinstance(node.body[0], ast.Expr)
                or not isinstance(node.body[0].value, ast.Call)
            ):
                return self.generic_visit(node)
            call = node.body[0].value
            if not (
                isinstance(call.func, ast.Name)
                and call.func.id == "setattr"
                and len(call.args) == 3
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id in scope_names
            ):
                return self.generic_visit(node)
            receiver, name, allocation = call.args
            if (
                shadowed
                or call.keywords
                or node.orelse
                or not isinstance(node.target, ast.Name)
                or not isinstance(name, ast.Name)
                or name.id != node.target.id
                or not isinstance(allocation, ast.Call)
                or not isinstance(allocation.func, ast.Attribute)
                or not isinstance(allocation.func.value, ast.Name)
                or allocation.func.value.id != receiver.id
                or allocation.func.attr not in ("field", "vector_field", "matrix_field", "struct_field")
                or any(isinstance(n, ast.Name) and n.id == name.id for n in ast.walk(allocation))
            ):
                fail(module, node, "Field-name loop requires an unshadowed literal member declaration")
            if not isinstance(node.iter, (ast.Tuple, ast.List)) or not node.iter.elts:
                fail(module, node.iter, "Field-name loop requires an explicit nonempty literal tuple/list")
            names = []
            for item in node.iter.elts:
                if not (
                    isinstance(item, ast.Constant)
                    and type(item.value) is str
                    and item.value.isidentifier()
                    and not item.value.startswith("__")
                    and item.value not in names
                ):
                    fail(module, item, "Field-name loop requires distinct non-mangled literal member names")
                names.append(item.value)
            result = []
            for member in names:
                # Retain the ordinary loop variable's last value and original
                # allocation order. Only fixed declarations expand, not kernels.
                result.append(ast.copy_location(ast.Assign([copy.deepcopy(node.target)], ast.Constant(member)), node))
                result.append(
                    ast.copy_location(
                        ast.Assign(
                            [ast.Attribute(copy.deepcopy(receiver), member, ast.Store())], copy.deepcopy(allocation)
                        ),
                        node.body[0],
                    )
                )
            return result

    Declarations().visit(constructor)


def scope_declarations(module, constructor, kind, fail, namespace):
    if constructor is None:
        return {}
    _normalize_field_loops(module, constructor, kind, fail, namespace)
    scopes, declarations, allowed = {}, {}, set()
    scope_groups, builders = {}, {}
    contexts, context_bindings, context_stores = {}, {}, {}
    for statement in ast.walk(constructor):
        recipe = getattr(statement, "_package_resource_context", None)
        if recipe is None:
            continue
        target = statement.items[0].optional_vars
        scope = (target.id, statement.lineno, statement.col_offset)
        call = copy.deepcopy(recipe["call"])
        # The declared group belongs to this lexical scope. Its Host generator
        # still owns finalization, exception handling and external bookkeeping.
        call.lineno, call.col_offset = statement.lineno, statement.col_offset
        allocation = ast.copy_location(ast.Assign([statement.items[0].optional_vars], call), statement)
        group = f"context_{statement.lineno}_{statement.col_offset}"
        scopes[scope], scope_groups[scope] = allocation, group
        builders[group] = dict(
            allocation=[statement.lineno, statement.col_offset],
            finalization=[statement.end_lineno + 1, statement.col_offset],
            context={k: v for k, v in recipe.items() if k != "call"},
        )
        contexts[id(statement)] = scope
        context_stores.setdefault(target.id, set()).add(id(target))
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and node.id == target.id:
                context_bindings[id(node)] = scope
    for node in ast.walk(constructor):
        if (
            isinstance(node, ast.Name)
            and node.id in context_stores
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and id(node) not in context_stores[node.id]
            or isinstance(node, ast.arg)
            and node.arg in context_stores
        ):
            fail(module, node, "Package ResourceScope aliases may only be rebound by distinct resource contexts")
    for statement in ast.walk(constructor):
        if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if qualified_kind(call, kind) != "experimental.ResourceScope":
            continue
        if not (
            statement in constructor.body
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "self"
        ):
            fail(module, statement, "Package ResourceScope requires one unconditional local scope for self")
        keywords = {item.arg: item.value for item in call.keywords}
        if len(keywords) != len(call.keywords) or set(keywords) - {"builder", "initialization"}:
            fail(module, call, "ResourceScope requires explicit builder/initialization arguments")
        policy = keywords.get("initialization", ast.Constant("consumer_explicit"))
        if not isinstance(policy, ast.Constant) or policy.value not in ("consumer_explicit", "zero"):
            fail(module, policy, "ResourceScope initialization requires its zero/consumer_explicit policy")
        builder = keywords.get("builder")
        if builder is not None:
            if not isinstance(builder, ast.Name):
                fail(module, builder, "ResourceScope builder requires a single local FieldsBuilder")
            allocations = [
                n
                for n in constructor.body
                if isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == builder.id
                and isinstance(n.value, ast.Call)
                and qualified_kind(n.value, kind) == "FieldsBuilder"
            ]
            if len(allocations) != 1 or allocations[0].value.args or allocations[0].value.keywords:
                fail(module, builder, "ResourceScope builder requires one unconditional FieldsBuilder allocation")
            allocation = allocations[0]
            uses = [n for n in ast.walk(constructor) if isinstance(n, ast.Name) and n.id == builder.id]
            finalizers = [
                n
                for n in constructor.body
                if isinstance(n, (ast.Expr, ast.Assign))
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute)
                and isinstance(n.value.func.value, ast.Name)
                and n.value.func.value.id == builder.id
                and n.value.func.attr == "finalize"
                and not n.value.args
                and not n.value.keywords
            ]
            if len(finalizers) != 1 or not allocation.lineno < statement.lineno < finalizers[0].lineno:
                fail(module, call, "Declared FieldsBuilder must finalize once after its resource scopes")
            finalize = finalizers[0].value
            accepted = {id(allocation.targets[0]), id(finalize.func.value)}
            accepted.update(
                id(k.value)
                for n in ast.walk(constructor)
                if isinstance(n, ast.Call) and qualified_kind(n, kind) == "experimental.ResourceScope"
                for k in n.keywords
                if k.arg == "builder" and isinstance(k.value, ast.Name) and k.value.id == builder.id
            )
            if any(id(n) not in accepted for n in uses):
                fail(
                    module,
                    call,
                    "Explicit ResourceScope builders/initialization require further package support: "
                    "raw SNode construction or escaping/reassigned builders",
                )
            builders["builder_" + builder.id] = dict(
                allocation=[allocation.value.lineno, allocation.value.col_offset],
                finalization=[finalize.lineno, finalize.col_offset],
            )
        scope = statement.targets[0].id
        if (
            sum(
                isinstance(n, ast.Name) and n.id == scope and isinstance(n.ctx, ast.Store)
                for n in ast.walk(constructor)
            )
            != 1
        ):
            fail(module, statement, "Package ResourceScope aliases must not be reassigned")
        scopes[scope] = statement
        scope_groups[scope] = "builder_" + builder.id if builder is not None else "fields"
    if not scopes:
        return {}
    from ._component import _capture_id

    scope_paths = {key: () for key in scopes}
    record_calls = set()
    # Records describe existing plain values, including frozen dataclasses.
    # Their constructors are never evaluated to discover numerical inputs.
    for statement in constructor.body:
        if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if not (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in scopes
            and call.func.attr == "record"
        ):
            continue
        parent = call.func.value.id
        if not (
            len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id not in scopes
            and len(call.args) == 2
            and (
                not call.keywords
                or len(call.keywords) == 1
                and call.keywords[0].arg == "optional"
                and isinstance(call.keywords[0].value, ast.Constant)
                and call.keywords[0].value.value is False
            )
            and isinstance(call.args[0], ast.Constant)
            and type(call.args[0].value) is str
            and call.args[0].value.isidentifier()
            and not call.args[0].value.startswith("__")
            and (statement.lineno, statement.col_offset) > (scopes[parent].lineno, scopes[parent].col_offset)
            and sum(
                isinstance(n, ast.Name)
                and n.id == statement.targets[0].id
                and isinstance(n.ctx, ast.Store)
                or isinstance(n, ast.arg)
                and n.arg == statement.targets[0].id
                for n in ast.walk(constructor)
            )
            == 1
        ):
            fail(module, statement, "Record declaration requires an unconditional unique alias and literal member name")
        alias = statement.targets[0].id
        path = (*scope_paths[parent], call.args[0].value)
        identity = _capture_id(path)
        if identity in declarations:
            fail(module, statement, "Record declaration cannot replace an existing member")
        declarations[identity] = dict(
            kind="record",
            path=list(path),
            scope_allocation=[scopes[parent].value.lineno, scopes[parent].value.col_offset],
        )
        scopes[alias], scope_paths[alias] = statement, path
        scope_groups[alias] = scope_groups[parent]
        record_calls.add(id(call))
        allowed.add(id(call.func.value))
    scope_names = {key[0] if isinstance(key, tuple) else key for key in scopes}

    def scope_binding(node):
        # Context variables retain Python's original names in emitted Host code.
        # Only declaration lookup distinguishes their lexical allocation sites.
        return context_bindings.get(id(node), node.id) if isinstance(node, ast.Name) else None

    def operation(node):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and scope_binding(node.func.value) in scopes
        ):
            return scope_binding(node.func.value), node.func.attr
        return None, None

    def add(name, spec, node):
        existing = declarations.get(name)
        if existing is not None and existing["kind"] == spec["kind"] == "dimension":
            # Two explicit groups may use the same owner count. Provenance of
            # its declaration does not create a different dimension value.
            if spec.get("expression"):
                if existing.get("expression") not in (None, spec["expression"]):
                    fail(module, node, f"Conflicting derived dimension {name!r}")
                existing["expression"] = spec["expression"]
            return
        if existing is not None and existing != spec:
            fail(module, node, f"Conflicting or repeated ResourceScope role for {name!r}")
        declarations[name] = spec

    def origin(scope):
        call = scopes[scope].value
        return [call.lineno, call.col_offset]

    def member(scope, name):
        return _capture_id((*scope_paths[scope], name))

    def member_spec(scope, name, **spec):
        return {**spec, **({"path": list((*scope_paths[scope], name))} if scope_paths[scope] else {})}

    source_aliases = {}

    def source_expression(node):
        class Forwarded(ast.NodeTransformer):
            def visit_Name(self, value):
                forwarded = getattr(value, "_package_forwarded_argument", None)
                return self.visit(copy.deepcopy(forwarded)) if forwarded is not None else value

            def visit_Attribute(self, value):
                if isinstance(value.value, ast.Name) and value.value.id == "self":
                    alias = source_aliases.get(value.attr)
                    if alias is not None and alias.lineno < getattr(value, "lineno", 0):
                        return copy.deepcopy(alias.value)
                return self.generic_visit(value)

        return ast.dump(Forwarded().visit(copy.deepcopy(node)), include_attributes=False)

    def dimension_source(scope, name):
        base = scopes[scope].value.args[1] if scope_paths[scope] else ast.Name("self", ast.Load())
        return source_expression(ast.Attribute(copy.deepcopy(base), name, ast.Load()))

    model_candidates, used_model_dimensions = {}, set()

    aliases, alias_conditions, placements = {}, {}, {}
    local_types = {}
    deferred_aliases, resolving_aliases, alias_checks = {}, set(), {}
    allow_member_dimensions = True
    extent_namespaces = {}
    constraint_declarations = {}
    scope_writes = []
    current_conditions = {}
    stores = {n.arg: 1 for n in ast.walk(constructor.args) if isinstance(n, ast.arg)}
    parameter_names = frozenset(stores)
    member_stores = {}
    for node in ast.walk(constructor):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            stores[node.id] = stores.get(node.id, 0) + 1
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id in ("self", *scope_names)
        ):
            member_stores[node.attr] = member_stores.get(node.attr, 0) + 1

    # A record may be forwarded as self.layout while its Field type uses the
    # original local layout. Compare this one-time binding as provenance only;
    # preserve both Host reads and the normal allocation/record validation.
    deleted = {n.id for n in ast.walk(constructor) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Del)}
    deleted_members = {
        n.attr for n in ast.walk(constructor) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Del)
    }
    record_members = {path[0] for path in scope_paths.values() if len(path) == 1}
    for statement in constructor.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr in record_members
                and member_stores[target.attr] == 1
                and target.attr not in deleted_members
                and isinstance(value, ast.Name)
                and stores.get(value.id) == 1
                and value.id not in deleted
            ):
                source_aliases[target.attr] = statement

    def dimension(node, bindings=None):
        """Only the already-declared dimension algebra; no model-value evaluation."""
        bindings = bindings or {}
        if isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 2**31 - 1:
            return Extent(("const", node.value))
        if isinstance(node, ast.Name):
            if node.id in bindings:
                return bindings[node.id]
            if any(
                current_conditions.get(key) is not value for key, value in alias_conditions.get(node.id, {}).items()
            ):
                fail(module, node, "Dimension alias must remain within its declaration branch")
            if node.id in deferred_aliases:
                if node.id in resolving_aliases:
                    return None
                resolving_aliases.add(node.id)
                try:
                    value = dimension(deferred_aliases[node.id], bindings)
                finally:
                    resolving_aliases.remove(node.id)
                if isinstance(value, Extent):
                    alias_checks[node.id] = value.expression
                return value
            return aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            if not allow_member_dimensions:
                return None
            for scope in scopes:
                name = member(scope, node.attr)
                if declarations.get(name, {}).get("kind") == "dimension" and source_expression(
                    node
                ) == dimension_source(scope, node.attr):
                    # Reading an ordinary integer member is not exporting a
                    # scope's Extent object. Its qualified declaration still
                    # determines the layout; explicit dim() aliases retain
                    # their namespace checks below.
                    return Extent(("dim", name))
        if isinstance(node, (ast.Tuple, ast.List)):
            values = tuple(dimension(item, bindings) for item in node.elts)
            return values if all(isinstance(v, Extent) for v in values) else None
        scope, method = operation(node)
        if method in ("derived_dim", "power_of_two_dim"):
            if (
                len(node.args) != 2
                or node.keywords
                or not isinstance(node.args[0], ast.Constant)
                or type(node.args[0].value) is not str
                or not node.args[0].value.isidentifier()
                or current_conditions
                or node.lineno < scopes[scope].lineno
            ):
                fail(module, node, "Derived dimensions require an unconditional literal name and declared expression")
            local_name = node.args[0].value
            name = member(scope, local_name)
            value = dimension(node.args[1], bindings)
            if not isinstance(value, Extent) or not extent_dependencies(value.expression):
                fail(module, node, "Derived dimension requires the existing declared dimension algebra")
            expression = value.expression
            namespace_check(scope, value, node)
            if method == "power_of_two_dim":
                expression = ("ceil_pow2", expression)

            # Dependencies are declared in source order. Check the new edge
            # against existing relations; do not infer an independent count.
            def check(key, visiting):
                if key in visiting:
                    fail(module, node, f"Cyclic derived dimension {key!r}")
                declaration = declarations.get(key, {})
                if declaration.get("kind") != "dimension":
                    fail(module, node, f"Undeclared dimension {key!r}")
                for child in extent_dependencies(declaration.get("expression", ("const", 0))):
                    check(child, (*visiting, key))

            for dependency in extent_dependencies(expression):
                check(dependency, (name,))
            allowed.add(id(node.func.value))
            add(
                name,
                member_spec(
                    scope,
                    local_name,
                    kind="dimension",
                    dtype="i32",
                    expression=expression,
                    scope_allocation=origin(scope),
                ),
                node,
            )
            return scoped_extent(scope, name)
        if method in ("dim", "parent_dim"):
            name = (
                (
                    node.args[0].value
                    if isinstance(node.args[0], ast.Constant)
                    else bindings.get(node.args[0].id) if isinstance(node.args[0], ast.Name) else None
                )
                if len(node.args) == 1
                else None
            )
            if len(node.args) != 1 or node.keywords or not isinstance(name, str) or not name.isidentifier():
                fail(module, node, "ResourceScope.dim requires a literal owner attribute name")
            if node.lineno < scopes[scope].lineno:
                fail(module, node, "Dimension declaration precedes its resource scope")
            allowed.add(id(node.func.value))
            path = scope_paths[scope]
            if method == "parent_dim":
                if not path:
                    fail(module, node, "parent_dim requires a declared record scope")
                path = path[:-1]
            identity = _capture_id((*path, name))
            add(
                identity,
                dict(
                    kind="dimension",
                    dtype="i32",
                    scope_allocation=origin(scope),
                    **({"path": list((*path, name))} if path else {}),
                ),
                node,
            )
            return scoped_extent(scope, identity)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "tuple"
            and node.func.id not in stores
            and namespace.get("tuple", tuple) is tuple
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.GeneratorExp)
        ):
            generator = node.args[0]
            if len(generator.generators) != 1:
                return None
            loop = generator.generators[0]
            if (
                not isinstance(loop.target, ast.Name)
                or loop.ifs
                or loop.is_async
                or not isinstance(loop.iter, (ast.Tuple, ast.List))
                or not all(isinstance(item, ast.Constant) and type(item.value) is str for item in loop.iter.elts)
            ):
                return None
            values = tuple(
                dimension(generator.elt, {**bindings, loop.target.id: item.value}) for item in loop.iter.elts
            )
            return values if all(isinstance(item, Extent) for item in values) else None
        if isinstance(node, ast.BinOp):
            methods = {ast.Add: "__add__", ast.Sub: "__sub__", ast.Mult: "__mul__", ast.FloorDiv: "__floordiv__"}
            method = methods.get(type(node.op))
            left, right = dimension(node.left, bindings), dimension(node.right, bindings)
            if method is not None and isinstance(left, Extent) and isinstance(right, Extent):
                result = getattr(left, method)(right)
                extent_objects.append(result)
                extent_namespaces[id(result)] = extent_namespaces.get(id(left), set()) | extent_namespaces.get(
                    id(right), set()
                )
                return result
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("maximum", "minimum")
            and len(node.args) == 1
            and not node.keywords
        ):
            left, right = dimension(node.func.value, bindings), dimension(node.args[0], bindings)
            if isinstance(left, Extent) and isinstance(right, Extent):
                result = getattr(left, node.func.attr)(right)
                extent_objects.append(result)
                extent_namespaces[id(result)] = extent_namespaces.get(id(left), set()) | extent_namespaces.get(
                    id(right), set()
                )
                return result
        return None

    def scoped_extent(scope, name):
        result = Extent(("dim", name))
        # Hold the Extent too: temporary-expression IDs cannot be reused.
        extent_objects.append(result)
        extent_namespaces[id(result)] = {scope_paths[scope]}
        return result

    extent_objects = []

    def namespace_check(scope, value, node):
        for extent in value if isinstance(value, tuple) else (value,):
            if extent_namespaces.get(id(extent), set()) - {scope_paths[scope]}:
                fail(
                    module,
                    node,
                    "Dimension expressions cannot escape their ResourceScope record namespace; use parent_dim",
                )

    parameters = (*constructor.args.posonlyargs, *constructor.args.args, *constructor.args.kwonlyargs)
    defaults = (
        dict(
            zip(
                [arg.arg for arg in (*constructor.args.posonlyargs, *constructor.args.args)][
                    -len(constructor.args.defaults) :
                ],
                constructor.args.defaults,
            )
        )
        if constructor.args.defaults
        else {}
    )
    defaults.update(zip((arg.arg for arg in constructor.args.kwonlyargs), constructor.args.kw_defaults))
    boolean_parameters = {
        arg.arg
        for arg in parameters
        if stores.get(arg.arg) == 1
        and (
            isinstance(defaults.get(arg.arg), ast.Constant)
            and type(defaults[arg.arg].value) is bool
            or isinstance(arg.annotation, ast.Name)
            and arg.annotation.id == "bool"
            and namespace.get("bool", bool) is bool
        )
    }
    optional_records = set()
    optional_members = set()

    def absent_record(call):
        if not (
            operation(call)[1] == "record"
            and len(call.args) == 2
            and isinstance(call.args[0], ast.Constant)
            and type(call.args[0].value) is str
            and call.args[0].value.isidentifier()
            and isinstance(call.args[1], ast.Constant)
            and call.args[1].value is None
            and len(call.keywords) == 1
            and call.keywords[0].arg == "optional"
            and isinstance(call.keywords[0].value, ast.Constant)
            and call.keywords[0].value.value is True
        ):
            return None
        return call.args[0].value

    def optional_arm(body):
        fields, records = set(), {}
        for node in body:
            call = node.value if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)) else None
            scope, method = operation(call)
            if method in ("field", "vector_field", "matrix_field"):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target] if isinstance(node, ast.AnnAssign) else []
                )
                if (
                    len(targets) != 1
                    or not isinstance(targets[0], ast.Attribute)
                    or not isinstance(targets[0].value, ast.Name)
                    or not (targets[0].value.id == "self" or scope_binding(targets[0].value) in scopes)
                ):
                    return None
                fields.add(targets[0].attr)
            elif method == "record":
                name = absent_record(call)
                if not isinstance(node, ast.Expr) or name is None or name in records:
                    return None
                records[name] = call
            elif method == "borrow" and isinstance(node, ast.Expr):
                if len(call.args) != 2 or not isinstance(call.args[0], ast.Constant):
                    return None
                fields.add(call.args[0].value)
        return fields, records

    def paired_records(statement):
        arms = [optional_arm(statement.body), optional_arm(statement.orelse)]
        if any(arm is None for arm in arms):
            return None
        (left, absent_left), (right, absent_right) = arms
        if not (left or right) or left != absent_right.keys() or right != absent_left.keys() or left & right:
            return None
        return [*absent_left.values(), *absent_right.values()]

    def presence_parameter(statement, parameter, *, nonnull=False):
        calls = paired_records(statement)
        if calls is None:
            return None
        name = (
            "_forge_package_presence_"
            + hashlib.sha256(f"{module}:{constructor.lineno}:{parameter}".encode()).hexdigest()
        )
        scope, _ = operation(calls[0])
        if name not in declarations:
            add(
                name,
                dict(
                    kind="static",
                    choices=[("bool", False), ("bool", True)],
                    kernel_only=False,
                    presence_parameter=parameter,
                    **({"presence_nonnull": True} if nonnull else {}),
                    scope_allocation=origin(scope),
                ),
                statement,
            )
        optional_records.update(id(call) for call in calls)
        return name

    def declare_static(scope, call, statement):
        from ._component import StaticValue
        from taichi_forge_runtime._static_values import static_key

        if (
            len(call.args) != 1
            or not isinstance(call.args[0], ast.Constant)
            or type(call.args[0].value) is not str
            or not call.args[0].value.isidentifier()
            or statement.lineno < scopes[scope].lineno
        ):
            fail(module, call, "Static declaration requires one literal owner attribute name")
        keywords = {k.arg: k.value for k in call.keywords}
        if (
            len(keywords) != len(call.keywords)
            or "choices" not in keywords
            or set(keywords) - {"choices", "kernel_only"}
        ):
            fail(module, call, "Static declaration requires explicit finite choices")
        try:
            choices = ast.literal_eval(keywords["choices"])
            kernel_only = ast.literal_eval(keywords.get("kernel_only", ast.Constant(False)))
            if not isinstance(choices, (tuple, list)):
                raise ValueError("choices must be a tuple or list")
            declaration = StaticValue(choices=choices, kernel_only=kernel_only)
        except (TypeError, ValueError) as error:
            choices = keywords["choices"]
            name = call.args[0].value
            # An existing singleton type declaration can describe the model's
            # tensor extent. Require identical declared provenance and later
            # prove a tensor-axis use; arbitrary computed static domains fail.
            if not (
                isinstance(choices, (ast.Tuple, ast.List))
                and len(choices.elts) == 1
                and source_expression(choices.elts[0]) == dimension_source(scope, name)
                and isinstance(keywords.get("kernel_only", ast.Constant(False)), ast.Constant)
                and keywords.get("kernel_only", ast.Constant(False)).value is False
            ):
                fail(module, call, f"Static domain must contain distinct immutable literal choices: {error}")
            identity = member(scope, name)
            model_candidates[identity] = call
            allowed.add(id(call.func.value))
            add(
                identity,
                member_spec(scope, name, kind="dimension", dtype="i32", scope_allocation=origin(scope)),
                statement,
            )
            return
        allowed.add(id(call.func.value))
        add(
            member(scope, call.args[0].value),
            member_spec(
                scope,
                call.args[0].value,
                kind="static",
                choices=[static_key(v) for v in declaration.choices],
                kernel_only=kernel_only,
                static_declaration=[statement.lineno, statement.col_offset],
                scope_allocation=origin(scope),
            ),
            statement,
        )

    def statements(body, conditions=None):
        conditions = conditions or {}
        for statement in body:
            if id(statement) in contexts:
                group = scope_groups[contexts[id(statement)]]
                builders[group]["conditions"] = dict(conditions)
                yield from statements(statement.body, conditions)
            elif isinstance(statement, ast.If) and any(operation(n)[0] for n in ast.walk(statement)):
                test, expected = statement.test, True
                if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                    test, expected = test.operand, False
                name = (
                    test.attr
                    if isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name) and test.value.id == "self"
                    else None
                )
                if isinstance(test, ast.Name) and test.id in boolean_parameters:
                    name = presence_parameter(statement, test.id)
                elif (
                    isinstance(test, ast.Compare)
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], (ast.Is, ast.IsNot))
                    and isinstance(test.left, ast.Name)
                    and test.left.id in parameter_names
                    and stores[test.left.id] == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value is None
                ):
                    name = presence_parameter(statement, test.left.id, nonnull=True)
                    expected = isinstance(test.ops[0], ast.IsNot) == expected
                declaration = declarations.get(name, {})
                if not (
                    declaration.get("kind") == "static"
                    and {tuple(v) for v in declaration["choices"]} == {("bool", False), ("bool", True)}
                    and not declaration["kernel_only"]
                    and name not in conditions
                ):
                    fail(
                        module,
                        statement,
                        "Conditional resources require an unconditional declared boolean owner static",
                    )
                calls = paired_records(statement)
                if calls is not None:
                    optional_records.update(id(call) for call in calls)
                yield from statements(statement.body, {**conditions, name: expected})
                yield from statements(statement.orelse, {**conditions, name: not expected})
            else:
                yield statement, conditions

    def unconditional_statements(body):
        for statement in body:
            if id(statement) in contexts:
                yield from unconditional_statements(statement.body)
            else:
                yield statement

    # A declaration describes the legal domain, independent of when Host Python
    # performs its check. Actual attribute values remain checked at allocation.
    for statement in unconditional_statements(constructor.body):
        if isinstance(statement, ast.Expr):
            scope, method = operation(statement.value)
            if method == "static":
                declare_static(scope, statement.value, statement)

    conditional_fields = set()
    for statement, conditions in statements(constructor.body):
        current_conditions = conditions
        call = statement.value if isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign)) else None
        scope, method = operation(call)
        if conditions and method in ("static", "constant", "value"):
            fail(module, statement, "Value/static declarations must remain unconditional")
        if method == "record" and id(call) in record_calls:
            continue
        if method == "record" and isinstance(statement, ast.Expr) and id(call) in optional_records:
            allowed.add(id(call.func.value))
            optional_members.add(absent_record(call))
        elif method == "static" and isinstance(statement, ast.Expr):
            declare_static(scope, call, statement)
        elif method == "constant" and isinstance(statement, ast.Expr):
            if (
                scope_paths[scope]
                or len(call.args) != 1
                or call.keywords
                or not isinstance(call.args[0], ast.Constant)
                or type(call.args[0].value) is not str
                or not call.args[0].value.isidentifier()
                or statement.lineno < scopes[scope].lineno
            ):
                fail(module, call, "Constant declaration requires one literal owner attribute name")
            allowed.add(id(call.func.value))
            add(
                call.args[0].value,
                dict(
                    kind="constant",
                    constant_declaration=[statement.lineno, statement.col_offset],
                    scope_allocation=origin(scope),
                ),
                statement,
            )
        elif method == "value" and isinstance(statement, ast.Expr):
            if statement.lineno < scopes[scope].lineno:
                fail(module, statement, "Value declaration precedes its resource scope")
            args = dict(zip(("name", "dtype"), call.args))
            if len(call.args) > 2 or any(
                k.arg not in ("name", "dtype", "affects_plan") or k.arg in args for k in call.keywords
            ):
                fail(module, statement, "ResourceScope.value requires explicit name and dtype")
            args.update((k.arg, k.value) for k in call.keywords)
            plan_option = args.pop("affects_plan", ast.Constant(False))
            if not isinstance(plan_option, ast.Constant) or type(plan_option.value) is not bool:
                fail(module, statement, "ResourceScope.value affects_plan requires a literal bool")
            name = args.get("name")
            if not (
                set(args) == {"name", "dtype"}
                and isinstance(name, ast.Constant)
                and isinstance(name.value, str)
                and name.value.isidentifier()
            ):
                fail(module, statement, "ResourceScope.value requires a literal attribute name and fixed type")
            allowed.add(id(call.func.value))
            add(
                member(scope, name.value),
                member_spec(
                    scope,
                    name.value,
                    kind="value",
                    type_node=args["dtype"],
                    **({"affects_plan": True} if plan_option.value else {}),
                    value_declaration=[statement.lineno, statement.col_offset],
                    scope_allocation=origin(scope),
                ),
                statement,
            )
        elif method == "owner_reference" and isinstance(statement, ast.Expr):
            if (
                conditions
                or scope_paths[scope]
                or len(call.args) != 2
                or call.keywords
                or not isinstance(call.args[0], ast.Constant)
                or type(call.args[0].value) is not str
                or not call.args[0].value.isidentifier()
                or not isinstance(call.args[1], ast.Name)
                or call.args[1].id not in parameter_names
                or stores[call.args[1].id] != 1
            ):
                fail(module, call, "Owner input requires an unconditional named constructor parameter")
            add(
                call.args[0].value,
                dict(
                    kind="owner_input",
                    input_parameter=call.args[1].id,
                    reference_declaration=[call.lineno, call.col_offset],
                    scope_allocation=origin(scope),
                ),
                call,
            )
            allowed.add(id(call.func.value))
        elif method == "borrow" and isinstance(statement, ast.Expr):
            from ._package_borrows import declaration

            name, spec = declaration(
                call, scope_paths[scope], declarations, dimension, lambda node, message: fail(module, node, message)
            )
            parameter = spec.get("input_parameter")
            if parameter is not None and (parameter not in parameter_names or stores[parameter] != 1):
                fail(module, call, "Borrowed input requires an unmodified constructor parameter")
            if parameter is not None and getattr(call, "_package_optional_borrow", False):
                spec["optional_capable"] = True
            if parameter is not None and getattr(call, "_package_field_obligation", False):
                spec["field_obligation"] = True
            if conditions:
                if not (
                    parameter is not None
                    and len(conditions) == 1
                    and all(
                        expected is True
                        and declarations[key].get("presence_nonnull")
                        and declarations[key].get("presence_parameter") == parameter
                        for key, expected in conditions.items()
                    )
                ):
                    fail(module, call, "Conditional borrow requires that input's proven non-None branch")
                spec.update(conditions=dict(conditions), optional_input=True)
            prior_input_binding = (
                parameter is not None
                and member_stores.get(name, 0) == 1
                and any(
                    isinstance(prior, ast.Assign)
                    and len(prior.targets) == 1
                    and isinstance(prior.targets[0], ast.Attribute)
                    and prior.targets[0].attr == name
                    and isinstance(prior.targets[0].value, ast.Name)
                    and prior.targets[0].value.id == "self"
                    and isinstance(prior.value, ast.Name)
                    and prior.value.id == parameter
                    and prior.lineno < call.lineno
                    for prior in constructor.body
                )
            )
            if member_stores.get(name, 0) and not prior_input_binding:
                fail(module, call, "Borrowed Field cannot replace an existing owner attribute")
            spec["scope_allocation"] = origin(scope)
            add(name, spec, call)
            allowed.add(id(call.func.value))
        elif method == "require_le" and isinstance(statement, ast.Expr):
            from taichi_forge_runtime._dimensions import validate_constraints

            if not (
                not conditions
                and len(call.args) == 3
                and len(call.keywords) == 1
                and call.keywords[0].arg == "when"
                and isinstance(call.args[0], ast.Constant)
                and type(call.args[0].value) is str
                and call.args[0].value.isidentifier()
                and isinstance(call.keywords[0].value, ast.Constant)
                and type(call.keywords[0].value.value) is str
                and call.keywords[0].value.value.isidentifier()
            ):
                fail(module, call, "Dimension constraint requires a literal name, declared expressions and trigger")
            # require_le receives Extent objects (or literal constants) in
            # ordinary Python. A raw member read is an integer snapshot, not a
            # symbolic constraint, even if layouts use that declared member.
            allow_member_dimensions = False
            try:
                left, right = dimension(call.args[1]), dimension(call.args[2])
            finally:
                allow_member_dimensions = True
            if not isinstance(left, Extent) or not isinstance(right, Extent):
                fail(module, call, "Dimension constraint requires the declared dimension algebra")
            namespace_check(scope, (left, right), call)
            trigger = member(scope, call.keywords[0].value.value)
            name = member(scope, call.args[0].value)
            spec = dict(left=left.expression, right=right.expression, when=trigger)
            dimensions = {k for k, v in declarations.items() if v["kind"] == "dimension"}
            roles = {
                k for k, v in declarations.items() if v["kind"] in ("dimension", "value", "dense_field", "ndarray")
            }
            try:
                validate_constraints({name: spec}, dimensions, roles)
            except ValueError as error:
                fail(module, call, str(error))
            constraints = declarations[trigger].setdefault("constraints", {})
            if name in constraint_declarations and constraint_declarations[name] != spec:
                fail(module, call, "Dimension constraint already has another declaration")
            constraint_declarations[name] = spec
            constraints[name] = dict(spec=spec, source=[module, statement.lineno])
            allowed.add(id(call.func.value))
        elif method == "dense":
            if scope_paths[scope]:
                fail(module, call, "A record describes existing members and cannot allocate a Dense group")
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and stores[statement.targets[0].id] == 1
                and len(call.args) == 2
                and not call.keywords
                and statement.lineno > scopes[scope].lineno
                and (
                    scope_groups[scope] not in builders
                    or statement.lineno < builders[scope_groups[scope]]["finalization"][0]
                )
            ):
                fail(module, statement, "Declared Dense placement requires one local alias and explicit axes/shape")
            axes = qualified_kind(call.args[0], kind)
            if axes not in ("i", "ij", "ijk", "ijkl"):
                fail(module, call.args[0], "Declared Dense placement requires canonical ordered axes")
            shape = dimension(call.args[1])
            namespace_check(scope, shape, call)
            shape = (shape,) if isinstance(shape, Extent) else shape
            if not isinstance(shape, tuple) or len(shape) != len(axes):
                fail(module, call, "Declared Dense placement rank differs from its declared axes")
            alias = statement.targets[0].id
            placements[alias] = dict(
                shape=shape,
                group=scope_groups[scope],
                axes=axes,
                declaration=[statement.lineno, statement.col_offset],
                expression=ast.unparse(call.args[1]),
                conditions=dict(conditions),
            )
            allowed.add(id(call.func.value))
        elif method in ("field", "vector_field", "matrix_field", "struct_field"):
            alias_checks.clear()
            if scope_paths[scope]:
                fail(module, call, "A record describes existing members and cannot allocate Fields")
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else ([statement.target] if isinstance(statement, ast.AnnAssign) else [])
            )
            target = targets[0] if len(targets) == 1 else None
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and (target.value.id == "self" or scope_binding(target.value) in scopes)
                and statement.lineno > scopes[scope].lineno
                and (
                    member_stores[target.attr] == 1
                    or (
                        conditions
                        and member_stores[target.attr] == 2
                        and any(
                            isinstance(n, ast.Assign)
                            and isinstance(n.value, ast.Constant)
                            and n.value.value is None
                            and n.lineno < statement.lineno
                            and any(
                                isinstance(initial, ast.Attribute)
                                and isinstance(initial.value, ast.Name)
                                and initial.value.id == "self"
                                and initial.attr == target.attr
                                for initial in n.targets
                            )
                            for n in constructor.body
                        )
                    )
                )
            ):
                fail(module, statement, "ResourceScope Field requires one unconditional owner attribute")
            if scope_binding(target.value) in scopes:
                allowed.add(id(target.value))
            allowed.add(id(call.func.value))
            normalized = copy.deepcopy(call)
            normalized.keywords = [
                k
                for k in call.keywords
                if k.arg not in ("initialization", "placement", "layout", "member_initialization")
            ]
            layout = "AOS"
            if method == "struct_field":
                if conditions:
                    fail(module, call, "Conditional Struct Field presence requires further package support")
                layouts = [k.value for k in call.keywords if k.arg == "layout"]
                if layouts:
                    layout = {"Layout.AOS": "AOS", "Layout.SOA": "SOA"}.get(qualified_kind(layouts[0], kind))
                    if len(layouts) != 1 or layout is None:
                        fail(module, call, "Struct Field layout requires the original AOS/SOA enum")
                if layout == "SOA" and scope_groups[scope] in builders:
                    fail(module, call, "Explicit ResourceScope groups require the original AOS Struct Field layout")
            elif any(k.arg == "layout" for k in call.keywords):
                fail(module, call, "Package Field declarations currently require plain default AOS layout")
            if method != "struct_field" and any(k.arg == "member_initialization" for k in call.keywords):
                fail(module, call, "Member initialization requires a Struct Field")
            for keyword in call.keywords:
                if keyword.arg == "initialization" and not (
                    isinstance(keyword.value, ast.Constant) and keyword.value.value in ("zero", "consumer_explicit")
                ):
                    fail(module, keyword, "Field initialization requires its existing zero/consumer_explicit policy")
            role = {
                "field": "field",
                "vector_field": "Vector.field",
                "matrix_field": "Matrix.field",
                "struct_field": "Struct.field",
            }[method]
            dtype, shape_node = field_type_shape(normalized, role, module, fail)
            source_type = copy.deepcopy(dtype)
            local_type = any(isinstance(n, ast.Name) and n.id in local_types for n in ast.walk(dtype))
            try:
                dtype = expand_local_type(dtype, local_types, kind)
            except ValueError as error:
                fail(module, call, str(error))
            # Use original shape nodes for declaration provenance.
            original_args = dict(
                zip(
                    {
                        "field": ("dtype", "shape"),
                        "vector_field": ("n", "dtype", "shape"),
                        "matrix_field": ("n", "m", "dtype", "shape"),
                        "struct_field": ("members", "shape"),
                    }[method],
                    call.args,
                )
            )
            original_args.update((k.arg, k.value) for k in call.keywords)
            shape_node = original_args["shape"]
            shape = dimension(shape_node)
            namespace_check(scope, shape, shape_node)
            if isinstance(shape, Extent):
                shape = (shape,)
            if not isinstance(shape, tuple):
                fail(module, shape_node, "ResourceScope Field shape requires declared dimensions and their algebra")
            group = scope_groups[scope]
            if group in builders and statement.lineno >= builders[group]["finalization"][0]:
                fail(module, statement, "Resource declaration follows FieldsBuilder finalization")
            placement_node = original_args.get("placement")
            placement = None
            if placement_node is not None:
                placement = placements.get(placement_node.id) if isinstance(placement_node, ast.Name) else None
                if (
                    placement is None
                    or placement["group"] != group
                    or placement["shape"] != shape
                    or placement["declaration"][0] >= statement.lineno
                    or any(conditions.get(key) is not value for key, value in placement["conditions"].items())
                ):
                    fail(module, placement_node, "Field placement must retain its declared group and extent relations")
                allowed.add(id(placement_node))
            extents, expressions, sources = [], {}, {}

            def project_extent(extent, node):
                expression = extent.expression
                if not extent_dependencies(expression):
                    value = evaluate_extent(expression, lambda _: None)
                    if value <= 0:
                        fail(module, node, "Field extents must be positive")
                    return value
                key = "_forge_package_extent_" + hashlib.sha256(repr(expression).encode()).hexdigest()
                expressions[key] = expression
                sources[key] = f"_forge_package_runtime.extent({expression!r}, self)"
                return key

            extents = [project_extent(extent, shape_node) for extent in shape]

            def model_axis(node):
                extent = dimension(node)
                if isinstance(extent, Extent):
                    used_model_dimensions.update(extent_dependencies(extent.expression))
                    return project_extent(extent, node)
                return None

            project_tensor_axes(dtype, kind, model_axis)
            add(
                target.attr,
                dict(
                    kind="dense_field",
                    type_node=dtype,
                    **({"local_type_source": source_type} if local_type else {}),
                    **(
                        {
                            "member_initialization_node": original_args["member_initialization"],
                            "member_initialization_source": source_type,
                        }
                        if "member_initialization" in original_args
                        else {}
                    ),
                    shape=extents,
                    dimension_sources=sources,
                    dimension_expressions=expressions,
                    **({"dimension_alias_checks": dict(alias_checks)} if alias_checks else {}),
                    allocation=[statement.lineno, statement.col_offset],
                    scope_allocation=origin(scope),
                    group=group,
                    **({"struct_field": True, "layout": layout} if method == "struct_field" else {}),
                    **({"conditions": dict(conditions)} if conditions else {}),
                    **({"group_builder": builders[group]} if group in builders else {}),
                    **(
                        {
                            "placement": {k: v for k, v in placement.items() if k != "shape"},
                            "placement_alias": placement_node.id,
                        }
                        if placement is not None
                        else {}
                    ),
                ),
                statement,
            )
            if conditions:
                conditional_fields.add(target.attr)
        elif isinstance(statement, ast.Expr) and method in ("dim", "parent_dim", "derived_dim", "power_of_two_dim"):
            dimension(call)
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if len(targets) == 1 and isinstance(targets[0], ast.Attribute):
                target = targets[0]
                receiver = scope_binding(target.value)
                if receiver in scopes and not scope_paths[receiver]:
                    scope_writes.append(target)
                    if isinstance(statement.value, ast.Call):
                        from ._package_factories import _source_value
                        import inspect

                        cls = _source_value(statement.value.func, namespace)
                        if inspect.isclass(cls):
                            root = statement.value.func
                            while isinstance(root, ast.Attribute):
                                root = root.value
                            if not isinstance(root, ast.Name) or root.id in stores:
                                fail(module, statement, "Declared child owner requires an unshadowed class definition")
                            if conditions or member_stores[target.attr] != 1:
                                fail(module, statement, "Declared child owner requires one unconditional construction")
                            add(
                                target.attr,
                                dict(
                                    kind="owner",
                                    owner_type=cls.__module__ + "." + cls.__qualname__,
                                    constructor_call=copy.deepcopy(statement.value),
                                    scope_allocation=origin(receiver),
                                ),
                                statement,
                            )
            if len(targets) == 1 and isinstance(targets[0], (ast.Name, ast.Tuple, ast.List)):
                target = targets[0]
                expression = statement.value
                if (
                    isinstance(target, ast.Name)
                    and not conditions
                    and stores[target.id] == 1
                    and target.id not in deleted
                ):
                    if qualified_kind(expression, kind) in ("types.vector", "types.matrix", "types.struct") or (
                        isinstance(expression, ast.Name) and expression.id in local_types
                    ):
                        local_types[target.id] = expand_local_type(expression, local_types, kind)
                if isinstance(target, (ast.Tuple, ast.List)) and isinstance(expression, ast.GeneratorExp):
                    # Tuple unpacking consumes the same literal-name generator
                    # once. Reuse the existing proof; do not run the generator.
                    expression = ast.Call(ast.Name("tuple", ast.Load()), [expression], [])
                value = dimension(expression)
                # Local integer aliases are often read before the resource
                # helper declares their dimensions. Retain only expression
                # provenance, then resolve against declarations at use. No
                # constructor or attribute getter is executed at export.
                names = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
                expressions = (
                    expression.elts
                    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(expression, (ast.Tuple, ast.List))
                    else [expression]
                )
                deferred = value is None or any(
                    isinstance(n, ast.Name) and n.id in deferred_aliases for n in ast.walk(expression)
                )
                if deferred and len(names) == len(expressions):
                    for name, source in zip(names, expressions):
                        if (
                            isinstance(name, ast.Name)
                            and stores[name.id] == 1
                            and name.id not in deleted
                            and isinstance(source, (ast.Attribute, ast.Name, ast.BinOp))
                            and not any(isinstance(n, (ast.Call, ast.Subscript)) for n in ast.walk(source))
                        ):
                            deferred_aliases[name.id] = source
                            alias_conditions[name.id] = dict(conditions)
                if value is not None:
                    names = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
                    values = (
                        value if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, tuple) else (value,)
                    )
                    if len(names) != len(values):
                        continue
                    for name, value in zip(names, values):
                        if not isinstance(name, ast.Name) or stores[name.id] != 1 + int(name.id in parameter_names):
                            continue
                        aliases[name.id] = value
                        alias_conditions[name.id] = dict(conditions)

    for target in scope_writes:
        if declarations.get(target.attr, {}).get("kind") in ("dimension", "value", "static", "constant", "owner"):
            allowed.add(id(target.value))
    for node in ast.walk(constructor):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in placements
            and id(node) not in allowed
        ):
            fail(module, node, "Declared Dense placement may only be used by its scoped Field allocations")
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in scope_names
            and id(node) not in allowed
        ):
            fail(module, node, "ResourceScope use requires an unconditional supported resource declaration")
    for name, placement in placements.items():
        members = [item for item in declarations.values() if item.get("placement_alias") == name]
        if not members:
            fail(module, constructor, "Unused Dense placement requires an explicit resource closure")
        if not any(
            all(placement["conditions"].get(key) is value for key, value in item.get("conditions", {}).items())
            for item in members
        ):
            fail(module, constructor, "Declared Dense placement requires a member whenever its branch is active")
    for name in conditional_fields:
        group = declarations[name]["group"]
        group_conditions = builders.get(group, {}).get("conditions", {})
        if not any(
            item["kind"] == "dense_field"
            and item.get("group") == group
            and item.get("conditions", {}) == group_conditions
            for item in declarations.values()
        ):
            fail(module, constructor, "Conditional Dense groups currently require an unconditional member")
    for name in optional_members:
        declarations[name]["optional_presence"] = True
    for name, call in model_candidates.items():
        if name not in used_model_dimensions:
            fail(
                module,
                call,
                "Computed static domain requires a proven model tensor axis; otherwise use literal choices",
            )
    return declarations
