"""Project existing resource declarations, never execute application constructors."""

import ast
import copy
import hashlib

from taichi_forge_runtime._dimensions import Extent, extent_dependencies, evaluate_extent
from ._package_types import qualified_kind
from ._package_fields import field_type_shape


def scope_declarations(module, constructor, kind, fail, namespace):
    if constructor is None:
        return {}
    scopes, declarations, allowed = {}, {}, set()
    scope_groups, builders = {}, {}
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

    def operation(node):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in scopes
        ):
            return node.func.value.id, node.func.attr
        return None, None

    def add(name, spec, node):
        existing = declarations.get(name)
        if existing is not None and existing["kind"] == spec["kind"] == "dimension":
            # Two explicit groups may use the same owner count. Provenance of
            # its declaration does not create a different dimension value.
            return
        if existing is not None and existing != spec:
            fail(module, node, f"Conflicting or repeated ResourceScope role for {name!r}")
        declarations[name] = spec

    def origin(scope):
        call = scopes[scope].value
        return [call.lineno, call.col_offset]

    aliases, placements = {}, {}
    stores = {n.arg: 1 for n in ast.walk(constructor.args) if isinstance(n, ast.arg)}
    member_stores = {}
    for node in ast.walk(constructor):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            stores[node.id] = stores.get(node.id, 0) + 1
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id in ("self", *scopes)
        ):
            member_stores[node.attr] = member_stores.get(node.attr, 0) + 1

    def dimension(node, bindings=None):
        """Only the already-declared dimension algebra; no model-value evaluation."""
        bindings = bindings or {}
        if isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 2**31 - 1:
            return Extent(("const", node.value))
        if isinstance(node, ast.Name):
            if node.id in bindings:
                return bindings[node.id]
            return aliases.get(node.id)
        if isinstance(node, (ast.Tuple, ast.List)):
            values = tuple(dimension(item, bindings) for item in node.elts)
            return values if all(isinstance(v, Extent) for v in values) else None
        scope, method = operation(node)
        if method == "dim":
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
            add(name, dict(kind="dimension", dtype="i32", scope_allocation=origin(scope)), node)
            return Extent(("dim", name))
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
                return getattr(left, method)(right)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("maximum", "minimum")
            and len(node.args) == 1
            and not node.keywords
        ):
            left, right = dimension(node.func.value, bindings), dimension(node.args[0], bindings)
            if isinstance(left, Extent) and isinstance(right, Extent):
                return getattr(left, node.func.attr)(right)
        return None

    for statement in constructor.body:
        call = statement.value if isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign)) else None
        scope, method = operation(call)
        if method == "value" and isinstance(statement, ast.Expr):
            if statement.lineno < scopes[scope].lineno:
                fail(module, statement, "Value declaration precedes its resource scope")
            args = dict(zip(("name", "dtype"), call.args))
            if len(call.args) > 2 or any(k.arg not in ("name", "dtype") or k.arg in args for k in call.keywords):
                fail(module, statement, "ResourceScope.value requires explicit name and dtype")
            args.update((k.arg, k.value) for k in call.keywords)
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
                name.value,
                dict(
                    kind="value",
                    type_node=args["dtype"],
                    value_declaration=[statement.lineno, statement.col_offset],
                    scope_allocation=origin(scope),
                ),
                statement,
            )
        elif method == "dense":
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
            )
            allowed.add(id(call.func.value))
        elif method in ("field", "vector_field", "matrix_field"):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else ([statement.target] if isinstance(statement, ast.AnnAssign) else [])
            )
            target = targets[0] if len(targets) == 1 else None
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in ("self", *scopes)
                and statement.lineno > scopes[scope].lineno
                and member_stores[target.attr] == 1
            ):
                fail(module, statement, "ResourceScope Field requires one unconditional owner attribute")
            if target.value.id in scopes:
                allowed.add(id(target.value))
            allowed.add(id(call.func.value))
            normalized = copy.deepcopy(call)
            normalized.keywords = [k for k in call.keywords if k.arg not in ("initialization", "placement")]
            for keyword in call.keywords:
                if keyword.arg == "initialization" and not (
                    isinstance(keyword.value, ast.Constant) and keyword.value.value in ("zero", "consumer_explicit")
                ):
                    fail(module, keyword, "Field initialization requires its existing zero/consumer_explicit policy")
            role = {"field": "field", "vector_field": "Vector.field", "matrix_field": "Matrix.field"}[method]
            dtype, shape_node = field_type_shape(normalized, role, module, fail)
            # Use original shape nodes for declaration provenance.
            original_args = dict(
                zip(
                    {
                        "field": ("dtype", "shape"),
                        "vector_field": ("n", "dtype", "shape"),
                        "matrix_field": ("n", "m", "dtype", "shape"),
                    }[method],
                    call.args,
                )
            )
            original_args.update((k.arg, k.value) for k in call.keywords)
            shape_node = original_args["shape"]
            shape = dimension(shape_node)
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
                ):
                    fail(module, placement_node, "Field placement must retain its declared group and extent relations")
                allowed.add(id(placement_node))
            extents, expressions, sources = [], {}, {}
            for extent in shape:
                expression = extent.expression
                if not extent_dependencies(expression):
                    value = evaluate_extent(expression, lambda _: None)
                    if value <= 0:
                        fail(module, shape_node, "Field extents must be positive")
                    extents.append(value)
                    continue
                key = "_forge_package_extent_" + hashlib.sha256(repr(expression).encode()).hexdigest()
                extents.append(key)
                expressions[key] = expression
                sources[key] = f"_forge_package_runtime.extent({expression!r}, self)"
            add(
                target.attr,
                dict(
                    kind="dense_field",
                    type_node=dtype,
                    shape=extents,
                    dimension_sources=sources,
                    dimension_expressions=expressions,
                    allocation=[statement.lineno, statement.col_offset],
                    scope_allocation=origin(scope),
                    group=group,
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
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if len(targets) == 1 and isinstance(targets[0], (ast.Name, ast.Tuple, ast.List)):
                target = targets[0]
                value = dimension(statement.value)
                if value is not None:
                    names = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
                    values = (
                        value if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, tuple) else (value,)
                    )
                    if len(names) != len(values):
                        continue
                    for name, value in zip(names, values):
                        if not isinstance(name, ast.Name) or stores[name.id] != 1:
                            continue
                        aliases[name.id] = value

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
            and node.id in scopes
            and id(node) not in allowed
        ):
            fail(module, node, "ResourceScope use requires an unconditional supported resource declaration")
    for name, placement in placements.items():
        if not any(item.get("placement_alias") == name for item in declarations.values()):
            fail(module, constructor, "Unused Dense placement requires an explicit resource closure")
    return declarations
