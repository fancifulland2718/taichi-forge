"""Recognize direct Field declarations; native SNode code owns every byte layout."""

import ast

from ._package_types import qualified_kind, resolve_type, type_spec


def field_type_shape(call, role, module, fail):
    names = {
        "field": ("dtype", "shape"),
        "Vector.field": ("n", "dtype", "shape"),
        "Matrix.field": ("n", "m", "dtype", "shape"),
    }[role]
    if len(call.args) > len(names) or any(k.arg not in names for k in call.keywords):
        fail(module, call, "Package Field declarations currently require plain default AOS layout")
    args = dict(zip(names, call.args))
    for keyword in call.keywords:
        if keyword.arg in args:
            fail(module, call, "Duplicate Field factory argument")
        args[keyword.arg] = keyword.value
    if not set(names) <= args.keys():
        fail(module, call, "Field declaration requires its type dimensions, dtype and shape")
    dtype = args["dtype"]
    if role != "field":
        factory = "_forge_package_vector" if role == "Vector.field" else "_forge_package_matrix"
        dimensions = [args["n"]] + ([args["m"]] if role == "Matrix.field" else [])
        dtype = ast.Call(ast.Name(factory, ast.Load()), [*dimensions, dtype], [])
    return dtype, args["shape"]


def field_declarations(module, constructor, kind, fail):
    """Read only a declarative constructor slice, without running user code."""
    if constructor is None:
        return {}
    parameters = {
        a.arg for a in (*constructor.args.posonlyargs, *constructor.args.args, *constructor.args.kwonlyargs)
    } - {"self"}
    aliases, writes = {}, {}
    for node in ast.walk(constructor):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            if isinstance(node.ctx, ast.Store):
                writes[node.attr] = writes.get(node.attr, 0) + 1
    for node in constructor.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and isinstance(node.value, ast.Name)
                and node.value.id in parameters
                and writes[target.attr] == 1
            ):
                aliases[target.attr] = (node.value.id, node.lineno)

    field_calls = [
        n
        for n in ast.walk(constructor)
        if isinstance(n, ast.Call) and qualified_kind(n, kind) in ("field", "Vector.field", "Matrix.field")
    ]
    first_allocation = min((n.lineno for n in field_calls), default=constructor.end_lineno)

    def extent(node, sources):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value > 0:
            return node.value
        parameter = node.id if isinstance(node, ast.Name) else None
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            alias = aliases.get(node.attr)
            parameter = alias[0] if alias is not None and alias[1] < first_allocation else None
        if parameter not in parameters:
            fail(
                module,
                node,
                "Field extents require constructor parameters, single-assignment aliases or positive constants",
            )
        # Preserve the actual Host expression, including self aliases. An
        # intervening Host call may mutate an attribute without an AST store in
        # __init__; replacing self.n with n would silently change its shape.
        suffix = "self_" + node.attr if isinstance(node, ast.Attribute) else "arg_" + parameter
        key = "_forge_package_extent_" + suffix
        sources[key] = ast.unparse(node)
        return key

    fields = {}
    for node in ast.walk(constructor):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, ast.Call):
            continue
        role = qualified_kind(node.value, kind)
        if role not in ("field", "Vector.field", "Matrix.field"):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        target = targets[0]
        if (
            len(targets) != 1
            or not isinstance(target, ast.Attribute)
            or not isinstance(target.value, ast.Name)
            or target.value.id != "self"
            or node not in constructor.body
            or writes[target.attr] != 1
        ):
            fail(module, node, "Field allocation requires one unconditional owner attribute")
        dtype, shape = field_type_shape(node.value, role, module, fail)
        shape = shape.elts if isinstance(shape, (ast.List, ast.Tuple)) else [shape]
        sources = {}
        fields[target.attr] = dict(
            kind="dense_field",
            type_node=dtype,
            shape=[extent(n, sources) for n in shape],
            dimension_sources=sources,
            allocation=[node.lineno, node.col_offset],
        )
    return fields


def construct_fields(owner, schema, namespace, module, *, imported_modules=None):
    """Use original FieldsBuilder and DenseField capture, not a layout emulator."""
    import taichi_forge as ti
    from ._component import DenseField, Dimension
    from taichi_forge_runtime._dimensions import evaluate_extent

    fields = dict(
        sorted(
            ((name, item) for name, item in schema.items() if item["kind"] == "dense_field"),
            key=lambda pair: pair[1]["allocation"],
        )
    )
    declarations = {}
    for name, item in schema.items():
        if item["kind"] == "dimension":
            declarations[name] = Dimension(kind="active")
            setattr(owner, name, 1)
    if not fields:
        return declarations
    typed = []
    for name, item in fields.items():
        declaration_namespace = (
            vars(imported_modules[item["declaration_module"]])
            if imported_modules is not None and "declaration_module" in item
            else namespace
        )
        dtype = resolve_type(
            item.pop("type_node"),
            {
                **declaration_namespace,
                "_forge_package_vector": ti.types.vector,
                "_forge_package_matrix": ti.types.matrix,
            },
        )
        item["element"] = type_spec(dtype, module)
        if item["element"]["kind"] not in ("scalar", "tensor"):
            raise ValueError("Package Field declaration currently requires scalar/vector/matrix elements")
        concrete = []
        for extent in item["shape"]:
            if isinstance(extent, str):
                expression = item.get("dimension_expressions", {}).get(extent, ())
                declarations[extent] = Dimension(expression=expression)
                value = evaluate_extent(expression, lambda key: getattr(owner, key)) if expression else 1
                if not 0 < value <= 2**31 - 1:
                    raise ValueError("Declared extent needs a legal symbolic construction witness")
                setattr(owner, extent, value)
                concrete.append(value)
            else:
                if not 0 < extent <= 2**31 - 1:
                    raise ValueError("Field extents must fit the existing positive i32 dimension contract")
                concrete.append(extent)
        axes = ti.lang.impl.index_nd(len(concrete))
        typed.append((name, item, dtype, concrete, axes))
    # Expected type/rank/extent rejection must not leave an unfinalized
    # FieldsBuilder behind and poison the caller's otherwise valid JIT.
    builder = ti.FieldsBuilder()
    for name, item, dtype, concrete, axes in typed:
        value = ti.field(dtype)
        # This is the documented default field(shape=...) placement through the
        # ordinary builder. Native capture verifies the complete group below.
        # Even shape=() uses the ordinary factory's empty-axis Dense node.
        # Placing a tensor directly at root would split its lanes into separate
        # implicit nodes, unlike Matrix.field(shape=())'s one packed element.
        builder.dense(axes, tuple(concrete)).place(value)
        setattr(owner, name, value)
        declarations[name] = DenseField(dtype, tuple(item["shape"]), "fields")
    owner._forge_package_field_tree = builder._finalize_for_aot()
    return declarations


def field_allocation(owner_key, name, schema):
    sources = {
        key: expression
        for item in schema.values()
        if item["kind"] == "dense_field"
        for key, expression in item["dimension_sources"].items()
    }
    dimensions = sorted(sources)
    return ast.Call(
        ast.Attribute(ast.Name("_forge_package_runtime", ast.Load()), "field", ast.Load()),
        [
            ast.Constant(owner_key),
            ast.Constant(name),
            ast.Dict(
                [ast.Constant(key) for key in dimensions],
                [ast.parse(sources[key], mode="eval").body for key in dimensions],
            ),
        ],
        [],
    )
