"""Recognize direct Field declarations; native SNode code owns every byte layout."""

import ast

from ._package_types import qualified_kind, resolve_type, type_spec, project_tensor_axes


def field_type_shape(call, role, module, fail):
    names = {
        "field": ("dtype", "shape"),
        "Vector.field": ("n", "dtype", "shape"),
        "Matrix.field": ("n", "m", "dtype", "shape"),
        "Struct.field": ("members", "shape"),
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
    dtype = args["members"] if role == "Struct.field" else args["dtype"]
    if role in ("Vector.field", "Matrix.field"):
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

        def model_axis(axis):
            if (
                isinstance(axis, ast.Name)
                and axis.id in parameters
                or (isinstance(axis, ast.Attribute) and isinstance(axis.value, ast.Name) and axis.value.id == "self")
            ):
                return extent(axis, sources)
            return None

        project_tensor_axes(dtype, kind, model_axis)
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
    from ._component import DenseField, Dimension, Record, _ConditionalDenseField, _ModelTensorField
    from taichi_forge_runtime._dimensions import evaluate_extent
    from ._package_records import read_member, write_member

    fields = dict(
        sorted(
            ((name, item) for name, item in schema.items() if item["kind"] == "dense_field"),
            key=lambda pair: pair[1]["allocation"],
        )
    )
    declarations = {}
    from types import SimpleNamespace

    for name, item in schema.items():
        if item["kind"] == "record":
            write_member(owner, name, SimpleNamespace(), schema)

    def witness(name, visiting=()):
        if name in visiting:
            raise ValueError(f"Cyclic derived dimension {name!r}")
        expression = schema[name].get("expression")
        value = evaluate_extent(expression, lambda key: witness(key, (*visiting, name))) if expression else 1
        if not 0 <= value <= 2**31 - 1:
            raise ValueError("Declared dimension needs a legal symbolic construction witness")
        write_member(owner, name, value, schema)
        return value

    for name, item in schema.items():
        if item["kind"] == "dimension":
            declarations[name] = Dimension(kind="active", expression=item.get("expression", ()))
            witness(name)
    if not fields:
        return declarations
    typed = []
    for name, item in fields.items():

        def field_extent(extent):
            expression = item.get("dimension_expressions", {}).get(extent, ())
            declarations[extent] = Dimension(expression=expression)
            value = evaluate_extent(expression, lambda key: read_member(owner, key, schema)) if expression else 1
            if not 0 < value <= 2**31 - 1:
                raise ValueError("Declared extent needs a legal symbolic construction witness")
            setattr(owner, extent, value)
            return value

        model_tensors = {}
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
            allow_members=item.get("struct_field", False),
            model_tensors=model_tensors,
            witness=field_extent,
        )
        item["element"] = type_spec(dtype, module, model_tensors=model_tensors)
        if model_tensors and item.get("conditions"):
            raise ValueError("Conditional model tensor Fields require projected layout selection")
        if item.get("struct_field"):
            if (
                item["element"]["kind"] != "struct"
                or not item["element"]["members"]
                or any(part["kind"] not in ("scalar", "tensor") for part in item["element"]["members"].values())
            ):
                raise ValueError("Package Struct Field requires flat scalar/vector/matrix members")
        elif item["element"]["kind"] not in ("scalar", "tensor"):
            raise ValueError("Package Field declaration currently requires scalar/vector/matrix elements")
        concrete = []
        for extent in item["shape"]:
            if isinstance(extent, str):
                concrete.append(field_extent(extent))
            else:
                if not 0 < extent <= 2**31 - 1:
                    raise ValueError("Field extents must fit the existing positive i32 dimension contract")
                concrete.append(extent)
        axes = ti.lang.impl.index_nd(len(concrete))
        typed.append((name, item, dtype, concrete, axes))
    # Expected type/rank/extent rejection must not leave an unfinalized
    # FieldsBuilder behind and poison the caller's otherwise valid JIT.
    builders, placements, events = {}, {}, {}
    for ordinal, (name, item, dtype, concrete, axes) in enumerate(typed):
        group = item.get("group", "fields")
        if group not in builders:
            builders[group] = ti.FieldsBuilder()
        # Several literal member declarations can originate at one source call.
        # Preserve their declaration order instead of replacing earlier members.
        events[(*item["allocation"], ordinal)] = ("field", name, item, dtype, concrete, axes)
        if "placement" in item:
            event = item["placement"]
            events[(*event["declaration"], -1)] = ("placement", name, item, dtype, concrete, axes)
    for position, (kind, name, item, dtype, concrete, axes) in sorted(events.items()):
        group = item.get("group", "fields")
        builder = builders[group]
        if kind == "placement":
            placements[position[:2]] = builder.dense(axes, tuple(concrete))
            continue
        value = ti.Struct.field(dtype.members) if item.get("struct_field") else ti.field(dtype)
        # This is the documented default field(shape=...) placement through the
        # ordinary builder. Native capture verifies the complete group below.
        # Even shape=() uses the ordinary factory's empty-axis Dense node.
        # Placing a tensor directly at root would split its lanes into separate
        # implicit nodes, unlike Matrix.field(shape=())'s one packed element.
        parent = (
            placements[tuple(item["placement"]["declaration"])]
            if "placement" in item
            else None if item.get("layout") == "SOA" else builder.dense(axes, tuple(concrete))
        )
        if item.get("layout") == "SOA":
            for member in value.field_dict.values():
                builder.dense(axes, tuple(concrete)).place(member)
        else:
            parent.place(value)
        setattr(owner, name, value)

        def field_declaration(element, spec):
            if spec.get("model_tensor"):
                return _ModelTensorField(element.dtype, tuple(item["shape"]), group, tuple(spec["element_shape"]))
            return DenseField(element, tuple(item["shape"]), group)

        declarations[name] = (
            Record(
                **{
                    member: field_declaration(element, item["element"]["members"][member])
                    for member, element in dtype.members.items()
                }
            )
            if item.get("struct_field")
            else (
                _ConditionalDenseField(dtype, tuple(item["shape"]), group, tuple(sorted(item["conditions"].items())))
                if item.get("conditions")
                else field_declaration(dtype, item["element"])
            )
        )
    owner._forge_package_field_trees = [builder._finalize_for_aot() for builder in builders.values()]
    return declarations


def field_environment(declarations, schema):
    """Retain the allocation scope of record dimensions, as from_owner does."""
    from dataclasses import replace
    from ._component import _ModelTensorField, _capture_id
    from ._package_records import record_environment

    environment = record_environment(declarations, schema)
    flattened = dict(environment.declarations)
    for name, item in schema.items():
        if item.get("struct_field"):
            for member in item["element"]["members"]:
                identity = _capture_id((name, member))
                original = flattened[identity]
                changes = dict(shape=tuple(item["shape"]))
                if isinstance(original, _ModelTensorField):
                    changes["element_shape"] = tuple(item["element"]["members"][member]["element_shape"])
                flattened[identity] = replace(original, **changes)
    environment._set_declarations(flattened)
    environment._derived_attributes = frozenset(
        name for name, item in schema.items() if item["kind"] == "dimension" and item.get("expression")
    )
    return environment


def field_allocation(owner_key, name, schema):
    fields = [
        item
        for item in schema.values()
        if item["kind"] == "dense_field" and item.get("group", "fields") == schema[name].get("group", "fields")
    ]
    sources = {key: expression for item in fields for key, expression in item["dimension_sources"].items()}
    dimensions = sorted(sources)

    def dimension_value(key):
        value = ast.parse(sources[key], mode="eval").body
        conditions = [
            item.get("placement", {}).get("conditions", item.get("conditions", {}))
            for item in fields
            if key in item["dimension_sources"]
        ]
        if any(not condition for condition in conditions):
            return value
        tests = []
        for condition in conditions:
            terms = [
                ast.Compare(
                    ast.Attribute(ast.Name("self", ast.Load()), attribute, ast.Load()),
                    [ast.Is()],
                    [ast.Constant(expected)],
                )
                for attribute, expected in condition.items()
            ]
            tests.append(terms[0] if len(terms) == 1 else ast.BoolOp(ast.And(), terms))
        test = tests[0] if len(tests) == 1 else ast.BoolOp(ast.Or(), tests)
        return ast.IfExp(test, value, ast.Constant(None))

    return ast.Call(
        ast.Attribute(ast.Name("_forge_package_runtime", ast.Load()), "field", ast.Load()),
        [
            ast.Constant(owner_key),
            ast.Constant(name),
            ast.Dict(
                [ast.Constant(key) for key in dimensions],
                [dimension_value(key) for key in dimensions],
            ),
        ],
        (
            [ast.keyword("placement", ast.Name(schema[name]["placement_alias"], ast.Load()))]
            if "placement_alias" in schema[name]
            else []
        ),
    )
