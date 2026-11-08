"""Recognize direct Field declarations; native SNode code owns every byte layout."""

import ast

from ._package_types import qualified_kind, resolve_type, type_spec, project_tensor_axes, member_initialization


def field_type_shape(call, role, module, fail):
    names = {
        "field": ("dtype", "shape"),
        "Vector.field": ("n", "dtype", "shape"),
        "Matrix.field": ("n", "m", "dtype", "shape"),
        "Struct.field": ("members", "shape"),
    }[role]
    if len(call.args) > len(names) or any(k.arg not in (*names, "order") for k in call.keywords):
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


def field_order(call, rank, module, fail):
    """Only a fixed factory order; model-selected layouts need a separate contract."""
    from taichi_forge.lang.impl import _field_layout_axes

    orders = [keyword.value for keyword in call.keywords if keyword.arg == "order"]
    if not orders:
        return None
    value = orders[0]
    if (
        len(orders) != 1
        or not isinstance(value, ast.Constant)
        or (value.value is not None and type(value.value) is not str)
    ):
        fail(module, value, "Field order requires a literal axis permutation or None")
    try:
        _field_layout_axes((1,) * rank, value.value)
    except (TypeError, ValueError, SyntaxError) as error:
        fail(module, value, str(error))
    return value.value


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
        order = field_order(node.value, len(shape), module, fail)
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
            **({"order": order} if order is not None else {}),
        )
    return fields


def construct_fields(owner, schema, namespace, module, *, imported_modules=None):
    """Use original FieldsBuilder and DenseField capture, not a layout emulator."""
    import taichi_forge as ti
    from ._component import (
        DenseField,
        Dimension,
        Record,
        _ConditionalDenseField,
        _ConditionalModelTensorField,
        _ConditionalRecord,
        _ModelTensorField,
    )
    from taichi_forge_runtime._dimensions import evaluate_extent
    from ._package_records import read_member, write_member

    fields = dict(
        sorted(
            ((name, item) for name, item in schema.items() if item["kind"] == "dense_field"),
            key=lambda pair: pair[1]["allocation"],
        )
    )
    # Field dtype metadata comes from a declared resource type, not a live
    # constructor sample. Resolve aliases before allocating any witness Field.
    type_nodes = {key: item["type_node"] for key, item in schema.items() if "type_node" in item}

    def resource_dtype(expression, visiting=()):
        from taichi_forge.types._argument_descriptor import describe_element_type

        path = []
        while isinstance(expression, ast.Attribute):
            path.insert(0, expression.attr)
            expression = expression.value
        if not isinstance(expression, ast.Name) or expression.id != "self":
            raise ValueError("Field dtype metadata requires a declared owner resource")
        while True:
            key = next((key for key, spec in schema.items() if spec.get("path", [key]) == path), None)
            if key is None or key in visiting:
                raise ValueError("Field dtype metadata has an unknown or cyclic resource source")
            visiting = (*visiting, key)
            spec = schema[key]
            if spec["kind"] != "borrowed_field":
                break
            if any(name in spec for name in ("input_parameter", "input_expression", "source_type")):
                raise ValueError("Field dtype metadata requires a resolved resource type contract")
            path = spec["source_path"]
        if spec["kind"] not in ("dense_field", "ndarray") or key not in type_nodes:
            raise ValueError("Field dtype metadata requires an explicit scalar or tensor resource type")
        node = type_nodes[key]
        source_namespace = (
            vars(imported_modules[spec["declaration_module"]])
            if imported_modules is not None and "declaration_module" in spec
            else namespace
        )
        source_namespace = {
            **source_namespace,
            "_forge_package_vector": ti.types.vector,
            "_forge_package_matrix": ti.types.matrix,
        }
        # .dtype means the scalar component type, also for model-sized tensors.
        if isinstance(node, ast.Call):
            from ._package_factories import _source_value

            factory = _source_value(node.func, source_namespace)
            axes = 1 if factory is ti.types.vector else 2 if factory is ti.types.matrix else 0
            if axes:
                node = next((k.value for k in node.keywords if k.arg == "dtype"), None) or node.args[axes]
        dtype = resolve_type(node, source_namespace, resource_dtype=lambda value: resource_dtype(value, visiting))
        return describe_element_type(dtype).scalar_type

    declarations = {}
    from types import SimpleNamespace

    # Direct fields retain the Host expression used at allocation. When a
    # model axis comes from an owner scalar, connect that scalar to the same
    # authoritative Dimension relation instead of treating it as an unrelated
    # runtime value (or specializing its construction witness).
    for item in fields.values():
        for node in ast.walk(item["type_node"]):
            axis = getattr(node, "_package_model_extent", None)
            source = item.get("dimension_sources", {}).get(axis)
            if source is None or axis in item.get("dimension_expressions", {}):
                continue
            expression = ast.parse(source, mode="eval").body
            if (isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name)
                    and expression.value.id == "self"):
                name = expression.attr
                scalar = schema.get(name, {})
                if scalar.get("kind") == "value" and scalar.get("dtype") == "i32":
                    schema[name] = {**scalar, "kind": "dimension"}
                if schema.get(name, {}).get("kind") == "dimension":
                    item.setdefault("dimension_expressions", {})[axis] = ("dim", name)

    for name, item in schema.items():
        if item["kind"] == "record":
            write_member(owner, name, SimpleNamespace(), schema)

    def witness(name, visiting=()):
        if name in visiting:
            raise ValueError(f"Cyclic derived dimension {name!r}")
        if "owner_dimension" in schema[name]:
            value = read_member(owner, name, schema)
            if type(value) is not int or not 0 <= value <= 2**31 - 1:
                raise ValueError("Referenced dimension needs a legal symbolic construction witness")
            return value
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
            resource_dtype=resource_dtype,
        )
        item["element"] = type_spec(dtype, module, model_tensors=model_tensors)
        if "local_type_source" in item and item["element"]["kind"] == "struct":
            item["local_type_order"] = list(item["element"]["members"])
        if "member_initialization_node" in item:
            item["member_initialization"] = member_initialization(
                item.pop("member_initialization_node"),
                item["element"].get("members", {}),
                item.pop("member_initialization_source"),
            )
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
        axis_seq, shape_seq, same_level = ti.lang.impl._field_layout_axes(concrete, item.get("order"))

        def ordinary_placement():
            return ti.lang.impl._create_snode(axis_seq, shape_seq, same_level, root_node=builder)

        # This is the documented default field(shape=...) placement through the
        # ordinary builder. Native capture verifies the complete group below.
        # Even shape=() uses the ordinary factory's empty-axis Dense node.
        # Placing a tensor directly at root would split its lanes into separate
        # implicit nodes, unlike Matrix.field(shape=())'s one packed element.
        parent = (
            placements[tuple(item["placement"]["declaration"])]
            if "placement" in item
            else None if item.get("layout") == "SOA" else ordinary_placement()
        )
        if item.get("layout") == "SOA":
            if item.get("struct_field"):
                members = value.field_dict.values()
            elif isinstance(value, ti.MatrixField):
                value._aot_tensor_soa = True
                members = (value.get_scalar_field(i, j) for i in range(value.n) for j in range(value.m))
            else:
                members = (value,)
            for member in members:
                ordinary_placement().place(member)
        else:
            parent.place(value)
        setattr(owner, name, value)

        def field_declaration(element, spec):
            if spec.get("model_tensor"):
                if item.get("conditions"):
                    return _ConditionalModelTensorField(
                        element.dtype,
                        tuple(item["shape"]),
                        group,
                        tuple(spec["element_shape"]),
                        tuple(sorted(item["conditions"].items())),
                    )
                return _ModelTensorField(element.dtype, tuple(item["shape"]), group, tuple(spec["element_shape"]))
            if item.get("conditions"):
                return _ConditionalDenseField(
                    element, tuple(item["shape"]), group, tuple(sorted(item["conditions"].items()))
                )
            return DenseField(element, tuple(item["shape"]), group)

        def record_declaration(**members):
            return (
                _ConditionalRecord(sorted(item["conditions"].items()), **members)
                if item.get("conditions")
                else Record(**members)
            )

        declarations[name] = (
            record_declaration(
                **{
                    member: field_declaration(element, item["element"]["members"][member])
                    for member, element in dtype.members.items()
                }
            )
            if item.get("struct_field")
            else field_declaration(dtype, item["element"])
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
    keywords = []
    if "order_source" in schema[name]:
        keywords.append(ast.keyword("order", schema[name]["order_source"]))
    if "like_source_path" in schema[name]:
        source = ast.Name("self", ast.Load())
        for part in schema[name]["like_source_path"]:
            source = ast.Attribute(source, part, ast.Load())
        keywords.append(ast.keyword("like_source", source))
    if "local_type_source" in schema[name]:
        keywords.append(ast.keyword("declared_type", schema[name]["local_type_source"]))
    if "placement_alias" in schema[name]:
        keywords.append(ast.keyword("placement", ast.Name(schema[name]["placement_alias"], ast.Load())))
    checks = schema[name].get("dimension_alias_checks", {})
    if checks:
        keywords.append(
            ast.keyword(
                "dimension_aliases",
                ast.List(
                    [
                        ast.Tuple(
                            [
                                ast.Constant(alias),
                                ast.Name(alias, ast.Load()),
                                ast.parse(repr(expression), mode="eval").body,
                            ],
                            ast.Load(),
                        )
                        for alias, expression in sorted(checks.items())
                    ],
                    ast.Load(),
                ),
            )
        )

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
            terms = []
            for attribute, expected in condition.items():
                declaration = schema[attribute]
                source = ast.Name("self", ast.Load())
                for part in declaration.get("path", (attribute,)):
                    source = ast.Attribute(source, part, ast.Load())
                optional = declaration["kind"] == "record" and declaration.get("optional")
                terms.append(
                    ast.Compare(
                        source,
                        [ast.IsNot() if optional and expected else ast.Is()],
                        [ast.Constant(None if optional else expected)],
                    )
                )
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
        keywords,
    )
