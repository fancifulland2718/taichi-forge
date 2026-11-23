"""Project declared pointer/bitmasked placement through normal SNode construction."""

import ast


def owner_member(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def read_node(statement, nodes, dimension, qualified, fail):
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1 or not isinstance(statement.value, ast.Call):
        return None
    name = owner_member(statement.targets[0])
    if name is None:
        return None
    call = statement.value
    if qualified(call.func) == "root.pointer":
        operation, parent = "pointer", None
    elif (isinstance(call.func, ast.Attribute) and call.func.attr == "bitmasked"
          and owner_member(call.func.value) in nodes):
        operation, parent = "bitmasked", owner_member(call.func.value)
    else:
        return None
    if name in nodes or len(call.args) != 2 or call.keywords:
        fail(call, "Sparse SNode construction requires a unique named owner member and explicit axes and extents")
    axes = qualified(call.args[0])
    if axes not in ("i", "ij", "ijk", "ijkl"):
        fail(call.args[0], "Sparse SNode axes must be canonical and ordered")
    extent = dimension(call.args[1])
    parts = extent if isinstance(extent, tuple) else (extent,) * len(axes)
    if len(parts) != len(axes) or any(not hasattr(value, "expression") for value in parts):
        fail(call.args[1], "Sparse SNode size requires declared dimensions or integer constants")
    local = tuple(value.expression for value in parts)
    if parent is None:
        shape = local
        chain = ()
    else:
        if axes != nodes[parent]["axes"]:
            fail(call, "Bitmasked SNode axes must match the parent pointer axes")
        shape = tuple(("mul", a, b) for a, b in zip(nodes[parent]["shape"], local))
        chain = nodes[parent]["chain"]
    node = dict(attribute=name, op=operation, parent=parent, axes=axes,
                shape_expressions=local, declaration=[statement.lineno, statement.col_offset])
    nodes[name] = dict(shape=shape, group=chain[0]["attribute"] if chain else name, axes=axes, chain=(*chain, node))
    return name


def projected_assignment(statement, schema, runtime_name):
    if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
        return None
    target = statement.targets[0] if isinstance(statement, ast.Assign) else statement.target
    if not isinstance(target, ast.Attribute):
        return None
    site = [statement.lineno, statement.col_offset]
    for name, spec in schema.items():
        if not spec.get("storage_nodes"):
            continue
        method, member = None, None
        if name == target.attr and spec["allocation"] == site:
            method, member = "native_storage_field", name
        else:
            for node in spec["storage_nodes"]:
                if node["attribute"] == target.attr and node["declaration"] == site:
                    method, member = "native_storage_node", target.attr
                    break
        if method is not None:
            statement.value = ast.Call(ast.Attribute(ast.Name(runtime_name, ast.Load()), method, ast.Load()),
                                       [ast.Name("self", ast.Load()), ast.Constant(spec["group"]), ast.Constant(member)], [])
            return statement
    return None

def create_nodes(builder, chain, dimension):
    from taichi_forge.lang.impl import index_nd
    from taichi_forge_runtime._dimensions import evaluate_extent

    nodes = {}
    params = []
    expressions = {}
    native = builder
    for record in chain:
        sizes = []
        names = []
        for axis, expr in enumerate(record["shape_expressions"]):
            size = evaluate_extent(expr, dimension)
            if not 0 < size <= 2**31 - 1:
                raise ValueError("Sparse SNode axis must be a positive i32 extent")
            sizes.append(size)
            if expr[0] != "const":
                name = f"n{len(nodes)}_axis{axis}"
                names.append((axis, name))
                expressions[name] = expr
        native = getattr(native, record["op"])(index_nd(len(sizes)), tuple(sizes))
        nodes[record["attribute"]] = native
        params.extend((native.ptr, axis, name) for axis, name in names)
    return nodes, params, expressions


def register_tree(module, builder, chain, construction, witness_scope, group, guards, fields):
    import json

    from taichi_forge import _kernels, ndarray

    nodes, parameters, expressions = construction
    registry = module.__dict__.setdefault("_package_native_storage_trees", {})
    root = builder.ptr
    tree_id = root.get_snode_tree_id()
    if tree_id in registry:
        return
    package, owner_name, path = witness_scope
    owner_type = package + "." + owner_name
    key = json.dumps([owner_type, list(path), group, sorted(guards.items())], separators=(",", ":"))
    slot = module._aot_builder._declare_unified_storage(root, parameters, key)
    deactivation = {}
    for attribute, node in nodes.items():
        symbols = []

        class Deactivations:
            def dispatch(self, kernel, *, template_args):
                symbol = f"storage_{slot}_clear_{attribute}_{len(symbols)}"
                if symbol not in module._exported_kernels:
                    module.add_kernel(kernel, name=symbol, template_args=template_args)
                    module._aot_builder._seal_kernel_export(symbol)
                symbols.append(symbol)

        node.deactivate_all(builder=Deactivations())
        deactivation[attribute] = symbols
    paths = {}

    def visit(node, path):
        paths[node.id] = path
        for child in range(node.get_num_ch()):
            visit(node.get_ch(child), [*path, child])

    visit(root, [])
    members, arrays = {}, {}
    for name, field in fields.items():
        named = field.field_dict.items() if hasattr(field, "field_dict") else [(None, field)]
        for child, value in named:
            member = name if child is None else f"{name}.{child}"
            matrix = hasattr(value, "n")
            lanes = tuple(value._get_field_members())
            element_shape = ((value.n,) if value.ndim == 1 else (value.n, value.m)) if matrix else ()
            record = dict(places=[paths[lane.ptr.snode().id] for lane in lanes], element_shape=element_shape)
            args = {"mat": value, "as_vector": value.ndim == 1} if matrix else {"tensor": value}
            array_type = value.dtype, len(value.shape) + len(element_shape)
            if array_type not in arrays:
                arrays[array_type] = ndarray(value.dtype, shape=(1,) * array_type[1])
            args["arr"] = arrays[array_type]
            for operation, kernel in (
                ("read", _kernels.matrix_to_ext_arr if matrix else _kernels.tensor_to_ext_arr),
                ("write", _kernels.ext_arr_to_matrix if matrix else _kernels.ext_arr_to_tensor),
            ):
                symbol = f"storage_{slot}_{operation}_{len(members)}"
                if symbol not in module._exported_kernels:
                    module.add_kernel(kernel, name=symbol, template_args=args)
                    module._aot_builder._seal_kernel_export(symbol)
                record[operation] = symbol
            members[member] = record
    registry[tree_id] = dict(slot=slot, owner_type=owner_type, owner_path=list(path), group=group,
                             deactivate=deactivation, members=members,
                             conditions=guards, nodes=chain, dimensions=expressions)


def public_groups(module, manifest):
    from taichi_forge_runtime._dimensions import extent_dependencies

    groups = manifest.setdefault("native_storage_groups", {})
    for tree in getattr(module, "_package_native_storage_trees", {}).values():
        group = tree["group"]
        owner_type = tree["owner_type"]
        # Witnesses for an inherited method are bound to the concrete owner type.
        resources = manifest["owners"].get(owner_type, {})
        fields = {key: item for key, item in resources.items() if item.get("kind") == "fixed_snode_field"
                  and item["group"] == group}
        if not fields:
            continue
        name = owner_type + ":" + group
        result = groups.setdefault(name, dict(
            nodes=[{k: v for k, v in node.items() if k != "declaration"} for node in tree["nodes"]],
            dimensions={
                key: dict(expression=expr, paths={dep: resources[dep].get("path", [dep])
                                                  for dep in extent_dependencies(expr)})
                for key, expr in tree["dimensions"].items()
            },
            members={key: dict(element=field["element"], shape_expressions=field["shape_expressions"],
                               placement=field["storage_nodes"][-1]["attribute"],
                               conditions=field.get("conditions", {})) for key, field in fields.items()},
            slots=[],
        ))
        slot = dict(slot=tree["slot"], conditions=tree["conditions"], deactivate=tree["deactivate"], members=tree["members"])
        if slot not in result["slots"]:
            result["slots"].append(slot)
    if not groups:
        manifest.pop("native_storage_groups")


def entry_storage(module, code):
    registry = getattr(module, "_package_native_storage_trees", {})
    entries = []
    for group, dependency in sorted(code.fixed_dependencies.items()):
        tree = registry.get(dependency[0])
        if tree is None:
            raise ValueError("Numerical SNode dependency has no native storage declaration")
        prefix, _, local = group.rpartition(".")
        entries.append(dict(slot=tree["slot"], owner_path=prefix.split(".") if prefix else [], group=local))
    return entries


def selection(schema, environment, code):
    """An entry depends on the physical shape of each SNode tree it uses."""
    guards = {}
    for group in code.fixed_dependencies:
        *prefix, local = group.split(".")
        for key, item in schema.items():
            if (item.get("storage_nodes") and item["group"] == local
                    and list(environment._paths.get(key, ())[:-1]) == prefix):
                guards.update(item.get("storage_guards", {}))
    return guards
