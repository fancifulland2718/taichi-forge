"""Resolve declared construction types without executing application constructors."""

import copy
from types import MappingProxyType


def owner_bindings(environment, schema, used):
    paths = [environment._paths[key] for key in used if key in environment._paths]
    paths.extend(environment._source_scope_paths[key] for key in used if key in environment._source_scope_paths)
    return {
        key: dict(
            path=item.get("path", [key]),
            owner_type=item["owner_type"],
            **({"reference_path": item["reference_path"]} if "reference_path" in item else {}),
        )
        for key, item in schema.items()
        if item["kind"] in ("owner", "owner_input")
        and any(tuple(path[: len(item.get("path", [key]))]) == tuple(item.get("path", [key])) for path in paths)
    }


def layout_owner(environment, schema, name):
    from ._component import _capture_id

    path = environment._source_scope_paths.get(name, environment._paths[name])
    owners = [
        tuple(item.get("path", [key])) for key, item in schema.items() if item["kind"] in ("owner", "owner_input")
    ]
    owner = max((p for p in owners if tuple(path[: len(p)]) == p), key=len, default=())
    return dict(layout_owner_path=list(owner), layout_member=_capture_id(path[len(owner) :])) if owner else {}


class OwnerGraph:
    def __init__(self, sources, imported, module):
        from ._package_input_families import source_class

        self.imported, self.module = imported, module
        self._contexts = {}
        self._resource_contexts = {}
        self.definitions = {
            name + "." + owner: (name, source_class(tree, owner), copy.deepcopy(schema))
            for name, _, _, _, tree, _, schemas, _ in sources
            for owner, schema in schemas.items()
        }
        for identity in self.definitions:
            self._validate(identity, ())

    def context(self, identity, choices=None):
        """Reuse definition witnesses across kernels, never consumer instances."""
        key = identity, tuple(sorted((choices or {}).items()))
        if key not in self._contexts:
            self._contexts[key] = self.construct(identity, choices=choices)
        return self._contexts[key]

    def _validate(self, identity, ancestors):
        if identity not in self.definitions:
            raise ValueError("Declared child owner is outside the supported definition closure: " + identity)
        if identity in ancestors:
            raise ValueError("Declared owner construction is cyclic: " + " -> ".join((*ancestors, identity)))
        for item in self.definitions[identity][2].values():
            if item["kind"] == "owner":
                for choice in item.get("owner_alternatives", [item]):
                    self._validate(choice["owner_type"], (*ancestors, identity))

    def construct(self, identity, *, resource_only=False, references=(), choices=None, path=(), resource_paths=None):
        import taichi_forge as ti
        from taichi_forge.lang.matrix import MatrixType
        from taichi_forge_runtime._static_values import read_static_key
        from ._component import (
            Array,
            Record,
            RuntimeValue,
            StaticValue,
            _ChoiceRecord,
            _ConditionalArray,
            _ConditionalOwnerRecord,
            _InputPresence,
            _OptionalArray,
        )
        from ._package_fields import construct_fields, field_environment
        from ._package_records import write_member
        from ._package_types import resolve_type, type_spec, producer_static
        from ._package_borrows import project as project_borrows

        name, class_name, original = self.definitions[identity]
        schema = original
        source_children = {}
        if resource_paths is not None:
            from ._package_resource_slice import allocation_schema

            schema, source_children = allocation_schema(schema, resource_paths)
        resource_key = None
        if resource_only:
            token = identity, None if resource_paths is None else tuple(sorted(resource_paths))
            if token in references:
                raise ValueError(
                    "Cyclic input resource definitions require a resolved allocation path: "
                    + identity
                    + " at "
                    + ".".join(path)
                )
            references = (*references, token)
            if resource_paths is not None and len(references) == 1:
                # Many input families borrow different members of one native
                # allocation group. Reuse that complete build-only witness,
                # not a new copy of every sibling Field for each member.
                # Keep input slots distinct: two unrelated inputs in a single
                # owner must never acquire a false allocation alias. Relative
                # choices retain every layout/child selection of this source.
                # Nested borrows still traverse their cycle guards normally.
                resource_key = (
                    identity,
                    path,
                    tuple(schema),
                    tuple(sorted((child, tuple(sorted(paths))) for child, paths in source_children.items())),
                    tuple(
                        sorted(
                            (key[len(path) :], value)
                            for key, value in (choices or {}).items()
                            if key[: len(path)] == path
                        )
                    ),
                )
                if resource_key in self._resource_contexts:
                    return self._resource_contexts[resource_key]
        # The slice is read-only. Copy its retained declarations before any
        # per-witness selection, rather than cloning the entire source owner
        # once for each borrowed allocation.
        schema = copy.deepcopy(schema)
        static_values = {
            key: (choices or {}).get((*path, key), read_static_key(item["choices"][0]))
            for key, item in schema.items()
            if item["kind"] == "static"
        }

        def layout_static(policy):
            if "condition_path" not in policy:
                return static_values[policy["condition"]]
            remaining = tuple(policy["condition_path"])
            source = original
            while True:
                member = next(
                    (item for key, item in source.items() if tuple(item.get("path", (key,))) == remaining), None
                )
                if member is not None:
                    if member["kind"] != "static":
                        raise ValueError("field_like layout selection requires a declared static source")
                    return (choices or {}).get(
                        (*path, *policy["condition_path"]), read_static_key(member["choices"][0])
                    )
                child = next(
                    (
                        (tuple(item.get("path", (key,))), item)
                        for key, item in source.items()
                        if item["kind"] in ("owner", "owner_input")
                        and remaining[: len(item.get("path", (key,)))] == tuple(item.get("path", (key,)))
                    ),
                    None,
                )
                if child is None:
                    raise ValueError("field_like layout selection has no declared source path")
                prefix, item = child
                remaining = remaining[len(prefix) :]
                source = self.definitions[item["owner_type"]][2]

        for attribute, item in schema.items():
            if "source_policy" in item:
                policy = item["source_policy"]
                # Validate both complete allocation paths after child schemas
                # have been composed, including StructField members.
                selected = static_values[policy["condition"]]
                item["source_path"] = policy["if_true" if selected else "if_false"]
                item["source_guards"] = {policy["condition"]: selected}
            if "dtype_policy" in item:
                policy = item["dtype_policy"]
                selected = layout_static(policy)
                item["type_node"] = policy["if_true" if selected else "if_false"]
                item["dtype_guards"] = {policy["condition"]: selected}
            if "shape_policy" in item:
                policy = item["shape_policy"]
                selected = layout_static(policy)
                item["shape"] = policy["if_true" if selected else "if_false"]
                item["shape_guards"] = {policy["condition"]: selected}
            alternatives = item.get("owner_alternatives")
            if alternatives:
                applicable = [
                    option
                    for option in alternatives
                    if all(
                        static_values.get(key) is value
                        for key, value in option.get("conditions", {}).items()
                        if key not in item.get("conditions", {})
                    )
                ]
                if len(applicable) != 1:
                    raise ValueError("Child implementation choices must cover their declared guard domain")
                selected = applicable[0]
                item["owner_type"] = selected["owner_type"]
                # Constructor identity proofs belong to the selected call site,
                # just like its concrete child type. Do not lose Child(self)
                # when that child is selected by a normal Host branch.
                item["input_references"] = selected.get("input_references", {})
                item["choice_guards"] = {
                    key: value
                    for key, value in selected.get("conditions", {}).items()
                    if key not in item.get("conditions", {})
                }
        if resource_only:
            # An input borrows its existing allocation scope. It does not create
            # or recursively compile that scope's owned solver/workspace objects.
            if resource_paths is None:
                schema = {key: item for key, item in schema.items() if item["kind"] != "owner"}
        imported = self.imported[name]
        owner = object.__new__(getattr(imported, class_name))
        # Dimensions and field_like may read child allocation metadata before
        # this owner's fields are allocated. Build each declared child once;
        # attach owned children after the parent's record namespaces exist.
        children, child_values = {}, {}
        for attribute, item in schema.items():
            if item["kind"] in ("owner", "owner_input"):
                child_choices = dict(choices or {})
                child_schema = self.definitions[item["owner_type"]][2]
                inputs = [
                    (tuple(child_schema[member].get("path", (member,))), target,
                     child_schema[member]["owner_type"])
                    for member, target in item.get("input_references", {}).items()
                ]
                if item["kind"] == "owner_input" and "reference_path" in item:
                    inputs.append(((), item["reference_path"], item["owner_type"]))
                for member, target, source_type in inputs:
                    source_path = (*path, *target)
                    input_path = (*path, attribute, *member)
                    # A borrowed view of the same instance must materialize
                    # the same layout before numerical variant exploration.
                    # Include default static values as well as explicit choices.
                    for key, source in self.definitions[source_type][2].items():
                        if source["kind"] == "static":
                            child_choices[(*input_path, key)] = (choices or {}).get(
                                (*source_path, key), read_static_key(source["choices"][0]))
                    for selected_path, value in (choices or {}).items():
                        if selected_path[:len(source_path)] == source_path:
                            child_choices[(*input_path, *selected_path[len(source_path):])] = value
                child_paths = source_children.get(attribute) or (
                    (tuple(schema[item["field_input"]]["source_member"]),) if "field_input" in item else None
                )
                if child_paths is None and item["kind"] == "owner_input":
                    prefix = tuple(item.get("path", (attribute,)))
                    borrowed = []
                    for declaration in schema.values():
                        if declaration["kind"] != "borrowed_field":
                            continue
                        policy = declaration.get("source_policy")
                        candidates = (
                            (policy["if_true"], policy["if_false"]) if policy else (declaration["source_path"],)
                        )
                        borrowed.extend(
                            tuple(candidate[len(prefix) :])
                            for candidate in candidates
                            if tuple(candidate[: len(prefix)]) == prefix and len(candidate) > len(prefix)
                        )
                    if any(child_schema.get(p[0], {}).get("kind") == "owner" for p in borrowed):
                        # Keep the existing input view, and include only the
                        # owned allocations explicitly borrowed through it.
                        # Do not construct the input's unrelated solver graph.
                        child_paths = tuple(
                            dict.fromkeys(
                                [
                                    *(
                                        tuple(part.get("path", (key,)))
                                        for key, part in child_schema.items()
                                        if part["kind"] != "owner"
                                    ),
                                    *borrowed,
                                ]
                            )
                        )
                value, _, child, expanded = self.construct(
                    item["owner_type"],
                    resource_only=resource_only or item["kind"] == "owner_input",
                    references=references,
                    choices=child_choices,
                    path=(*path, attribute),
                    resource_paths=child_paths,
                )
                children[attribute] = (child, expanded)
                child_values[attribute] = value
                if item["kind"] == "owner_input":
                    write_member(owner, attribute, value, schema)

        def source_dimension(parts):
            # Owned child fields may supply a clone's dimensions before they are
            # attached to the parent. Reuse that same construction witness.
            if parts[0] in child_values:
                value, parts = child_values[parts[0]], parts[1:]
            else:
                value = owner
            for part in parts:
                value = getattr(value, part)
            return value

        declarations = construct_fields(owner, schema, vars(imported), self.module,
                                        imported_modules=self.imported, source_dimension=source_dimension)
        for attribute, item in schema.items():
            if item["kind"] in ("dense_field", "dimension", "borrowed_field", "host_owner_sequence"):
                continue
            if item["kind"] == "record":
                declarations[attribute] = Record(optional=item.get("optional", False))
                continue
            if item["kind"] in ("owner", "owner_input"):
                record = (
                    _ConditionalOwnerRecord(item["conditions"].items())
                    if item.get("conditions") else Record()
                )
                if item["kind"] == "owner_input":
                    declarations[attribute] = record
                    continue
                value = child_values[attribute]
                declarations[attribute] = (
                    _ChoiceRecord(item["choice_guards"].items(), optional=bool(item.get("conditions")))
                    if item.get("owner_alternatives")
                    else record
                )
            elif item["kind"] == "static":
                domain = tuple(producer_static(read_static_key(key)) for key in item["choices"])
                declarations[attribute] = StaticValue(choices=domain, kernel_only=item["kernel_only"])
                value = producer_static(static_values[attribute])
            elif item["kind"] == "constant":
                declarations[attribute] = StaticValue()
                value = read_static_key(item["value"])
            elif item["kind"] == "ndarray":
                namespace = vars(self.imported[item.get("declaration_module", name)])
                if "dtype_policy" in item:
                    from ._package_types import resolve_scalar_dtype_policy

                    item["dtype_policy"] = resolve_scalar_dtype_policy(item["dtype_policy"], namespace, self.module)
                dtype = resolve_type(item.pop("type_node"), namespace)
                item["element"] = type_spec(dtype, self.module)
                if item.get("external") and item["element"]["kind"] not in ("scalar", "tensor"):
                    raise ValueError("Typed external borrow requires scalar/vector/matrix elements")
                conditions = {**item.get("conditions", {}), **item.get("dtype_guards", {})}
                declarations[attribute] = (
                    _OptionalArray(dtype, item["ndim"], presence=item["input_presence_attribute"])
                    if item.get("external") and item.get("optional_input")
                    else (
                        _ConditionalArray(dtype, item["ndim"], conditions=tuple(conditions.items()))
                        if conditions
                        else Array(dtype, item["ndim"])
                    )
                )
                value = ti.ndarray(dtype, (1,) * item["ndim"])
            else:
                if "type_node" in item:
                    dtype = resolve_type(
                        item.pop("type_node"), vars(self.imported[item.get("declaration_module", name)])
                    )
                    item["element"] = type_spec(dtype, self.module)
                    if item["element"]["kind"] not in ("scalar", "tensor"):
                        raise ValueError("ResourceScope.value requires a scalar/vector/matrix type")
                    item["dtype"] = item["element"]["dtype"]
                else:
                    dtype = getattr(ti, item["dtype"])
                value_type = _InputPresence if item.get("input_presence") else RuntimeValue
                declarations[attribute] = value_type(dtype, affects_plan=item.get("affects_plan", False))
                value = dtype(0) if isinstance(dtype, MatrixType) else (0.0 if item["dtype"].startswith("f") else 0)
            write_member(owner, attribute, value, schema)
        from ._package_optional import project_input_records, project_boolean_groups

        project_input_records(owner, schema, declarations)
        project_boolean_groups(schema, declarations)
        environment, expanded = _compose(field_environment(declarations, schema), schema, children)
        from ._component import _capture_id

        for attribute, item in schema.items():
            for member, target in item.get("input_references", {}).items():
                child_schema = children[attribute][1]
                source = child_schema[member]
                key = _capture_id((attribute, *source.get("path", (member,))))
                expanded[key] = {**expanded[key], "reference_path": list(target)}
        from ._package_field_like import project_field_likes

        project_borrows(owner, environment, expanded)
        project_field_likes(environment, expanded)
        from ._package_optional import project_struct_member_inputs

        project_struct_member_inputs(environment, expanded)
        result = owner, schema, environment, expanded
        if resource_key is not None:
            self._resource_contexts[resource_key] = result
        return result


def _compose(environment, schema, children):
    """Qualify existing Component declarations; native layouts remain authoritative."""
    from dataclasses import replace
    from ._component import (
        Array,
        DenseField,
        Dimension,
        Record,
        _ConditionalRecord,
        _ChoiceRecord,
        _InputRecord,
        _ModelTensorField,
        _capture_id,
    )

    declarations, paths = dict(environment.declarations), dict(environment._paths)
    constraints, source_paths = dict(environment._constraints), dict(environment._source_scope_paths)
    derived, expanded = set(environment._derived_attributes), dict(schema)
    for attribute, (child, child_schema) in children.items():
        names = {key: _capture_id((attribute, *path)) for key, path in child._paths.items()}
        # Presence controls whether the namespace exists. Implementation choice
        # controls the logical owner binding, not every field's numerical ABI.
        # Each alternative still constructs and validates its original layout.
        owner_conditions = schema[attribute].get("conditions", {})

        def expression(value):
            op, *args = value
            if op == "dim":
                return op, names[args[0]]
            return value if op == "const" else (op, *(expression(arg) for arg in args))

        for key, original in child.declarations.items():
            value = original
            if isinstance(value, Dimension):
                value = Dimension(
                    value.kind,
                    names[value.capacity] if value.capacity else "",
                    expression(value.expression) if value.expression else (),
                )
            elif isinstance(value, DenseField):
                shape = tuple(names[n] if isinstance(n, str) else n for n in value.shape)
                group = attribute + "." + value.group
                changes = dict(shape=shape, group=group)
                if isinstance(value, _ModelTensorField):
                    changes["element_shape"] = tuple(names[n] if isinstance(n, str) else n for n in value.element_shape)
                if getattr(value, "conditions", ()):
                    changes["conditions"] = tuple((names[n], expected) for n, expected in value.conditions)
                value = replace(value, **changes)
            elif isinstance(value, Array):
                changes = dict(
                    shape=(
                        None
                        if value.shape is None
                        else tuple(names[n] if isinstance(n, str) else n for n in value.shape)
                    ),
                    resource_id=names[value.resource_id] if value.resource_id else "",
                )
                if getattr(value, "conditions", ()):
                    changes["conditions"] = tuple((names[n], expected) for n, expected in value.conditions)
                if getattr(value, "presence", ""):
                    changes["presence"] = names[value.presence]
                value = replace(value, **changes)
            elif isinstance(value, _ConditionalRecord):
                value = type(value)(tuple((names[n], expected) for n, expected in value.conditions))
            elif isinstance(value, _InputRecord):
                value = _InputRecord(names[value.presence])
            elif isinstance(value, _ChoiceRecord):
                value = _ChoiceRecord(
                    tuple((names[n], expected) for n, expected in value.guards), optional=value.optional
                )
            elif isinstance(value, Record):
                value = Record(optional=value.optional)
            declarations[names[key]], paths[names[key]] = value, (attribute, *child._paths[key])
        for key, binding in child._source_scope_paths.items():
            source_paths[names[key]] = (attribute, *binding)
        for key, constraint in child._constraints.items():
            spec = constraint["spec"]
            constraints[_capture_id((attribute, key))] = dict(
                source=constraint["source"],
                spec=dict(left=expression(spec["left"]), right=expression(spec["right"]), when=names[spec["when"]]),
            )
        derived.update(names[key] for key in child._derived_attributes)
        for key, item in child_schema.items():
            if item["kind"] == "host_owner_sequence":
                # Independent algorithm instances stay in their child's Host
                # contract. The collection has no device allocation/path to
                # compose into its parent; its entries are exported separately.
                continue
            transformed = {**item, "path": [attribute, *item.get("path", (key,))]}
            if "source_path" in item:
                transformed["source_path"] = [attribute, *item["source_path"]]
            if "source_policy" in item:
                policy = item["source_policy"]
                transformed["source_policy"] = {
                    "condition": names[policy["condition"]],
                    "if_true": [attribute, *policy["if_true"]],
                    "if_false": [attribute, *policy["if_false"]],
                    **(
                        {
                            "presence": {
                                branch: [
                                    {**predicate, "path": [attribute, *predicate["path"]]} for predicate in predicates
                                ]
                                for branch, predicates in policy["presence"].items()
                            }
                        }
                        if "presence" in policy
                        else {}
                    ),
                }
                transformed["source_guards"] = {names[n]: expected for n, expected in item["source_guards"].items()}
            if "input_presence_attribute" in item:
                transformed["input_presence_attribute"] = names[item["input_presence_attribute"]]
            if "input_record" in item:
                transformed["input_record"] = names[item["input_record"]]
            if "input_conditions" in item:
                transformed["input_conditions"] = [names[key] for key in item["input_conditions"]]
            if item.get("external"):
                transformed["shape_expressions"] = [expression(axis) for axis in item["shape_expressions"]]
            if "reference_path" in item:
                transformed["reference_path"] = [attribute, *item["reference_path"]]
            if "like_source_path" in item:
                transformed["like_source_path"] = [attribute, *item["like_source_path"]]
            if "source_presence" in item:
                transformed["source_presence"] = [
                    {**predicate, "path": [attribute, *predicate["path"]]} for predicate in item["source_presence"]
                ]
            if "conditions" in item:
                transformed["conditions"] = {names[n]: expected for n, expected in item["conditions"].items()}
            if "choice_guards" in item:
                transformed["choice_guards"] = {names[n]: expected for n, expected in item["choice_guards"].items()}
            if "shape_policy" in item:
                transformed["shape_policy"] = {
                    **item["shape_policy"],
                    "condition": names[item["shape_policy"]["condition"]],
                    **(
                        {"condition_path": [attribute, *item["shape_policy"]["condition_path"]]}
                        if "condition_path" in item["shape_policy"]
                        else {}
                    ),
                }
                transformed["shape_guards"] = {names[n]: expected for n, expected in item["shape_guards"].items()}
            if "dtype_policy" in item:
                transformed["dtype_policy"] = {
                    **item["dtype_policy"],
                    "condition": names[item["dtype_policy"]["condition"]],
                    **(
                        {"condition_path": [attribute, *item["dtype_policy"]["condition_path"]]}
                        if "condition_path" in item["dtype_policy"]
                        else {}
                    ),
                }
                transformed["dtype_guards"] = {names[n]: expected for n, expected in item["dtype_guards"].items()}
            if "owner_alternatives" in item:
                transformed["owner_alternatives"] = [
                    {
                        **choice,
                        "conditions": {names[n]: expected for n, expected in choice.get("conditions", {}).items()},
                    }
                    for choice in item["owner_alternatives"]
                ]
            if owner_conditions:
                transformed["conditions"] = {**transformed.get("conditions", {}), **owner_conditions}
            expanded[names[key]] = transformed
    environment._set_declarations(declarations)
    environment._paths = MappingProxyType(paths)
    environment._constraints = MappingProxyType(constraints)
    environment._source_scope_paths = MappingProxyType(source_paths)
    environment._derived_attributes = frozenset(derived)
    return environment, expanded
