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

    def construct(self, identity, *, resource_only=False, references=(), choices=None, path=()):
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
            _InputPresence,
            _OptionalArray,
        )
        from ._package_fields import construct_fields, field_environment
        from ._package_records import write_member
        from ._package_types import resolve_type, type_spec
        from ._package_borrows import project as project_borrows

        name, class_name, original = self.definitions[identity]
        schema = copy.deepcopy(original)
        static_values = {
            key: (choices or {}).get((*path, key), read_static_key(item["choices"][0]))
            for key, item in schema.items()
            if item["kind"] == "static"
        }
        for attribute, item in schema.items():
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
                item["choice_guards"] = {
                    key: value
                    for key, value in selected.get("conditions", {}).items()
                    if key not in item.get("conditions", {})
                }
        if resource_only:
            # An input borrows its existing allocation scope. It does not create
            # or recursively compile that scope's owned solver/workspace objects.
            if identity in references:
                raise ValueError("Cyclic input resource definitions require a resolved allocation path")
            references = (*references, identity)
            schema = {key: item for key, item in schema.items() if item["kind"] != "owner"}
        imported = self.imported[name]
        owner = object.__new__(getattr(imported, class_name))
        # Dimension relations may read a non-owning source before this owner's
        # fields are allocated. Construct only those resource definitions first;
        # owned algorithms remain in their original phase below.
        children = {}
        for attribute, item in schema.items():
            if item["kind"] == "owner_input":
                value, _, child, expanded = self.construct(
                    item["owner_type"], resource_only=True, references=references
                )
                children[attribute] = (child, expanded)
                write_member(owner, attribute, value, schema)
        declarations = construct_fields(owner, schema, vars(imported), self.module, imported_modules=self.imported)
        for attribute, item in schema.items():
            if item["kind"] in ("dense_field", "dimension", "borrowed_field", "host_owner_sequence"):
                continue
            if item["kind"] == "record":
                declarations[attribute] = Record(optional=item.get("optional", False))
                continue
            if item["kind"] in ("owner", "owner_input"):
                if item["kind"] == "owner_input":
                    declarations[attribute] = Record(optional=bool(item.get("conditions")))
                    continue
                value, _, child, expanded = self.construct(
                    item["owner_type"], references=references, choices=choices, path=(*path, attribute)
                )
                declarations[attribute] = (
                    _ChoiceRecord(item["choice_guards"].items(), optional=bool(item.get("conditions")))
                    if item.get("owner_alternatives")
                    else Record(optional=bool(item.get("conditions")))
                )
                children[attribute] = (child, expanded)
            elif item["kind"] == "static":
                choices = tuple(read_static_key(key) for key in item["choices"])
                declarations[attribute] = StaticValue(choices=choices, kernel_only=item["kernel_only"])
                value = choices[0]
            elif item["kind"] == "constant":
                declarations[attribute] = StaticValue()
                value = read_static_key(item["value"])
            elif item["kind"] == "ndarray":
                dtype = resolve_type(item.pop("type_node"), vars(self.imported[item.get("declaration_module", name)]))
                item["element"] = type_spec(dtype, self.module)
                if item.get("external") and item["element"]["kind"] not in ("scalar", "tensor"):
                    raise ValueError("Typed external borrow requires scalar/vector/matrix elements")
                declarations[attribute] = (
                    _OptionalArray(dtype, item["ndim"], presence=item["input_presence_attribute"])
                    if item.get("external") and item.get("optional_input")
                    else (
                        _ConditionalArray(dtype, item["ndim"], conditions=tuple(item["conditions"].items()))
                        if item.get("external") and item.get("conditions")
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
        from ._package_optional import project_input_records

        project_input_records(owner, schema, declarations)
        environment, expanded = _compose(field_environment(declarations, schema), schema, children)
        from ._package_field_like import project_field_likes

        project_borrows(owner, environment, expanded)
        project_field_likes(environment, expanded)
        return owner, schema, environment, expanded


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
        owner_conditions = {**schema[attribute].get("conditions", {}), **schema[attribute].get("choice_guards", {})}

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
                value = _ConditionalRecord(tuple((names[n], expected) for n, expected in value.conditions))
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
            transformed = {**item, "path": [attribute, *item.get("path", (key,))]}
            if "source_path" in item:
                transformed["source_path"] = [attribute, *item["source_path"]]
            if "input_presence_attribute" in item:
                transformed["input_presence_attribute"] = names[item["input_presence_attribute"]]
            if "input_record" in item:
                transformed["input_record"] = names[item["input_record"]]
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
