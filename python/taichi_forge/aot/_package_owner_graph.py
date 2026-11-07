"""Resolve declared construction types without executing application constructors."""

import copy
from types import MappingProxyType


def owner_bindings(environment, schema, used):
    paths = [environment._paths[key] for key in used if key in environment._paths]
    paths.extend(environment._source_scope_paths[key] for key in used if key in environment._source_scope_paths)
    return {
        key: dict(path=item.get("path", [key]), owner_type=item["owner_type"])
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
        self.imported, self.module = imported, module
        self.definitions = {
            name + "." + owner: (name, owner, copy.deepcopy(schema))
            for name, _, _, _, _, _, schemas, _ in sources
            for owner, schema in schemas.items()
        }
        for identity in self.definitions:
            self._validate(identity, ())

    def _validate(self, identity, ancestors):
        if identity not in self.definitions:
            raise ValueError("Declared child owner is outside the supported definition closure: " + identity)
        if identity in ancestors:
            raise ValueError("Declared owner construction is cyclic: " + " -> ".join((*ancestors, identity)))
        for item in self.definitions[identity][2].values():
            if item["kind"] == "owner":
                self._validate(item["owner_type"], (*ancestors, identity))

    def construct(self, identity, *, resource_only=False, references=()):
        import taichi_forge as ti
        from taichi_forge.lang.matrix import MatrixType
        from taichi_forge_runtime._static_values import read_static_key
        from ._component import Array, Record, RuntimeValue, StaticValue
        from ._package_fields import construct_fields, field_environment
        from ._package_records import write_member
        from ._package_types import resolve_type, type_spec
        from ._package_borrows import project as project_borrows

        name, class_name, original = self.definitions[identity]
        schema = copy.deepcopy(original)
        if resource_only:
            # An input borrows its existing allocation scope. It does not create
            # or recursively compile that scope's owned solver/workspace objects.
            if identity in references:
                raise ValueError("Cyclic input resource definitions require a resolved allocation path")
            references = (*references, identity)
            schema = {key: item for key, item in schema.items() if item["kind"] != "owner"}
        imported = self.imported[name]
        owner = object.__new__(getattr(imported, class_name))
        declarations = construct_fields(owner, schema, vars(imported), self.module, imported_modules=self.imported)
        children = {}
        for attribute, item in schema.items():
            if item["kind"] in ("dense_field", "dimension", "borrowed_field"):
                continue
            if item["kind"] == "record":
                declarations[attribute] = Record()
                continue
            if item["kind"] in ("owner", "owner_input"):
                value, _, child, expanded = self.construct(
                    item["owner_type"], resource_only=item["kind"] == "owner_input", references=references
                )
                declarations[attribute] = Record()
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
                declarations[attribute] = Array(dtype, item["ndim"])
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
                declarations[attribute] = RuntimeValue(dtype, affects_plan=item.get("affects_plan", False))
                value = dtype(0) if isinstance(dtype, MatrixType) else (0.0 if item["dtype"].startswith("f") else 0)
            write_member(owner, attribute, value, schema)
        environment, expanded = _compose(field_environment(declarations, schema), schema, children)
        project_borrows(owner, environment, expanded)
        return owner, schema, environment, expanded


def _compose(environment, schema, children):
    """Qualify existing Component declarations; native layouts remain authoritative."""
    from ._component import Array, DenseField, Dimension, Record, _ConditionalDenseField, _ModelTensorField, _capture_id

    declarations, paths = dict(environment.declarations), dict(environment._paths)
    constraints, source_paths = dict(environment._constraints), dict(environment._source_scope_paths)
    derived, expanded = set(environment._derived_attributes), dict(schema)
    for attribute, (child, child_schema) in children.items():
        names = {key: _capture_id((attribute, *path)) for key, path in child._paths.items()}

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
                value = (
                    _ModelTensorField(
                        value.dtype,
                        shape,
                        group,
                        tuple(names[n] if isinstance(n, str) else n for n in value.element_shape),
                    )
                    if isinstance(value, _ModelTensorField)
                    else (
                        _ConditionalDenseField(
                            value.dtype, shape, group, tuple((names[n], expected) for n, expected in value.conditions)
                        )
                        if isinstance(value, _ConditionalDenseField)
                        else DenseField(value.dtype, shape, group)
                    )
                )
            elif isinstance(value, Array):
                value = Array(
                    value.dtype,
                    value.ndim,
                    None if value.shape is None else tuple(names[n] if isinstance(n, str) else n for n in value.shape),
                    names[value.resource_id] if value.resource_id else "",
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
            if "conditions" in item:
                transformed["conditions"] = {names[n]: expected for n, expected in item["conditions"].items()}
            expanded[names[key]] = transformed
    environment._set_declarations(declarations)
    environment._paths = MappingProxyType(paths)
    environment._constraints = MappingProxyType(constraints)
    environment._source_scope_paths = MappingProxyType(source_paths)
    environment._derived_attributes = frozenset(derived)
    return environment, expanded
