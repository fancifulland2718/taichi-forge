"""Host tensor values and namespaces over native Field members.

Tensor values own CPU data only; Struct Field namespaces delegate transfers to
native members. Neither interprets numerical definitions or computes layouts.
"""

from collections.abc import Sequence
import numbers
import operator

from .._dimensions import evaluate_extent, check_constraints, validate_constraints
from .._static_values import read_static_key, static_key


def path_value(owner, path):
    for part in path:
        owner = getattr(owner, part)
    return owner


def dimension_value(owner, name):
    return path_value(owner, getattr(owner, "_forge_package_dimension_paths", {}).get(name, (name,)))


def owner_type(owner):
    return getattr(owner, "_forge_package_contract", getattr(type(owner), "_forge_package_type", None))


def selected_owner_declaration(owner, item):
    alternatives = item.get("owner_alternatives")
    if not alternatives:
        return item
    matches = [
        choice
        for choice in alternatives
        if all(getattr(owner, flag) is expected for flag, expected in choice.get("conditions", {}).items())
    ]
    if len(matches) != 1:
        raise ValueError("Child construction has missing or ambiguous declared guards")
    return {**item, **matches[0]}


def check_admission(owner, contract):
    admission = contract.get("admission")
    if admission:
        for binding in admission.get("constants", {}).values():
            check_constant(owner, binding["path"], binding["value"])
        for binding in admission.get("input_presence", {}).values():
            if (path_value(owner, binding["path"]) is not None) is not binding["present"]:
                raise ValueError("Optional Field input presence changed after construction")
        presence = {}
        for name, binding in admission.get("dynamic_input_presence", {}).items():
            actual = path_value(owner, binding["path"]) is not None
            if path_value(owner, binding["flag_path"]) is not actual:
                raise ValueError("Optional input presence changed outside its preparation binding")
            presence[name] = int(actual)

        def required(paths):
            return any(
                all(bool(evaluate_extent(expression, presence.__getitem__)) is truth for expression, truth in path)
                for path in paths
            )

        records = {}
        for name, record in admission.get("input_records", {}).items():
            records[name] = path_value(owner, record["path"]) is not None
            if not records[name] and required(record["required_when"]):
                raise ValueError("Optional record presence is required by the selected execution branch")
        for resource in admission.get("external_resources", {}).values():
            if resource.get("input_record") and not records[resource["input_record"]]:
                continue
            value = path_value(owner, resource["path"])
            if value is None and "required_when" in resource:
                if required(resource["required_when"]):
                    raise ValueError("Optional input presence is required by the selected execution branch")
                continue
            check_external_resource(owner, resource, value)
        for name, resource in admission.get("conditional_resources", {}).items():
            if not records[resource["input_record"]]:
                continue
            value = path_value(owner, resource["path"])
            allocated_owner = path_value(owner, resource.get("layout_owner_path", ()))
            member = resource.get("layout_member", name)
            schema = owner._forge_package_owner.manifest["owners"][owner_type(allocated_owner)]
            source = (allocated_owner, tuple(schema[member].get("path", (member,))))
            if getattr(value, "_forge_package_field_source", None) != source:
                raise ValueError("Conditional workspace no longer refers to its owned Field")
            check_external_resource(owner, resource, value)
        check_owner_bindings(owner, admission.get("owner_bindings", {}).values())
        for binding in admission.get("source_bindings", {}).values():
            for predicate in binding.get("presence", ()):
                actual = path_value(owner, predicate["path"])
                if predicate.get("record"):
                    actual = actual is not None
                if actual is not predicate["expected"]:
                    raise ValueError("Borrowed input source allocation presence changed")
            if path_value(owner, binding["path"]) is not path_value(owner, binding["source_path"]):
                raise ValueError("Borrowed Field no longer refers to its declared allocation")
        constraints, dimensions = admission["constraints"], admission["dimensions"]
        validate_constraints(constraints, dimensions.keys(), {spec["when"] for spec in constraints.values()})
        check_constraints(
            {
                key: spec
                for key, spec in constraints.items()
                if key not in admission.get("constraint_records", {}) or records[admission["constraint_records"][key]]
            },
            lambda name: capture_value(owner, dict(name=name, dimension_relations=dimensions)),
        )


def check_external_resource(owner, contract, value):
    """Check the caller's storage contract, without requiring allocation provenance."""
    from ._runtime import NdArray

    value = owner._forge_package_owner.runtime._adapt_package_array(value)
    if not isinstance(value, NdArray):
        raise ValueError("External resource requires an explicitly imported device array or Field view")
    value._descriptor(owner._forge_package_owner.runtime)
    element = contract["element"]

    def dimension(key):
        if "dimensions" in contract:
            return capture_value(owner, dict(name=key, dimension_relations=contract["dimensions"]))
        return dimension_value(owner, key)

    expected = tuple(evaluate_extent(axis, dimension) for axis in contract["shape_expressions"])
    if (
        value.dtype != element["dtype"]
        or value.element_shape != tuple(element.get("element_shape", ()))
        or value.shape != expected
    ):
        raise ValueError("External resource dtype, element shape or logical shape differs from its declaration")


def check_constant(owner, path, value):
    if static_key(path_value(owner, path)) != static_key(read_static_key(value)):
        raise ValueError(f"Definition constant {'.'.join(path)!r} changed; rebuild the package")


def check_owner_bindings(owner, bindings):
    for binding in bindings:
        child = path_value(owner, binding["path"])
        if "reference_path" in binding and child is not path_value(owner, binding["reference_path"]):
            raise ValueError("Owner reference no longer refers to its declared source")
        if (
            getattr(child, "_forge_package_owner", None) is not owner._forge_package_owner
            or owner_type(child) != binding["owner_type"]
        ):
            raise ValueError("Declared child owner differs from its package definition")


def capture_value(owner, spec):
    root = owner
    if spec.get("nullable_record") and path_value(owner, spec["nullable_record"]) is None:
        if "optional_array" in spec:
            return root._forge_package_owner._empty_input(spec["optional_array"])
        # The record access is unreachable according to admission. This scalar
        # only completes the immutable ABI snapshot; it is not a model value.
        return 0
    relations = spec.get("dimension_relations")
    if relations:

        def resolve(name, visiting=()):
            if name in visiting or name not in relations:
                raise ValueError("Invalid or cyclic package dimension relation")
            declaration = relations[name]
            actual = owner
            for part in declaration["path"]:
                actual = getattr(actual, part)
            actual = operator.index(actual)
            expression = declaration["expression"]
            value = (
                evaluate_extent(expression, lambda child: resolve(child, (*visiting, name))) if expression else actual
            )
            if not (1 if declaration["kind"] == "extent" else 0) <= value <= 2**31 - 1:
                raise ValueError(f"Dimension {name!r} is outside its declared i32 range")
            if value != actual:
                raise ValueError(f"Derived dimension {name!r} does not match its normal attribute")
            return value

        return resolve(spec["name"])
    for name in spec.get("path", (spec["name"],)):
        owner = getattr(owner, name)
    if owner is None and "optional_array" in spec:
        # A valid zero-work descriptor is an ABI token, not fabricated model
        # storage. Admission has already ruled out every active read/write.
        return root._forge_package_owner._empty_input(spec["optional_array"])
    if spec.get("kind") == "model_tensor_field":
        expected = tuple(
            (
                capture_value(root, dict(name=axis, dimension_relations=spec["model_dimension_relations"]))
                if isinstance(axis, str)
                else axis
            )
            for axis in spec["model_element_shape"]
        )
        if owner.element_shape != expected:
            raise ValueError("Model tensor shape differs from its allocated Field")
        view = getattr(owner, "_forge_package_model_scalar_view", None)
        if view is None:
            raise ValueError("Model tensor capture requires its native scalar-axis Field view")
        return view
    return owner


class StructField:
    """Host namespace over native Field members; no independent storage ABI."""

    def __init__(self, members):
        self.field_dict = dict(members)

    def __getattr__(self, name):
        try:
            return self.field_dict[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def shape(self):
        return next(iter(self.field_dict.values())).shape

    def to_numpy(self):
        return {name: field.to_numpy() for name, field in self.field_dict.items()}

    def from_numpy(self, arrays):
        if not isinstance(arrays, dict) or arrays.keys() != self.field_dict.keys():
            raise ValueError("Struct Field upload requires its complete member mapping")
        for name, field in self.field_dict.items():
            if getattr(arrays[name], "shape", None) != field.shape + field.element_shape:
                raise ValueError("Struct Field upload shape differs from its declared member")
        for name, field in self.field_dict.items():
            field.from_numpy(arrays[name])


def _numpy_dtype(dtype):
    if dtype is None:
        return None
    if dtype not in ("f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"):
        raise TypeError("Host tensor dtype must be a supported scalar type")
    return {"f": "float", "i": "int", "u": "uint"}[dtype[0]] + dtype[1:]


class HostTensor(Sequence):
    def __init__(self, values, dt=None):
        import numpy as np

        if not isinstance(values, (list, tuple, np.ndarray)):
            raise TypeError("Host tensor construction requires a list, tuple or NumPy array")
        self.entries = np.array(values, dtype=_numpy_dtype(dt), copy=True)
        if self.entries.ndim not in (1, 2) or self.entries.dtype.kind not in "bfiu" or not self.entries.size:
            raise TypeError("Host tensor values require a nonempty rank-one or rank-two numeric array")
        self.ndim = self.entries.ndim
        self.n = self.entries.shape[0]
        self.m = self.entries.shape[1] if self.ndim == 2 else 1

    def __len__(self):
        return self.n

    def __iter__(self):
        if self.ndim == 1:
            return iter(self.entries)
        return (row.tolist() for row in self.entries)

    def _indices(self, indices):
        indices = tuple(indices) if isinstance(indices, (tuple, list)) else (indices,)
        if len(indices) != self.ndim:
            raise IndexError(f"Expected {self.ndim} Host tensor indices, got {len(indices)}")
        if any(not isinstance(index, numbers.Integral) for index in indices):
            raise TypeError("Host tensor indexing requires integer indices")
        return indices

    def __getitem__(self, indices):
        return self.entries[self._indices(indices)]

    def __setitem__(self, indices, value):
        self.entries[self._indices(indices)] = value

    def get_shape(self):
        return self.entries.shape

    def to_numpy(self):
        return self.entries

    def to_list(self):
        return self.entries.tolist()

    def __bool__(self):
        raise TypeError("Host tensor truth evaluation is not supported by package projection")

    def __eq__(self, other):
        raise TypeError("Host tensor comparison is not supported by package projection")

    def __ne__(self, other):
        return self.__eq__(other)


def _component(index):
    def get(self):
        if self.ndim != 1 or index >= self.n:
            raise AttributeError("Host vector component is outside its shape")
        return self[index]

    def set(self, value):
        get(self)
        self[index] = value

    return property(get, set)


for _index, _name in enumerate("xyzw"):
    setattr(HostTensor, _name, _component(_index))


def typed_tensor(shape, dtype, args):
    import numpy as np

    if not args:
        raise TypeError("Typed Host tensor construction requires an initial value")
    if len(args) == 1 and isinstance(args[0], numbers.Number):
        values = np.full(shape, args[0], dtype=_numpy_dtype(dtype))
    else:
        values = args[0] if len(args) == 1 else args
        if isinstance(values, HostTensor):
            values = values.to_list()
        flattened = []
        for value in values:
            if isinstance(value, HostTensor):
                flattened.extend(value.entries.ravel())
            elif isinstance(value, (list, tuple, np.ndarray)):
                flattened.extend(np.asarray(value).ravel())
            else:
                flattened.append(value)
        values = np.asarray(flattened, dtype=_numpy_dtype(dtype)).reshape(shape)
    return HostTensor(values)
