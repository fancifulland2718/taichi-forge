"""Host tensor values and namespaces over native Field members.

Tensor values own CPU data only; Struct Field namespaces delegate transfers to
native members. Neither interprets numerical definitions or computes layouts.
"""

from collections.abc import Sequence
import numbers
import operator

from .._dimensions import evaluate_extent


def capture_value(owner, spec):
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
