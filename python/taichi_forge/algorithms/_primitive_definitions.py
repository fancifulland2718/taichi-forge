"""Cold native algorithm semantics, independent of execution and export owners."""

from dataclasses import dataclass, field
import struct

from taichi_forge.types.primitive_types import f32, f64, i32, i64, u32, u64

NATIVE_VALUE_TYPE = {i32: 0, f32: 1, u32: 2, u64: 3, i64: 4, f64: 5}
HISTOGRAM_VALUE_TYPE = {i32: 0, u32: 2}
HISTOGRAM_BIN_TYPE = {i32: 0, i64: 4}
GROUPED_REDUCE_OPS = {"sum": 0}
CHECK_OPS = {"nonzero": 0, "zero": 1, "nan": 2, "inf": 3, "not_finite": 4, "index_oob": 5}
METRIC_OPS = {"max_abs": 0, "max_abs_delta": 1}
METRIC_VALUE_TYPE = {f32: 1, f64: 5}


@dataclass(frozen=True)
class BucketBuilderDefinition:
    size: int
    num_bins: int
    dtype: object
    element_shape: tuple = ()
    raw_payload: bool = False

    def __post_init__(self):
        object.__setattr__(self, "element_shape", tuple(self.element_shape))

    @property
    def value_type(self):
        # The existing native driver receives the actual raw element byte size
        # from the ndarray. Its scalar tag remains i32 for opaque payloads.
        return 0 if self.raw_payload else NATIVE_VALUE_TYPE[self.dtype]

    def bind(self, keys, values, offsets, output, *, cpu_max_threads=None, vulkan_policy=None):
        from taichi_forge.graph._ir import GraphAccess
        from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding

        if self.raw_payload:
            raise ValueError("AOT bucket requires a packed numeric payload")
        parameters = dict(
            size=self.size, num_bins=self.num_bins, dtype=str(self.dtype), element_shape=self.element_shape
        )
        if cpu_max_threads is not None:
            parameters["cpu_max_threads"] = cpu_max_threads
        if vulkan_policy is not None:
            parameters["vulkan_policy"] = vulkan_policy
        return NativeOperationBinding(
            NativeDefinition("builtin", "bucket_builder", parameters),
            (
                NativeArrayBinding("keys", keys, GraphAccess.READ),
                NativeArrayBinding("values", values, GraphAccess.READ),
                NativeArrayBinding("offsets", offsets, GraphAccess.WRITE),
                # Invalid keys leave an unused tail in output unchanged.
                NativeArrayBinding("output", output, GraphAccess.READ_WRITE),
            ),
        )


def _diagnostic_binding(operation, parameters, values, output, other, *, cpu_max_threads, vulkan_items_per_group):
    from taichi_forge.graph._ir import GraphAccess
    from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding

    if cpu_max_threads is not None:
        parameters["cpu_max_threads"] = cpu_max_threads
    if vulkan_items_per_group is not None:
        parameters["vulkan_items_per_group"] = vulkan_items_per_group
    arrays = (
        NativeArrayBinding("values", values, GraphAccess.READ),
        NativeArrayBinding("output", output, GraphAccess.WRITE),
    )
    if other is not None:
        arrays += (NativeArrayBinding("other", other, GraphAccess.READ),)
    return NativeOperationBinding(NativeDefinition("builtin", operation, parameters), arrays)


@dataclass(frozen=True)
class CheckDefinition:
    size: int
    dtype: object
    op: str
    lower: int = 0
    upper: int = 0

    def __post_init__(self):
        object.__setattr__(self, "lower", int(self.lower))
        object.__setattr__(self, "upper", int(self.upper))

    @property
    def value_type(self):
        return NATIVE_VALUE_TYPE[self.dtype]

    @property
    def op_id(self):
        return CHECK_OPS[self.op]

    def bind(self, values, output, *, cpu_max_threads=None, vulkan_items_per_group=None):
        return _diagnostic_binding(
            "check_count",
            dict(size=self.size, dtype=str(self.dtype), op=self.op, lower=self.lower, upper=self.upper),
            values,
            output,
            None,
            cpu_max_threads=cpu_max_threads,
            vulkan_items_per_group=vulkan_items_per_group,
        )


@dataclass(frozen=True)
class MetricDefinition:
    size: int
    dtype: object
    op: str

    @property
    def value_type(self):
        return METRIC_VALUE_TYPE[self.dtype]

    @property
    def op_id(self):
        return METRIC_OPS[self.op]

    def bind(self, values, output, *, other=None, cpu_max_threads=None, vulkan_items_per_group=None):
        return _diagnostic_binding(
            "metric_reduce",
            dict(size=self.size, dtype=str(self.dtype), op=self.op),
            values,
            output,
            other,
            cpu_max_threads=cpu_max_threads,
            vulkan_items_per_group=vulkan_items_per_group,
        )


@dataclass(frozen=True)
class GroupedReduceDefinition:
    size: int
    num_groups: int
    dtype: object
    op: str = "sum"

    def __post_init__(self):
        if self.op not in GROUPED_REDUCE_OPS:
            raise ValueError(f"grouped reduce op must be one of {sorted(GROUPED_REDUCE_OPS)}.")
        if self.dtype not in NATIVE_VALUE_TYPE:
            raise TypeError("unsupported grouped reduce dtype")

    @property
    def value_type(self):
        return NATIVE_VALUE_TYPE[self.dtype]

    @property
    def op_id(self):
        return GROUPED_REDUCE_OPS[self.op]

    def bind(self, keys, values, output, *, cpu_max_threads=None):
        from taichi_forge.graph._ir import GraphAccess
        from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding

        parameters = dict(size=self.size, num_groups=self.num_groups, dtype=str(self.dtype))
        if cpu_max_threads is not None:
            parameters["cpu_max_threads"] = cpu_max_threads
        return NativeOperationBinding(
            NativeDefinition("builtin", "grouped_reduce", parameters),
            (
                NativeArrayBinding("keys", keys, GraphAccess.READ),
                NativeArrayBinding("values", values, GraphAccess.READ),
                NativeArrayBinding("output", output, GraphAccess.WRITE),
            ),
        )


@dataclass(frozen=True)
class HistogramDefinition:
    size: int
    num_bins: int
    value_dtype: object
    counter_dtype: object

    def __post_init__(self):
        if self.value_dtype not in HISTOGRAM_VALUE_TYPE:
            raise TypeError("unsupported histogram value dtype")
        if self.counter_dtype not in HISTOGRAM_BIN_TYPE:
            raise TypeError("unsupported histogram bin dtype")

    @property
    def value_type(self):
        return HISTOGRAM_VALUE_TYPE[self.value_dtype]

    @property
    def bin_type(self):
        return HISTOGRAM_BIN_TYPE[self.counter_dtype]

    def bind(self, values, bins, *, cpu_max_threads=None, vulkan_policy=None):
        from taichi_forge.graph._ir import GraphAccess
        from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding

        parameters = dict(
            size=self.size,
            num_bins=self.num_bins,
            value_dtype=str(self.value_dtype),
            counter_dtype=str(self.counter_dtype),
        )
        if cpu_max_threads is not None:
            parameters["cpu_max_threads"] = cpu_max_threads
        if vulkan_policy is not None:
            parameters["vulkan_policy"] = vulkan_policy
        return NativeOperationBinding(
            NativeDefinition("builtin", "histogram", parameters),
            (
                NativeArrayBinding("values", values, GraphAccess.READ),
                NativeArrayBinding("bins", bins, GraphAccess.WRITE),
            ),
        )


@dataclass(frozen=True)
class IndexedCopyDefinition:
    size: int
    src_items: int
    dst_items: int
    dtype: object
    element_shape: tuple
    scatter: bool

    @staticmethod
    def check_size(size, src_items, dst_items, scatter):
        if scatter and src_items != size:
            raise ValueError("experimental_scatter() expects source and indices sizes to match.")
        if not scatter and dst_items != size:
            raise ValueError("experimental_gather() expects indices and destination sizes to match.")
        return size

    def __post_init__(self):
        object.__setattr__(self, "element_shape", tuple(self.element_shape))
        self.check_size(self.size, self.src_items, self.dst_items, self.scatter)

    @property
    def mode(self):
        return "scatter" if self.scatter else "gather"

    @property
    def value_type(self):
        return NATIVE_VALUE_TYPE.get(self.dtype)

    def bind(self, src, indices, dst):
        from taichi_forge.graph._ir import GraphAccess
        from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding

        return NativeOperationBinding(
            NativeDefinition(
                "builtin",
                "indexed_copy",
                dict(
                    size=self.size,
                    src_items=self.src_items,
                    dst_items=self.dst_items,
                    dtype=str(self.dtype),
                    element_shape=self.element_shape,
                    op=self.mode,
                ),
            ),
            (
                NativeArrayBinding("src", src, GraphAccess.READ),
                NativeArrayBinding("indices", indices, GraphAccess.READ),
                # Scatter preserves destinations not named by valid indices.
                NativeArrayBinding("dst", dst, GraphAccess.READ_WRITE if self.scatter else GraphAccess.WRITE),
            ),
        )


@dataclass(frozen=True)
class ScatterAddDefinition:
    size: int
    dst_items: int
    dtype: object
    element_shape: tuple = ()

    @staticmethod
    def value_type_for(dtype):
        if dtype not in NATIVE_VALUE_TYPE:
            raise TypeError("unsupported scatter_add dtype")
        return NATIVE_VALUE_TYPE[dtype]

    def __post_init__(self):
        object.__setattr__(self, "element_shape", tuple(self.element_shape))
        self.value_type_for(self.dtype)

    @property
    def value_type(self):
        return self.value_type_for(self.dtype)

    def bind(self, src, indices, dst, *, strategy=None, cpu_max_threads=None):
        from taichi_forge.graph._ir import GraphAccess
        from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding

        parameters = dict(size=self.size, dst_items=self.dst_items, dtype=str(self.dtype))
        if strategy is not None:
            parameters["strategy"] = strategy
        if self.element_shape:
            parameters["element_shape"] = self.element_shape
        if cpu_max_threads is not None:
            parameters["cpu_max_threads"] = cpu_max_threads
        return NativeOperationBinding(
            NativeDefinition("builtin", "scatter_add", parameters),
            (
                NativeArrayBinding("src", src, GraphAccess.READ),
                NativeArrayBinding("indices", indices, GraphAccess.READ),
                NativeArrayBinding("dst", dst, GraphAccess.READ_WRITE),
            ),
        )


@dataclass(frozen=True)
class ScanDefinition:
    """The inclusive scalar scan performed by a selected native call.

    Storage addressing and backend admission remain with the caller. In
    particular, a JIT member-view call does not imply AOT member-view support.
    """

    size: int
    dtype: object

    def __post_init__(self):
        if self.dtype not in NATIVE_VALUE_TYPE:
            raise RuntimeError("unsupported PrefixSumExecutor ndarray dtype.")

    @property
    def value_type(self):
        return NATIVE_VALUE_TYPE[self.dtype]

    def bind(self, values):
        from taichi_forge.graph._ir import GraphAccess
        from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding

        return NativeOperationBinding(
            NativeDefinition("builtin", "scan", dict(size=self.size, dtype=str(self.dtype))),
            (NativeArrayBinding("values", values, GraphAccess.READ_WRITE),),
        )


REDUCE_OPS = {"sum": 0, "min": 1, "max": 2}


def _as_integral_transform_arg(name, value, *, bits, signed):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"experimental_transform() integer {name} must be integral.") from exc
    if result != value:
        raise TypeError(f"experimental_transform() integer {name} must be integral.")
    lo = -(1 << (bits - 1)) if signed else 0
    hi = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    if result < lo or result > hi:
        raise ValueError(f"experimental_transform() integer {name} is out of range for {'i' if signed else 'u'}{bits}.")
    if bits == 64 and abs(result) > (1 << 53) - 1:
        raise ValueError(
            "experimental_transform() 64-bit integer scale/bias must be exactly "
            "representable by the current native binding."
        )
    return result


def normalize_transform_args(dtype, scale, bias):
    if dtype == i32:
        return (
            _as_integral_transform_arg("scale", scale, bits=32, signed=True),
            _as_integral_transform_arg("bias", bias, bits=32, signed=True),
        )
    if dtype == u32:
        return (
            _as_integral_transform_arg("scale", scale, bits=32, signed=False),
            _as_integral_transform_arg("bias", bias, bits=32, signed=False),
        )
    if dtype == i64:
        return (
            _as_integral_transform_arg("scale", scale, bits=64, signed=True),
            _as_integral_transform_arg("bias", bias, bits=64, signed=True),
        )
    if dtype == u64:
        return (
            _as_integral_transform_arg("scale", scale, bits=64, signed=False),
            _as_integral_transform_arg("bias", bias, bits=64, signed=False),
        )
    if dtype == f32:
        return float(scale), float(bias)
    if dtype == f64:
        return float(scale), float(bias)
    raise TypeError(
        "experimental_transform() currently supports ti.u32, ti.i32, ti.f32, "
        "ti.u64, ti.i64, and ti.f64 ndarray values."
    )


@dataclass(frozen=True)
class ReduceDefinition:
    size: int
    dtype: object
    op: str

    def __post_init__(self):
        if self.op not in REDUCE_OPS:
            raise ValueError(f"reduce op must be one of {sorted(REDUCE_OPS)}.")
        if self.dtype not in NATIVE_VALUE_TYPE:
            raise TypeError("unsupported reduce dtype")

    @property
    def value_type(self):
        return NATIVE_VALUE_TYPE[self.dtype]

    @property
    def op_id(self):
        return REDUCE_OPS[self.op]

    def bind(self, values, output, *, strategy="packed_array", element_shape=(), cpu_max_threads=None):
        from taichi_forge.graph._ir import GraphAccess
        from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding

        parameters = dict(size=self.size, dtype=str(self.dtype), op=self.op)
        if strategy != "packed_array":
            parameters["strategy"] = strategy
        if element_shape:
            parameters["element_shape"] = tuple(element_shape)
        if cpu_max_threads is not None:
            parameters["cpu_max_threads"] = cpu_max_threads
        return NativeOperationBinding(
            NativeDefinition("builtin", "reduce", parameters),
            (
                NativeArrayBinding("values", values, GraphAccess.READ),
                NativeArrayBinding("output", output, GraphAccess.WRITE),
            ),
        )


@dataclass(frozen=True)
class TransformDefinition:
    size: int
    dtype: object
    scale: object = field(compare=False)
    bias: object = field(compare=False)
    scale_bits: str = field(init=False)
    bias_bits: str = field(init=False)

    def __post_init__(self):
        scale, bias = normalize_transform_args(self.dtype, self.scale, self.bias)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "bias", bias)
        # Preserve signed zero and non-finite floating coefficients. These are
        # legal transform inputs, encoded as bits rather than JSON numbers.
        object.__setattr__(self, "scale_bits", struct.pack(">d", float(scale)).hex())
        object.__setattr__(self, "bias_bits", struct.pack(">d", float(bias)).hex())

    @property
    def value_type(self):
        return NATIVE_VALUE_TYPE[self.dtype]

    def bind(self, src, dst, *, strategy=None, element_shape=()):
        from taichi_forge.graph._ir import GraphAccess
        from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding

        parameters = dict(size=self.size, dtype=str(self.dtype), scale_bits=self.scale_bits, bias_bits=self.bias_bits)
        if strategy is not None:
            parameters["strategy"] = strategy
        if element_shape:
            parameters["element_shape"] = tuple(element_shape)
        return NativeOperationBinding(
            NativeDefinition("builtin", "transform", parameters),
            (NativeArrayBinding("src", src, GraphAccess.READ), NativeArrayBinding("dst", dst, GraphAccess.WRITE)),
        )
