"""Prepared stable compaction with one device prefix shared across columns."""

import hashlib
import json

from taichi_forge._lib import core as _ti_core
from taichi_forge.algorithms._prepared_primitive import _PreparedNativePlan
from taichi_forge.lang import impl
from taichi_forge.lang._storage_view import describe_storage
from taichi_forge.lang.exception import TaichiRuntimeError


class PreparedCompactPlan(_PreparedNativePlan):
    """Compact records with nonzero i32 flags, preserving input order.

    Bindings must be compact, naturally aligned CUDA/Vulkan dense ranges.
    Values/output are matching scalar, vector or matrix records with 32/64-bit
    scalar lanes. Output capacity must cover the input capacity. Count is a
    scalar i32 field or an i32 vector; the result stays in count[0] (or count[None]
    for scalar fields). Only output[:count] is defined; input data is unchanged.

    Preparation performs no mathematics or GPU submission. In-place content
    updates reuse the plan; new storage/layout requires preparation again.
    Native scratch remains lazy and Program-owned. This is an explicit
    runtime-ordered primitive, not a capture-safe command or a raw search axis.
    """

    _ad_kind = "compact"

    def __init__(self, values, flags, output, count):
        self._prepare_columns(((values, output),), flags, count)

    @classmethod
    def _from_columns(cls, columns, flags, count):
        columns = tuple(tuple(column) for column in columns)
        if not columns or any(len(column) != 2 for column in columns):
            raise ValueError("Compact columns require one or more (values, output) pairs")
        plan = cls.__new__(cls)
        plan._prepare_columns(columns, flags, count)
        return plan

    def _prepare_columns(self, columns, flags, count):
        from taichi_forge.graph._ir import GraphAccess
        from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding

        owners = (flags, count, *(value for column in columns for value in column))
        access = ("read", "write", *(mode for _ in columns for mode in ("read", "write")))
        definition = NativeDefinition("builtin", "compact", {"column_count": len(columns)})
        ports = (
            "flags",
            "count",
            *(port for i in range(definition.parameters["column_count"]) for port in (f"values_{i}", f"output_{i}")),
        )
        native_binding = NativeOperationBinding(
            definition,
            tuple(
                NativeArrayBinding(port, owner, GraphAccess(mode)) for port, owner, mode in zip(ports, owners, access)
            ),
        )
        descriptions = tuple(
            describe_storage(binding.value, access="read" if binding.access == GraphAccess.READ else "readwrite")
            for binding in native_binding.arrays
        )
        for description in descriptions:
            if not description.supported:
                raise TaichiRuntimeError(f"Prepared compact storage is unsupported: {description.failure_reason}")
        descriptors = tuple(description.descriptor for description in descriptions)
        program = impl.get_runtime().prog
        prepare = getattr(program, "_prepare_primitive_compact", None)
        if prepare is None:
            raise TaichiRuntimeError("Prepared compact requires native prepared-compaction support")
        command = prepare(descriptors[2::2], descriptors[0], descriptors[3::2], descriptors[1])
        self._definition, self._native_binding = definition, native_binding
        self._program, self._command = program, command
        self._execute = program._execute_primitive_compact
        self._owners = owners
        self._workspace_bytes = None
        self._workspace_effect = "forge-native-compact-workspace"
        self._backend = "cuda" if impl.current_cfg().arch == _ti_core.Arch.cuda else "vulkan"
        self._description = {
            "kind": "stable_compact",
            "backend": self._backend,
            "capacity": int(descriptors[0].index_shape[0]),
            "columns": tuple(
                {"dtype": str(descriptor.scalar_type), "element_shape": tuple(descriptor.element_shape)}
                for descriptor in descriptors[2::2]
            ),
            "predicate": "nonzero_i32",
            "prefix_evaluations": 1,
            "count_location": "device",
        }
        canonical = json.dumps(self._description, sort_keys=True, separators=(",", ":"))
        self._physical_id = "prepared-native-compact-v1:" + hashlib.sha256(canonical.encode()).hexdigest()


def prepare_compact(values, flags, output, count):
    """Prepare stable fixed-binding compaction; see :class:`PreparedCompactPlan`."""
    return PreparedCompactPlan(values, flags, output, count)


def prepare_compact_columns(columns, flags, count):
    """Prepare stable selection of aligned columns with one shared flag scan.

    ``columns`` is a nonempty iterable of ``(values, output)`` pairs. Each
    input has the flag capacity; each output has at least that capacity.
    Columns may have different scalar types or record shapes. Every pair
    preserves the storage, alias, output-prefix and lifetime contract of
    :class:`PreparedCompactPlan`. One device count applies to all outputs.
    Bindings are frozen at preparation; execution reads their current contents.
    """
    return PreparedCompactPlan._from_columns(columns, flags, count)


__all__ = ["PreparedCompactPlan", "prepare_compact", "prepare_compact_columns"]
