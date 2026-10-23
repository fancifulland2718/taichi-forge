"""Fixed-binding native primitives, without data-mutating preparation."""

import hashlib
import json

from taichi_forge._lib import core as _ti_core
from taichi_forge.algorithms import _autodiff
from taichi_forge.lang import impl
from taichi_forge.lang._storage_view import describe_storage
from taichi_forge.lang.exception import TaichiRuntimeError


class _PreparedNativePlan:
    """Shared fixed-binding lifetime and recording, not a new dispatch registry."""

    def _validate_lifetime(self):
        if self._command is None:
            raise TaichiRuntimeError(f"{type(self).__name__} is closed")
        if impl.get_runtime().prog is not self._program:
            raise TaichiRuntimeError(f"{type(self).__name__} belongs to another runtime")

    def run(self):
        """Execute on current contents without host readback."""
        self._validate_lifetime()
        _autodiff.reject_unsupported_automatic_ad(self._ad_kind)
        self._workspace_bytes = self._execute(self._command)

    def close(self):
        """Drop bindings without clearing shared scratch or waiting for GPU work."""
        self._command = None
        self._owners = ()
        self._native_binding = None
        self._execute = None
        self._program = None

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, *_):
        self.close()

    def report(self):
        from ._prepared_compact import PreparedCompactPlan

        persistent_available = (
            type(self) in (PreparedSortPlan, PreparedCompactPlan)
            and self._command is not None
            and self._backend in ("cuda", "vulkan")
            and not self._program.config().debug
            and (self._backend != "vulkan" or not self._program.config().kernel_profiler)
            and self._definition.parameters.get("nan_policy", "last") == "last"
            and callable(getattr(self._program, "_record_primitive_" + self._definition.operation, None))
        )
        return {
            **self._description,
            "physical_plan_id": self._physical_id,
            "closed": self._command is None,
            "binding_policy": "fixed_dense_storage",
            "replay_mode": "rerecord",
            "stream_binding": "runtime_ordered",
            "workspace_owner": getattr(self, "_workspace_owner", "program_primitive_arena"),
            "workspace_preparation": "lazy_first_execution",
            "workspace_bytes_last_observed": self._workspace_bytes,
            "device_capture": False,
            "persistent_recording_available": persistent_available,
        }

    def record(self, *, persistent=False):
        """Return a fixed-binding root-Graph action.

        The action retains these fixed bindings. Explicitly closing this plan
        invalidates its recordings; an enclosing Graph may use segmented or
        ordinary execution. With ``persistent=True``, GPU NaNs-last sort and
        compact explicitly allocate a private recording workspace. That
        immutable snapshot retains its bindings independently of this plan;
        closing the source plan does not invalidate the snapshot. Runtime reset
        or destruction of a referenced Field tree still retires its Graph.
        """
        self._validate_lifetime()
        if type(persistent) is not bool:
            raise TypeError("persistent must be a bool")
        if persistent:
            from ._prepared_recording import persistent_recording

            return persistent_recording(self)
        return _native_recording(self)


class PreparedSortPlan(_PreparedNativePlan):
    """A stable ascending in-place sort over fixed CPU/CUDA/Vulkan dense storage.

    Keys are compact 1D scalar ranges; optional payloads are equally sized
    scalar/vector/matrix ranges with i32/u32/f32/i64/u64/f64 lanes. Ndarrays and
    qualified dense fields/views share this path. Equal keys preserve
    payload order. CPU/Vulkan support ``nan_policy="last"`` only; CUDA also
    supports ``"bitwise"`` (the existing sortable-bit order). Other algorithms,
    descending order and raw provider selection are not options of this plan.

    Preparation validates layout, access and lifetime without sorting, issuing
    GPU work, synchronizing or allocating scratch. Backend pipelines/workspace
    remain lazy; GPU scratch is Program-owned and CPU scratch is plan-owned,
    so the first run is not a steady-state timing sample. GPU workspace growth
    can synchronize as in ordinary sort.

    Contents may change in place between runs. Replacement, shape/layout change
    or runtime reset requires a new plan. ``close()`` invalidates this plan and
    ordinary recordings made from it, but does not clear shared Program scratch
    or wait for the GPU. Explicit ``record(persistent=True)`` snapshots have an
    independent lifetime and private scratch. Existing runtime completion owns
    in-flight allocation release.
    """

    _ad_kind = "sort"

    def __init__(self, keys, values=None, *, nan_policy="last"):
        from taichi_forge.graph._ir import GraphAccess
        from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding

        if nan_policy not in ("last", "bitwise"):
            raise ValueError("Prepared sort nan_policy must be 'last' or 'bitwise'")
        owners = (keys,) if values is None else (keys, values)
        definition = NativeDefinition("builtin", "sort", {"nan_policy": nan_policy})
        native_binding = NativeOperationBinding(
            definition,
            tuple(
                NativeArrayBinding(port, owner, GraphAccess.READ_WRITE)
                for port, owner in zip(("keys", "values"), owners)
            ),
        )
        descriptions = tuple(describe_storage(binding.value, access="readwrite") for binding in native_binding.arrays)
        for name, description in zip(("keys", "values"), descriptions):
            if not description.supported:
                raise TaichiRuntimeError(
                    f"Prepared sort {name} is not supported dense storage: {description.failure_reason}"
                )
        program = impl.get_runtime().prog
        prepare = getattr(program, "_prepare_primitive_sort", None)
        if prepare is None:
            raise TaichiRuntimeError("Prepared sort requires native prepared-primitive support")
        command = prepare(
            descriptions[0].descriptor,
            None if values is None else descriptions[1].descriptor,
            0 if definition.parameters["nan_policy"] == "last" else 1,
        )
        self._definition, self._native_binding = definition, native_binding
        self._program = program
        self._owners = owners
        self._command = command
        self._execute = program._execute_primitive_sort
        self._workspace_bytes = None
        arch = impl.current_cfg().arch
        self._backend = "cuda" if arch == _ti_core.Arch.cuda else "vulkan" if arch == _ti_core.Arch.vulkan else "cpu"
        self._workspace_owner = "prepared_plan" if self._backend == "cpu" else "program_primitive_arena"
        self._description = {
            "kind": "stable_sort" if self._backend == "cpu" else "stable_radix_sort",
            "backend": self._backend,
            "count": int(descriptions[0].descriptor.index_shape[0]),
            "key_dtype": str(descriptions[0].descriptor.scalar_type),
            "value_dtype": None if values is None else str(descriptions[1].descriptor.scalar_type),
            "value_element_shape": [] if values is None else list(descriptions[1].descriptor.element_shape),
            "nan_policy": definition.parameters["nan_policy"],
            "ascending": True,
            "stable": True,
        }
        canonical = json.dumps(self._description, sort_keys=True, separators=(",", ":"))
        self._physical_id = "prepared-native-sort-v1:" + hashlib.sha256(canonical.encode()).hexdigest()
        self._workspace_effect = "forge-native-sort-workspace"


def prepare_sort(keys, values=None, *, count=None, nan_policy="last"):
    """Prepare a stable full-range or device-count prefix sort.

    With ``count``, use :class:`PreparedPrefixSortPlan`: preallocated scratch,
    preserved tail, device overflow status, and a capturable Graph action.
    """
    if count is not None:
        from ._prepared_prefix_sort import PreparedPrefixSortPlan

        return PreparedPrefixSortPlan(keys, values, count=count, nan_policy=nan_policy)
    return PreparedSortPlan(keys, values, nan_policy=nan_policy)


def _native_recording(plan):
    # Construct the executable recording only when requested. Definitions and
    # bound roles never replace the direct prepared command on ordinary replay.
    from taichi_forge.graph._ir import GraphAccess, ResourceEffect
    from taichi_forge.graph._native import BackendCommandRecording
    from taichi_forge.hardware._native_adapter import native_recording_node

    class PrimitiveRecording(BackendCommandRecording):
        def __init__(self):
            super().__init__(
                backend=plan._backend,
                binding_names=plan._native_binding.binding_names,
                command_count=1,
                barrier_policy="internal",
                workspace_ownership="provider_generation",
                replay_mode="rerecord",
            )
            object.__setattr__(self, "_plan", plan)
            # Freeze effects before close() can release the plan's references.
            object.__setattr__(
                self,
                "_effects",
                plan._native_binding.resource_effects
                + (ResourceEffect(plan._workspace_effect, GraphAccess.READ_WRITE, runtime_bound=False),),
            )
            object.__setattr__(self, "_graph_physical_plan_id", plan._physical_id)

        @property
        def resource_effects(self):
            return self._effects

        def execute(self, bindings):
            self._plan.run()

        def prepare_graph_execute(self, bindings):
            self._plan._validate_lifetime()
            return self._plan.run

        def _export_aot(self, context):
            from ._prepared_export import export_prepared

            export_prepared(self._plan, context)
            return ()

        def _as_graph_native_node(self):
            return native_recording_node(
                self,
                runtime_bindings=(),
                lifetime_leases=(self._plan,),
                debug_info=self._plan._description,
            )

    return PrimitiveRecording()


__all__ = ["PreparedSortPlan", "prepare_sort"]
