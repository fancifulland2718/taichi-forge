"""Cold adoption of a complete JIT recipe into an independent CUDA instance."""

import ctypes

from taichi_forge._lib import core
from taichi_forge.graph._lowering import GraphLoweringContext, lower_frozen_graph
from taichi_forge.graph._lowering_array_sources import SourceArrayBindings
from taichi_forge.graph._lowering_arrays import RetainedArrayBindings
from taichi_forge.graph._lowering_constraints import export_constraints
from taichi_forge.graph._lowering_resource_bindings import ResourceInputBindings
from taichi_forge.graph._lowering_resource_sources import SourceResourceBindings
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray, StructNdarray
from taichi_forge.lang.argpack import ArgPack
from taichi_forge.lang._storage_view import DenseNdarrayView, ndarray_view
from taichi_forge.lang.field import ScalarField
from taichi_forge.lang.matrix import MatrixField
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge_runtime.aot import ExecutionPlan, Scalar
from taichi_forge_runtime.aot._domain import CudaExecutionDomain
from taichi_forge_runtime.aot._runtime import Module


def execution_domain(*, capacity=2, instance_capacity=64, priority=0, gpu_timing=False):
    """Create a bounded persistent CUDA queue on the current JIT Program.

    Prepare immutable instances with ``graph.prepare(..., execution_domain=...)``.
    Explicit close/reset drains accepted work. Dropping a domain handle only
    releases that handle; the Program retains its resources until retirement.
    """
    from taichi_forge.aot import borrow_runtime

    owner = impl.get_runtime()
    with owner._graph_execution_runtime_lock:
        if impl.current_cfg().arch != core.Arch.cuda:
            raise NotImplementedError("Graph execution domains require CUDA")
        runtime = owner._graph_execution_runtime
        if runtime is None:
            runtime = borrow_runtime()
            owner._graph_execution_runtime = runtime
        return runtime.execution_domain(
            capacity=capacity, instance_capacity=instance_capacity, priority=priority, gpu_timing=gpu_timing
        )


class _JitRetainedArrays(RetainedArrayBindings):
    def declare_snapshot(self, *args, **kwargs):
        return self.context.data_resource()

    def require(self, owner):
        if not isinstance(owner, (Ndarray, DenseNdarrayView, ScalarField, MatrixField)):
            return super().require(owner)
        context = self.context
        identity = id(owner)
        previous = context.live_names.get(identity)
        if previous is not None:
            return previous
        view = owner if isinstance(owner, (Ndarray, DenseNdarrayView)) else ndarray_view(owner)
        name = context._name("live")
        context.live_arrays[name] = view
        context.live_names[identity] = name
        return name


class _JitContext(GraphLoweringContext):
    def __init__(self, graph, arguments):
        spec = graph._spec
        super().__init__(
            core.Arch.cuda,
            set(spec.runtime_arg_names) | set(spec.fixed_runtime_args),
            source_arrays=SourceArrayBindings(None, spec.fixed_runtime_args),
            source_resources=SourceResourceBindings(
                {name: value for name, value in arguments.items() if isinstance(value, ArgPack)}, spec.fixed_runtime_args
            ),
            resource_inputs=ResourceInputBindings(None, spec.fixed_runtime_args),
        )
        self.imports = []
        self.scalar_types = {}
        self.live_arrays = {}
        self.live_names = {id(value): name for name, value in arguments.items() if isinstance(value, (Ndarray, DenseNdarrayView))}
        self.live_blobs = {}
        self.retained_arrays = _JitRetainedArrays(self)

    def data_resource(self, *args, **kwargs):
        raise TaichiRuntimeError("JIT preparation requires a live storage loan for retained Field views/provider data")

    provider_data_binding = data_resource

    def host_blob_resource(self, data):
        if type(data) is not bytes:
            raise TypeError("JIT host blobs must be immutable bytes")
        name = self._name("blob")
        self.live_blobs[name] = data
        return name

    def _kernel_entry(self, kernel, launch_abi, jit_source):
        if jit_source is None:
            raise TaichiRuntimeError(
                "JIT preparation requires a frozen physical dispatch"
            )
        graph, index = jit_source
        # A shared Kernel can occur with different frozen launch policies.
        # Native import borrows the selected dispatch, never recompiles Kernel.
        key = (id(graph), index, launch_abi)
        name = self._kernel_names.get(key)
        if name is None:
            name = self._name("kernel")
            self._kernel_names[key] = name
            self.imports.append((name, graph, index, launch_abi != "ordinary"))
        return name

    def field_state(self, dependencies):
        for dependency in dependencies:
            dependency = tuple(dependency)
            if (
                self._state_dependencies.setdefault(dependency[0], dependency)
                != dependency
            ):
                raise TaichiRuntimeError(
                    "JIT Graph has conflicting Field lifetime dependencies"
                )
        # The native loan validates these generations against the compiled
        # payload and retires dependent domains before freeing a live root.

    def kernel_binding(self, argument):
        name = super().kernel_binding(argument)
        if (
            argument.tag == core.ArgKind.SCALAR
            and argument.name not in self._scalar_values
        ):
            dtype = str(argument.dtype())
            if self.scalar_types.setdefault(name, dtype) != dtype:
                raise TaichiRuntimeError(
                    "One Graph scalar binding has conflicting compiled types"
                )
        return name


class GraphExecution:
    """An immutable complete Graph instance bound to one CUDA execution domain.

    The native closure survives source Graph.close(). Closing this host handle
    leaves its bounded resources retained until domain close or Program reset.
    Explicit adjoint kernels are supported; automatic Tape/FwdMode is not.
    """

    def __init__(self, prepared, owner):
        self._prepared = prepared
        self._owner = owner

    def _begin(self):
        owner = self._owner
        if owner is not impl.get_runtime():
            raise TaichiRuntimeError("Graph execution was prepared before ti.reset()")
        count = owner._active_graph_submissions
        if (
            count < 0
            or owner.target_tape is not None
            or owner.fwd_mode_manager is not None
        ):
            raise TaichiRuntimeError(
                "Graph execution cannot run inside or during setup of automatic AD"
            )
        # Publish before ctypes/pybind releases the GIL. AD setup uses the same
        # counter as ordinary Graph submissions; no global submission lock.
        owner._active_graph_submissions = count + 1

    def initialize(self):
        self._begin()
        try:
            self._prepared.initialize()
        finally:
            self._owner._active_graph_submissions -= 1

    def submit(self, *, wait_for=(), ready_only=False, values=None, sample_gpu=False):
        """Return an exact completion, or None if busy, without a host wait."""
        self._begin()
        try:
            return self._prepared.submit(wait_for=wait_for, ready_only=ready_only, values=values, sample_gpu=sample_gpu)
        finally:
            self._owner._active_graph_submissions -= 1

    @property
    def info(self):
        return self._prepared.info

    def close(self):
        self._prepared.close()

    def __enter__(self):
        self._prepared._check()
        return self

    def __exit__(self, *_):
        self.close()


def prepare_graph(graph, arguments, *, execution_domain, mutable_parameters=()):
    """Resolve immutable bindings once; submit reuses the native closure."""
    if not isinstance(execution_domain, CudaExecutionDomain):
        raise TypeError("execution_domain must be a CudaExecutionDomain")
    runtime = execution_domain._runtime
    if runtime._borrower is None or runtime.arch != "cuda":
        raise ValueError("A JIT Graph requires a domain borrowed from its CUDA Program")
    runtime._check()
    with graph._lifecycle_lock:
        graph._check_runtime_valid()
        spec = graph._spec
        if (
            graph._execution_arch != "cuda"
            or graph._workspace_lane_capacity != 1
            or graph._qualified_fusion_selector is not None
            or getattr(spec, "_binding_executor_factory", None) is not None
        ):
            raise TaichiRuntimeError(
                "JIT preparation requires a fixed single-frame CUDA recipe"
            )
        arguments, version = graph._snapshot_binding_source(arguments)
        spec._validate_runtime_arg_names(arguments, "Graph.prepare")
        context = _JitContext(graph, arguments)
        lower_frozen_graph(context, graph)
        plan = ExecutionPlan(
            context.operations,
            initialization=context.initialization or None,
            resources=context.resources,
            constraints=export_constraints(
                spec, context.binding, additional=context.binding_constraints
            )
            or None,
        )
        if (
            set(plan.argument_names)
            != (set(spec.runtime_arg_names) - context.source_resources.argpacks.used)
            | context.live_arrays.keys() | context.live_blobs.keys()
        ):
            raise TaichiRuntimeError(
                "JIT preparation changed the complete Graph binding contract"
            )
        if not context.imports:
            raise TaichiRuntimeError(
                "JIT preparation requires a nonempty physical kernel recipe"
            )
        # Reject unsupported inputs before importing code or storage. Original
        # fixed workspace is never silently shared between independent calls.
        arguments = dict(
            ((name, value) for name, value in arguments.items() if name not in context.source_resources.argpacks.used),
            **context.live_arrays,
        )
        for name, value in arguments.items():
            if name in context.scalar_types:
                if type(value) not in (bool, int, float):
                    raise TypeError("Graph scalar inputs require bool, int or float")
            elif not isinstance(value, (Ndarray, DenseNdarrayView)) or isinstance(
                value, StructNdarray
            ):
                raise TypeError(
                    "Graph preparation requires scalar/tensor JIT ndarrays or dense views"
                )
            elif (
                isinstance(value, Ndarray) and value.arr is None
            ) or value._runtime_prog is not impl.get_runtime().prog:
                raise TaichiRuntimeError(
                    "Graph preparation requires arrays from the current Program"
                )
        # Native import takes SNode lifecycle before Program submission locks.
        # It must not run inside a borrowed Runtime scope (the inverse order).
        module = Module(runtime, runtime._borrower.import_jit_kernels(context.imports))
        imported = {}
        prepared = None
        try:

            def borrow(array):
                if not isinstance(array, (Ndarray, DenseNdarrayView)) or isinstance(
                    array, StructNdarray
                ):
                    raise TypeError(
                        "Graph gradients require scalar/tensor JIT ndarrays"
                    )
                if (
                    isinstance(array, Ndarray) and array.arr is None
                ) or array._runtime_prog is not impl.get_runtime().prog:
                    raise TaichiRuntimeError(
                        "Graph storage belongs to a retired or different Program"
                    )
                identity = (
                    ("view", id(array.descriptor))
                    if isinstance(array, DenseNdarrayView)
                    else array._runtime_allocation_identity
                )
                if identity not in imported:
                    imported[identity] = (
                        runtime.import_dense_view(array)
                        if isinstance(array, DenseNdarrayView)
                        else runtime.import_ndarray(array)
                    )
                return imported[identity]

            bindings = {}
            gradients = {}
            for name, value in arguments.items():
                dtype = context.scalar_types.get(name)
                if dtype is not None:
                    if dtype in ("f16", "f32"):
                        value = ctypes.c_float(float(value)).value
                    bindings[name] = Scalar(dtype, value)
                else:
                    bindings[name] = borrow(value)
                    if value.grad is not None:
                        gradients[name] = borrow(value.grad)
            for name, payload in context.live_blobs.items():
                blob = runtime.host_blob(payload)
                imported[("host_blob", name)] = bindings[name] = blob
            with runtime._lock:
                prepared = module.prepare_execution(
                    plan,
                    bindings,
                    gradients=gradients,
                    execution_domain=execution_domain,
                    mutable_parameters=mutable_parameters,
                )
                # Keep source wrappers until cold domain retirement as well as
                # native code/storage leases. Dropping a call must not trigger
                # source Graph capture destruction or resource frees in submit.
                owners = (*prepared._owners, graph, tuple(arguments.values()), version)
                prepared._owners = owners
                runtime._domain_owners[execution_domain._native_handle()][
                    prepared._handle
                ] = owners
            return GraphExecution(prepared, impl.get_runtime())
        except BaseException:
            if prepared is not None:
                prepared.close()
            module.close()
            for array in imported.values():
                array.close()
            raise
