"""Fixed-factor recordings and pointer-free CPU/CUDA resource export."""

from taichi_forge._lib import core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect, RuntimeBinding
from taichi_forge.graph._native import BackendCommandRecording, NativeGraphExecutable, NativeGraphNode
from taichi_forge.graph._native_definition import (
    NativeArrayBinding,
    NativeOperationBinding,
    NativeResourceBinding,
    NativeResourceReference,
)
from taichi_forge.hardware._native_adapter import native_recording_node, validate_exact_bindings
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError


class SparseSolveRecording(BackendCommandRecording):
    def __init__(self, solver, rhs, solution):
        if solver.selected_provider not in ("eigen", "cusolver_sp") or solver.matrix is None:
            raise TaichiRuntimeError("SparseSolver.record_solve requires a factored Eigen or cuSOLVERSp solver")
        if rhs == solution:
            raise ValueError("SparseSolver RHS and solution names must differ")
        solver.matrix._ensure_valid()
        solver.solver.validate_factorization(solver.matrix.matrix)
        super().__init__(
            backend="cpu" if solver.selected_provider == "eigen" else "cuda",
            binding_names=(rhs, solution),
            command_count=1,
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=solver.selected_provider == "eigen" or solver._solver_type == "LLT",
        )
        object.__setattr__(self, "_solver", solver)
        object.__setattr__(self, "_matrix", solver.matrix)
        object.__setattr__(self, "_program", impl.get_runtime().prog)
        object.__setattr__(self, "_generation", solver._factorization_generation)
        object.__setattr__(self, "_definition", solver._definition)
        object.__setattr__(self, "_operation", solver._definition.solve(solver.selected_provider))
        object.__setattr__(
            self,
            "_arrays",
            (
                NativeArrayBinding("rhs_0", rhs, GraphAccess.READ),
                NativeArrayBinding("output_0", solution, GraphAccess.WRITE),
            ),
        )

    def validate_graph_lifetime(self):
        solver = self._solver
        if (
            impl.get_runtime().prog is not self._program
            or solver.matrix is not self._matrix
            or solver._factorization_generation != self._generation
        ):
            raise TaichiRuntimeError("SparseSolver recording factor changed; record a new cold generation")
        self._matrix._ensure_valid()
        solver.solver.validate_factorization(self._matrix.matrix)

    def validate_graph_bindings(self, bindings):
        self.validate_graph_lifetime()
        validate_exact_bindings(self, bindings, "SparseSolver solve")
        for name in self.binding_names:
            value = bindings[name]
            if (
                not isinstance(value, Ndarray)
                or value.dtype != self._solver.dtype
                or tuple(value.shape) != (self._matrix.n,)
                or tuple(value.element_shape)
            ):
                raise TaichiRuntimeError("SparseSolver recording requires matching one-dimensional scalar ndarrays")

    @property
    def resource_effects(self):
        return tuple(array.effect for array in self._arrays) + (
            ResourceEffect(self._solver, GraphAccess.READ_WRITE, runtime_bound=False),
        )

    def execute(self, bindings):
        self.validate_graph_bindings(bindings)
        rhs, output = (bindings[name] for name in self.binding_names)
        self._solver.solver.solve_rf(self._program, self._matrix.matrix, rhs.arr, output.arr)
        self._solver._record_solve_dispatch(None)

    def _export_aot(self, context):
        self.validate_graph_lifetime()
        solver = self._solver
        implementation = solver.selected_provider
        cpu = implementation == "eigen"
        if context.arch not in ((core.Arch.x64, core.Arch.arm64) if cpu else (core.Arch.cuda,)):
            raise TaichiRuntimeError("SparseSolver recording and AOT target backends must match")

        def snapshot():
            if cpu:
                rows, count, dtype, offsets, indices, values = core._snapshot_eigen_sparse_solver(
                    solver.solver, self._matrix.matrix
                )
            else:
                rows, columns, offsets, indices, values = self._matrix.matrix._snapshot_cuda_csr(self._program)
                if columns != rows:
                    raise TaichiRuntimeError("SparseSolver export requires a square CSR matrix")
                count, dtype = len(values) // 4, "f32"
                self.validate_graph_lifetime()
            arrays = {}
            for name, scalar, payload, size in (
                ("offsets", "i32", offsets, rows + 1),
                ("indices", "i32", indices, count),
                ("values", dtype, values, count),
            ):
                arrays[name] = context.data_resource(object(), payload, dtype=scalar, shape=[size], readonly=True)
            return NativeResourceBinding(
                solver,
                self._definition.resource(implementation, rows, count, dtype),
                arrays=tuple(NativeArrayBinding(port, name, GraphAccess.READ) for port, name in arrays.items()),
            ).lower(context)

        resource = context.resource_once(solver, snapshot, policy=(implementation, self._generation))
        NativeOperationBinding(self._operation, self._arrays, (("plan", NativeResourceReference(resource)),)).lower(
            context
        )
        context.reconstructed_binding_validators.add(id(self))
        return ()

    def _as_graph_native_node(self):
        if not self.no_host_readback:
            return _HostSparseSolveNode(self)
        return native_recording_node(
            self,
            lifetime_leases=(self._solver, self._matrix),
            debug_info=dict(kind="sparse_direct_solve", implementation=self._solver.selected_provider),
        )


class _HostSparseSolveNode(NativeGraphNode, NativeGraphExecutable):
    """Explicit root host boundary; never a no-readback device recording."""

    def __init__(self, recording):
        self._recording = recording

    def compile(self):
        return self

    def run(self, runtime_args):
        self._recording.execute(runtime_args)

    def validate_graph_lifetime(self):
        self._recording.validate_graph_lifetime()

    def validate_graph_bindings(self, runtime_args):
        self._recording.validate_graph_bindings(runtime_args)

    @property
    def runtime_arg_schema(self):
        return tuple(RuntimeBinding(name, "ndarray") for name in self._recording.binding_names)

    @property
    def resource_effects(self):
        return self._recording.resource_effects

    @property
    def lifetime_leases(self):
        return (self._recording,)

    @property
    def debug_info(self):
        return dict(kind="sparse_direct_host_lu", implementation="cusolver_sp")

    def _export_aot(self, context):
        self._recording._export_aot(context)
        context.reconstructed_binding_validators.add(id(self))
