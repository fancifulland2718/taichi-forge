"""Fixed-factor CPU recordings and pointer-free resource export."""

from taichi_forge._lib import core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.hardware._native_adapter import native_recording_node, validate_exact_bindings
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError


class SparseSolveRecording(BackendCommandRecording):
    def __init__(self, solver, rhs, solution):
        if solver.selected_provider != "eigen" or solver.matrix is None:
            raise TaichiRuntimeError("SparseSolver.record_solve currently requires a factored CPU Eigen solver")
        if rhs == solution:
            raise ValueError("SparseSolver RHS and solution names must differ")
        solver.matrix._ensure_valid()
        solver.solver.validate_factorization(solver.matrix.matrix)
        super().__init__(
            backend="cpu",
            binding_names=(rhs, solution),
            command_count=1,
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
        )
        object.__setattr__(self, "_solver", solver)
        object.__setattr__(self, "_matrix", solver.matrix)
        object.__setattr__(self, "_program", impl.get_runtime().prog)
        object.__setattr__(self, "_generation", solver._factorization_generation)

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
        return (
            ResourceEffect(self.binding_names[0], GraphAccess.READ),
            ResourceEffect(self.binding_names[1], GraphAccess.WRITE),
            ResourceEffect(self._solver, GraphAccess.READ_WRITE, runtime_bound=False),
        )

    def execute(self, bindings):
        self.validate_graph_bindings(bindings)
        rhs, output = (bindings[name] for name in self.binding_names)
        self._solver.solver.solve_rf(self._program, self._matrix.matrix, rhs.arr, output.arr)
        self._solver._record_solve_dispatch(None)

    def _export_aot(self, context):
        self.validate_graph_lifetime()
        if context.arch not in (core.Arch.x64, core.Arch.arm64):
            raise TaichiRuntimeError("CPU SparseSolver recording requires a CPU AOT target")
        solver = self._solver

        def snapshot():
            rows, count, dtype, offsets, indices, values = core._snapshot_eigen_sparse_solver(
                solver.solver, self._matrix.matrix
            )
            arrays = {}
            for name, scalar, payload, size in (
                ("offsets", "i32", offsets, rows + 1),
                ("indices", "i32", indices, count),
                ("values", dtype, values, count),
            ):
                arrays[name] = context.data_resource(object(), payload, dtype=scalar, shape=[size], readonly=True)
            return context.resource(
                solver,
                "sparse_direct",
                "plan",
                parameters=dict(
                    rows=rows,
                    nonzeros=count,
                    dtype=dtype,
                    implementation="eigen",
                    method=solver._solver_type,
                    ordering=solver._ordering,
                    format="csc",
                    phase="solve",
                ),
                bindings=arrays,
            )

        resource = context.resource_once(solver, snapshot, policy=("eigen_factor", self._generation))
        context.native(
            "sparse_direct",
            "solve",
            parameters=dict(rhs_count=1),
            bindings=dict(plan=resource, rhs_0=self.binding_names[0], output_0=self.binding_names[1]),
        )
        context.reconstructed_binding_validators.add(id(self))
        return ()

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=(self._solver, self._matrix),
            debug_info=dict(kind="sparse_direct_solve", implementation="eigen"),
        )
