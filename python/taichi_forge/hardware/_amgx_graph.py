"""Explicit host-controlled AmgX Graph actions and their shared cold recipe."""

from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode
from taichi_forge.graph._native_definition import (
    NativeArrayBinding,
    NativeDefinition,
    NativeOperationBinding,
    NativeResourceBinding,
    NativeResourceReference,
)
from taichi_forge.lang.exception import TaichiRuntimeError


class AmgxGraphNode(NativeGraphNode):
    def __init__(self, binding, update):
        from ._amgx import AmgxDeviceBinding, AmgxSolver

        if type(binding) is not AmgxDeviceBinding or type(binding._solver) is not AmgxSolver:
            raise TypeError("AmgX Graph variants require their own reconstruction contract")
        if type(update) is not bool:
            raise TypeError("update_coefficients must be a bool")
        if update and binding._arrays[2] is None:
            raise ValueError("AmgX coefficient updates require bound values")
        self._binding, self._update = binding, update

    def compile(self):
        return AmgxGraphExecutable(self._binding, self._update)


class AmgxGraphExecutable(NativeGraphExecutable):
    def __init__(self, binding, update):
        self._binding, self._update = binding, update
        self._run = binding.update_and_solve if update else binding.solve
        self.validate_graph_lifetime()

    def validate_graph_lifetime(self):
        if self._binding._closed:
            raise TaichiRuntimeError("AmgX device binding has been closed")
        self._binding._solver._validate_lifetime()

    def run(self):
        return self._run()

    @property
    def lifetime_leases(self):
        binding = self._binding
        return (binding, binding._solver, binding._solver.provider, *tuple(x for x in binding._arrays if x is not None))

    @property
    def resource_effects(self):
        rhs, solution, values = self._binding._arrays
        effects = [
            ResourceEffect(rhs, GraphAccess.READ, runtime_bound=False),
            ResourceEffect(solution, GraphAccess.READ_WRITE, runtime_bound=False),
            ResourceEffect(self._binding._solver, GraphAccess.READ_WRITE, runtime_bound=False),
        ]
        if self._update:
            effects.append(ResourceEffect(values, GraphAccess.READ, runtime_bound=False))
        return tuple(effects)

    @property
    def debug_info(self):
        return dict(kind="amgx_host_solve", update_coefficients=self._update, cuda_capture=False)

    def _export_aot(self, context):
        binding, solver = self._binding, self._binding._solver
        with solver.provider._lock, solver._lock:
            self.validate_graph_lifetime()

            def build_resource():
                # No vendor snapshot API certifies its current coefficients or
                # retained solution. Require an explicit seed for a new session.
                seed = context.resource_inputs.require(solver, ("values",))["values"]
                resource = NativeResourceBinding(solver, solver._definition).lower(context)
                definition = solver._definition.parameters
                arrays = [NativeArrayBinding("values", seed, GraphAccess.READ)]
                for port, data, size in zip(
                    ("row_offsets", "column_indices"),
                    solver._topology,
                    (definition["rows"] + 1, definition["nonzeros"]),
                    strict=True,
                ):
                    name = context.data_resource(object(), data, dtype="i32", shape=[size], readonly=True)
                    arrays.append(NativeArrayBinding(port, name, GraphAccess.READ))
                setup = NativeOperationBinding(
                    NativeDefinition("amgx", "setup_csr", {}),
                    tuple(arrays),
                    (("solver", NativeResourceReference(resource)),),
                )
                context.initialization.extend(context.region(lambda ctx, _: setup.lower(ctx), None))
                return resource

            resource = context.resource_once(solver, build_resource, policy=("amgx", "fresh_session"))
            # An identity local to this solver's reconstructed session, shared
            # by repeated occurrences of the same source binding. Resolve it in
            # prepare; execution never looks up names or touches this context.
            identities = context._native_export_facts.setdefault(("amgx", "bindings"), {})
            if binding not in identities:
                identities[binding] = context._name("amgx_binding")
            definition = binding._definition
            solve = NativeDefinition(
                definition.provider,
                definition.operation,
                dict(definition.parameters, binding_id=identities[binding]),
                definition.version,
            )
            reference = (("solver", NativeResourceReference(resource)),)
            rhs, solution, values = binding._arrays
            if self._update:
                NativeOperationBinding(
                    NativeDefinition("amgx", "replace_coefficients", {}),
                    (NativeArrayBinding("values", values, GraphAccess.READ),),
                    reference,
                ).lower(context)
            NativeOperationBinding(
                solve,
                (
                    NativeArrayBinding("rhs", rhs, GraphAccess.READ),
                    NativeArrayBinding("solution", solution, GraphAccess.READ_WRITE),
                ),
                reference,
            ).lower(context)
