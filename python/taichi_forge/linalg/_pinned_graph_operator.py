"""Symbolic operator ports over an explicitly pinned provider generation."""

from taichi_forge.graph._native import DispatchGraphAction, NativeGraphNode, ProviderOwnedNdarrayBinding
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.linalg._runtime import _LinearOperatorGraphExecutable


class PinnedOperatorGraphExecutable(_LinearOperatorGraphExecutable):
    def __init__(self, operator, input_arg, output_arg, adjoint):
        if operator._composition_spec is not None or not operator._handle._supports_recordable_kernel():
            raise TaichiRuntimeError("Pinned symbolic operator actions currently require a from_kernel provider")
        if adjoint and not operator.capabilities.adjoint_apply:
            raise TaichiRuntimeError("LinearOperator provider does not expose an adjoint action")
        super().__init__(operator, input_arg, output_arg, adjoint)
        if self._record._graph_dispatches or self._record._state_dependencies:
            raise TaichiRuntimeError("Pinned symbolic operator actions currently require a from_kernel provider")
        self._action = DispatchGraphAction(
            self._action.dispatches,
            backends=self._action.capabilities.backends,
            conditional_body_safe=True,
            fixed_bindings=self._action.fixed_bindings,
            update_policy="immutable",
        )

    def _current_compatible_record(self):
        self._operator._ensure_valid()
        return self._record

    def run(self, runtime_args):
        # Calling the operator's ordinary apply here would select its current
        # generation. This action must instead be lowered into its parent Graph.
        raise TaichiRuntimeError("Pinned operator actions require native Graph recording")

    @property
    def debug_info(self):
        return dict(super().debug_info, update_policy="pinned")

    def _export_aot_bindings(self, context):
        self.validate_graph_lifetime()
        for name, owner in self._action.fixed_bindings.items():
            if isinstance(owner, ProviderOwnedNdarrayBinding):
                context.provider_data_binding(name, owner)
            elif type(owner) is not int:
                raise TaichiRuntimeError("Pinned operator has an unsupported fixed binding")
        for name, extent in (
            (self._input_name, self._operator.shape[0 if self._adjoint else 1]),
            (self._output_name, self._operator.shape[1 if self._adjoint else 0]),
        ):
            context.binding_constraints.append(
                dict(
                    kind="array_layout",
                    array=context.binding(name),
                    exact_shape=[extent],
                    element_shape=[],
                    record_bytes=4,
                    alignment=4,
                )
            )
        context.binding_constraints.append(
            dict(kind="disjoint", left=context.binding(self._input_name), right=context.binding(self._output_name))
        )
        context.reconstructed_argument_binders.add(id(self))
        context.reconstructed_binding_validators.add(id(self))


class PinnedOperatorGraphNode(NativeGraphNode):
    def __init__(self, operator, input_arg, output_arg, adjoint):
        self._arguments = (operator, input_arg, output_arg, adjoint)

    def compile(self):
        return PinnedOperatorGraphExecutable(*self._arguments)
