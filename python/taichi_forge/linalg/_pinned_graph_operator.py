"""Symbolic operator ports over an explicitly pinned provider generation."""

from taichi_forge.graph._ir import GraphAccess, NativeCallNode, ResourceEffect
from taichi_forge.graph._native import (
    DispatchGraphAction,
    NativeGraphExecutable,
    NativeGraphNode,
    PreparedGraphBindings,
    ProviderOwnedNdarrayBinding,
)
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.linalg._prepared_operator import _operator_graph_workspace
from taichi_forge.linalg._runtime import (
    _LinearOperatorCompositionGraphExecutable,
    _LinearOperatorGraphExecutable,
    _OperatorGraphPorts,
)


def _qualify_pinned_source(source):
    """Keep derived views and live/writable provider state out of data snapshots."""
    if source.derived_runtime_arg_schema:
        raise TaichiRuntimeError("Pinned symbolic operator derived views require a reconstruction contract")
    pending = [source]
    while pending:
        executable = pending.pop()
        if type(executable) is _LinearOperatorCompositionGraphExecutable:
            pending.extend(executable._children)
        elif type(executable) is _LinearOperatorGraphExecutable:
            if executable._record._graph_dispatches or executable._record._state_dependencies:
                raise TaichiRuntimeError("Pinned symbolic operators currently require from_kernel leaves")
        else:
            raise TaichiRuntimeError("Pinned symbolic operator provider requires an explicit generation contract")


class PinnedOperatorGraphExecutable(_OperatorGraphPorts, NativeGraphExecutable):
    def __init__(self, operator, input_arg, output_arg, adjoint):
        self._init_graph_ports(operator, input_arg, output_arg, adjoint)
        source = operator.graph_action(input_arg, output_arg, adjoint=adjoint).compile()
        _qualify_pinned_source(source)
        prepared = source._bind_provider_generation()
        action = source.recordable_action
        if not prepared.replacements.keys() <= action.fixed_bindings.keys():
            raise TaichiRuntimeError("Pinned operator generation changed its fixed binding schema")
        fixed = dict(action.fixed_bindings)
        fixed.update(prepared.replacements)
        workspace, self._workspace_bytes = _operator_graph_workspace(source)
        for name, requirement in action.temporary_bindings.items():
            fixed[name] = workspace[requirement]
        self._source = source
        self._generation_owners = tuple(prepared.submission_owners)
        self._effects = tuple(source.resource_effects) + tuple(
            ResourceEffect(owner, GraphAccess.READ_WRITE, runtime_bound=False) for owner in workspace.values()
        )
        self._action = DispatchGraphAction(
            action.dispatches,
            backends=action.capabilities.backends,
            conditional_body_safe=action.capabilities.conditional_body_safe,
            fixed_bindings=fixed,
            update_policy="immutable",
        )

    def validate_graph_lifetime(self):
        self._operator._ensure_valid()

    def bind_graph_arguments(self, runtime_args):
        replacements = {}
        self._bind_dense_ports(runtime_args, replacements)
        return PreparedGraphBindings(replacements, self._generation_owners)

    @property
    def runtime_arg_schema(self):
        return self._source.runtime_arg_schema

    @property
    def resource_effects(self):
        return self._effects

    @property
    def recordable_action(self):
        return self._action

    @property
    def graph_ir_node(self):
        if not self._workspace_bytes:
            return self._source.graph_ir_node
        return NativeCallNode(
            name="pinned_operator_composition", effects=self._effects, bindings=self.runtime_arg_schema, opaque=True
        )

    def run(self, runtime_args):
        # Calling the operator's ordinary apply here would select its current
        # generation. This action must instead be lowered into its parent Graph.
        raise TaichiRuntimeError("Pinned operator actions require native Graph recording")

    @property
    def debug_info(self):
        return dict(self._source.debug_info, update_policy="pinned", graph_workspace_bytes=self._workspace_bytes)

    def _export_aot_bindings(self, context):
        from taichi_forge.graph._graph import GraphOwnedNdarray

        self.validate_graph_lifetime()
        for name, owner in self._action.fixed_bindings.items():
            if isinstance(owner, ProviderOwnedNdarrayBinding):
                context.provider_data_binding(name, owner)
            elif type(owner) not in (int, float) and not isinstance(owner, GraphOwnedNdarray):
                raise TaichiRuntimeError("Pinned operator has an unsupported fixed binding")
        for name, extent in self._port_extents():
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
