"""Capture qualified native definitions into the existing deployment lowering.

Only declaration identities cross this boundary. Neither producer arrays nor
their workspace are retained by the exported component.
"""

import json
from types import SimpleNamespace

from taichi_forge.graph._ir import GraphAccess, NativeCallNode, ResourceEffect, RuntimeBinding
from taichi_forge.graph._native import NativeGraphExecutable
from taichi_forge.lang.exception import TaichiCompilationError, TaichiRuntimeError


def capture_sequence(sequence, capture):
    from ._algorithms import PrimitiveSequence

    if type(sequence) is not PrimitiveSequence or sequence._fused_plan is not None:
        raise TaichiCompilationError(
            "Component native capture requires an unfused qualified primitive sequence"
        )
    return capture_calls(tuple(sequence._calls), capture)


def capture_calls(calls, capture):
    from ._sequence_export import _compact, _sort

    if not calls or any(call.kind not in ("sort", "sort_by_key", "compact") for call in calls):
        raise TaichiCompilationError(
            "Component native capture currently qualifies single-key sort and compact sequences"
        )
    for call in calls:
        if (
            call.kind == "sort_by_key"
            and isinstance(call.args[0], (list, tuple))
            and len(call.args[0]) != 1
        ):
            # The ordinary exporter also supports multi-key recipes. Those
            # need their own component resource/operation qualification; only
            # the public single-key spelling shares the existing sort ABI.
            raise TaichiCompilationError(
                "Component native capture requires exactly one sort_by_key key part"
            )
    operations = []
    context = SimpleNamespace(
        retained_arrays=SimpleNamespace(require=capture.native_array), operations=operations
    )
    for call in calls:
        # Reuse the ordinary exporter: exact ordering/NaN policy, native route,
        # packed storage, dtype, workspace and selected strategy validation.
        (_compact if call.kind == "compact" else _sort)(call, context)
    return _ComponentSortDefinition(operations)


def _capture_context(capture, operations):
    def native(provider, operation, *, bindings, parameters=None, version=1):
        operations.append(dict(kind="native", provider=provider, operation=operation,
                               version=version, parameters=parameters or {}, bindings=bindings))

    return SimpleNamespace(retained_arrays=SimpleNamespace(require=capture.native_array), native=native)


def capture_prepared(plan, capture):
    from ._prepared_export import export_prepared

    operations = []
    export_prepared(plan, _capture_context(capture, operations))
    return _ComponentSortDefinition(operations)


def capture_recording(recording, capture):
    from taichi_forge.lang import impl

    if impl.get_runtime().prog is not recording._program:
        raise TaichiCompilationError("Persistent component primitive belongs to another runtime")
    # The immutable recording keeps this definition after the source plan is
    # closed. Only declared resource symbols survive, not its native command,
    # producer workload dimensions or private recording scratch.
    operations = []
    recording._binding.lower(_capture_context(capture, operations))
    return _ComponentSortDefinition(operations)


class _ComponentSortDefinition(NativeGraphExecutable):
    def __init__(self, operations):
        from taichi_forge.aot._component import _digest
        from taichi_forge_runtime.aot._component_native import operation_bindings

        self._operations = json.dumps(operations, sort_keys=True)
        self.semantic_key = _digest(operations)
        self._names = tuple(
            sorted({value for op in operations for _, value in operation_bindings(op)})
        )

    @property
    def control_kind(self):
        return "native_sort" if all(op["kind"] == "sort" for op in self.operations) else "native_primitives"

    @property
    def operations(self):
        return json.loads(self._operations)

    @property
    def runtime_arg_schema(self):
        return tuple(RuntimeBinding(name, "ndarray") for name in self._names)

    @property
    def resource_effects(self):
        return tuple(ResourceEffect(name, GraphAccess.READ_WRITE) for name in self._names)

    @property
    def graph_ir_node(self):
        return NativeCallNode(
            name="component_sort_" + self.semantic_key,
            bindings=self.runtime_arg_schema,
            effects=self.resource_effects,
            opaque=False,
            semantic_fingerprint=self.semantic_key,
        )

    def _export_aot(self, context):
        from taichi_forge_runtime.aot._component_native import map_operation_bindings

        # Length and scratch are resolved by existing Runtime preparation
        # from this invocation's arrays, never from the producer's shape.
        for operation in self.operations:
            context.operations.append(
                map_operation_bindings(operation, lambda value: context.binding(value, resource=True))
            )

    def run(self, runtime_args=None):
        raise TaichiRuntimeError(
            "A component native definition must execute through its exported instance plan"
        )
